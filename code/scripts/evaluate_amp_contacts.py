"""Measure actual GPU solver contacts during AMP rollout, before environment resets.

Contact quantities are sampled at every 5 ms physics substep from the solver's
pre-integration contact geometry and cvel, immediately after that step returns.
They are not reconstructed by replaying poses or inferred from ankle height.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from amp_gait_metrics import inspect_arrays
from evaluate_amp_gait_robustness import summarize_trials
from supermimic import networks
from supermimic.amp_eval import evaluate_command, make_amp_eval_env, parse_commands
from supermimic.amp_train import load_amp_actor
from supermimic.motion_lib import MotionLibrary


def point_velocity(cvel, point, reference):
    """World point velocity from MuJoCo's angular:linear, root-COM-based cvel."""
    return cvel[..., 3:] + torch.linalg.cross(cvel[..., :3], point - reference, dim=-1)


def tangent_speed(relative_velocity, normal):
    tangent = relative_velocity - (relative_velocity * normal).sum(-1, keepdim=True) * normal
    return tangent.norm(dim=-1)


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return dict(mean=float(values.mean()), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)), max=float(values.max()))


class ContactRecorder:
    """Read-only sim.step wrapper; keeps the first original episode only."""
    def __init__(self, env, point_force_threshold=1., foot_force_threshold=5.):
        self.env, self.sim = env, env.sim
        self.point_threshold, self.foot_threshold = point_force_threshold, foot_force_threshold
        self.dt = float(env.mj_model.opt.timestep)
        self.n = env.num_envs
        m, d = env.mj_model, self.sim.wp_data
        self.wp_device = self.sim.wp_device
        self.contact_ids = wp.array(np.arange(d.naconmax, dtype=np.int32), dtype=wp.int32, device=self.wp_device)
        self.forces = wp.zeros(d.naconmax, dtype=wp.spatial_vector, device=self.wp_device)
        self.force_tensor = wp.to_torch(self.forces)
        self.ids = wp.to_torch(self.contact_ids)
        self.contacts = {name: wp.to_torch(getattr(d.contact, name)) for name in
                         ('geom', 'worldid', 'pos', 'frame', 'dist', 'efc_address')}
        self.nacon = wp.to_torch(d.nacon)
        self.cvel = wp.to_torch(d.cvel)
        self.com = wp.to_torch(d.subtree_com)
        self.torque = wp.to_torch(d.qfrc_actuator)
        self.geom_body = torch.as_tensor(m.geom_bodyid.copy(), device=env.device, dtype=torch.long)
        self.body_root = torch.as_tensor(m.body_rootid.copy(), device=env.device, dtype=torch.long)
        foot = np.full(m.ngeom, -1, np.int64)
        for geom in range(m.ngeom):
            name = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom) or '').split('/')[-1]
            match = re.fullmatch(r'(left|right)_foot[1-7]_collision', name)
            if match:
                foot[geom] = 0 if match.group(1) == 'left' else 1
        if (foot >= 0).sum() != 14:
            raise ValueError('Expected 14 G1 foot collision geoms')
        self.foot_side = torch.as_tensor(foot, device=env.device)
        self.ground_geom = torch.as_tensor((m.body_weldid[m.geom_bodyid] == 0)
            & (m.geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE)), device=env.device)
        if not bool(self.ground_geom.any()):
            raise ValueError('Expected a fixed plane ground geom')
        self.old_step = self.sim.step
        self.sim.step = self.step
        self.begin()

    def begin(self):
        self.active = torch.ones(self.n, dtype=torch.bool, device=self.env.device)
        self.records = []
        self.point_samples = []
        self.substep = 0

    def close(self):
        self.sim.step = self.old_step

    def step(self):
        self.old_step()
        self.capture()

    @torch.no_grad()
    def capture(self):
        d = self.sim.wp_data
        with wp.ScopedDevice(self.wp_device):
            mjwarp.contact_force(self.sim.wp_model, d, self.contact_ids, False, self.forces)
        c = self.contacts
        geom = c['geom'].long()
        safe_geom = geom.clamp(0, len(self.geom_body) - 1)
        bodies = self.geom_body[safe_geom]
        sides = self.foot_side[safe_geom]
        ground = self.ground_geom[safe_geom]
        side = torch.where((sides[:, 0] >= 0) & ground[:, 1], sides[:, 0],
                           torch.where((sides[:, 1] >= 0) & ground[:, 0], sides[:, 1], -1))
        world = c['worldid'].long()
        safe_world = world.clamp(0, self.n - 1)
        valid = ((self.ids < self.nacon[0]) & (side >= 0) & (geom >= 0).all(-1)
                 & (world >= 0) & (world < self.n) & (c['efc_address'][:, 0] >= 0)
                 & self.active[safe_world])
        selected = valid.nonzero().flatten()
        wi, si = world[selected], side[selected]
        body0, body1 = bodies[selected].unbind(-1)
        p = c['pos'][selected]
        v0 = point_velocity(self.cvel[wi, body0], p, self.com[wi, self.body_root[body0]])
        v1 = point_velocity(self.cvel[wi, body1], p, self.com[wi, self.body_root[body1]])
        speed = tangent_speed(v1 - v0, c['frame'][selected, 0])
        fn = self.force_tensor[selected, 0].clamp_min(0)
        ft = self.force_tensor[selected, 1:3].norm(dim=-1)
        penetration = (-c['dist'][selected]).clamp_min(0)
        values = torch.stack((wi.float(), si.float(), fn, ft, speed, penetration), -1).cpu().numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError('Non-finite actual contact force, slip, or penetration')
        active = self.active.cpu().numpy().copy()
        # Keep per-foot totals even when the force is below the loaded-point threshold.
        row = np.zeros((self.n, 2, 8), dtype=np.float32)
        if len(values):
            world_np, side_np = values[:, 0].astype(int), values[:, 1].astype(int)
            for column, source in ((0, 2), (1, 3)):
                np.add.at(row[..., column], (world_np, side_np), values[:, source])
            loaded = values[:, 2] > self.point_threshold
            w, s, f, slip, pen = world_np[loaded], side_np[loaded], values[loaded, 2], values[loaded, 4], values[loaded, 5]
            np.add.at(row[..., 2], (w, s), f)
            np.add.at(row[..., 3], (w, s), f * slip)
            np.add.at(row[..., 4], (w, s), slip)
            np.add.at(row[..., 5], (w, s), 1.)
            np.maximum.at(row[..., 6], (w, s), pen)
            np.maximum.at(row[..., 7], (w, s), slip)
            if loaded.any():
                # world, foot, solver time, normal force, tangent slip, penetration.
                self.point_samples.append(np.column_stack((w, s, np.full(len(w), self.substep * self.dt), f, slip, pen)).astype(np.float32))
        torques = self.torque[:, self.env.qd_adr].abs().max(-1).values.cpu().numpy()
        if not np.isfinite(torques[active]).all():
            raise FloatingPointError('Non-finite active actuator torque')
        self.records.append((row, active, torques))
        self.substep += 1

    def observe(self, step, state, info, active):
        # Called after all four physics snapshots, before the next control action.
        self.active = active & ~info['done']

    def finish(self, output, skip_seconds=2.):
        data = np.stack([r[0] for r in self.records])
        active = np.stack([r[1] for r in self.records])
        torques = np.stack([r[2] for r in self.records])
        times = np.arange(len(data)) * self.dt
        points = np.concatenate(self.point_samples) if self.point_samples else np.empty((0, 6), np.float32)
        output = Path(output)
        np.savez_compressed(output, time_s=times, foot_metrics=data, active_first_episode=active,
                            max_abs_joint_actuator_torque_nm=torques, contact_points=points,
                            foot_names=np.array(['left', 'right']),
                            foot_metric_columns=np.array(['normal_force_N', 'sum_tangential_force_norm_N',
                                'loaded_normal_force_N', 'force_times_slip_N_mps', 'point_slip_sum_mps',
                                'loaded_point_count', 'max_loaded_penetration_m', 'max_loaded_slip_mps']),
                            contact_point_columns=np.array(['world', 'foot', 'time_s', 'normal_force_N', 'slip_mps', 'penetration_m']))
        def make(mask, point_mask):
            usable = active & mask[:, None]
            selected_points = points[point_mask]
            rows = []
            for world in range(self.n):
                valid = usable[:, world]
                observed = valid.sum()
                foot_rows = []
                for side in range(2):
                    arr = data[:, world, side]
                    stance = arr[:, 0] > self.foot_threshold
                    pp = selected_points[(selected_points[:, 0] == world) & (selected_points[:, 1] == side)]
                    adjacent = valid[1:] & valid[:-1]
                    starts = int(((~stance[:-1]) & stance[1:] & adjacent).sum())
                    ends = int((stance[:-1] & (~stance[1:]) & adjacent).sum())
                    load = float(arr[valid, 2].sum())
                    foot_rows.append(dict(foot=('left', 'right')[side],
                        observed_seconds=float(observed * self.dt),
                        stance_fraction=float(stance[valid].mean()) if observed else None,
                        contact_onsets=starts, contact_offsets=ends,
                        contact_onsets_per_second=starts / (observed * self.dt) if observed else None,
                        normal_impulse_Ns=float(arr[valid, 0].sum() * self.dt),
                        normal_force_N=summary(arr[valid, 0]),
                        loaded_point_slip_mps=summary(pp[:, 4]),
                        force_weighted_slip_mps=float(arr[valid, 3].sum() / load) if load else None,
                        loaded_penetration_m=summary(pp[:, 5])))
                rows.append(dict(copy=world, feet=foot_rows,
                                 max_abs_joint_actuator_torque_nm=summary(torques[valid, world])))
            total_load = float(selected_points[:, 3].sum())
            stance = data[..., 0] > self.foot_threshold
            return dict(observed_world_seconds=float(usable.sum() * self.dt),
                        loaded_contact_point_samples=len(selected_points),
                        loaded_point_slip_mps=summary(selected_points[:, 4]),
                        force_weighted_slip_mps=float((selected_points[:, 3] * selected_points[:, 4]).sum() / total_load) if total_load else None,
                        loaded_penetration_m=summary(selected_points[:, 5]),
                        both_feet_stance_fraction=float(stance.all(-1)[usable].mean()) if usable.any() else None,
                        neither_foot_stance_fraction=float((~stance.any(-1))[usable].mean()) if usable.any() else None,
                        normal_impulse_per_copy_Ns=float(data[..., 0][usable].sum() * self.dt / self.n),
                        max_abs_joint_actuator_torque_nm=summary(torques[usable]), trials=rows)
        return dict(trace=output.name, physics_dt_seconds=self.dt,
                    loaded_point_force_threshold_N=self.point_threshold, foot_stance_force_threshold_N=self.foot_threshold,
                    full_episode=make(np.ones(len(times), bool), np.ones(len(points), bool)),
                    after_transient=make(times >= skip_seconds, points[:, 2] >= skip_seconds),
                    transient_excluded_seconds=skip_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--motions', default='data/processed/amp_walk_50hz.npz')
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--commands', default='0.3,0,0;0.5,0,0;0.8,0,0')
    parser.add_argument('--seed', type=int, default=59)
    parser.add_argument('--perturbation', type=float, default=1.)
    parser.add_argument('--skip-seconds', type=float, default=2.)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.num_envs < 1 or not np.isfinite(args.seconds) or args.seconds <= 0 or not 0 <= args.skip_seconds < args.seconds:
        parser.error('Need positive count/duration and 0 <= skip-seconds < seconds')
    if not np.isfinite(args.perturbation) or args.perturbation < 0:
        parser.error('Invalid perturbation scale')
    if (args.out / 'summary.json').exists():
        parser.error('Output summary already exists; choose a fresh directory')
    args.out.mkdir(parents=True, exist_ok=True)
    networks.AMP['enabled'] = False
    actor, cfg, ck = load_amp_actor(args.ckpt, args.device)
    lib = MotionLibrary(args.motions, args.device)
    env = make_amp_eval_env(lib, cfg, args.num_envs, args.device, args.seconds)
    recorder = ContactRecorder(env)
    results = []
    feet = [env.task.kin.body_names.index(f'{side}_ankle_roll_link') for side in ('left', 'right')]
    try:
        for i, command in enumerate(parse_commands(args.commands)):
            torch.manual_seed(args.seed)
            recorder.begin()
            gait_records = {key: [] for key in ('ankle_pos_w', 'ankle_vel_w', 'action', 'velocity_heading', 'yaw_rate')}
            last_action = None
            def policy(observation):
                nonlocal last_action
                last_action = actor(observation)
                return last_action
            def observe(step, state, info, active):
                recorder.observe(step, state, info, active)
                positions, _, velocities, _ = env.task.body_kin(state)
                values = dict(ankle_pos_w=positions[:, feet], ankle_vel_w=velocities[:, feet],
                              action=last_action, velocity_heading=info['velocity_heading'], yaw_rate=info['yaw_rate'])
                for key, value in values.items():
                    gait_records[key].append(value.cpu().numpy().copy())
            report = evaluate_command(policy, env, command, round(args.seconds / env.tcfg.dt),
                                      observer=observe, reset_perturbation=args.perturbation)
            report['contacts'] = recorder.finish(args.out / f'command_{i:02d}_contacts.npz', args.skip_seconds)
            gait_records = {key: np.stack(value) for key, value in gait_records.items()}
            gait_dir = args.out / f'command_{i:02d}_gait'
            gait_dir.mkdir(exist_ok=True)
            for trial in report['trials']:
                copy, length = trial['copy'], trial['observed_steps']
                trace = {key: value[:length, copy] for key, value in gait_records.items()}
                trace.update(time_s=(np.arange(length) + 1) * env.tcfg.dt, dt=env.tcfg.dt,
                             command=np.array(command), ankle_names=np.array(['left', 'right']))
                name = f'copy_{copy:02d}_trace.npz'
                np.savez_compressed(gait_dir / name, **trace)
                trial['gait_trace'] = f'{gait_dir.name}/{name}'
                trial['gait_after_2s'] = inspect_arrays(trace, skip_seconds=2.)
                trial['gait_final_half'] = inspect_arrays(trace, skip_seconds=max(2., args.seconds / 2))
            report['style_observations'] = summarize_trials(report['trials'])
            results.append(report)
            contact = report['contacts']['after_transient']
            print(f"command={command} survived={report['survived']}/{args.num_envs} "
                  f"force_weighted_contact_slip={contact['force_weighted_slip_mps']}", flush=True)
    finally:
        recorder.close()
    result = dict(schema='amp_actual_solver_contact_eval_v1', checkpoint=str(args.ckpt.resolve()),
                  checkpoint_iteration=int(ck['it']), checkpoint_sha256=hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
                  training_config=vars(cfg), seconds=args.seconds, num_envs=args.num_envs, reset_seed=args.seed,
                  reset_perturbation_scale=args.perturbation, friction=1., observation_noise=False,
                  domain_randomization=False, deterministic_policy=True,
                  protocol={
                      'sampling': 'Every 5 ms physics solve, contact/cvel/constraint force at its pre-integration state, read immediately after sim.step and before episode reset',
                      'solver': 'Existing GPU MuJoCo-Warp AMPEnv; unchanged physics/control',
                      'integrator': mujoco.mjtIntegrator(env.mj_model.opt.integrator).name,
                      'contact_events': 'Onsets/offsets are crossings of the 5 N foot-load threshold, not gait step frequency',
                      'force': 'Official mujoco_warp.contact_force in contact frame; normal index 0; tangent indices 1:3',
                      'slip': 'Relative velocities of both rigid bodies at actual contact midpoint, projected into contact tangent plane; includes rotation around root subtree COM',
                      'contact_selection': '14 G1 left/right foot collision geoms against fixed plane geoms welded to world with active efc address; loaded point Fn > 1 N, foot stance total Fn > 5 N',
                      'episode_mask': 'First original episode only, terminal control interval included, subsequent reset episodes excluded',
                      'limitations': 'Simulator soft-contact force and signed distance; point sample percentiles weight contact multiplicity; force-weighted slip also provided; no real-hardware claim',
                  }, commands=results)
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Saved {args.out / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
