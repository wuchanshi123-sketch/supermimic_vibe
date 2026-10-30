"""Compare standing and time-scaled forward-motion starts at one fixed command.

This is an initialization diagnostic, not a training curriculum. Each copy is
recorded only through its original episode's first termination.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from supermimic import networks
from supermimic.amp_eval import make_amp_eval_env
from supermimic.amp_train import load_amp_actor, task_config
from supermimic.motion_lib import MotionLibrary
from supermimic.rotations import heading_mat, mtv


FORWARD_CLIPS = ("walk_forward_loop_002__A022", "walk_forward_loop_002__A024")


def diverse_indices(features, count):
    """Deterministic farthest-point selection avoids repeated similar gait phases."""
    if count < 1 or count > len(features):
        raise ValueError("Need enough eligible, distinct frames")
    normalized = (features - features.mean(0)) / features.std(0, unbiased=False).clamp_min(.05)
    chosen, distance = [], torch.full((len(features),), float("inf"), device=features.device)
    index = 0
    for _ in range(count):
        chosen.append(index)
        distance = torch.minimum(distance, (normalized - normalized[index]).square().mean(-1))
        distance[chosen] = -1
        index = int(distance.argmax())
    return torch.tensor(chosen, device=features.device)


@torch.no_grad()
def select_frames(lib, joint_names, count):
    legs = [i for i, name in enumerate(joint_names) if any(part in name for part in ("hip", "knee", "ankle"))]
    if not legs:
        raise ValueError("No leg joints found")
    selected, metadata = [], []
    for clip_index, name in enumerate(FORWARD_CLIPS):
        clip = lib.names.index(name)
        start, length = int(lib.starts[clip]), int(lib.lengths[clip])
        frames = torch.arange(start, start + length - 1, device=lib.device)
        state = lib.frame(frames)
        forward = mtv(heading_mat(state["rot"]), state["v"])[:, 0]
        leg_rms = state["qd"][:, legs].square().mean(-1).sqrt()
        eligible = (forward > .5) & (leg_rms > .3)
        candidates = frames[eligible]
        features = torch.cat((state["q"][eligible][:, legs], state["qd"][eligible][:, legs]), -1)
        needed = count // 2 + (clip_index < count % 2)
        chosen = diverse_indices(features, needed)
        selected.append(candidates[chosen])
        for index in chosen.tolist():
            frame = int(candidates[index])
            metadata.append({"clip": name, "frame": frame, "clip_frame": frame - start,
                             "unscaled_root_heading_forward_mps": float(forward[eligible][index]),
                             "unscaled_leg_velocity_rms_radps": float(leg_rms[eligible][index])})
    return torch.cat(selected), metadata


@torch.no_grad()
def reset_scaled_reference(env, frames, scale):
    """Retain normal reset bookkeeping, then replace only physical velocities."""
    ids = torch.arange(env.num_envs, device=env.device)
    original_probability = env.rsi_probability
    try:
        env.rsi_probability = 1.
        env.reset_envs(ids, frames=frames)
    finally:
        env.rsi_probability = original_probability
    reference = env.lib.frame(frames)
    env.write_state(ids, reference["pos"], reference["quat"], reference["q"],
                    reference["v"] * scale, reference["w"] * scale, reference["qd"] * scale)
    env._finish_resets(ids, force=True)


def gait_metrics(position, velocity, heading_velocity, yaw, dt, command):
    if not len(position):
        return {"samples": 0}
    height = position[..., 2]
    speed = np.linalg.norm(velocity[..., :2], axis=-1)
    low = height < .05
    result = {
        "samples": len(height), "seconds": len(height) * dt,
        "height_p95_minus_p05_m": (np.quantile(height, .95, axis=0) - np.quantile(height, .05, axis=0)).tolist(),
        "both_ankles_low_fraction": float(low.all(-1).mean()),
        "one_ankle_low_other_high_fraction": float(((low[:, 0] & (height[:, 1] > .06))
                                                   | (low[:, 1] & (height[:, 0] > .06))).mean()),
        "low_ankle_planar_speed_mps": float(speed[low].mean()) if low.any() else None,
        "mean_velocity_heading_mps": heading_velocity.mean(0).tolist(),
        "planar_rmse_mps": float(np.sqrt(np.square(heading_velocity[:, :2] - command[:2]).sum(-1).mean())),
        "mean_abs_yaw_error_radps": float(np.abs(yaw - command[2]).mean()),
        "height_dominant_hz": None, "height_left_right_phase_degrees": None,
    }
    if len(height) * dt >= 1. and np.std(height, axis=0).min() > 1e-5:
        time = np.linspace(-1, 1, len(height))
        detrended = height - height.mean(0) - time[:, None] * (time @ height / (time @ time))[None]
        spectrum = np.fft.rfft(detrended * np.hanning(len(height))[:, None], axis=0)
        frequencies = np.fft.rfftfreq(len(height), dt)
        valid = np.flatnonzero((frequencies >= .25) & (frequencies <= 12.))
        peak = valid[np.argmax((np.abs(spectrum[valid]) ** 2).sum(-1))]
        result["height_dominant_hz"] = float(frequencies[peak])
        result["height_left_right_phase_degrees"] = float(np.degrees(np.angle(spectrum[peak, 0] * spectrum[peak, 1].conj())))
    return result


@torch.no_grad()
def rollout(actor, env, seconds, output, label, provenance):
    steps = max(1, round(seconds / env.tcfg.dt))
    feet = [env.task.kin.body_names.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    fell = torch.zeros_like(alive)
    truncated = torch.zeros_like(alive)
    keys = ("pos", "rot", "q", "v", "w", "qd", "action", "ankle_pos_w", "ankle_vel_w", "velocity_heading", "yaw_rate", "valid")
    records = {key: [] for key in keys}
    initial = {key: value.cpu().numpy().copy() for key, value in env.state.items()}
    for _ in range(steps):
        action = actor(env.policy_inputs())
        state, info = env.step(action)
        positions, _, velocities, _ = env.task.body_kin(state)
        values = {**state, "action": action, "ankle_pos_w": positions[:, feet],
                  "ankle_vel_w": velocities[:, feet], "velocity_heading": info["velocity_heading"],
                  "yaw_rate": info["yaw_rate"], "valid": alive}
        for key in keys:
            value = values[key]
            if key != "valid":
                if not bool(torch.isfinite(value[alive]).all()):
                    raise FloatingPointError(f"Non-finite active state: {key}")
                mask = alive.reshape((len(alive),) + (1,) * (value.ndim - 1))
                value = torch.where(mask, value, float("nan"))
            records[key].append(value.cpu().numpy().copy())
        fell |= info["terminated"] & alive
        truncated |= info["truncated"] & alive
        alive &= ~info["done"]
        if not bool(alive.any()):
            break
    records = {key: np.stack(value) for key, value in records.items()}
    copies = []
    for i in range(env.num_envs):
        length = int(records["valid"][:, i].sum())
        trace = {key: value[:length, i] for key, value in records.items() if key != "valid"}
        time = (np.arange(length) + 1) * env.tcfg.dt
        trace_name = f"{label}_copy_{i:02d}_trace.npz"
        np.savez_compressed(output / trace_name, **trace, time_s=time, dt=env.tcfg.dt,
                            command=env.fixed_command.cpu().numpy(), joint_names=np.asarray(env.joint_names),
                            **{"initial_" + key: value[i] for key, value in initial.items()})
        periods = {}
        for name, low, high in (("first_2s", 0., 2.), ("after_2s", 2., seconds),
                                ("2_to_5s", 2., 5.), ("5_to_10s", 5., 10.), ("after_10s", 10., seconds)):
            select = (time > low) & (time <= high)
            periods[name] = gait_metrics(trace["ankle_pos_w"][select], trace["ankle_vel_w"][select],
                                         trace["velocity_heading"][select], trace["yaw_rate"][select],
                                         env.tcfg.dt, env.fixed_command.cpu().numpy())
        copies.append({"copy": i, "initialization": provenance[i], "survival_seconds": length * env.tcfg.dt,
                       "fell": bool(fell[i]), "truncated": bool(truncated[i]),
                       "survived_full_horizon": bool(alive[i]), "trace": trace_name, "periods": periods})
    print(f"{label}: survived={int(alive.sum())}/{env.num_envs}; falls={int(fell.sum())}", flush=True)
    return {"initialization": label, "survived": int(alive.sum()), "falls": int(fell.sum()), "copies": copies}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--motions", type=Path, default=Path("data/processed/amp_walk_50hz.npz"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--speed", type=float, default=.3)
    parser.add_argument("--time-scale", type=float, default=.4)
    parser.add_argument("--seed", type=int, default=103)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.num_envs < 2 or not math.isfinite(args.seconds) or args.seconds <= 0
            or not math.isfinite(args.speed) or not 0 < args.time_scale <= 1):
        parser.error("Need at least two copies, positive duration, finite speed, and time scale in (0,1]")
    networks.AMP["enabled"] = False
    torch.manual_seed(args.seed)
    actor, cfg, checkpoint = load_amp_actor(args.ckpt, args.device)
    lib = MotionLibrary(str(args.motions), args.device)
    if abs(lib.dt - task_config(cfg).dt) > 1e-8:
        parser.error("Motion/control intervals differ")
    env = make_amp_eval_env(lib, cfg, args.num_envs, args.device, args.seconds)
    env.fixed_command = torch.tensor((args.speed, 0., 0.), device=args.device)
    frames, provenance = select_frames(lib, env.joint_names, args.num_envs)
    args.out.mkdir(parents=True, exist_ok=True)
    env.reset()
    standing = rollout(actor, env, args.seconds, args.out, "standing", ["nominal standing"] * args.num_envs)
    env.reset()
    reset_scaled_reference(env, frames, args.time_scale)
    expert = rollout(actor, env, args.seconds, args.out, "scaled_expert", provenance)
    report = {"checkpoint": str(args.ckpt), "iteration": int(checkpoint["it"]), "command": [args.speed, 0., 0.],
              "seconds": args.seconds, "seed": args.seed, "reference_velocity_time_scale": args.time_scale,
              "protocol": "No DR/noise; reference pose unchanged; only v/w/qd scaled; first termination included; resets excluded",
              "observation_history": "Reset history repeats current observation; previous actions zero for both initializations",
              "gait_statistics": "Ankle-origin height proxies, not measured contact; frequency/phase estimates use detrended Hann spectra",
              "results": [standing, expert]}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
