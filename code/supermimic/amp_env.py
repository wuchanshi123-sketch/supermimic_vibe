"""Flat-ground command locomotion; reference clips are used only for reset states."""

from __future__ import annotations

import torch

from .amp_task import AMPTask, AMPTaskCfg, COMMAND_LO, COMMAND_HI
from .rotations import heading_mat, mtv, quat_from_euler_xyz
from .sim_env import EnvCfg, TrackingSimEnv
from .task import TaskCfg
from mjlab.managers.event_manager import RecomputeLevel


class AMPEnv(TrackingSimEnv):
    def __init__(
        self, lib, cfg: EnvCfg, device, task_cfg=None, gamma=0.99,
        rsi_probability=0.5, stand_height=0.76, command_seconds=(3.0, 8.0),
        command_low=COMMAND_LO, command_high=COMMAND_HI, standing_probability=0.05,
    ):
        ac = task_cfg or AMPTaskCfg()
        if not 0 <= rsi_probability <= 1 or not 0 <= standing_probability <= 1:
            raise ValueError("Reset and standing-command probabilities must be in [0,1]")
        if command_seconds[0] <= 0 or command_seconds[1] < command_seconds[0]:
            raise ValueError("Invalid command duration range")
        super().__init__(lib, cfg, TaskCfg(dt=ac.dt, history=ac.history), device, gamma)
        self.amp_task = AMPTask(self.task, ac, mj_model=self.mj_model)
        self.rsi_probability = rsi_probability
        self.stand_height = stand_height
        self.command_seconds = command_seconds
        self.standing_probability = standing_probability
        self.command_low = torch.tensor(command_low, device=device)
        self.command_high = torch.tensor(command_high, device=device)
        if bool((self.command_low > self.command_high).any()):
            raise ValueError("Command lower bounds exceed upper bounds")
        self.command = torch.zeros(self.num_envs, 3, device=device)
        self.command_age = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.command_duration = torch.ones_like(self.command_age)
        self.fixed_command = None
        self.obs_hist = torch.zeros(
            self.num_envs, ac.history, self.amp_task.obs_dim, device=device
        )

    def _sample_start_frames(self, k):
        # Uniform clip weighting, independent of tracking return or reference phase.
        clip = torch.randint(self.lib.num_clips, (k,), device=self.device)
        offset = (torch.rand(k, device=self.device) * self.lib.lengths[clip]).long()
        frames = self.lib.starts[clip] + offset
        return frames, self.lib.bin_of_frame[frames]

    def _resample_commands(self, ids):
        n = len(ids)
        if not n:
            return
        if self.fixed_command is not None:
            fixed = torch.as_tensor(self.fixed_command, device=self.device, dtype=torch.float32)
            self.command[ids] = fixed if fixed.ndim == 1 else fixed[ids]
        else:
            self.command[ids] = self.command_low + torch.rand(n, 3, device=self.device) * (
                self.command_high - self.command_low
            )
            standing = torch.rand(n, device=self.device) < self.standing_probability
            self.command[ids[standing]] = 0
        lo, hi = self.command_seconds
        self.command_duration[ids] = (
            (lo + (hi - lo) * torch.rand(n, device=self.device)) / self.tcfg.dt
        ).long().clamp_min(1)
        self.command_age[ids] = 0

    def reset_envs(self, env_ids, frames=None):
        super().reset_envs(env_ids, frames)
        # Mix reference-state initialization and a nominal standing pose.
        # This initializes physical states, never actor or world-model weights.
        standing = env_ids[
            torch.rand(len(env_ids), device=self.device) >= self.rsi_probability
        ]
        if len(standing):
            n = len(standing)
            p = torch.zeros(n, 3, device=self.device)
            p[:, 2] = self.stand_height
            zero = torch.zeros(n, device=self.device)
            yaw = torch.zeros_like(zero) if self.eval_mode else (torch.rand_like(zero) * 2 - 1) * torch.pi
            quat = quat_from_euler_xyz(zero, zero, yaw)
            q = self.q0.expand(n, -1).clone()
            if self.cfg.reset_noise and not self.eval_mode:
                q += torch.empty_like(q).uniform_(-0.02, 0.02)
                p[:, 2] += torch.empty_like(zero).uniform_(-0.01, 0.01)
            q = torch.minimum(torch.maximum(q, self.task.qmin), self.task.qmax)
            z3 = torch.zeros(n, 3, device=self.device)
            self.write_state(standing, p, quat, q, z3, z3, torch.zeros_like(q))
        self._resample_commands(env_ids)

    def _finish_resets(self, env_ids, force=False):
        self._steps_since_const += 1
        if self.cfg.domain_rand and self._need_set_const and (
            force or self._steps_since_const >= self.cfg.set_const_every
        ):
            self.sim.recompute_constants(RecomputeLevel.set_const)
            self._need_set_const = False
            self._steps_since_const = 0
        self.state = self.read_state()
        if len(env_ids):
            s = {k: v[env_ids] for k, v in self.state.items()}
            o = self.amp_task.observation(
                s, self.prev_action[env_ids], self.command[env_ids], self.xi[env_ids]
            )
            if self.cfg.obs_noise and not self.eval_mode:
                o = o + self.amp_task.sample_noise((len(env_ids),))
            self.obs_hist[env_ids] = o[:, None].expand(-1, self.amp_task.cfg.history, -1)

    def policy_inputs(self):
        return self.obs_hist.flatten(1)

    def step(self, clean_action, executed_action=None):
        """Return pre-reset state; the discriminator must use this successor."""
        executed = clean_action if executed_action is None else executed_action
        command = self.command.clone()
        previous = self.prev_action.clone()
        xi = self.xi.clone()
        self.sim.data.ctrl[:] = (
            self.q0 + self.action_scale * executed - self.enc_bias
        )[:, self.act_to_joint]
        for _ in range(self.decimation):
            self.sim.step()
        s_next = self.read_state()
        kin = self.kin_c(
            s_next["pos"], s_next["rot"], s_next["q"],
            s_next["v"], s_next["w"], s_next["qd"],
        )
        task_reward, terms = self.amp_task.reward(
            s_next, clean_action, previous, command, kin=kin
        )
        hard, _ = self.amp_task.termination(s_next)
        self.ep_len += 1
        timeout = self.ep_len >= self.max_ep_len
        truncated = timeout & ~hard
        done = hard | truncated
        self.ep_ret += task_reward
        ti = self.amp_task.torso_index
        v_heading = mtv(heading_mat(kin[1][:, ti]), kin[2][:, ti])
        info = {
            "task_reward": task_reward, "task_terms": terms,
            "terminated": hard, "truncated": truncated, "done": done,
            "ep_len": self.ep_len.clone(), "command": command,
            "linear_error": (v_heading[:, :2] - command[:, :2]).norm(dim=-1),
            "yaw_error": (kin[3][:, ti, 2] - command[:, 2]).abs(),
            "velocity_heading": v_heading, "yaw_rate": kin[3][:, ti, 2],
        }
        self.prev_action = clean_action.clone()
        self.command_age += 1
        refresh = ((self.command_age >= self.command_duration) & ~done).nonzero().flatten()
        self._resample_commands(refresh)
        o = self.amp_task.observation(s_next, clean_action, self.command, xi)
        if self.cfg.obs_noise and not self.eval_mode:
            o = o + self.amp_task.sample_noise((self.num_envs,))
        self.obs_hist = torch.cat((self.obs_hist[:, 1:], o[:, None]), dim=1)
        ids = done.nonzero().flatten()
        if len(ids):
            info["done_ids"] = ids
            info["done_ep_len"] = self.ep_len[ids].clone()
            info["done_ep_task_return"] = self.ep_ret[ids].clone()
            info["done_terminated"] = hard[ids].clone()
            self.reset_envs(ids)
        # self.frame is an RSI provenance index, not a target timeline.
        # There is deliberately no clip-end termination or future-motion goal.
        self._finish_resets(ids)
        return s_next, info
