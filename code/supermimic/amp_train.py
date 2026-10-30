"""Train command-conditioned AMP walking from random weights through a world model."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn

from . import networks
from .amp_augmentation import (
    MixedExpertPairs, TimeScaledExpertPairs, command_style_gate, parse_clip_prefixes,
    parse_time_scales, select_expert_clips,
)
from .amp_env import AMPEnv
from .amp_networks import CommandActor
from .amp_prior import AMPPrior, AMPFeatureExtractor, ExpertFeaturePairs
from .amp_replay import AMPReplay
from .amp_task import AMPTaskCfg
from .motion_lib import MotionLibrary
from .networks import Critic, WorldModel, apply_delta, apply_delta_c, run_mlp, state_delta, state_error
from .sim_env import EnvCfg
from .task import NJ, XI_DIM, flat_to_state, state_to_flat

CHECKPOINT_TYPE = "supermimic_vibe_amp_from_scratch_v1"
REWARD_RESUME_FIELDS = (
    "task_weight", "style_scale", "linear_std", "angular_std", "action_rate_weight", "foot_slip_weight",
    "foot_slip_height_threshold", "foot_slip_width",
    "expert_time_scales", "expert_augmented_fraction", "expert_motion_threshold", "style_command_gate",
    "command_low", "command_high", "standing_command_probability",
    "expert_clip_prefixes",
    "yaw_rate_weight", "foot_landing_weight", "foot_penetration_weight",
    "refinement_speed_start", "refinement_speed_full", "landing_height", "landing_width",
    "penetration_scale", "penetration_slop", "low_speed_anchor_weight",
)


@dataclasses.dataclass
class AMPTrainCfg:
    motions: str = "data/processed/amp_walk_50hz.npz"
    out: str = "runs/amp_walk_24gb_train"
    seed: int = 42
    num_envs: int = 1024
    buffer_steps: int = 1000
    iterations: int = 50_000
    utd: float = 0.4
    warmup_steps: int = 64
    wm_only_iters: int = 500
    k: int = 8
    wm_batch: int = 512
    wm_batches: int = 10
    wm_width: int = 2048
    horizon: int = 8
    pi_batch: int = 256
    history: int = 4
    linear_std: float = 0.5
    angular_std: float = 0.5
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 2e-4
    actor_lr_scale: float = 1.0
    low_speed_anchor_weight: float = 0.0
    lr_final_frac: float = 0.03
    lr_decay_iters: int = 50_000
    critic_epochs: int = 5
    critic_minibatches: int = 4
    action_noise: float = 0.1
    action_limit: float = 3.0
    det_frac: float = 0.25
    disc_batch: int = 512
    disc_batches: int = 2
    disc_lr: float = 1e-4
    disc_gp: float = 10.0
    style_scale: float = 0.1
    task_weight: float = 0.75
    action_rate_weight: float = -0.01
    foot_slip_weight: float = 0.0
    yaw_rate_weight: float = 0.0
    foot_landing_weight: float = 0.0
    foot_penetration_weight: float = 0.0
    refinement_speed_start: float = 0.5
    refinement_speed_full: float = 0.8
    landing_height: float = 0.02
    landing_width: float = 0.01
    penetration_scale: float = 0.005
    penetration_slop: float = 0.0005
    foot_slip_height_threshold: float = 0.045
    foot_slip_width: float = 0.005
    expert_time_scales: str = "1.0"
    expert_clip_prefixes: str = ""
    expert_augmented_fraction: float = 0.0
    expert_motion_threshold: float = 0.15
    style_command_gate: float = 0.0
    command_low: str = "-0.5,-0.5,-1.0"
    command_high: str = "1.0,0.5,1.0"
    standing_command_probability: float = 0.05
    episode_seconds: float = 20.0
    rsi_probability: float = 0.5
    stand_height: float = 0.76
    domain_rand: bool = True
    obs_noise: bool = True
    reset_noise: bool = True
    amp: bool = False
    compile_env: bool = True
    compile_wm: bool = True
    value_clip: bool = True
    norm_update_every: int = 5
    log_every: int = 50
    ckpt_every: int = 1000
    tensorboard: bool = True
    wall_hours: float = 1e9
    resume: str = ""
    allow_reward_change: bool = False
    device: str = "cuda"


def task_config(cfg):
    return AMPTaskCfg(
        history=cfg.history, linear_std=cfg.linear_std, angular_std=cfg.angular_std,
        w_action_rate=cfg.action_rate_weight,
        w_foot_slip=cfg.foot_slip_weight,
        w_yaw_rate=cfg.yaw_rate_weight,
        w_foot_landing=cfg.foot_landing_weight,
        w_foot_penetration=cfg.foot_penetration_weight,
        refinement_speed_start=cfg.refinement_speed_start,
        refinement_speed_full=cfg.refinement_speed_full,
        landing_height=cfg.landing_height, landing_width=cfg.landing_width,
        penetration_scale=cfg.penetration_scale, penetration_slop=cfg.penetration_slop,
        foot_slip_height_threshold=cfg.foot_slip_height_threshold,
        foot_slip_width=cfg.foot_slip_width,
    )


def parse_command_vector(text):
    """Parse exactly three finite heading-frame vx/vy and world-yaw values."""
    try:
        parts = text.split(",") if isinstance(text, str) else tuple(text)
        if len(parts) != 3:
            raise ValueError
        values = tuple(float(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError("Command bounds must contain exactly three finite CSV values") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Command bounds must contain exactly three finite CSV values")
    return values


def reward_config(cfg):
    """Read reward and actor-regularization fields, including legacy defaults."""
    defaults = AMPTrainCfg()
    values = {
        key: cfg.get(key, getattr(defaults, key)) if isinstance(cfg, dict) else getattr(cfg, key)
        for key in REWARD_RESUME_FIELDS
    }
    values["expert_time_scales"] = list(parse_time_scales(values["expert_time_scales"]))
    values["expert_clip_prefixes"] = list(parse_clip_prefixes(values["expert_clip_prefixes"]))
    for key in ("command_low", "command_high"):
        values[key] = list(parse_command_vector(values[key]))
    return values


def validate_resume_config(saved_cfg, cfg):
    """Validate structural compatibility and explicitly authorize reward curricula.

    New fields absent from a legacy checkpoint retain their historical defaults.
    Returns only changed reward fields; it never mutates learner or critic state.
    """
    defaults = AMPTrainCfg()
    for key in ("history", "wm_width", "action_limit", "gamma"):
        if saved_cfg.get(key, getattr(defaults, key)) != getattr(cfg, key):
            raise ValueError(f"Resume requires the same {key}")
    old, new = reward_config(saved_cfg), reward_config(cfg)
    changes = {key: {"old": old[key], "new": new[key]} for key in REWARD_RESUME_FIELDS if old[key] != new[key]}
    if changes and not cfg.allow_reward_change:
        raise ValueError(
            "Resume reward fields differ: " + ", ".join(changes)
            + "; use --allow-reward-change true for an intentional reward curriculum"
        )
    return changes


def validate_cfg(c):
    for name in (
        "num_envs", "buffer_steps", "iterations", "k", "wm_batch", "wm_batches",
        "wm_width", "horizon", "pi_batch", "history", "critic_epochs",
        "critic_minibatches", "disc_batch", "disc_batches", "norm_update_every",
        "log_every", "ckpt_every", "lr_decay_iters",
    ):
        if getattr(c, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if c.buffer_steps < max(c.k, c.history, c.warmup_steps):
        raise ValueError("buffer_steps must cover k, history and warmup_steps")
    if c.warmup_steps < max(c.k, c.history) or c.wm_only_iters < 0:
        raise ValueError("warm-up is too short")
    if not (0 < c.gamma < 1 and 0 <= c.lam <= 1 and c.utd > 0):
        raise ValueError("Invalid discount, lambda or UTD")
    if not (0 <= c.task_weight <= 1 and 0 <= c.det_frac <= 1 and 0 <= c.rsi_probability <= 1):
        raise ValueError("Mixture probabilities must be in [0,1]")
    low, high = parse_command_vector(c.command_low), parse_command_vector(c.command_high)
    if any(lower > upper for lower, upper in zip(low, high)):
        raise ValueError("command_low must not exceed command_high on any axis")
    if not math.isfinite(c.standing_command_probability) or not 0 <= c.standing_command_probability <= 1:
        raise ValueError("standing_command_probability must be finite and in [0, 1]")
    if c.pi_batch * c.horizon < c.critic_minibatches:
        raise ValueError("Too few imagined samples for critic minibatches")
    if c.action_limit <= 0 or c.action_noise < 0 or c.lr <= 0 or c.episode_seconds <= 0:
        raise ValueError("Invalid action, learning-rate or episode setting")
    if not math.isfinite(c.actor_lr_scale) or c.actor_lr_scale <= 0:
        raise ValueError("actor_lr_scale must be finite and positive")
    if not math.isfinite(c.low_speed_anchor_weight) or c.low_speed_anchor_weight < 0:
        raise ValueError("low_speed_anchor_weight must be finite and nonnegative")
    if c.low_speed_anchor_weight > 0 and not c.resume:
        raise ValueError("Low-speed anchoring requires resuming this project's AMP checkpoint")
    if not math.isfinite(c.action_rate_weight) or c.action_rate_weight > 0:
        raise ValueError("action_rate_weight must be finite and nonpositive")
    if not math.isfinite(c.foot_slip_weight) or c.foot_slip_weight > 0:
        raise ValueError("foot_slip_weight must be finite and nonpositive")
    if (not math.isfinite(c.foot_slip_height_threshold) or c.foot_slip_height_threshold < 0
            or not math.isfinite(c.foot_slip_width) or c.foot_slip_width <= 0):
        raise ValueError("foot-slip height must be nonnegative and width must be positive and finite")
    for name in ("yaw_rate_weight", "foot_landing_weight", "foot_penetration_weight"):
        if not math.isfinite(getattr(c, name)) or getattr(c, name) > 0:
            raise ValueError(f"{name} must be finite and nonpositive")
    if (not math.isfinite(c.refinement_speed_start) or c.refinement_speed_start < 0
            or not math.isfinite(c.refinement_speed_full)
            or c.refinement_speed_full <= c.refinement_speed_start):
        raise ValueError("Refinement speed gate requires 0 <= start < full")
    for name in ("landing_width", "penetration_scale"):
        if not math.isfinite(getattr(c, name)) or getattr(c, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("landing_height", "penetration_slop"):
        if not math.isfinite(getattr(c, name)) or getattr(c, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    scales = parse_time_scales(c.expert_time_scales)
    parse_clip_prefixes(c.expert_clip_prefixes)
    if not math.isfinite(c.expert_augmented_fraction) or not 0 <= c.expert_augmented_fraction <= 1:
        raise ValueError("expert_augmented_fraction must be finite and in [0, 1]")
    if c.expert_augmented_fraction > 0 and not any(scale < 1 for scale in scales):
        raise ValueError("Augmented expert positives require a time scale below 1")
    for name in ("expert_motion_threshold", "style_command_gate"):
        if not math.isfinite(getattr(c, name)) or getattr(c, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This scratch AMP entry point currently supports one GPU")


def weight_hash(module):
    h = hashlib.sha256()
    for name, p in module.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def actor_state_hash(actor):
    """Hash parameters and normalization buffers of a frozen policy reference."""
    digest = hashlib.sha256()
    for name, value in actor.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def low_speed_anchor_penalty(action, reference_action, command, cfg):
    """Training-only MSE at real replay starts; zero above the refinement range.

    This is added to the actor objective, never to simulator rewards or critic
    targets. Detach the target even if a caller forgot to freeze its producer.
    """
    high = ((command[..., :2].norm(dim=-1) - cfg.refinement_speed_start)
            / (cfg.refinement_speed_full - cfg.refinement_speed_start)).clamp(0, 1)
    weighted_mse = ((action - reference_action.detach()).square().mean(-1) * (1 - high)).mean()
    return cfg.low_speed_anchor_weight * weighted_mse, weighted_mse


@contextmanager
def freeze_parameters(*modules):
    previous = [(p, p.requires_grad) for m in modules for p in m.parameters()]
    try:
        for p, _ in previous:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in previous:
            p.requires_grad_(flag)


class AMPTrainer:
    def __init__(self, cfg: AMPTrainCfg):
        validate_cfg(cfg)
        self.cfg, self.device = cfg, cfg.device
        if not cfg.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("AMP locomotion training requires a CUDA device")
        self.out = Path(cfg.out)
        if (self.out / "last.pt").exists() and not cfg.resume:
            raise FileExistsError(f"{self.out}/last.pt exists; use --resume or a fresh --out")
        self.out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        networks.AMP["enabled"] = cfg.amp
        self.lib = MotionLibrary(cfg.motions, cfg.device)
        self.expert_clip_prefixes = parse_clip_prefixes(cfg.expert_clip_prefixes)
        self.expert_clip_ids = (
            select_expert_clips(self.lib.names, self.expert_clip_prefixes) if self.expert_clip_prefixes else None
        )
        self.motion_hash = hashlib.sha256(Path(cfg.motions).read_bytes()).hexdigest()
        if abs(self.lib.dt - 0.02) > 1e-8:
            raise ValueError("Expert library must be at the 50 Hz control rate")
        ecfg = EnvCfg(
            num_envs=cfg.num_envs, episode_length_s=cfg.episode_seconds,
            domain_rand=cfg.domain_rand, obs_noise=cfg.obs_noise, reset_noise=cfg.reset_noise,
            compile=cfg.compile_env, nconmax=128, njmax=500,
            pose_range=((-0.02, 0.02), (-0.02, 0.02), (-0.01, 0.01),
                        (-0.05, 0.05), (-0.05, 0.05), (-0.1, 0.1)),
            vel_range=((-0.1, 0.1), (-0.1, 0.1), (-0.05, 0.05),
                       (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)),
            joint_pos_range=(-0.02, 0.02),
        )
        self.env = AMPEnv(
            self.lib, ecfg, cfg.device, task_config(cfg),
            gamma=cfg.gamma, rsi_probability=cfg.rsi_probability, stand_height=cfg.stand_height,
            command_low=parse_command_vector(cfg.command_low),
            command_high=parse_command_vector(cfg.command_high),
            standing_probability=cfg.standing_command_probability,
        )
        self.env.reset()
        self.task = self.env.task
        self.amp_task = self.env.amp_task
        if cfg.compile_env:
            self.task.compile_sizes |= {cfg.wm_batch, cfg.pi_batch}
        feat_dim = self.task.state_features(self.env.state).shape[-1]
        self.wm = WorldModel(self.task, feat_dim, XI_DIM, hidden=(cfg.wm_width,) * 3).to(cfg.device)
        self.actor = CommandActor(history=cfg.history, action_limit=cfg.action_limit).to(cfg.device)
        self.critic = Critic(feat_dim + 3 + XI_DIM + NJ).to(cfg.device)
        self.prior = AMPPrior(
            lr=cfg.disc_lr, gp_weight=cfg.disc_gp,
            style_scale=cfg.style_scale, device=cfg.device,
        )
        self.features = AMPFeatureExtractor(self.task)
        with torch.no_grad():
            expert = []
            for start in range(0, self.lib.num_frames, 4096):
                idx = torch.arange(start, min(start + 4096, self.lib.num_frames), device=cfg.device)
                expert.append(self.features(self.lib.frame(idx)))
            expert = torch.cat(expert)
            self.prior.normalizer.update(expert)
        if self.expert_clip_prefixes:
            selected_ids = torch.tensor(self.expert_clip_ids, dtype=torch.long, device=cfg.device)
            positive_starts, positive_lengths = self.lib.starts[selected_ids], self.lib.lengths[selected_ids]
        else:
            self.expert_clip_ids = list(range(self.lib.num_clips))
            positive_starts, positive_lengths = self.lib.starts, self.lib.lengths
        self.expert_pairs = ExpertFeaturePairs(expert, positive_starts, positive_lengths)
        self.expert_sampling_metadata = {
            "original_fraction": 1 - cfg.expert_augmented_fraction,
            "augmented_fraction": cfg.expert_augmented_fraction,
            "original_clips": len(self.expert_clip_ids),
            "original_pairs": int((positive_lengths - 1).sum()),
            "source_motion_library_clips": self.lib.num_clips,
            "expert_clip_prefixes": list(self.expert_clip_prefixes),
            "selected_clip_ids": list(self.expert_clip_ids),
            "selected_clip_names": [self.lib.names[i] for i in self.expert_clip_ids],
            "source_motion_library_sha256": self.motion_hash,
            "control_interval_seconds": self.lib.dt,
            "style_command_gate": cfg.style_command_gate,
            "style_gate_motion_score": "norm(command_xy) + 0.5 * abs(command_yaw)",
        }
        if cfg.expert_augmented_fraction > 0:
            augmented = TimeScaledExpertPairs(
                self.lib, self.features, cfg.expert_time_scales,
                motion_threshold=cfg.expert_motion_threshold,
                clip_ids=self.expert_clip_ids if self.expert_clip_prefixes else None,
            )
            self.expert_pairs = MixedExpertPairs(self.expert_pairs, augmented, cfg.expert_augmented_fraction)
            self.expert_sampling_metadata["augmentation"] = augmented.sampling_metadata
            original_speed = sum(
                float(self.lib.root_lin_vel[start:start + length - 1, :2].norm(dim=-1).mean())
                for start, length in zip(positive_starts.tolist(), positive_lengths.tolist())
            ) / len(self.expert_clip_ids)
            augmented_speed = sum(
                row["root_planar_speed_mps"]
                for row in augmented.sampling_metadata["calibration_clip_uniform_first_state_by_scale"]
            ) / len(augmented.scales)
            self.expert_sampling_metadata["calibration_original_clip_uniform_mean_root_planar_speed_mps"] = original_speed
            self.expert_sampling_metadata["calibration_mixture_mean_root_planar_speed_mps"] = (
                (1 - cfg.expert_augmented_fraction) * original_speed + cfg.expert_augmented_fraction * augmented_speed
            )
        self.opt_wm = torch.optim.Adam(self.wm.net.parameters(), lr=cfg.lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
        self.opt_critic = torch.optim.Adam(self.critic.net.parameters(), lr=cfg.lr)
        self.buffer = AMPReplay(cfg.num_envs, cfg.buffer_steps, XI_DIM, cfg.device)
        self.integrate = apply_delta_c if cfg.compile_wm else apply_delta
        self._wm_loss = torch.compile(self.wm_loss, dynamic=False) if cfg.compile_wm else self.wm_loss
        self.max_reward = cfg.task_weight * self.amp_task.max_step_reward + (1 - cfg.task_weight) * cfg.style_scale
        self.vmax = self.max_reward / (1 - cfg.gamma)
        self.it = self.updates = self.actor_updates = self.disc_updates = self.env_steps = 0
        self.update_credit = self.train_time = 0.0
        self.ep_return = torch.zeros(cfg.num_envs, device=cfg.device)
        self.episodes = {"return": [], "length": [], "failed": []}
        self.initialization = {
            "type": "random", "seed": cfg.seed, "teacher": None, "pretrained_checkpoint": None,
            "initial_weight_hashes": {k: weight_hash(m) for k, m in self.models().items()},
            "reference_state_initialization_probability": cfg.rsi_probability,
        }
        self.writer = None
        if cfg.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.out / "tb"))
        self.anchor_actor = None
        self.anchor_metadata = None
        if cfg.resume:
            self.load(cfg.resume)
        if (cfg.expert_augmented_fraction > 0 or cfg.style_command_gate > 0
                or self.expert_clip_prefixes or "expert_positive_sampling" in self.initialization):
            self.initialization["expert_positive_sampling"] = self.expert_sampling_metadata
            print("AMP expert positive sampling: " + json.dumps(self.expert_sampling_metadata, sort_keys=True), flush=True)
        (self.out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2) + "\n")
        (self.out / "initialization.json").write_text(json.dumps(self.initialization, indent=2) + "\n")
        self.log = (self.out / "log.jsonl").open("a")
        clip_description = f"{self.lib.num_clips} expert clips"
        if self.expert_clip_prefixes:
            clip_description = f"{self.lib.num_clips} source clips; {len(self.expert_clip_ids)} selected positive clips"
        print(f"AMP random-initialization task: {clip_description}, "
              f"actor input {cfg.history * self.amp_task.obs_dim}, AMP pair 390. No teacher.", flush=True)

    def models(self):
        return {"wm": self.wm, "actor": self.actor, "critic": self.critic, "discriminator": self.prior.discriminator}

    def predict(self, state, action, xi):
        delta = run_mlp(self.wm.net, self.wm.in_norm(self.wm.inputs(state, action, xi)))
        return self.integrate(state, delta * self.wm.out_norm.std + self.wm.out_norm.mean)

    def _body_kin(self, state):
        # AMPEnv selects compiled or eager FK from cfg.compile_env.
        return self.env.kin_c(
            state["pos"], state["rot"], state["q"],
            state["v"], state["w"], state["qd"],
        )

    def _amp_features(self, state):
        return self.features(state, self._body_kin(state))

    def style_reward(self, f0, f1, command):
        style, score = self.prior.reward(f0, f1)
        if self.cfg.style_command_gate == 0:
            return style, score
        return style * command_style_gate(command, self.cfg.style_command_gate), score

    @torch.no_grad()
    def collect(self):
        obs = self.env.policy_inputs()
        self.actor.obs_norm.update(obs)
        clean = self.actor(obs)
        noise = self.cfg.action_noise * torch.randn_like(clean)
        noise[:int(self.cfg.det_frac * self.cfg.num_envs)] = 0
        executed = (clean + noise).clamp(-self.cfg.action_limit, self.cfg.action_limit)
        s0 = state_to_flat(self.env.state)
        xi = self.env.xi.clone()
        command = self.env.command.clone()
        previous = self.env.prev_action.clone()
        frame, episode = self.env.frame.clone(), self.env.ep_id.clone()
        next_state, info = self.env.step(clean, executed)
        s1 = state_to_flat(next_state)
        if not bool(torch.isfinite(s0).all() and torch.isfinite(s1).all()):
            raise FloatingPointError("Non-finite real transition; check simulator/reset settings")
        self.buffer.add(s0, executed, s1, xi, frame, episode)
        self.buffer.add_amp(command, previous, clean)
        f0, f1 = self._amp_features(flat_to_state(s0)), self._amp_features(next_state)
        style, _ = self.style_reward(f0, f1, command)
        reward = self.cfg.task_weight * info["task_reward"] + (1 - self.cfg.task_weight) * style
        self.ep_return += reward
        if "done_ids" in info:
            ids = info["done_ids"]
            self.episodes["return"].extend(
                (self.ep_return[ids] / (self.max_reward * self.env.max_ep_len)).cpu().tolist()
            )
            self.episodes["length"].extend(info["done_ep_len"].cpu().tolist())
            self.episodes["failed"].extend(info["done_terminated"].float().cpu().tolist())
            self.ep_return[ids] = 0
        self.env_steps += self.cfg.num_envs
        return {
            "sim_task_reward": float(info["task_reward"].mean()),
            "sim_style_reward": float(style.mean()),
            "sim_reward": float(reward.mean()),
            "linear_error": float(info["linear_error"].mean()),
            "yaw_error": float(info["yaw_error"].mean()),
        }

    def wm_loss(self, s0, actions, targets, xi, mask):
        state = flat_to_state(s0)
        errors = []
        for j in range(self.cfg.k):
            state = self.predict(state, actions[:, j], xi)
            errors.append((state_error(state, flat_to_state(targets[:, j])) / self.wm.out_norm.std).square().mean(-1))
        denom = mask.sum().clamp_min(1)
        return (
            (torch.stack(errors).sum(0) * mask).sum() / denom,
            (errors[0].detach() * mask).sum() / denom,
            (errors[-1].detach() * mask).sum() / denom,
        )

    def update_wm(self):
        totals = {}
        for bi in range(self.cfg.wm_batches):
            for _ in range(8):
                s0, actions, targets, xi, valid = self.buffer.sample_windows(self.cfg.wm_batch, self.cfg.k)
                if bool(valid.any()):
                    break
            else:
                raise RuntimeError("No valid world-model windows: check fall rate/reset states or increase warm-up")
            if bi == 0 and self.updates % self.cfg.norm_update_every == 0:
                with torch.no_grad():
                    previous = flat_to_state(s0)
                    inputs, deltas = [], []
                    for j in range(self.cfg.k):
                        target = flat_to_state(targets[:, j])
                        inputs.append(self.wm.inputs(previous, actions[:, j], xi)[valid])
                        deltas.append(state_delta(previous, target)[valid])
                        previous = target
                    self.wm.in_norm.update(torch.cat(inputs))
                    self.wm.out_norm.update(torch.cat(deltas))
            loss, err1, errk = self._wm_loss(s0, actions, targets, xi, valid.float())
            self.opt_wm.zero_grad(set_to_none=True)
            loss.backward()
            grad = nn.utils.clip_grad_norm_(self.wm.net.parameters(), 1.0, error_if_nonfinite=True)
            self.opt_wm.step()
            for k, v in {"wm_loss": loss, "wm_err1": err1, "wm_errk": errk, "wm_grad": grad}.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach()) / self.cfg.wm_batches
        self.opt_wm.zero_grad(set_to_none=True)
        return totals

    def update_prior(self):
        stats = {}
        for _ in range(self.cfg.disc_batches):
            s0, s1 = self.buffer.sample_transitions(self.cfg.disc_batch)
            with torch.no_grad():
                policy0, policy1 = self._amp_features(flat_to_state(s0)), self._amp_features(flat_to_state(s1))
                expert0, expert1 = self.expert_pairs.sample(self.cfg.disc_batch)
            for k, v in self.prior.update(policy0, policy1, expert0, expert1).items():
                stats[k] = stats.get(k, 0.0) + v / self.cfg.disc_batches
            self.disc_updates += 1
        return stats

    def critic_input(self, state, command, xi, previous):
        return torch.cat((self.task.features_fast(state), command, xi, previous), -1)

    def value(self, x):
        v = self.critic(x)
        return v.clamp(-self.vmax, self.vmax) if self.cfg.value_clip else v

    def update_actor(self):
        c = self.cfg
        s0, xi, command, history, previous = self.buffer.sample_starts_amp(c.pi_batch, c.history, self.amp_task)
        state = flat_to_state(s0)
        if c.obs_noise:
            history = history + self.amp_task.sample_noise((c.pi_batch, c.history))
        self.wm.zero_grad(set_to_none=True)
        self.prior.zero_grad(set_to_none=True)
        self.critic.zero_grad(set_to_none=True)
        self.opt_actor.zero_grad(set_to_none=True)
        record = {"x": [], "r": [], "d": [], "weight": []}
        anchor_enabled = c.low_speed_anchor_weight > 0
        first_action = None
        if anchor_enabled:
            if self.anchor_actor is None:
                raise RuntimeError("Low-speed anchor was not initialized from a checkpoint")
            start_observation = history.flatten(1)
            with torch.no_grad():
                reference_action = self.anchor_actor(start_observation)
            first_action = self.actor(start_observation)
            anchor_loss, anchor_mse = low_speed_anchor_penalty(
                first_action, reference_action, command, c)
        with freeze_parameters(self.wm, self.prior, self.critic):
            alive = torch.ones(c.pi_batch, device=self.device)
            objective = torch.zeros_like(alive)
            phi = self._amp_features(state)
            task_sum = style_sum = 0.0
            for t in range(c.horizon):
                record["x"].append(self.critic_input(state, command, xi, previous).detach())
                action = first_action if t == 0 and anchor_enabled else self.actor(history.flatten(1))
                next_state = self.predict(state, action, xi)
                kin = self._body_kin(next_state)
                next_phi = self.features(next_state, kin)
                task_reward, _ = self.amp_task.reward(next_state, action, previous, command, kin=kin)
                style_reward, _ = self.style_reward(phi, next_phi, command)
                reward = c.task_weight * task_reward + (1 - c.task_weight) * style_reward
                _, terminal = self.amp_task.termination(next_state)
                objective = objective + c.gamma**t * alive * reward
                record["r"].append(reward.detach())
                record["d"].append(terminal.detach())
                record["weight"].append(alive.detach())
                alive = alive * (1 - terminal)
                obs = self.amp_task.observation(next_state, action, command, xi)
                if c.obs_noise:
                    obs = obs + self.amp_task.sample_noise((c.pi_batch,))
                history = torch.cat((history[:, 1:], obs[:, None]), 1)
                task_sum += float(task_reward.detach().mean()) / c.horizon
                style_sum += float(style_reward.detach().mean()) / c.horizon
                state, previous, phi = next_state, action, next_phi
            xh = self.critic_input(state, command, xi, previous)
            objective = objective + c.gamma**c.horizon * alive * self.value(xh)
            loss = -objective.mean() / c.horizon
            if anchor_enabled:
                loss = loss + anchor_loss
            loss.backward()
            grad = nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0, error_if_nonfinite=True)
            nn.utils.clip_grad_value_(self.actor.parameters(), 0.1)
            self.opt_actor.step()
        if any(p.grad is not None for m in (self.wm, self.prior, self.critic) for p in m.parameters()):
            raise AssertionError("Actor backward leaked into frozen world-model/discriminator/critic parameters")
        self.actor_updates += 1
        stats = {
            "actor_loss": float(loss.detach()), "imagined_return": float(objective.detach().mean()),
            "actor_grad": float(grad), "imagined_survival": float(alive.detach().mean()),
            "imagined_task_reward": task_sum, "imagined_style_reward": style_sum,
            "frozen_parameter_gradients_clear": 1.0,
        }
        if anchor_enabled:
            if any(p.grad is not None for p in self.anchor_actor.parameters()):
                raise AssertionError("Actor update leaked gradients into the frozen low-speed anchor")
            stats.update(anchor_loss=float(anchor_loss.detach()), anchor_action_mse=float(anchor_mse.detach()))
        stats.update(self.update_critic(record, xh.detach()))
        return stats

    def update_critic(self, record, xh):
        c = self.cfg
        with torch.no_grad():
            x = torch.stack(record["x"] + [xh], dim=1)
            self.critic.norm.update(x)
            values = self.value(x)
            target = values[:, -1].clone()
            targets = []
            for t in range(c.horizon - 1, -1, -1):
                target = record["r"][t] + c.gamma * (1 - record["d"][t]) * (
                    (1 - c.lam) * values[:, t + 1] + c.lam * target
                )
                if c.value_clip:
                    target = target.clamp(-self.vmax, self.vmax)
                targets.append(target)
            y = torch.stack(targets[::-1], dim=1).flatten()
            weight = torch.stack(record["weight"], dim=1).flatten()
            xf = x[:, :-1].reshape(-1, x.shape[-1])
        n = len(y)
        losses = []
        for _ in range(c.critic_epochs):
            for ids in torch.randperm(n, device=self.device).tensor_split(c.critic_minibatches):
                error = (self.critic(xf[ids]) - y[ids]).square()
                loss = (error * weight[ids]).sum() / weight[ids].sum().clamp_min(1)
                self.opt_critic.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.net.parameters(), 1.0, error_if_nonfinite=True)
                self.opt_critic.step()
                losses.append(float(loss.detach()))
        self.opt_critic.zero_grad(set_to_none=True)
        return {"critic_loss": sum(losses) / len(losses), "critic_value": float(values[:, 0].mean())}

    def _restore_low_speed_anchor(self, checkpoint, path):
        """Preserve the first self-checkpoint policy across continuation stages."""
        saved = checkpoint.get("low_speed_anchor")
        if saved is None and (checkpoint["cfg"].get("low_speed_anchor_weight", 0.0) > 0
                              or "low_speed_anchor" in checkpoint.get("initialization", {})):
            raise ValueError("Anchored AMP checkpoint is missing its frozen reference policy")
        if saved is None and self.cfg.low_speed_anchor_weight == 0:
            self.anchor_actor = None
            self.anchor_metadata = None
            return
        # deepcopy consumes no RNG and copies the observation normalizer exactly.
        self.anchor_actor = copy.deepcopy(self.actor).requires_grad_(False).eval()
        self.anchor_actor.zero_grad(set_to_none=True)
        if saved is not None:
            self.anchor_actor.load_state_dict(saved)
            metadata = checkpoint.get("low_speed_anchor_metadata")
            if not isinstance(metadata, dict) or metadata.get("actor_state_sha256") != actor_state_hash(self.anchor_actor):
                raise ValueError("Frozen anchor state does not match its saved provenance hash")
            self.anchor_metadata = copy.deepcopy(metadata)
        else:
            self.anchor_metadata = {
                "kind": "self_checkpoint_actor", "source_checkpoint": str(path),
                "source_iteration": int(checkpoint["it"]),
                "actor_state_sha256": actor_state_hash(self.anchor_actor),
                "source_initialization": copy.deepcopy(checkpoint.get("initialization", {})),
                "normalizer": "Frozen with the original actor state; never updated",
                "objective": "Command-gated action MSE on real replay start observations",
            }
        self.initialization["low_speed_anchor"] = copy.deepcopy(self.anchor_metadata)

    def save(self, path):
        ck = {
            "checkpoint_type": CHECKPOINT_TYPE, "cfg": dataclasses.asdict(self.cfg),
            "it": self.it, "updates": self.updates, "actor_updates": self.actor_updates,
            "disc_updates": self.disc_updates, "env_steps": self.env_steps, "train_time": self.train_time,
            "update_credit": self.update_credit,
            "initialization": self.initialization, "expert_library_sha256": self.motion_hash,
            "wm": self.wm.state_dict(), "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(), "amp_prior": self.prior.state_dict(),
            "opt_wm": self.opt_wm.state_dict(), "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(), "opt_discriminator": self.prior.optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        if getattr(self, "anchor_actor", None) is not None:
            ck["low_speed_anchor"] = self.anchor_actor.state_dict()
            ck["low_speed_anchor_metadata"] = copy.deepcopy(self.anchor_metadata)
        path = Path(path)
        torch.save(ck, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if ck.get("checkpoint_type") != CHECKPOINT_TYPE:
            raise ValueError("Only this AMP format can be resumed; a motion-tracking/teacher checkpoint is not a scratch AMP run")
        if ck["expert_library_sha256"] != self.motion_hash:
            raise ValueError("Resume expert library differs from the saved AMP library")
        reward_changes = validate_resume_config(ck["cfg"], self.cfg)
        for key, module in (("wm", self.wm), ("actor", self.actor), ("critic", self.critic), ("amp_prior", self.prior)):
            module.load_state_dict(ck[key])
        for key, opt in (("opt_wm", self.opt_wm), ("opt_actor", self.opt_actor),
                         ("opt_critic", self.opt_critic), ("opt_discriminator", self.prior.optimizer)):
            opt.load_state_dict(ck[key])
        for key in ("it", "updates", "actor_updates", "disc_updates", "env_steps", "train_time", "initialization"):
            setattr(self, key, ck[key])
        self.update_credit = ck.get("update_credit", 0.0)
        self._restore_low_speed_anchor(ck, path)
        if reward_changes:
            entry = {
                "iteration": self.it, "checkpoint": str(path),
                "changes": reward_changes,
                "old_reward_config": reward_config(ck["cfg"]),
                "new_reward_config": reward_config(self.cfg),
                "old_actor_lr_scale": ck["cfg"].get("actor_lr_scale", 1.0),
                "new_actor_lr_scale": self.cfg.actor_lr_scale,
                "critic_reset": False,
            }
            if hasattr(self, "expert_sampling_metadata"):
                entry["expert_positive_sampling"] = self.expert_sampling_metadata
            self.initialization.setdefault("reward_curriculum", []).append(entry)
            print(
                f"Reward curriculum at iteration {self.it}: {json.dumps(reward_changes, sort_keys=True)}. "
                f"Actor LR scale {self.cfg.actor_lr_scale}; critic weights and optimizer state preserved.",
                flush=True,
            )
        torch.set_rng_state(ck["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in ck["cuda_rng"]])
        print(f"Resumed AMP learner at iteration {self.it}; refilling real replay.", flush=True)

    def run(self):
        c = self.cfg
        aggregate, count = {}, 0
        normal_exit = False
        try:
            for _ in range(c.warmup_steps):
                self.collect()
            last_log = time.monotonic()
            while self.it < c.iterations and self.train_time / 3600 < c.wall_hours:
                start = time.monotonic()
                lr = c.lr * (c.lr_final_frac + (1 - c.lr_final_frac) * 0.5 * (
                    1 + math.cos(math.pi * min(self.it, c.lr_decay_iters) / c.lr_decay_iters)
                ))
                for opt in (self.opt_wm, self.opt_actor, self.opt_critic):
                    for group in opt.param_groups:
                        group["lr"] = lr * (c.actor_lr_scale if opt is self.opt_actor else 1.0)
                metrics = self.collect()
                self.update_credit += c.utd
                while self.update_credit >= 1:
                    self.update_credit -= 1
                    metrics.update(self.update_wm())
                    metrics.update(self.update_prior())
                    if self.it >= c.wm_only_iters:
                        metrics.update(self.update_actor())
                    self.updates += 1
                torch.cuda.synchronize()
                self.it += 1
                self.train_time += time.monotonic() - start
                for k, v in metrics.items():
                    if not math.isfinite(v):
                        raise FloatingPointError(f"Non-finite metric {k}: {v}")
                    total, n = aggregate.get(k, (0.0, 0))
                    aggregate[k] = (total + v, n + 1)
                count += 1
                if self.it % c.log_every == 0 or self.it == c.iterations:
                    record = {k: total / n for k, (total, n) in aggregate.items()}
                    mean = lambda xs: sum(xs) / max(len(xs), 1)
                    record.update(
                        it=self.it, updates=self.updates, actor_updates=self.actor_updates,
                        disc_updates=self.disc_updates, env_steps=self.env_steps,
                        hours=self.train_time / 3600, lr=lr, actor_lr=lr * c.actor_lr_scale,
                        iter_s=(time.monotonic() - last_log) / count,
                        episode_return=mean(self.episodes["return"]),
                        episode_seconds=mean(self.episodes["length"]) * self.amp_task.cfg.dt,
                        episode_fall_fraction=mean(self.episodes["failed"]),
                        completed_episodes=len(self.episodes["failed"]),
                        mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                        actor_warmup=int(self.it <= c.wm_only_iters),
                    )
                    self.log.write(json.dumps(record) + "\n"); self.log.flush()
                    if self.writer:
                        for k, v in record.items():
                            self.writer.add_scalar(k, v, self.it)
                        self.writer.flush()
                    print(
                        f"AMP it {self.it} upd {self.updates} actor {self.actor_updates} "
                        f"len {record['episode_seconds']:.2f}s fall {record['episode_fall_fraction']:.3f} "
                        f"vel_err {record['linear_error']:.3f} style {record['sim_style_reward']:.4f} "
                        f"wm8 {record.get('wm_errk', 0):.4f} "
                        f"D {record.get('discriminator_loss', 0):.4f} "
                        f"J {record.get('imagined_return', 0):.3f} "
                        f"mem {record['mem_gb']:.2f}GB", flush=True,
                    )
                    aggregate, count = {}, 0
                    self.episodes = {"return": [], "length": [], "failed": []}
                    last_log = time.monotonic()
                if self.it % c.ckpt_every == 0:
                    self.save(self.out / f"ckpt_{self.it:07d}.pt")
                    self.save(self.out / "last.pt")
            normal_exit = True
        except KeyboardInterrupt:
            print("Interrupted; saving AMP learner checkpoint.", flush=True)
            normal_exit = True
        finally:
            if normal_exit:
                self.save(self.out / "last.pt")
                (self.out / ("DONE" if self.it >= c.iterations else "INTERRUPTED")).touch()
            self.log.close()
            if self.writer:
                self.writer.close()


def load_amp_actor(path, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError("Expected a scratch AMP checkpoint")
    cfg = AMPTrainCfg(**ck["cfg"])
    actor = CommandActor(history=cfg.history, action_limit=cfg.action_limit)
    actor.load_state_dict(ck["actor"])
    return actor.to(device).eval(), cfg, ck


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in dataclasses.fields(AMPTrainCfg):
        kind = type(field.default)
        if kind is bool:
            def boolean(text):
                if text.lower() not in ("true", "false", "1", "0", "yes", "no"):
                    raise argparse.ArgumentTypeError("Expected true or false")
                return text.lower() in ("true", "1", "yes")
            kind = boolean
        parser.add_argument("--" + field.name.replace("_", "-"), type=kind, default=field.default)
    return AMPTrainCfg(**vars(parser.parse_args()))


def main():
    cfg = parse_args()
    print(cfg, flush=True)
    AMPTrainer(cfg).run()


if __name__ == "__main__":
    main()
