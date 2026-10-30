"""Fixed-duration command evaluation of a from-scratch AMP walking checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from . import networks
from .amp_env import AMPEnv
from .amp_train import load_amp_actor, task_config
from .motion_lib import MotionLibrary
from .sim_env import EnvCfg


DEFAULT_COMMANDS = "0,0,0;0.5,0,0;0,0.3,0;0,0,0.5"


def parse_commands(text):
    """Read heading-frame vx/vy [m/s] and world yaw rate [rad/s]."""
    result = []
    for item in text.split(";"):
        if not item.strip():
            continue
        try:
            values = tuple(float(v.strip()) for v in item.split(","))
        except ValueError as exc:
            raise ValueError(f"Invalid velocity command: {item!r}") from exc
        if len(values) != 3 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Command must contain three finite values: {item!r}")
        result.append(values)
    if not result:
        raise ValueError("At least one command is required")
    return result


def make_amp_eval_env(lib, cfg, num_envs, device, seconds):
    task_cfg = task_config(cfg)
    env = AMPEnv(
        lib,
        EnvCfg(
            num_envs=num_envs, episode_length_s=seconds + 2 * task_cfg.dt,
            obs_noise=False, reset_noise=False, domain_rand=True,
            compile=False, nconmax=128, njmax=500,
        ),
        device, task_cfg, gamma=cfg.gamma,
        rsi_probability=0.0, stand_height=cfg.stand_height,
    )
    # Keep expanded friction/COM fields, then set their deterministic evaluation values.
    env.eval_mode = True
    return env


@torch.no_grad()
def perturb_standing_reset(env, scale):
    """Apply documented mild, uniform initial-state variation for distinct trials."""
    from .rotations import mat_to_quat, quat_from_euler_xyz, quat_mul

    state = env.state
    uniform = lambda like, width: (2 * torch.rand_like(like) - 1) * width * scale
    position = state["pos"].clone()
    position[:, :2] += uniform(position[:, :2], 0.02)
    euler = uniform(state["pos"], 1.0) * state["pos"].new_tensor((0.03, 0.03, 0.1))
    quaternion = quat_mul(quat_from_euler_xyz(*euler.unbind(-1)), mat_to_quat(state["rot"]))
    q = (state["q"] + uniform(state["q"], 0.02)).clamp(env.task.qmin, env.task.qmax)
    ids = torch.arange(env.num_envs, device=env.device)
    env.write_state(ids, position, quaternion, q,
                    state["v"] + uniform(state["v"], 0.05),
                    state["w"] + uniform(state["w"], 0.05),
                    state["qd"] + uniform(state["qd"], 0.1))
    env._finish_resets(ids)


@torch.no_grad()
def evaluate_command(actor, env, command, horizon_steps, diagnostics=None, observer=None,
                     reset_perturbation=0.0):
    """Freeze each row's statistics at its first termination, before auto-reset."""
    n, device = env.num_envs, env.device
    fixed = torch.tensor(command, dtype=torch.float32, device=device)
    env.fixed_command = fixed
    env.reset()
    if reset_perturbation:
        perturb_standing_reset(env, reset_perturbation)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    fell = torch.zeros_like(alive)
    truncated = torch.zeros_like(alive)
    steps = torch.zeros(n, dtype=torch.long, device=device)
    linear_sum = torch.zeros(n, device=device)
    linear_sq = torch.zeros(n, device=device)
    xyz_sq = torch.zeros(n, device=device)
    yaw_abs = torch.zeros(n, device=device)
    yaw_sq = torch.zeros(n, device=device)
    velocity_sum = torch.zeros(n, 3, device=device)
    yaw_rate_sum = torch.zeros(n, device=device)
    task_return = torch.zeros(n, device=device)
    target_xyz = torch.tensor((command[0], command[1], 0.0), device=device)
    for step in range(horizon_steps):
        before = env.state if diagnostics is not None else None
        previous = env.prev_action.clone() if diagnostics is not None else None
        action = actor(env.policy_inputs())
        state, info = env.step(action)
        if diagnostics is not None:
            diagnostics.observe(before, state, action, previous, alive)
        if observer is not None:
            observer(step, state, info, alive)
        velocity, yaw_rate = info["velocity_heading"], info["yaw_rate"]
        if not (bool(torch.isfinite(velocity[alive]).all())
                and bool(torch.isfinite(yaw_rate[alive]).all())):
            raise FloatingPointError("Non-finite active evaluation state")
        error = velocity - target_xyz
        planar_sq = error[:, :2].square().sum(-1)
        yaw_error = yaw_rate - fixed[2]
        steps += alive.long()
        linear_sum += torch.where(alive, planar_sq.sqrt(), 0.0)
        linear_sq += torch.where(alive, planar_sq, 0.0)
        xyz_sq += torch.where(alive, error.square().sum(-1), 0.0)
        yaw_abs += torch.where(alive, yaw_error.abs(), 0.0)
        yaw_sq += torch.where(alive, yaw_error.square(), 0.0)
        velocity_sum += torch.where(alive[:, None], velocity, 0.0)
        yaw_rate_sum += torch.where(alive, yaw_rate, 0.0)
        task_return += torch.where(alive, info["task_reward"], 0.0)
        fell |= info["terminated"] & alive
        truncated |= info["truncated"] & alive
        alive &= ~info["done"]
        if not bool(alive.any()):
            break
        # env.step() can reset inactive worlds, but the mask prevents those
        # episodes from entering these fixed-initial-state measurements.
    denominator = steps.float().clamp_min(1)
    total = denominator.sum()
    trials = []
    for i in range(n):
        trials.append({
            "copy": i,
            "survived_full_horizon": bool(alive[i]),
            "fell": bool(fell[i]),
            "truncated_early": bool(truncated[i]),
            "observed_steps": int(steps[i]),
            "survival_seconds": float(steps[i]) * env.tcfg.dt,
            "coverage": float(steps[i]) / horizon_steps,
            "mean_planar_error_mps_alive": float(linear_sum[i] / denominator[i]),
            "planar_rmse_mps_alive": float((linear_sq[i] / denominator[i]).sqrt()),
            "linear_xyz_rmse_mps_alive": float((xyz_sq[i] / denominator[i]).sqrt()),
            "mean_abs_yaw_error_radps_alive": float(yaw_abs[i] / denominator[i]),
            "yaw_rmse_radps_alive": float((yaw_sq[i] / denominator[i]).sqrt()),
            "mean_velocity_heading_mps_alive": (velocity_sum[i] / denominator[i]).cpu().tolist(),
            "mean_yaw_rate_radps_alive": float(yaw_rate_sum[i] / denominator[i]),
            "task_return_alive": float(task_return[i]),
        })
    result = {
        "command": list(command),
        "copies": n,
        "survived": int(alive.sum()),
        "falls": int(fell.sum()),
        "survival_rate": float(alive.float().mean()),
        "fall_rate": float(fell.float().mean()),
        "mean_survival_seconds": float(steps.float().mean()) * env.tcfg.dt,
        "mean_coverage": float(steps.float().mean()) / horizon_steps,
        "mean_planar_error_mps_alive": float(linear_sum.sum() / total),
        "planar_rmse_mps_alive": float((linear_sq.sum() / total).sqrt()),
        "linear_xyz_rmse_mps_alive": float((xyz_sq.sum() / total).sqrt()),
        "mean_abs_yaw_error_radps_alive": float(yaw_abs.sum() / total),
        "yaw_rmse_radps_alive": float((yaw_sq.sum() / total).sqrt()),
        "trials": trials,
    }
    if diagnostics is not None:
        result["gait_diagnostics_alive"] = diagnostics.report()
    return result


def evaluate_checkpoint(checkpoint, motions, commands, seconds=20.0, num_envs=32, device="cuda",
                        diagnostics=False, expert_bank_size=2048,
                        reset_perturbation=0.0, reset_seed=0):
    networks.AMP["enabled"] = False
    torch.manual_seed(reset_seed)
    actor, cfg, ck = load_amp_actor(checkpoint, device)
    lib = MotionLibrary(str(motions), device)
    dt = task_config(cfg).dt
    if abs(lib.dt - dt) > 1e-8:
        raise ValueError("Expert library/control intervals differ")
    horizon_steps = max(1, round(seconds / dt))
    env = make_amp_eval_env(lib, cfg, num_envs, device, horizon_steps * dt)
    expert_style = None
    if diagnostics:
        from .amp_diagnostics import FixedExpertStyle, GaitDiagnostics
        expert_style = FixedExpertStyle.from_library(env.task, lib, bank_size=expert_bank_size)
    results = []
    for command in commands:
        # Commands receive the same initial perturbations for direct comparison.
        torch.manual_seed(reset_seed)
        measures = GaitDiagnostics(env, expert_style) if diagnostics else None
        result = evaluate_command(actor, env, command, horizon_steps, diagnostics=measures,
                                  reset_perturbation=reset_perturbation)
        results.append(result)
        print(
            f"command={tuple(command)} survived={result['survived']}/{num_envs} "
            f"falls={result['falls']} planar_RMSE={result['planar_rmse_mps_alive']:.3f}m/s "
            f"yaw_MAE={result['mean_abs_yaw_error_radps_alive']:.3f}rad/s "
            f"coverage={result['mean_coverage']:.3f}", flush=True,
        )
    report = {
        "checkpoint": str(checkpoint), "checkpoint_iteration": int(ck["it"]),
        "motions": str(motions), "expert_library_sha256": hashlib.sha256(Path(motions).read_bytes()).hexdigest(),
        "actor_updates": int(ck.get("actor_updates", 0)),
        "initialization": "nominal standing pose; no reference-state initialization",
        "stand_height_m": cfg.stand_height,
        "requested_seconds": seconds, "horizon_steps": horizon_steps,
        "horizon_seconds": horizon_steps * dt, "num_envs_per_command": num_envs,
        "friction": 1.0, "observation_noise": False, "reset_noise": False,
        "randomization": False, "deterministic_policy": True,
        "same_initial_state_copies": reset_perturbation == 0,
        "error_statistics": "Until each copy's first termination, including its terminal transition; reset episodes excluded",
        "evaluation_scope": "Checkpoint measurement, not a claim of converged walking or independent random seeds",
        "commands": results,
    }
    if expert_style is not None:
        report["fixed_expert_style_reference"] = expert_style.metadata
    if reset_perturbation:
        report["initial_state_perturbations"] = {
            "seed": reset_seed, "scale": reset_perturbation, "distribution": "independent uniform per copy",
            "maximum_absolute_offsets_at_scale_one": {
                "root_xy_m": 0.02, "root_roll_pitch_rad": 0.03, "root_yaw_rad": 0.1,
                "joint_position_rad": 0.02, "root_linear_velocity_mps": 0.05,
                "root_angular_velocity_radps": 0.05, "joint_velocity_radps": 0.1,
            },
            "same_perturbations_for_each_command": True,
            "root_height_perturbed": False,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--motions", type=Path, default=Path("data/processed/amp_walk_50hz.npz"))
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--commands", default=DEFAULT_COMMANDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--diagnostics", action="store_true",
                        help="Measure gait, actions and distance to a fixed expert motion bank")
    parser.add_argument("--expert-bank-size", type=int, default=2048)
    parser.add_argument("--reset-perturbation", type=float, default=0.0,
                        help="Scale mild initial-state offsets for distinct trials (0 nominal; 1 mild)")
    parser.add_argument("--reset-seed", type=int, default=0)
    args = parser.parse_args()
    if (not math.isfinite(args.seconds) or args.seconds <= 0 or args.num_envs <= 0
            or args.expert_bank_size < 1):
        parser.error("--seconds, --num-envs and --expert-bank-size must be positive")
    if not math.isfinite(args.reset_perturbation) or args.reset_perturbation < 0:
        parser.error("--reset-perturbation must be finite and nonnegative")
    if not args.ckpt.is_file() or not args.motions.is_file():
        parser.error("Checkpoint and motion library must exist")
    try:
        commands = parse_commands(args.commands)
    except ValueError as exc:
        parser.error(str(exc))
    report = evaluate_checkpoint(
        args.ckpt, args.motions, commands, args.seconds, args.num_envs, args.device,
        diagnostics=args.diagnostics, expert_bank_size=args.expert_bank_size,
        reset_perturbation=args.reset_perturbation, reset_seed=args.reset_seed,
    )
    output = args.out or args.ckpt.parent / f"eval_commands_{report['checkpoint_iteration']:07d}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
