"""Evaluate a fixed 40 s AMP command sequence without resetting between commands."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from supermimic import networks
from supermimic.amp_eval import make_amp_eval_env, perturb_standing_reset
from supermimic.amp_train import load_amp_actor, task_config
from supermimic.motion_lib import MotionLibrary


SEQUENCE = (
    (2., (0., 0., 0.)), (8., (0.5, 0., 0.)), (8., (0.8, 0., 0.)),
    (6., (0.3, 0., 0.)), (4., (0., 0., 0.)), (4., (0., 0., 0.5)),
    (8., (0.5, 0., 0.)),
)


@torch.no_grad()
def set_command(env, command):
    """Mirror native playback: update the command and latest historical frame."""
    env.fixed_command = torch.as_tensor(command, dtype=torch.float32, device=env.device).clone()
    env.command[:] = env.fixed_command
    env.command_age.zero_()
    env.obs_hist[:, -1] = env.amp_task.observation(
        env.state, env.prev_action, env.command, env.xi,
    )


def error_report(totals):
    count, planar_sq, yaw_abs, vx, vy, vz, yaw = totals.cpu().tolist()
    if count == 0:
        return {"observed_samples": 0, "planar_rmse_mps": None,
                "mean_abs_yaw_error_radps": None, "mean_velocity_heading_mps": None,
                "mean_yaw_rate_radps": None}
    return {
        "observed_samples": int(count), "planar_rmse_mps": math.sqrt(planar_sq / count),
        "mean_abs_yaw_error_radps": yaw_abs / count,
        "mean_velocity_heading_mps": [vx / count, vy / count, vz / count],
        "mean_yaw_rate_radps": yaw / count,
    }


@torch.no_grad()
def evaluate_sequence(actor, env, schedule=SEQUENCE, perturbation=1.0, seed=101):
    """Measure each original episode until its first done, including that step."""
    torch.manual_seed(seed)
    env.fixed_command = torch.as_tensor(schedule[0][1], device=env.device)
    env.reset()
    if perturbation:
        perturb_standing_reset(env, perturbation)
    n, device, dt = env.num_envs, env.device, env.tcfg.dt
    alive = torch.ones(n, dtype=torch.bool, device=device)
    fell, truncated = torch.zeros_like(alive), torch.zeros_like(alive)
    total_steps = torch.zeros(n, dtype=torch.long, device=device)
    segments, offset_steps = [], 0
    for index, (seconds, command) in enumerate(schedule):
        horizon = max(1, round(seconds / dt))
        set_command(env, command)
        entered = int(alive.sum())
        segment_steps = torch.zeros_like(total_steps)
        totals, settled = torch.zeros(7, device=device), torch.zeros(7, device=device)
        before_falls, before_truncated = int(fell.sum()), int(truncated.sum())
        for step in range(horizon):
            if not bool(alive.any()):
                break
            _, info = env.step(actor(env.policy_inputs()))
            velocity, yaw_rate = info["velocity_heading"], info["yaw_rate"]
            if not (bool(torch.isfinite(velocity[alive]).all())
                    and bool(torch.isfinite(yaw_rate[alive]).all())):
                raise FloatingPointError("Non-finite active sequence evaluation state")
            planar_sq = (velocity[:, :2] - velocity.new_tensor(command[:2])).square().sum(-1)
            yaw_abs = (yaw_rate - command[2]).abs()
            add = torch.cat((
                torch.stack((alive.sum(), torch.where(alive, planar_sq, 0.).sum(),
                             torch.where(alive, yaw_abs, 0.).sum())),
                torch.where(alive[:, None], velocity, 0.).sum(0),
                torch.where(alive, yaw_rate, 0.).sum().reshape(1),
            ))
            totals += add
            if step * dt >= 1.0:
                settled += add
            total_steps += alive.long()
            segment_steps += alive.long()
            fell |= info["terminated"] & alive
            truncated |= info["truncated"] & alive
            alive &= ~info["done"]
        result = {
            "segment": index, "start_seconds": offset_steps * dt,
            "end_seconds": (offset_steps + horizon) * dt, "command": list(command),
            "copies_entered": entered, "survived_to_segment_end": int(alive.sum()),
            "survival_rate_of_entered": float(alive.sum()) / entered if entered else None,
            "new_falls": int(fell.sum()) - before_falls,
            "new_truncations": int(truncated.sum()) - before_truncated,
            "coverage_of_entered": float(segment_steps.sum()) / (entered * horizon) if entered else 0.,
            "errors_alive": error_report(totals),
            "errors_after_first_second_alive": error_report(settled),
        }
        segments.append(result)
        offset_steps += horizon
        print(f"segment={index} command={command} alive={int(alive.sum())}/{n} "
              f"planar_RMSE={result['errors_alive']['planar_rmse_mps']}", flush=True)
    return {
        "copies": n, "horizon_seconds": offset_steps * dt, "seed": seed,
        "initial_state_perturbation_scale": perturbation,
        "initial_state_perturbation_protocol": "amp_eval.perturb_standing_reset; independent uniform mild offsets per copy",
        "survived_full_sequence": int(alive.sum()), "falls": int(fell.sum()),
        "truncations": int(truncated.sum()), "segments": segments,
        "statistics_protocol": "First original episode only, including terminal transition; no reset episode contributes",
        "command_change_protocol": "No reset; replace fixed command and latest observation; retain older observation history",
        "trials": [{"copy": i, "survived_full_sequence": bool(alive[i]),
                    "fell": bool(fell[i]), "truncated": bool(truncated[i]),
                    "observed_seconds": int(total_steps[i]) * dt} for i in range(n)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--motions", type=Path, default=Path("data/processed/amp_walk_50hz.npz"))
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", "--reset-seed", type=int, default=101)
    parser.add_argument("--perturbation", "--reset-perturbation", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.num_envs < 1 or not math.isfinite(args.perturbation) or args.perturbation < 0:
        parser.error("Environment count must be positive and perturbation finite and nonnegative")
    if not args.ckpt.is_file() or not args.motions.is_file():
        parser.error("Checkpoint and motion library must exist")
    networks.AMP["enabled"] = False
    actor, cfg, checkpoint = load_amp_actor(args.ckpt, args.device)
    lib = MotionLibrary(str(args.motions), args.device)
    if abs(lib.dt - task_config(cfg).dt) > 1e-8:
        parser.error("Expert library/control intervals differ")
    env = make_amp_eval_env(lib, cfg, args.num_envs, args.device, sum(s for s, _ in SEQUENCE))
    report = evaluate_sequence(actor, env, perturbation=args.perturbation, seed=args.seed)
    report.update(checkpoint=str(args.ckpt), checkpoint_iteration=int(checkpoint["it"]),
                  friction=1.0, observation_noise=False, reset_noise=False,
                  domain_randomization=False, deterministic_policy=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.out}: falls={report['falls']}/{args.num_envs}", flush=True)


if __name__ == "__main__":
    main()
