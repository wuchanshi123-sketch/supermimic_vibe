"""Per-copy low-speed gait observations under seeded mild reset perturbations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import torch

from amp_gait_metrics import inspect_arrays
from supermimic import networks
from supermimic.amp_eval import evaluate_command, make_amp_eval_env
from supermimic.amp_train import load_amp_actor, task_config
from supermimic.motion_lib import MotionLibrary


def summarize_trials(trials):
    n = len(trials)
    alternating = sum(trial["gait_after_2s"]["alternating_steps_observed"] for trial in trials)
    sustained = sum(
        trial["survived_full_horizon"]
        and all(trial[key]["alternating_steps_observed"] and not trial[key]["flags"]
                for key in ("gait_after_2s", "gait_final_half"))
        for trial in trials
    )
    flags = Counter(flag["code"] for trial in trials for flag in trial["gait_after_2s"]["flags"])
    return {
        "copies": n, "alternating_steps_observed_after_2s_count": alternating,
        "alternating_steps_observed_after_2s_fraction": alternating / n if n else 0.,
        "surviving_alternating_unflagged_in_both_windows_count": sustained,
        "surviving_alternating_unflagged_in_both_windows_fraction": sustained / n if n else 0.,
        "flags_after_2s_by_code": dict(flags),
        "target_fraction_of_all_copies": .9,
        "meets_observation_target": bool(n and sustained / n >= .9),
        "quality_certified": False,
    }


@torch.no_grad()
def evaluate_robustness(actor, env, output, seconds=30., seed=59, perturbation=1., speed=.3):
    """Reuse fixed-command evaluation; the observer sees pre-reset successors."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    feet = [env.task.kin.body_names.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
    records = {key: [] for key in ("pos", "rot", "q", "v", "w", "qd", "action",
                                    "ankle_pos_w", "ankle_vel_w", "velocity_heading", "yaw_rate", "terminated")}
    last_action = None

    def policy(observation):
        nonlocal last_action
        last_action = actor(observation)
        return last_action

    def observe(step, state, info, active):
        positions, _, velocities, _ = env.task.body_kin(state)
        values = {**state, "action": last_action, "ankle_pos_w": positions[:, feet],
                  "ankle_vel_w": velocities[:, feet], "velocity_heading": info["velocity_heading"],
                  "yaw_rate": info["yaw_rate"], "terminated": info["terminated"]}
        for key in records:
            value = values[key]
            if value.dtype != torch.bool:
                if not bool(torch.isfinite(value[active]).all()):
                    raise FloatingPointError(f"Non-finite active gait trace: {key}")
                mask = active.reshape((len(active),) + (1,) * (value.ndim - 1))
                value = torch.where(mask, value, float("nan"))
            records[key].append(value.cpu().numpy().copy())

    steps = max(1, round(seconds / env.tcfg.dt))
    report = evaluate_command(policy, env, (speed, 0., 0.), steps,
                              observer=observe, reset_perturbation=perturbation)
    records = {key: np.stack(value) for key, value in records.items()}
    final_half_start = max(2., steps * env.tcfg.dt / 2)
    for trial in report["trials"]:
        index, length = trial["copy"], trial["observed_steps"]
        trace = {key: value[:length, index] for key, value in records.items()}
        trace.update(time_s=(np.arange(length) + 1) * env.tcfg.dt, dt=env.tcfg.dt,
                     command=np.array([speed, 0., 0.]), ankle_names=np.array(["left", "right"]),
                     joint_names=np.asarray(env.joint_names))
        trace_name = f"copy_{index:02d}_trace.npz"
        np.savez_compressed(output / trace_name, **trace)
        trial["trace"] = trace_name
        trial["gait_after_2s"] = inspect_arrays(trace, skip_seconds=2.)
        trial["gait_final_half"] = inspect_arrays(trace, skip_seconds=final_half_start)
    report.update(
        seed=seed, requested_seconds=seconds, reset_perturbation_scale=perturbation,
        final_half_window_starts_seconds=final_half_start,
        style_observations=summarize_trials(report["trials"]),
        protocol="First original episode only including terminal transition; trace stops at first done; reset episodes excluded",
        style_protocol="Existing amp_gait_metrics rules; observation count after 2 s and again over final half; denominator is all copies",
        initialization="Nominal standing plus amp_eval seeded mild uniform reset perturbations",
        friction=1., observation_noise=False, reset_noise=False, domain_randomization=False,
        deterministic_policy=True, requires_visual_review=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--motions", type=Path, default=Path("data/processed/amp_walk_50hz.npz"))
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument("--perturbation", type=float, default=1.)
    parser.add_argument("--speed", type=float, default=.3)
    parser.add_argument("--out", type=Path, required=True, help="Directory for summary.json and per-copy traces")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.num_envs < 1 or not math.isfinite(args.seconds) or args.seconds < 4.
            or not math.isfinite(args.speed) or not math.isfinite(args.perturbation) or args.perturbation < 0):
        parser.error("Need positive copy count, >=4 seconds, finite speed, and nonnegative finite perturbation")
    if not args.ckpt.is_file() or not args.motions.is_file():
        parser.error("Checkpoint and motion library must exist")
    networks.AMP["enabled"] = False
    actor, cfg, checkpoint = load_amp_actor(args.ckpt, args.device)
    lib = MotionLibrary(str(args.motions), args.device)
    if abs(lib.dt - task_config(cfg).dt) > 1e-8:
        parser.error("Motion/control intervals differ")
    env = make_amp_eval_env(lib, cfg, args.num_envs, args.device, args.seconds)
    report = evaluate_robustness(actor, env, args.out, args.seconds, args.seed, args.perturbation, args.speed)
    report.update(checkpoint=str(args.ckpt), checkpoint_iteration=int(checkpoint["it"]))
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["style_observations"], sort_keys=True), flush=True)
    print(f"Saved {args.out / 'summary.json'}; survived={report['survived']}/{args.num_envs}", flush=True)


if __name__ == "__main__":
    main()
