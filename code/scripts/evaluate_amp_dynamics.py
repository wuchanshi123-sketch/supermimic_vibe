"""Measure AMP world-model errors on fresh deterministic policy transitions.

Run from the repository root with the local runtime/PYTHONPATH configured. This
loads only the actor, world model and a small simulator batch, never a trainer or
its replay. Predictions replay recorded actions open loop; no policy is run on
predicted states. Windows may end at a terminal successor but never cross resets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from supermimic import networks
from supermimic.amp_eval import DEFAULT_COMMANDS, make_amp_eval_env, parse_commands
from supermimic.amp_train import load_amp_actor
from supermimic.motion_lib import MotionLibrary
from supermimic.networks import WorldModel, apply_delta, run_mlp, state_error
from supermimic.task import NJ, XI_DIM, flat_to_state, state_to_flat


def parse_horizons(text):
    try:
        horizons = sorted({int(x.strip()) for x in text.split(",")})
    except ValueError as exc:
        raise ValueError("Horizons must be comma-separated positive integers") from exc
    if not horizons or horizons[0] < 1:
        raise ValueError("Horizons must be positive")
    return horizons


def valid_windows(episodes, horizon):
    """Return (time, env) starts with every transition in the same episode."""
    if horizon > episodes.shape[0]:
        empty = torch.empty(0, dtype=torch.long, device=episodes.device)
        return empty, empty
    windows = episodes.unfold(0, horizon, 1)
    valid = (windows == windows[..., :1]).all(-1)
    return valid.nonzero(as_tuple=True)


@torch.no_grad()
def collect_transitions(actor, env, command, steps):
    env.fixed_command = torch.tensor(command, device=env.device)
    env.reset()
    records = {key: [] for key in ("state", "successor", "action", "xi", "episode", "done", "observation")}
    for _ in range(steps):
        obs = env.policy_inputs().clone()
        action = actor(obs)
        records["state"].append(state_to_flat(env.state))
        records["action"].append(action)
        records["xi"].append(env.xi.clone())
        records["episode"].append(env.ep_id.clone())
        records["observation"].append(obs)
        successor, info = env.step(action)
        records["successor"].append(state_to_flat(successor))
        records["done"].append(info["done"].clone())
    return {key: torch.stack(values) for key, values in records.items()}


def clipping_metrics(raw, normalizer):
    z = (raw - normalizer.mean) / normalizer.std
    clipped = torch.zeros_like(z, dtype=torch.bool) if normalizer.clip is None else z.abs() > normalizer.clip
    return {
        "element_fraction": float(clipped.float().mean()),
        "sample_fraction": float(clipped.any(-1).float().mean()),
        "normalized_abs_max": float(z.abs().max()),
        "clip_threshold": normalizer.clip,
    }


@torch.no_grad()
def action_metrics(actor, wm, data):
    action = data["action"].flatten(0, 1)
    state = flat_to_state(data["state"].flatten(0, 1))
    xi = data["xi"].flatten(0, 1)
    inputs = wm.inputs(state, action, xi)
    feat_dim = inputs.shape[-1] - NJ - XI_DIM
    raw = (inputs - wm.in_norm.mean) / wm.in_norm.std
    action_z = raw[:, feat_dim:feat_dim + NJ]
    threshold = wm.in_norm.clip
    clipped = torch.zeros_like(action_z, dtype=torch.bool) if threshold is None else action_z.abs() > threshold
    return {
        "mean_abs": float(action.abs().mean()),
        "rms": float(action.square().mean().sqrt()),
        "abs_max": float(action.abs().max()),
        "action_limit": actor.action_limit,
        "fraction_at_95_percent_limit": float((action.abs() >= 0.95 * actor.action_limit).float().mean()),
        "fraction_at_99_percent_limit": float((action.abs() >= 0.99 * actor.action_limit).float().mean()),
        "wm_action_normalizer": {
            "element_fraction_clipped": float(clipped.float().mean()),
            "sample_fraction_clipped": float(clipped.any(-1).float().mean()),
            "normalized_abs_max": float(action_z.abs().max()),
            "clip_threshold": threshold,
        },
        "wm_all_input_normalizer": clipping_metrics(inputs, wm.in_norm),
        "actor_observation_normalizer": clipping_metrics(data["observation"].flatten(0, 1), actor.obs_norm),
    }


def physical_squared_errors(predicted, target, wm):
    dp = predicted["pos"] - target["pos"]
    rel = target["rot"].transpose(-1, -2) @ predicted["rot"]
    cosine = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    skew = torch.stack((rel[..., 2, 1] - rel[..., 1, 2],
                        rel[..., 0, 2] - rel[..., 2, 0],
                        rel[..., 1, 0] - rel[..., 0, 1]), -1)
    angle = torch.atan2(skew.norm(dim=-1) / 2, cosine)
    return {
        "root_position_vector_rmse_m": dp.square().sum(-1),
        "root_height_rmse_m": dp[..., 2].square(),
        "root_rotation_angle_rmse_rad": angle.square(),
        "root_planar_velocity_vector_rmse_mps": (predicted["v"][..., :2] - target["v"][..., :2]).square().sum(-1),
        "root_angular_velocity_vector_rmse_radps": (predicted["w"] - target["w"]).square().sum(-1),
        "joint_position_per_joint_rmse_rad": (predicted["q"] - target["q"]).square().mean(-1),
        "joint_velocity_per_joint_rmse_radps": (predicted["qd"] - target["qd"]).square().mean(-1),
        "normalized_state_mse": (state_error(predicted, target) / wm.out_norm.std).square().mean(-1),
    }


@torch.no_grad()
def evaluate_horizon(wm, data, horizon, batch_size):
    time, env = valid_windows(data["episode"], horizon)
    count = len(time)
    if not count:
        return {"horizon_steps": horizon, "samples": 0, "metrics": None}
    totals = {}
    finite_count = terminal_count = 0
    for start in range(0, count, batch_size):
        t, e = time[start:start + batch_size], env[start:start + batch_size]
        predicted = flat_to_state(data["state"][t, e])
        xi = data["xi"][t, e]
        for j in range(horizon):
            action = data["action"][t + j, e]
            delta = run_mlp(wm.net, wm.in_norm(wm.inputs(predicted, action, xi)))
            predicted = apply_delta(predicted, delta * wm.out_norm.std + wm.out_norm.mean)
        target = flat_to_state(data["successor"][t + horizon - 1, e])
        errors = physical_squared_errors(predicted, target, wm)
        finite = torch.stack([torch.isfinite(value) for value in errors.values()]).all(0)
        finite_count += int(finite.sum())
        terminal_count += int(data["done"][t + horizon - 1, e].sum())
        for key, value in errors.items():
            totals[key] = totals.get(key, 0.0) + float(value[finite].double().sum())
    metrics = None
    if finite_count:
        means = {key: value / finite_count for key, value in totals.items()}
        metrics = {key: value if key.endswith("mse") else math.sqrt(max(0, value)) for key, value in means.items()}
        metrics["normalized_state_rmse"] = math.sqrt(max(0, means["normalized_state_mse"]))
    return {
        "horizon_steps": horizon,
        "samples": count,
        "finite_samples": finite_count,
        "nonfinite_samples": count - finite_count,
        "terminal_successor_samples": terminal_count,
        "metrics": metrics,
    }


@torch.no_grad()
def evaluate(args):
    networks.AMP["enabled"] = False
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    actor, cfg, checkpoint = load_amp_actor(args.ckpt, args.device)
    library_path = args.motions or Path(cfg.motions)
    lib = MotionLibrary(str(library_path), args.device)
    env = make_amp_eval_env(lib, cfg, args.num_envs, args.device, args.seconds)
    env.rsi_probability = args.rsi_probability
    env.fixed_command = torch.tensor(args.commands[0], device=args.device)
    env.reset()
    feat_dim = env.task.features_fast(env.state).shape[-1]
    wm = WorldModel(env.task, feat_dim, XI_DIM, hidden=(cfg.wm_width,) * 3).to(args.device)
    wm.load_state_dict(checkpoint["wm"])
    wm.eval()
    iteration = int(checkpoint["it"])
    del checkpoint
    steps = max(1, round(args.seconds / env.tcfg.dt))
    commands = []
    for command in args.commands:
        data = collect_transitions(actor, env, command, steps)
        result = {
            "command": command,
            "transitions": steps * args.num_envs,
            "episodes_observed": int(data["episode"].unique().numel()),
            "completed_episodes": int(data["done"].sum()),
            "actions": action_metrics(actor, wm, data),
            "horizons": [evaluate_horizon(wm, data, horizon, args.batch_size) for horizon in args.horizons],
        }
        commands.append(result)
        print(f"command={tuple(command)} transitions={result['transitions']} resets={result['completed_episodes']}", flush=True)
        for row in result["horizons"]:
            metrics = row["metrics"]
            if metrics:
                print(f"  H={row['horizon_steps']} n={row['finite_samples']}/{row['samples']} "
                      f"pos={metrics['root_position_vector_rmse_m']:.4f}m "
                      f"q={metrics['joint_position_per_joint_rmse_rad']:.4f}rad "
                      f"vxy={metrics['root_planar_velocity_vector_rmse_mps']:.4f}m/s "
                      f"normalized_MSE={metrics['normalized_state_mse']:.4f}", flush=True)
        del data
    return {
        "checkpoint": str(args.ckpt), "checkpoint_iteration": iteration,
        "motions": str(library_path), "seed": args.seed,
        "seconds_per_command": steps * env.tcfg.dt, "num_envs": args.num_envs,
        "rsi_probability": args.rsi_probability,
        "observation_noise": False, "reset_noise": False, "domain_randomization": False,
        "friction": 1.0, "control_dt_seconds": env.tcfg.dt,
        "policy": "Frozen checkpoint actor, deterministic; no normalizer updates or exploration noise",
        "prediction_protocol": "Fresh current-policy simulator transitions; replay recorded actions open loop with frozen checkpoint world model",
        "window_protocol": "All available overlapping windows per horizon; same episode throughout; terminal pre-reset successor included",
        "metric_protocol": "Final-state errors at each horizon; vector RMSE is sqrt(mean squared vector norm), per-joint RMSE averages joints and samples; normalized MSE matches trainer state_error/out_norm.std",
        "limitations": [
            "Overlapping windows are correlated and horizon-specific sample populations differ.",
            "Nominal-standing deterministic copies share the same initial state; RSI provides sampled motion-state variation.",
            "This measures open-loop dynamics accuracy, not closed-loop imagined policy accuracy or walking success.",
        ],
        "commands": commands,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--motions", type=Path)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--horizons", default="1,8,16,32")
    parser.add_argument("--rsi-probability", type=float, default=0.0)
    parser.add_argument("--commands", default=DEFAULT_COMMANDS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if not args.ckpt.is_file():
        parser.error("Checkpoint does not exist")
    if not math.isfinite(args.seconds) or args.seconds <= 0 or min(args.num_envs, args.batch_size, args.cpu_threads) < 1:
        parser.error("Duration and batch/thread counts must be positive")
    if not 0 <= args.rsi_probability <= 1:
        parser.error("RSI probability must be in [0,1]")
    try:
        args.horizons = parse_horizons(args.horizons)
        args.commands = parse_commands(args.commands)
    except ValueError as exc:
        parser.error(str(exc))
    report = evaluate(args)
    output = args.out or args.ckpt.parent / f"eval_dynamics_{report['checkpoint_iteration']:07d}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
