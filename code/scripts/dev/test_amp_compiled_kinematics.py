"""Compare AMP compiled/eager FK values and state/action gradients on a checkpoint.

Example, from the repository root after sourcing scripts/local_runtime.sh:
  "$SMIMIC_PY" code/scripts/dev/test_amp_compiled_kinematics.py \
    --ckpt runs/amp_walk_24gb_train/ckpt_0002000.pt \
    --out runs/setup/amp_compiled_kinematics_parity.json

Uses one simulator environment to construct the robot kinematics. Batch inputs
are expert states with bounded actions passed through the checkpoint's frozen
world model; no replay or trainer is allocated and no weights are updated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from supermimic import networks
from supermimic.amp_eval import make_amp_eval_env
from supermimic.amp_prior import AMPFeatureExtractor
from supermimic.amp_train import AMPTrainer, load_amp_actor
from supermimic.motion_lib import MotionLibrary
from supermimic.networks import WorldModel, apply_delta, run_mlp
from supermimic.task import XI_DIM


def run_parity(args):
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    networks.AMP["enabled"] = False
    actor, cfg, checkpoint = load_amp_actor(args.ckpt, args.device)
    lib = MotionLibrary(str(args.motions or cfg.motions), args.device)
    env = make_amp_eval_env(lib, cfg, 1, args.device, 5)
    env.fixed_command = torch.zeros(3, device=args.device)
    env.reset()
    feat_dim = env.task.features_fast(env.state).shape[-1]
    wm = WorldModel(env.task, feat_dim, XI_DIM, hidden=(cfg.wm_width,) * 3).to(args.device)
    wm.load_state_dict(checkpoint["wm"])
    wm.requires_grad_(False)
    wm.eval()
    iteration = int(checkpoint["it"])
    del checkpoint, actor
    trainer = AMPTrainer.__new__(AMPTrainer)
    trainer.env, trainer.task = env, env.task
    trainer.features = AMPFeatureExtractor(env.task)

    # The helper must preserve the original eager path when compilation is off.
    for actual, expected in zip(trainer._body_kin(env.state), env.task.body_kin(env.state)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    env.kin_c = torch.compile(env.kin.forward, dynamic=False)
    rows = []
    for batch in args.batches:
        frame = torch.randint(lib.num_frames, (batch,), device=args.device)
        base = {key: value for key, value in lib.frame(frame).items() if key != "quat"}
        action0 = (torch.rand(batch, 29, device=args.device) * 2 - 1) * min(2.0, cfg.action_limit)
        xi = torch.zeros(batch, XI_DIM, device=args.device)
        xi[:, 0] = 1
        previous = torch.zeros_like(action0)
        command = torch.rand(batch, 3, device=args.device) - 0.5
        projection = torch.randn(195, device=args.device) / 195**0.5
        results = []
        for compiled in (False, True):
            state = {key: value.clone().requires_grad_(True) for key, value in base.items()}
            action = action0.clone().requires_grad_(True)
            delta = run_mlp(wm.net, wm.in_norm(wm.inputs(state, action, xi)))
            next_state = apply_delta(state, delta * wm.out_norm.std + wm.out_norm.mean)
            kin = trainer._body_kin(next_state) if compiled else env.task.body_kin(next_state)
            features = trainer.features(next_state, kin)
            reward, _ = env.amp_task.reward(next_state, action, previous, command, kin=kin)
            loss = (features * projection).sum() + reward.sum()
            gradients = torch.autograd.grad(loss, (action, *state.values()))
            results.append((features.detach(), reward.detach(), tuple(g.detach() for g in gradients)))
        ef, er, eg = results[0]
        cf, cr, cg = results[1]
        if not all(bool(torch.isfinite(value).all()) for value in (ef, er, cf, cr)):
            raise AssertionError(f"Non-finite AMP features or reward at batch {batch}")
        torch.testing.assert_close(cf, ef, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(cr, er, rtol=1e-4, atol=1e-5)
        gradient_errors = {}
        for key, actual, expected in zip(("action", *base.keys()), cg, eg):
            if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
                raise AssertionError(f"Non-finite {key} gradient at batch {batch}")
            torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-5)
            gradient_errors[key] = float((actual - expected).abs().max())
        row = {
            "batch": batch,
            "feature_max_absolute_error": float((cf - ef).abs().max()),
            "reward_max_absolute_error": float((cr - er).abs().max()),
            "gradient_max_absolute_errors": gradient_errors,
            "all_finite": True,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {
        "checkpoint": str(args.ckpt), "checkpoint_iteration": iteration,
        "device": args.device, "seed": args.seed,
        "eager_helper_exact": True, "batches": rows,
        "feature_reward_tolerance": {"rtol": 1e-4, "atol": 1e-5},
        "gradient_tolerance": {"rtol": 5e-4, "atol": 5e-5},
        "gradient_path": "state/action -> frozen world model -> compiled/eager FK -> AMP features + analytic task reward",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--motions", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[256, 512, 4096])
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.ckpt.is_file() or (args.motions is not None and not args.motions.is_file()):
        parser.error("Checkpoint and optional motion file must exist")
    if min(*args.batches, args.cpu_threads) < 1:
        parser.error("Batch and CPU thread counts must be positive")
    report = run_parity(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
