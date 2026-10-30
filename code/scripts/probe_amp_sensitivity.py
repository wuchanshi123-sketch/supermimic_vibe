"""Compare AMP world-model action derivatives with simulator finite differences.

Default protocol: clone one nominal standing state into 59 worlds, perturb each
of 29 actions by +/-epsilon on the first step, then replay the unperturbed real
actor's action sequence open loop in every world and in the model. Report final
root vx/vy, joint position/velocity Jacobians and discounted mean reward gradients.
No checkpoint weights or normalizer statistics are updated. Run --self-test for
dependency-light CPU checks without a checkpoint or simulator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def central_jacobian(samples, epsilon):
    """Rows are baseline, positive input perturbations, negative perturbations."""
    if samples.ndim != 2 or samples.shape[0] < 3 or samples.shape[0] % 2 != 1 or epsilon <= 0:
        raise ValueError("Need (1+2*inputs, outputs) samples and positive epsilon")
    inputs = (samples.shape[0] - 1) // 2
    return ((samples[1:1 + inputs] - samples[1 + inputs:]) / (2 * epsilon)).T


def compare_jacobians(predicted, actual, valid=None, sign_threshold=1e-6):
    """Norm, cosine and active-entry sign agreement; zero references stay explicit."""
    if predicted.shape != actual.shape or predicted.ndim != 2:
        raise ValueError("Jacobian shapes must match (outputs, inputs)")
    if valid is None:
        valid = torch.ones(actual.shape[1], dtype=torch.bool, device=actual.device)
    valid = valid & torch.isfinite(predicted).all(0) & torch.isfinite(actual).all(0)
    p, a = predicted[:, valid].double().flatten(), actual[:, valid].double().flatten()
    pn, an = p.norm(), a.norm()
    active = a.abs() > sign_threshold
    return {
        "valid_input_columns": int(valid.sum()),
        "predicted_frobenius_norm": float(pn),
        "actual_frobenius_norm": float(an),
        "norm_ratio": float(pn / an) if an > 0 else None,
        "relative_frobenius_error": float((p - a).norm() / an) if an > 0 else None,
        "cosine": float(torch.dot(p, a) / (pn * an)) if pn > 0 and an > 0 else None,
        "active_sign_entries": int(active.sum()),
        "sign_agreement": float((p[active].sign() == a[active].sign()).double().mean()) if bool(active.any()) else None,
        "sign_threshold": sign_threshold,
    }


def reward_direction(predicted, actual, valid, sign_threshold):
    result = compare_jacobians(predicted[None], actual[None], valid, sign_threshold)
    valid = valid & torch.isfinite(predicted) & torch.isfinite(actual)
    p, a = predicted[valid].double(), actual[valid].double()
    result["real_slope_along_normalized_predicted_ascent"] = float(torch.dot(a, p) / p.norm()) if p.norm() > 0 else None
    result["predicted_gradient_by_joint"] = [float(x) if bool(ok) else None for x, ok in zip(predicted, valid)]
    result["real_fd_gradient_by_joint"] = [float(x) if bool(ok) else None for x, ok in zip(actual, valid)]
    return result


def cpu_self_test():
    matrix = torch.tensor([[1., 2., -1.], [0., 3., 4.]], dtype=torch.float64)
    epsilon = 0.1
    offsets = torch.cat((torch.zeros(1, 3), epsilon * torch.eye(3), -epsilon * torch.eye(3))).double()
    samples = offsets @ matrix.T + torch.tensor([0.4, -0.7])
    fd = central_jacobian(samples, epsilon)
    torch.testing.assert_close(fd, matrix, atol=1e-6, rtol=1e-6)
    analytic = torch.autograd.functional.jacobian(lambda delta: matrix @ delta, torch.zeros(3, dtype=torch.float64), vectorize=True)
    torch.testing.assert_close(fd, analytic, atol=1e-6, rtol=1e-6)
    exact = compare_jacobians(matrix, matrix)
    reversed_sign = compare_jacobians(-matrix, matrix)
    assert abs(exact["cosine"] - 1) < 1e-12 and exact["sign_agreement"] == 1
    assert abs(reversed_sign["cosine"] + 1) < 1e-12 and reversed_sign["sign_agreement"] == 0
    zero = compare_jacobians(torch.zeros_like(matrix), torch.zeros_like(matrix))
    assert zero["cosine"] is None and zero["sign_agreement"] is None
    valid = torch.tensor([True, False, True])
    corrupted = matrix.clone(); corrupted[:, 1] = float("nan")
    masked = compare_jacobians(corrupted, matrix, valid)
    assert masked["valid_input_columns"] == 2 and abs(masked["cosine"] - 1) < 1e-12
    direction = reward_direction(torch.tensor([1., 0.]), torch.tensor([-2., 3.]), torch.ones(2, dtype=torch.bool), 1e-6)
    assert direction["real_slope_along_normalized_predicted_ascent"] == -2
    print("CPU linear finite-difference, cosine/sign, zero/mask and reward-direction checks passed", flush=True)


def state_readout(state):
    return torch.cat((state["v"][..., :2], state["q"], state["qd"]), -1)


def probe(args):
    # Lazy simulator imports keep --self-test usable with only CPU torch.
    from supermimic import networks
    from supermimic.amp_eval import make_amp_eval_env, parse_commands
    from supermimic.amp_prior import AMPFeatureExtractor, AMPPrior
    from supermimic.amp_train import load_amp_actor
    from supermimic.motion_lib import MotionLibrary
    from supermimic.networks import WorldModel, apply_delta, run_mlp
    from supermimic.task import NJ, XI_DIM, state_to_flat

    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    networks.AMP["enabled"] = False
    command = parse_commands(args.command)
    if len(command) != 1:
        raise ValueError("Probe exactly one fixed command per invocation")
    actor, cfg, ck = load_amp_actor(args.ckpt, args.device)
    lib = MotionLibrary(str(args.motions or cfg.motions), args.device)
    count = 1 + 2 * NJ
    env = make_amp_eval_env(lib, cfg, count, args.device, (max(args.horizons) + args.settle_steps + 2) * lib.dt)
    env.fixed_command = torch.tensor(command[0], device=args.device)
    env.rsi_probability = float(args.rsi_frame is not None)
    env.reset()
    if args.rsi_frame is not None:
        if not 0 <= args.rsi_frame < lib.num_frames:
            raise ValueError("RSI frame is outside the motion library")
        ids = torch.arange(count, device=args.device)
        env.reset_envs(ids, torch.full((count,), args.rsi_frame, dtype=torch.long, device=args.device))
        env._finish_resets(ids, force=True)
    with torch.no_grad():
        for _ in range(args.settle_steps):
            baseline = actor(env.policy_inputs()[:1])
            _, info = env.step(baseline.expand(count, -1))
            if bool(info["done"].any()):
                raise ValueError("Preparation terminated; reduce --settle-steps or choose another state")
        # Reset all solver warm-start state, then clone physical state/history.
        qpos, qvel = env.sim.data.qpos[:1].clone(), env.sim.data.qvel[:1].clone()
        history, previous = env.obs_hist[:1].clone(), env.prev_action[:1].clone()
        env.reset()
        env.sim.data.qpos[:] = qpos.expand(count, -1)
        env.sim.data.qvel[:] = qvel.expand(count, -1)
        env.obs_hist = history.expand(count, -1, -1).clone()
        env.prev_action = previous.expand(count, -1).clone()
        env.state = env.read_state()
        initial = {key: value[:1].clone() for key, value in env.state.items()}
        xi = env.xi[:1].clone()
        initial_difference = float((state_to_flat(env.state) - state_to_flat(env.state)[:1]).abs().max())
        context_difference = float((env.xi - xi).abs().max())
    feat_dim = env.task.features_fast(initial).shape[-1]
    wm = WorldModel(env.task, feat_dim, XI_DIM, hidden=(cfg.wm_width,) * 3).to(args.device)
    wm.load_state_dict(ck["wm"]); wm.requires_grad_(False); wm.eval()
    prior = AMPPrior(style_scale=cfg.style_scale, device=args.device)
    prior.load_state_dict(ck["amp_prior"]); prior.requires_grad_(False); prior.eval()
    features = AMPFeatureExtractor(env.task)
    iteration = int(ck["it"])
    del ck
    identity = torch.eye(NJ, device=args.device)
    offsets = torch.cat((torch.zeros(1, NJ, device=args.device), args.epsilon * identity, -args.epsilon * identity))
    real, baseline_actions = {}, []
    clipped_inputs = torch.zeros(NJ, dtype=torch.bool, device=args.device)
    with torch.no_grad():
        alive = torch.ones(count, dtype=torch.bool, device=args.device)
        phi = features(env.state)
        task_return = mixed_return = torch.zeros(count, device=args.device)
        for t in range(max(args.horizons)):
            baseline = actor(env.policy_inputs()[:1]).detach()
            baseline_actions.append(baseline)
            raw = baseline.expand(count, -1) + (offsets if t == 0 or args.perturbation_mode == "constant-offset" else 0)
            action = raw.clamp(-cfg.action_limit, cfg.action_limit)
            clipped = (raw != action).any(-1)
            clipped_inputs |= clipped[1:1 + NJ] | clipped[1 + NJ:]
            valid_endpoint = alive.clone()
            successor, info = env.step(action)
            next_phi = features(successor)
            style, _ = prior.reward(phi, next_phi)
            task = info["task_reward"]
            mixed = cfg.task_weight * task + (1 - cfg.task_weight) * style
            task_return = task_return + cfg.gamma**t * alive * task
            mixed_return = mixed_return + cfg.gamma**t * alive * mixed
            horizon = t + 1
            if horizon in args.horizons:
                real[horizon] = {
                    "output": torch.cat((state_readout(successor), task_return[:, None] / horizon, mixed_return[:, None] / horizon), -1),
                    "valid": valid_endpoint,
                    "terminal": info["done"].clone(),
                }
            alive &= ~info["done"]
            phi = next_phi

    def imagined(offset):
        batch = len(offset)
        state = {key: value.expand((batch,) + value.shape[1:]) for key, value in initial.items()}
        prev = previous.expand(batch, -1)
        context = xi.expand(batch, -1)
        cmd = env.fixed_command.expand(batch, -1)
        phi = features(state)
        alive = torch.ones(batch, device=args.device)
        task_return = mixed_return = torch.zeros(batch, device=args.device)
        outputs = {}
        for t, baseline in enumerate(baseline_actions):
            action = (baseline + (offset if t == 0 or args.perturbation_mode == "constant-offset" else 0)).clamp(-cfg.action_limit, cfg.action_limit)
            if action.shape[0] != batch:
                action = action.expand(batch, -1)
            delta = run_mlp(wm.net, wm.in_norm(wm.inputs(state, action, context)))
            successor = apply_delta(state, delta * wm.out_norm.std + wm.out_norm.mean)
            kin = env.task.body_kin(successor)
            next_phi = features(successor, kin)
            task, _ = env.amp_task.reward(successor, action, prev, cmd, kin=kin)
            style, _ = prior.reward(phi, next_phi)
            mixed = cfg.task_weight * task + (1 - cfg.task_weight) * style
            task_return = task_return + cfg.gamma**t * alive * task
            mixed_return = mixed_return + cfg.gamma**t * alive * mixed
            _, terminal = env.amp_task.termination(successor)
            alive = alive * (1 - terminal)
            horizon = t + 1
            if horizon in args.horizons:
                outputs[horizon] = torch.cat((state_readout(successor), task_return[:, None] / horizon, mixed_return[:, None] / horizon), -1)
            state, prev, phi = successor, action, next_phi
        return outputs

    with torch.no_grad():
        wm_finite = imagined(offsets)
    zero = torch.zeros(NJ, device=args.device)
    def flattened(offset):
        values = imagined(offset[None])
        return torch.cat([values[h][0] for h in args.horizons])
    autograd_jacobian = torch.autograd.functional.jacobian(flattened, zero, vectorize=True)
    groups = {"root_vxy_world": slice(0, 2), "root_vx_world": slice(0, 1), "root_vy_world": slice(1, 2),
              "joint_position": slice(2, 2 + NJ), "joint_velocity": slice(2 + NJ, 2 + 2 * NJ)}
    output_dim = 2 + 2 * NJ + 2
    rows = []
    for i, horizon in enumerate(args.horizons):
        observed = real[horizon]
        valid = observed["valid"][1:1 + NJ] & observed["valid"][1 + NJ:] & observed["valid"][0]
        sim_fd = central_jacobian(observed["output"], args.epsilon)
        model_fd = central_jacobian(wm_finite[horizon], args.epsilon)
        analytic = autograd_jacobian[i * output_dim:(i + 1) * output_dim]
        row = {"horizon_steps": horizon, "valid_joint_pairs": int(valid.sum()),
               "terminal_endpoint_worlds": int(observed["terminal"].sum()), "state_groups": {}}
        for name, selection in groups.items():
            row["state_groups"][name] = {
                "wm_autograd_vs_sim_fd": compare_jacobians(analytic[selection], sim_fd[selection], valid, args.sign_threshold),
                "wm_fd_vs_sim_fd": compare_jacobians(model_fd[selection], sim_fd[selection], valid, args.sign_threshold),
                "wm_autograd_vs_wm_fd": compare_jacobians(analytic[selection], model_fd[selection], valid, args.sign_threshold),
            }
        row["reward_directions"] = {}
        for name, selection in (("task", -2), ("mixed_task_amp", -1)):
            row["reward_directions"][name] = reward_direction(analytic[selection], sim_fd[selection], valid, args.sign_threshold)
            row["reward_directions"][name]["wm_autograd_vs_wm_fd"] = compare_jacobians(analytic[selection][None], model_fd[selection][None], valid, args.sign_threshold)
        row["baseline_outputs"] = {"sim": observed["output"][0].cpu().tolist(), "wm": wm_finite[horizon][0].cpu().tolist()}
        if args.include_jacobians:
            row["jacobians_outputs_by_inputs"] = {"sim_fd": sim_fd.cpu().tolist(), "wm_fd": model_fd.cpu().tolist(), "wm_autograd": analytic.cpu().tolist()}
        rows.append(row)
    return {
        "checkpoint": str(args.ckpt), "checkpoint_iteration": iteration, "seed": args.seed,
        "command": command[0], "epsilon_normalized_action": args.epsilon,
        "initialization": "nominal standing" if args.rsi_frame is None else f"motion frame {args.rsi_frame}",
        "settle_steps": args.settle_steps, "cloned_worlds": count,
        "initial_physical_state_max_difference": initial_difference,
        "initial_context_max_difference": context_difference,
        "perturbation_mode": args.perturbation_mode,
        "protocol": "First-step impulse by default, followed by the same recorded real baseline actor sequence in every simulator/model world; no policy feedback in perturbed worlds",
        "reward_protocol": "Mean discounted task and task/AMP mixture, no critic tail; real returns stop at hard termination, model returns use actor soft continuation",
        "state_output_order": ["root_vx_world_mps", "root_vy_world_mps", "joint_position_rad[29]", "joint_velocity_radps[29]", "discounted_mean_task_reward", "discounted_mean_mixed_reward"],
        "joint_names": env.joint_names,
        "action_clipped_joint_indices": clipped_inputs.nonzero().flatten().cpu().tolist(),
        "baseline_action_sequence": torch.cat(baseline_actions).cpu().tolist(),
        "rows": rows,
        "limitations": [
            "Finite differences are epsilon-dependent secants through contact dynamics, not exact simulator derivatives; compare multiple epsilon values.",
            "Terminal successors are included; a pair is excluded after an earlier termination or baseline termination.",
            "Action bounds are applied identically; clipped perturbations are explicitly listed.",
            "Initial solver warm-start data are reset in every clone; results describe this local state/context, not all policy states.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--motions", type=Path)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--command", default="0.5,0,0")
    parser.add_argument("--rsi-frame", type=int)
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--perturbation-mode", choices=("first-step", "constant-offset"), default="first-step")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--sign-threshold", type=float, default=1e-6)
    parser.add_argument("--include-jacobians", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        cpu_self_test()
        return
    if args.ckpt is None or not args.ckpt.is_file():
        parser.error("--ckpt must identify an existing AMP checkpoint")
    if not math.isfinite(args.epsilon) or args.epsilon <= 0 or min(args.horizons) < 1:
        parser.error("Epsilon and horizons must be positive")
    if args.settle_steps < 0 or args.cpu_threads < 1 or not math.isfinite(args.sign_threshold) or args.sign_threshold < 0:
        parser.error("Invalid preparation, thread or sign-threshold setting")
    args.horizons = sorted(set(args.horizons))
    report = probe(args)
    output = args.out or args.ckpt.parent / f"probe_sensitivity_{report['checkpoint_iteration']:07d}_eps{args.epsilon:g}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
