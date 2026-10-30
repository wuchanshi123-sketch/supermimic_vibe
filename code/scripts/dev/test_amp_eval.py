"""CPU checks for fixed-command evaluation and reset-episode exclusion."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
import unittest

import torch

from supermimic.amp_eval import evaluate_command, parse_commands


class FakeEnv:
    """Supply known first-episode data, then poisoned reset-episode samples."""

    def __init__(self, velocities, yaw_rates, fall_steps, truncation_steps=None, poison_resets=True):
        self.velocity = torch.tensor(velocities, dtype=torch.float32)
        self.yaw = torch.tensor(yaw_rates, dtype=torch.float32)
        self.num_envs = len(fall_steps)
        self.device = "cpu"
        self.tcfg = SimpleNamespace(dt=0.02)
        self.fall_steps = fall_steps
        self.truncation_steps = truncation_steps or [None] * self.num_envs
        self.poison_resets = poison_resets
        self.fixed_command = None

    def reset(self):
        self.steps_taken = 0
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool)
        self.command_at_reset = self.fixed_command.clone()

    def policy_inputs(self):
        return torch.zeros(self.num_envs, 384)

    def step(self, action):
        self.steps_taken += 1
        velocity = self.velocity.clone()
        yaw = self.yaw.clone()
        reward = torch.full((self.num_envs,), 0.02)
        if self.poison_resets:
            # A fixed-initial-state metric must ignore everything after first done,
            # including subsequent reset samples that are non-finite.
            velocity[self.finished] = float("nan")
            yaw[self.finished] = float("nan")
            reward[self.finished] = float("nan")
        terminated = torch.tensor([s == self.steps_taken for s in self.fall_steps])
        truncated = torch.tensor([s == self.steps_taken for s in self.truncation_steps])
        done = terminated | truncated
        self.finished |= done
        return {}, {
            "velocity_heading": velocity,
            "yaw_rate": yaw,
            "task_reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "done": done,
        }


def zero_actor(obs):
    return torch.zeros(obs.shape[0], 29)


class AMPCommandEvaluationTests(unittest.TestCase):
    def test_parses_command_triples(self):
        self.assertEqual(
            parse_commands(" 0,0,0 ; 0.5,-0.1,0.2; "),
            [(0.0, 0.0, 0.0), (0.5, -0.1, 0.2)],
        )

    def test_rejects_invalid_commands(self):
        invalid = ("", ";", "0,0", "0,0,0,0", "bad,0,0", "nan,0,0", "inf,0,0", "-inf,0,0")
        for command in invalid:
            with self.subTest(command=command), self.assertRaises(ValueError):
                parse_commands(command)

    def test_inactive_reset_nan_does_not_pollute_metrics(self):
        env = FakeEnv(
            velocities=[[0.2, 0.0, 0.0], [0.5, 0.0, 0.0]],
            yaw_rates=[0.0, 0.0],
            fall_steps=[2, None],
        )
        report = evaluate_command(zero_actor, env, (0.5, 0.0, 0.0), horizon_steps=4)
        torch.testing.assert_close(env.command_at_reset, torch.tensor([0.5, 0.0, 0.0]))
        self.assertEqual([r["observed_steps"] for r in report["trials"]], [2, 4])
        self.assertEqual(report["falls"], 1)
        self.assertEqual(report["survived"], 1)
        self.assertAlmostEqual(report["mean_coverage"], 0.75)
        self.assertAlmostEqual(report["mean_survival_seconds"], 0.06)
        self.assertAlmostEqual(report["mean_planar_error_mps_alive"], 0.1, places=6)
        self.assertAlmostEqual(report["planar_rmse_mps_alive"], math.sqrt(0.03), places=6)
        self.assertAlmostEqual(report["trials"][0]["task_return_alive"], 0.04)
        self.assertAlmostEqual(report["trials"][1]["task_return_alive"], 0.08)
        self.assertEqual(env.steps_taken, 4)
        # Recursively verifies every float, including per-copy quantities.
        json.dumps(report, allow_nan=False)

    def test_full_horizon_survivors_have_complete_coverage(self):
        env = FakeEnv(
            velocities=[[0.4, -0.1, 0.0]] * 3,
            yaw_rates=[0.3] * 3,
            fall_steps=[None] * 3,
        )
        report = evaluate_command(zero_actor, env, (0.4, -0.1, 0.3), horizon_steps=5)
        self.assertEqual(env.steps_taken, 5)
        self.assertEqual(report["survived"], 3)
        self.assertEqual(report["falls"], 0)
        self.assertEqual(report["survival_rate"], 1.0)
        self.assertEqual(report["mean_coverage"], 1.0)
        self.assertAlmostEqual(report["mean_survival_seconds"], 0.1)
        self.assertAlmostEqual(report["planar_rmse_mps_alive"], 0.0)
        self.assertAlmostEqual(report["yaw_rmse_radps_alive"], 0.0)
        for trial in report["trials"]:
            self.assertTrue(trial["survived_full_horizon"])
            self.assertFalse(trial["fell"])
            self.assertEqual(trial["observed_steps"], 5)
            self.assertAlmostEqual(trial["task_return_alive"], 0.1)

    def test_all_fall_break_keeps_original_horizon_coverage(self):
        env = FakeEnv(
            velocities=[[0.0, 0.0, 0.0]] * 3,
            yaw_rates=[0.0] * 3,
            fall_steps=[1, 2, 3],
        )
        report = evaluate_command(zero_actor, env, (0.0, 0.0, 0.0), horizon_steps=100)
        self.assertEqual(env.steps_taken, 3)
        self.assertEqual(report["falls"], 3)
        self.assertEqual(report["survived"], 0)
        self.assertEqual([r["observed_steps"] for r in report["trials"]], [1, 2, 3])
        self.assertAlmostEqual(report["mean_coverage"], 0.02)
        self.assertAlmostEqual(report["mean_survival_seconds"], 0.04)
        for trial, expected in zip(report["trials"], (0.01, 0.02, 0.03)):
            self.assertAlmostEqual(trial["coverage"], expected)
        json.dumps(report, allow_nan=False)

    def test_early_truncation_is_not_counted_as_a_fall(self):
        env = FakeEnv(
            velocities=[[0.0, 0.0, 0.0]],
            yaw_rates=[0.0],
            fall_steps=[None],
            truncation_steps=[2],
        )
        report = evaluate_command(zero_actor, env, (0.0, 0.0, 0.0), horizon_steps=4)
        self.assertEqual(env.steps_taken, 2)
        self.assertEqual(report["falls"], 0)
        self.assertEqual(report["survived"], 0)
        self.assertTrue(report["trials"][0]["truncated_early"])
        self.assertAlmostEqual(report["mean_coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
