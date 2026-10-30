"""CPU regression checks for continuous command changes and first-episode masks."""

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_amp_sequence import evaluate_sequence, set_command


class FakeEnv:
    def __init__(self):
        self.num_envs, self.device, self.resets = 2, "cpu", 0
        self.tcfg = SimpleNamespace(dt=.02)
        self.amp_task = SimpleNamespace(observation=lambda state, previous, command, xi: command.clone())

    def reset(self):
        self.resets += 1
        self.step_count = 0
        self.command = self.fixed_command.repeat(2, 1)
        self.command_age = torch.zeros(2, dtype=torch.long)
        self.obs_hist = self.command[:, None].repeat(1, 4, 1)
        self.prev_action, self.xi, self.state = torch.zeros(2, 29), torch.zeros(2, 33), {}

    def policy_inputs(self):
        return self.obs_hist.flatten(1)

    def step(self, action):
        self.step_count += 1
        velocity = torch.cat((self.command[:, :2], torch.zeros(2, 1)), -1)
        yaw = self.command[:, 2].clone()
        if self.step_count > 3:
            velocity[0], yaw[0] = float("nan"), float("nan")
        terminated = torch.tensor([self.step_count == 3, False])
        return {}, {"velocity_heading": velocity, "yaw_rate": yaw,
                    "terminated": terminated, "truncated": torch.zeros(2, dtype=torch.bool),
                    "done": terminated}


class SequenceTests(unittest.TestCase):
    def test_command_change_updates_latest_observation_without_reset(self):
        env = FakeEnv()
        env.fixed_command = torch.zeros(3)
        env.reset()
        old = env.obs_hist.clone()
        set_command(env, (.5, 0., 0.))
        torch.testing.assert_close(env.obs_hist[:, :-1], old[:, :-1])
        torch.testing.assert_close(env.obs_hist[:, -1], torch.tensor([[.5, 0., 0.]]).repeat(2, 1))
        self.assertEqual(env.resets, 1)

    def test_first_fall_remains_masked_across_segment_boundaries(self):
        env = FakeEnv()
        report = evaluate_sequence(lambda obs: torch.zeros(2, 29), env,
                                   schedule=((.04, (0., 0., 0.)), (.06, (.5, 0., 0.)),
                                             (.04, (0., 0., .5))), perturbation=0.)
        self.assertEqual(env.resets, 1)
        self.assertEqual(report["falls"], 1)
        self.assertEqual(report["survived_full_sequence"], 1)
        self.assertEqual([s["copies_entered"] for s in report["segments"]], [2, 2, 1])
        self.assertEqual([s["new_falls"] for s in report["segments"]], [0, 1, 0])
        self.assertAlmostEqual(report["trials"][0]["observed_seconds"], .06)
        self.assertAlmostEqual(report["trials"][1]["observed_seconds"], .14)
        for segment in report["segments"]:
            self.assertAlmostEqual(segment["errors_alive"]["planar_rmse_mps"], 0.)
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
