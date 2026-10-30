"""CPU checks of the independent AMP locomotion task heads.

Run from the repository root after sourcing scripts/local_runtime.sh:
  python code/scripts/dev/test_amp_task.py
No MuJoCo instance, GPU, reference motion, or pretrained policy is needed.
"""

import math
from types import SimpleNamespace
import unittest

import torch

from supermimic.amp_task import AMPTask, AMPTaskCfg, COMMAND_HI, COMMAND_LO
from supermimic.rotations import axis_angle_to_mat


class FakeBase:
    def __init__(self):
        self.q0 = torch.linspace(-0.1, 0.1, 29)
        self.qmin = torch.full((29,), -1.0)
        self.qmax = torch.full((29,), 1.0)
        self.kin = SimpleNamespace(body_names=["pelvis", "torso_link"])

    def body_kin(self, state):
        return (
            state["pos"].unsqueeze(-2).expand(state["pos"].shape[:-1] + (2, 3)),
            state["rot"].unsqueeze(-3).expand(state["rot"].shape[:-2] + (2, 3, 3)),
            state["v"].unsqueeze(-2).expand(state["v"].shape[:-1] + (2, 3)),
            state["w"].unsqueeze(-2).expand(state["w"].shape[:-1] + (2, 3)),
        )


def state(batch=(3,)):
    return {
        "pos": torch.tensor([0.0, 0.0, 0.8]).expand(batch + (3,)).clone(),
        "rot": torch.eye(3).expand(batch + (3, 3)).clone(),
        "q": torch.zeros(batch + (29,)),
        "qd": torch.zeros(batch + (29,)),
        "v": torch.zeros(batch + (3,)),
        "w": torch.zeros(batch + (3,)),
    }


class AMPTaskTest(unittest.TestCase):
    def setUp(self):
        self.base = FakeBase()
        self.task = AMPTask(self.base)

    def test_observation_order_bias_and_command(self):
        s = state()
        action = torch.full((3, 29), 0.4)
        command = torch.tensor([[0.2, -0.1, 0.3]]).expand(3, 3)
        xi = torch.zeros(3, 33)
        xi[:, 1:30] = 0.01
        obs = self.task.observation(s, action, command, xi)
        self.assertEqual(obs.shape, (3, 96))
        torch.testing.assert_close(obs[:, :3], s["w"])
        torch.testing.assert_close(obs[:, 3:6], torch.tensor([[0., 0., -1.]]).expand(3, 3))
        torch.testing.assert_close(obs[:, 6:9], command)
        torch.testing.assert_close(obs[:, 9:38], s["q"] - self.base.q0 + 0.01)
        torch.testing.assert_close(obs[:, 38:67], s["qd"])
        torch.testing.assert_close(obs[:, 67:], action)
        changed = self.task.observation(s, action, command + 0.5, xi)
        torch.testing.assert_close(changed[:, 6:9] - obs[:, 6:9], torch.full((3, 3), 0.5))

    def test_history_broadcast_and_noise(self):
        s = state((3, 4))
        a = torch.zeros(3, 29)
        cmd = torch.randn(3, 3)
        xi = torch.zeros(3, 33)
        obs = self.task.observation(s, a, cmd, xi)
        self.assertEqual(obs.shape, (3, 4, 96))
        torch.testing.assert_close(obs[:, :, 6:9], cmd[:, None, :].expand(3, 4, 3))
        noise = self.task.sample_noise((64, 4))
        self.assertEqual(noise.shape, (64, 4, 96))
        self.assertTrue(bool((noise.abs() <= self.task.obs_noise).all()))
        self.assertEqual(int(torch.count_nonzero(noise[..., 6:9])), 0)
        self.assertEqual(int(torch.count_nonzero(noise[..., 67:])), 0)
        self.assertEqual(len(COMMAND_LO), 3)
        self.assertEqual(len(COMMAND_HI), 3)

    def test_reward_velocity_yaw_and_dt(self):
        s = state()
        s["v"][:, 0] = 0.7
        s["w"][:, 2] = 0.4
        a = torch.zeros(3, 29)
        matched = torch.tensor([[0.7, 0., 0.4]]).expand(3, 3)
        reward, terms = self.task.reward(s, a, a, matched)
        torch.testing.assert_close(reward, torch.full((3,), 0.05))
        torch.testing.assert_close(sum(terms.values()), reward)
        bad, bad_terms = self.task.reward(s, a, a, torch.zeros(3, 3))
        self.assertTrue(bool((bad < reward).all()))
        self.assertTrue(bool((bad_terms["linear"] < terms["linear"]).all()))
        self.assertTrue(bool((bad_terms["yaw"] < terms["yaw"]).all()))
        s["v"][:, 2] = 0.5
        vertical, _ = self.task.reward(s, a, a, matched)
        self.assertTrue(bool((vertical < reward).all()))

    def test_heading_frame_and_regularization(self):
        s = state()
        s["rot"] = axis_angle_to_mat(torch.tensor([[0., 0., math.pi / 2]])).expand(3, 3, 3)
        s["v"][:, 1] = 0.7
        a = torch.zeros(3, 29)
        command = torch.tensor([[0.7, 0., 0.]]).expand(3, 3)
        reward, terms = self.task.reward(s, a, a, command)
        torch.testing.assert_close(terms["linear"], torch.full((3,), 0.02), atol=1e-6, rtol=1e-6)
        s["q"][:, 0] = 1.2
        penalized, terms = self.task.reward(s, a + 0.1, a, command)
        self.assertTrue(bool((penalized < reward).all()))
        torch.testing.assert_close(terms["joint_limit"], torch.full((3,), -0.04), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(terms["action_rate"], torch.full((3,), -0.000058), atol=1e-8, rtol=1e-6)

    def test_fall_thresholds_and_continuation_gradient(self):
        s = state((4,))
        s["pos"][:, 2] = torch.tensor([0.8, 0.49, 0.8, 0.8])
        angles = torch.tensor([0., 0., math.radians(69), math.radians(71)])
        s["rot"] = axis_angle_to_mat(torch.stack((angles, angles * 0, angles * 0), -1))
        hard, soft = self.task.termination(s)
        torch.testing.assert_close(hard, torch.tensor([False, True, False, True]))
        self.assertTrue(bool(((soft >= 0) & (soft <= 1)).all()))
        self.assertLess(float(soft[0]), 1e-5)
        s["pos"] = s["pos"].clone().requires_grad_(True)
        _, soft = self.task.termination(s)
        soft.sum().backward()
        self.assertTrue(bool(torch.isfinite(s["pos"].grad).all()))
        self.assertLess(float(s["pos"].grad[1, 2]), 0.0)

    def test_reward_and_observation_gradients(self):
        s = state((2, 3))
        s["v"] = torch.full((2, 3, 3), 0.2, requires_grad=True)
        s["q"] = torch.full((2, 3, 29), 1.1, requires_grad=True)
        a = torch.full((2, 3, 29), 0.1, requires_grad=True)
        previous = torch.zeros(2, 29)
        cmd = torch.tensor([[0.7, 0., 0.], [0.4, 0., 0.]], requires_grad=True)
        r, _ = self.task.reward(s, a, previous, cmd)
        obs = self.task.observation(s, previous, cmd, torch.zeros(2, 33))
        self.assertEqual(r.shape, (2, 3))
        self.assertEqual(obs.shape, (2, 3, 96))
        (r.sum() + obs[..., 6:9].sum() * 0.01).backward()
        for value in (s["v"], s["q"], a, cmd):
            self.assertIsNotNone(value.grad)
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
            self.assertGreater(float(value.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
