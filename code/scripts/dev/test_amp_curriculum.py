"""CPU checks for the optional low-ankle slip penalty and reward resume guard.

Run from the repository root with PYTHONPATH=code. No simulator, GPU, motion
library, checkpoint or pretrained actor is constructed by these tests.
"""

from __future__ import annotations

import dataclasses
import copy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from supermimic.amp_task import AMPTask, AMPTaskCfg
from supermimic.amp_train import AMPTrainer, AMPTrainCfg, CHECKPOINT_TYPE, task_config, validate_cfg, validate_resume_config


class FakeBase:
    def __init__(self, feet=False):
        self.q0 = torch.zeros(29)
        self.qmin = torch.full((29,), -1.0)
        self.qmax = torch.full((29,), 1.0)
        names = ["pelvis", "torso_link"]
        if feet:
            names += ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.kin = SimpleNamespace(body_names=names)

    def body_kin(self, state):
        raise AssertionError("Reward must reuse the explicitly supplied kinematics")


def state(batch=2):
    return {
        "pos": torch.tensor([0., 0., 0.8]).expand(batch, 3).clone(),
        "rot": torch.eye(3).expand(batch, 3, 3).clone(),
        "q": torch.zeros(batch, 29), "qd": torch.zeros(batch, 29),
        "v": torch.zeros(batch, 3), "w": torch.zeros(batch, 3),
    }


def kinematics(batch=2, bodies=4, foot_height=0.045, foot_vxy=(1., 0.)):
    positions = torch.zeros(batch, bodies, 3)
    positions[..., 2] = 0.8
    velocities = torch.zeros_like(positions)
    if bodies == 4:
        positions[:, 2:, 2] = foot_height
        velocities[:, 2:, :2] = torch.tensor(foot_vxy)
    return positions, torch.eye(3).expand(batch, bodies, 3, 3).clone(), velocities, torch.zeros_like(positions)


def reward(task, kin):
    batch = kin[0].shape[0]
    action = torch.zeros(batch, 29)
    return task.reward(state(batch), action, action, torch.zeros(batch, 3), kin=kin)


class FootSlipRewardTests(unittest.TestCase):
    def test_default_reward_is_exact_and_does_not_require_foot_bodies(self):
        task = AMPTask(FakeBase())
        actual, terms = reward(task, kinematics(bodies=2))
        # Preserve the historical fp32 addition order, which differs slightly
        # from rounding a scalar 0.05 directly to fp32.
        historical = torch.full((2,), 0.02) + torch.full((2,), 0.02) + torch.full((2,), 0.01)
        torch.testing.assert_close(actual, historical, rtol=0, atol=0)
        self.assertEqual(set(terms), {"linear", "yaw", "upright", "action_rate", "joint_limit"})
        self.assertNotIn("foot_slip", terms)

    def test_penalty_is_zero_for_stationary_feet(self):
        task = AMPTask(FakeBase(feet=True), AMPTaskCfg(w_foot_slip=-0.25))
        actual, terms = reward(task, kinematics(foot_height=0., foot_vxy=(0., 0.)))
        torch.testing.assert_close(terms["foot_slip"], torch.zeros(2), rtol=0, atol=0)
        baseline, _ = reward(AMPTask(FakeBase()), kinematics(bodies=2))
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)

    def test_low_moving_feet_are_penalized_more_than_raised_feet(self):
        task = AMPTask(FakeBase(feet=True), AMPTaskCfg(w_foot_slip=-0.25))
        low, low_terms = reward(task, kinematics(foot_height=0., foot_vxy=(1., 2.)))
        raised, raised_terms = reward(task, kinematics(foot_height=0.145, foot_vxy=(1., 2.)))
        self.assertTrue(bool((low < raised).all()))
        self.assertTrue(bool((low_terms["foot_slip"] < -0.04).all()))
        self.assertTrue(bool((raised_terms["foot_slip"].abs() < 1e-8).all()))

    def test_threshold_gate_and_dt_weight_scaling(self):
        first = AMPTask(FakeBase(feet=True), AMPTaskCfg(w_foot_slip=-0.25))
        second = AMPTask(FakeBase(feet=True), AMPTaskCfg(dt=0.04, w_foot_slip=-0.5))
        _, a = reward(first, kinematics())
        _, b = reward(second, kinematics())
        # Two feet, unit planar speed and 0.5 gate give a unit gated cost.
        torch.testing.assert_close(a["foot_slip"], torch.full((2,), -0.005))
        torch.testing.assert_close(b["foot_slip"], 4 * a["foot_slip"])

    def test_penalty_gradients_reduce_planar_speed_and_raise_low_feet(self):
        task = AMPTask(FakeBase(feet=True), AMPTaskCfg(w_foot_slip=-0.25))
        positions, rotations, velocities, angular = kinematics(foot_vxy=(1., -2.))
        positions.requires_grad_(True)
        velocities.requires_grad_(True)
        _, terms = reward(task, (positions, rotations, velocities, angular))
        terms["foot_slip"].sum().backward()
        self.assertTrue(bool(torch.isfinite(positions.grad).all()))
        self.assertTrue(bool(torch.isfinite(velocities.grad).all()))
        self.assertTrue(bool((positions.grad[:, 2:, 2] > 0).all()))
        self.assertTrue(bool((velocities.grad[:, 2:, 0] < 0).all()))
        self.assertTrue(bool((velocities.grad[:, 2:, 1] > 0).all()))
        torch.testing.assert_close(velocities.grad[:, 2:, 2], torch.zeros(2, 2), rtol=0, atol=0)
        torch.testing.assert_close(positions.grad[:, :2], torch.zeros(2, 2, 3), rtol=0, atol=0)

    def test_invalid_penalty_settings_are_rejected(self):
        for weight in (0.1, float("inf"), -float("inf"), float("nan")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                AMPTask(FakeBase(feet=True), AMPTaskCfg(w_foot_slip=weight))
        for width in (0., -0.1, float("inf"), float("nan")):
            with self.subTest(width=width), self.assertRaises(ValueError):
                AMPTask(FakeBase(feet=True), AMPTaskCfg(foot_slip_width=width))


class RewardResumeTests(unittest.TestCase):
    def setUp(self):
        self.world = patch.dict(os.environ, {"WORLD_SIZE": "1"})
        self.world.start()
        self.addCleanup(self.world.stop)
        self.cfg = AMPTrainCfg()
        self.saved = dataclasses.asdict(self.cfg)

    def test_old_checkpoint_configs_acquire_disabled_curriculum_defaults(self):
        legacy = dict(self.saved)
        for name in ("action_rate_weight", "foot_slip_weight", "foot_slip_height_threshold", "foot_slip_width", "actor_lr_scale", "allow_reward_change"):
            legacy.pop(name)
        loaded = AMPTrainCfg(**legacy)
        self.assertEqual(loaded.foot_slip_weight, 0.)
        self.assertEqual(loaded.actor_lr_scale, 1.)
        self.assertFalse(loaded.allow_reward_change)
        self.assertEqual(validate_resume_config(legacy, loaded), {})

    def test_every_changed_reward_field_requires_explicit_curriculum_flag(self):
        changes = {
            "task_weight": 0.5, "style_scale": 0.2,
            "linear_std": 1., "angular_std": 1., "action_rate_weight": -0.05, "foot_slip_weight": -0.1,
            "foot_slip_height_threshold": 0.05, "foot_slip_width": 0.01,
        }
        for name, value in changes.items():
            with self.subTest(field=name):
                changed = dataclasses.replace(self.cfg, **{name: value})
                with self.assertRaises(ValueError):
                    validate_resume_config(self.saved, changed)
                allowed = dataclasses.replace(changed, allow_reward_change=True)
                self.assertEqual(validate_resume_config(self.saved, allowed), {
                    name: {"old": self.saved[name], "new": value},
                })

    def test_allowing_reward_change_does_not_allow_architecture_or_discount_changes(self):
        for name, value in (("history", 5), ("wm_width", 1024), ("action_limit", 2.), ("gamma", 0.98)):
            with self.subTest(field=name), self.assertRaises(ValueError):
                validate_resume_config(self.saved, dataclasses.replace(self.cfg, allow_reward_change=True, **{name: value}))

    def test_actor_lr_scale_may_change_without_relabeling_the_reward(self):
        original = dict(self.saved)
        scaled = dataclasses.replace(self.cfg, actor_lr_scale=0.25)
        self.assertEqual(validate_resume_config(self.saved, scaled), {})
        self.assertEqual(self.saved, original)
        validate_cfg(scaled)
        for scale in (0., -0.1, float("inf"), float("nan")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                validate_cfg(dataclasses.replace(self.cfg, actor_lr_scale=scale))

    def test_training_configuration_reaches_task_heads(self):
        configured = dataclasses.replace(self.cfg, foot_slip_weight=-0.125,
                                         foot_slip_height_threshold=0.05, foot_slip_width=0.01)
        task = task_config(configured)
        self.assertEqual(task.w_foot_slip, -0.125)
        self.assertEqual(task.foot_slip_height_threshold, 0.05)
        self.assertEqual(task.foot_slip_width, 0.01)

    def test_actual_resume_restores_critic_and_adam_state_and_records_curriculum(self):
        torch.manual_seed(23)
        source = {name: torch.nn.Linear(3, 2) for name in ("wm", "actor", "critic", "amp_prior")}
        source_opts = {name: torch.optim.Adam(module.parameters(), lr=0.001) for name, module in source.items()}
        source["critic"](torch.tensor([[1., -2., 3.]])).sum().backward()
        source_opts["critic"].step()
        expected_critic = copy.deepcopy(source["critic"].state_dict())
        expected_optimizer = copy.deepcopy(source_opts["critic"].state_dict())
        legacy_cfg = dict(self.saved)
        for name in ("action_rate_weight", "foot_slip_weight", "foot_slip_height_threshold", "foot_slip_width", "actor_lr_scale", "allow_reward_change"):
            legacy_cfg.pop(name)
        checkpoint = {
            "checkpoint_type": CHECKPOINT_TYPE, "expert_library_sha256": "test-motion-hash",
            "cfg": legacy_cfg, **{name: module.state_dict() for name, module in source.items()},
            "opt_wm": source_opts["wm"].state_dict(), "opt_actor": source_opts["actor"].state_dict(),
            "opt_critic": expected_optimizer, "opt_discriminator": source_opts["amp_prior"].state_dict(),
            "it": 17, "updates": 9, "actor_updates": 8, "disc_updates": 18,
            "env_steps": 544, "train_time": 3.5, "update_credit": 0.4,
            "initialization": {"type": "random", "reward_curriculum": [{"iteration": 1, "prior": "retained"}]},
            "torch_rng": torch.get_rng_state(), "cuda_rng": [],
        }
        trainer = AMPTrainer.__new__(AMPTrainer)
        trainer.cfg = dataclasses.replace(self.cfg, foot_slip_weight=-0.1, action_rate_weight=-0.05,
                                          actor_lr_scale=0.5, allow_reward_change=True)
        trainer.device, trainer.motion_hash = "cpu", "test-motion-hash"
        trainer.wm, trainer.actor, trainer.critic, trainer.prior = (torch.nn.Linear(3, 2) for _ in range(4))
        trainer.opt_wm = torch.optim.Adam(trainer.wm.parameters(), lr=0.9)
        trainer.opt_actor = torch.optim.Adam(trainer.actor.parameters(), lr=0.9)
        trainer.opt_critic = torch.optim.Adam(trainer.critic.parameters(), lr=0.9)
        trainer.prior.optimizer = torch.optim.Adam(trainer.prior.parameters(), lr=0.9)
        with patch("supermimic.amp_train.torch.load", return_value=checkpoint), \
             patch("supermimic.amp_train.torch.set_rng_state"), \
             patch("supermimic.amp_train.torch.cuda.set_rng_state_all"), \
             patch("builtins.print"):
            trainer.load("synthetic-old-checkpoint.pt")
        for name, value in expected_critic.items():
            torch.testing.assert_close(trainer.critic.state_dict()[name], value, rtol=0, atol=0)
        restored = trainer.opt_critic.state_dict()
        self.assertEqual(restored["param_groups"], expected_optimizer["param_groups"])
        self.assertEqual(restored["state"].keys(), expected_optimizer["state"].keys())
        for parameter, values in expected_optimizer["state"].items():
            for name, expected in values.items():
                torch.testing.assert_close(restored["state"][parameter][name], expected, rtol=0, atol=0)
        self.assertEqual(trainer.it, 17)
        self.assertEqual(trainer.update_credit, 0.4)
        history = trainer.initialization["reward_curriculum"]
        self.assertEqual(history[0], {"iteration": 1, "prior": "retained"})
        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1]["iteration"], 17)
        self.assertEqual(history[-1]["changes"], {
            "foot_slip_weight": {"old": 0., "new": -0.1},
            "action_rate_weight": {"old": -0.01, "new": -0.05},
        })
        self.assertEqual(history[-1]["old_reward_config"]["action_rate_weight"], -0.01)
        self.assertEqual(history[-1]["new_reward_config"]["action_rate_weight"], -0.05)
        self.assertEqual(history[-1]["old_reward_config"]["foot_slip_weight"], 0.)
        self.assertEqual(history[-1]["new_reward_config"]["foot_slip_weight"], -0.1)
        self.assertEqual(history[-1]["old_actor_lr_scale"], 1.)
        self.assertEqual(history[-1]["new_actor_lr_scale"], 0.5)
        self.assertFalse(history[-1]["critic_reset"])


if __name__ == "__main__":
    unittest.main()
