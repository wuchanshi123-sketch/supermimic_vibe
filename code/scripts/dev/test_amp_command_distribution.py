"""CPU checks for command-distribution configuration, wiring and resume history.

Run from the repository root with PYTHONPATH=code. Constructor tests stop at a
mock AMPEnv constructor, before any simulator or CUDA tensor can be allocated.
"""

from __future__ import annotations

import copy
import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from supermimic.amp_task import COMMAND_LO, COMMAND_HI
from supermimic.amp_train import (
    AMPTrainer, AMPTrainCfg, CHECKPOINT_TYPE, parse_command_vector,
    reward_config, validate_cfg, validate_resume_config,
)


class CommandDistributionTests(unittest.TestCase):
    def setUp(self):
        world = patch.dict(os.environ, {"WORLD_SIZE": "1"})
        world.start()
        self.addCleanup(world.stop)
        self.cfg = AMPTrainCfg()
        self.saved = dataclasses.asdict(self.cfg)

    def test_parser_requires_exactly_three_finite_numbers(self):
        self.assertEqual(parse_command_vector(" -.5, 0, 1e-1 "), (-0.5, 0., 0.1))
        self.assertEqual(parse_command_vector([0.3, 0, -1]), (0.3, 0., -1.))
        self.assertEqual(parse_command_vector((1, 2, 3)), (1., 2., 3.))
        for value in ("", " ", "0,1", "0,1,2,3", "0,,2", "0,1,2,", "x,1,2",
                      "nan,0,0", "0,inf,0", "0,0,-inf", [0, 1], [0, 1, float("nan")]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_command_vector(value)

    def test_default_distribution_matches_original_environment_constants(self):
        self.assertEqual(parse_command_vector(self.cfg.command_low), tuple(COMMAND_LO))
        self.assertEqual(parse_command_vector(self.cfg.command_high), tuple(COMMAND_HI))
        self.assertEqual(self.cfg.standing_command_probability, 0.05)
        validate_cfg(self.cfg)

    def test_parsing_validation_and_reward_provenance_do_not_advance_rng(self):
        with torch.random.fork_rng(devices=[]):
            before = torch.get_rng_state().clone()
            parse_command_vector(self.cfg.command_low)
            parse_command_vector(self.cfg.command_high)
            validate_cfg(self.cfg)
            reward_config(self.cfg)
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)

    def test_inverted_bounds_are_rejected_on_every_axis(self):
        for low in ("2,-0.5,-1", "-0.5,0.6,-1", "-0.5,-0.5,2"):
            with self.subTest(low=low), self.assertRaises(ValueError):
                validate_cfg(dataclasses.replace(self.cfg, command_low=low))

    def test_degenerate_axes_and_all_zero_commands_are_valid(self):
        for low, high in (("0.3,0,0", "0.8,0,0"), ("0.5,0,0", "0.5,0,0"), ("0,0,0", "0,0,0")):
            with self.subTest(low=low, high=high):
                validate_cfg(dataclasses.replace(self.cfg, command_low=low, command_high=high))

    def test_probability_is_finite_and_bounded(self):
        for probability in (0., 0.05, 1.):
            validate_cfg(dataclasses.replace(self.cfg, standing_command_probability=probability))
        for probability in (-0.1, 1.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                validate_cfg(dataclasses.replace(self.cfg, standing_command_probability=probability))

    def test_validation_rejects_bad_vectors_in_either_bound(self):
        for field in ("command_low", "command_high"):
            for value in ("0,1", "0,nan,0", "0,0,inf"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_cfg(dataclasses.replace(self.cfg, **{field: value}))

    def test_legacy_checkpoint_defaults_preserve_distribution_and_reward_config(self):
        legacy = dict(self.saved)
        for field in ("command_low", "command_high", "standing_command_probability"):
            legacy.pop(field)
        restored = AMPTrainCfg(**legacy)
        self.assertEqual(parse_command_vector(restored.command_low), tuple(COMMAND_LO))
        self.assertEqual(parse_command_vector(restored.command_high), tuple(COMMAND_HI))
        self.assertEqual(restored.standing_command_probability, 0.05)
        self.assertEqual(reward_config(legacy), reward_config(restored))
        self.assertEqual(validate_resume_config(legacy, restored), {})

    def test_every_distribution_change_requires_explicit_resume_authorization(self):
        changes = {"command_low": "0.3,0,0", "command_high": "0.8,0,0", "standing_command_probability": 0.}
        for field, value in changes.items():
            with self.subTest(field=field):
                candidate = dataclasses.replace(self.cfg, **{field: value})
                with self.assertRaises(ValueError):
                    validate_resume_config(self.saved, candidate)
                candidate.allow_reward_change = True
                self.assertEqual(validate_resume_config(self.saved, candidate), {
                    field: {"old": reward_config(self.saved)[field], "new": reward_config(candidate)[field]},
                })

    def test_equivalent_numeric_formats_have_no_semantic_distribution_change(self):
        candidate = dataclasses.replace(self.cfg, command_low="-.50,-5e-1,-1.000", command_high="1,.500,1.0")
        self.assertEqual(validate_resume_config(self.saved, candidate), {})
        self.assertEqual(reward_config(candidate)["command_low"], [-0.5, -0.5, -1.])
        saved_sequences = dict(self.saved, command_low=(-0.5, -0.5, -1), command_high=[1, 0.5, 1])
        self.assertEqual(validate_resume_config(saved_sequences, candidate), {})

    def _capture_environment_kwargs(self, cfg):
        class StopAtEnvironment(Exception):
            pass
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", old_tf32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            motions = root / "motions.npz"
            motions.write_bytes(b"mock motion bytes, only hashed")
            configured = dataclasses.replace(cfg, motions=str(motions), out=str(root / "out"), device="cuda")
            with patch("supermimic.amp_train.torch.cuda.is_available", return_value=True), \
                 patch("supermimic.amp_train.torch.manual_seed"), \
                 patch.dict("supermimic.amp_train.networks.AMP", {"enabled": False}), \
                 patch("supermimic.amp_train.MotionLibrary", return_value=SimpleNamespace(dt=0.02)) as library, \
                 patch("supermimic.amp_train.AMPEnv", side_effect=StopAtEnvironment) as environment:
                with self.assertRaises(StopAtEnvironment):
                    AMPTrainer(configured)
                library.assert_called_once_with(str(motions), "cuda")
                environment.assert_called_once()
                return environment.call_args.kwargs

    def test_actual_trainer_constructor_passes_historical_defaults_to_environment(self):
        kwargs = self._capture_environment_kwargs(self.cfg)
        self.assertEqual(kwargs["command_low"], tuple(COMMAND_LO))
        self.assertEqual(kwargs["command_high"], tuple(COMMAND_HI))
        self.assertEqual(kwargs["standing_probability"], 0.05)

    def test_actual_trainer_constructor_passes_focused_command_distribution(self):
        focused = dataclasses.replace(self.cfg, command_low="0.3,0,0", command_high="0.8,0,0", standing_command_probability=0.)
        kwargs = self._capture_environment_kwargs(focused)
        self.assertEqual(kwargs["command_low"], (0.3, 0., 0.))
        self.assertEqual(kwargs["command_high"], (0.8, 0., 0.))
        self.assertEqual(kwargs["standing_probability"], 0.)

    def test_actual_load_records_old_and_new_distribution_and_preserves_critic(self):
        source = {name: torch.nn.Linear(3, 2) for name in ("wm", "actor", "critic", "amp_prior")}
        source_optimizers = {name: torch.optim.Adam(module.parameters(), lr=0.001) for name, module in source.items()}
        source["critic"](torch.tensor([[1., -2., 3.]])).sum().backward()
        source_optimizers["critic"].step()
        critic_weights = copy.deepcopy(source["critic"].state_dict())
        critic_optimizer = copy.deepcopy(source_optimizers["critic"].state_dict())
        legacy = dict(self.saved)
        for field in ("command_low", "command_high", "standing_command_probability"):
            legacy.pop(field)
        checkpoint = {
            "checkpoint_type": CHECKPOINT_TYPE, "expert_library_sha256": "command-test-library",
            "cfg": legacy, **{name: module.state_dict() for name, module in source.items()},
            "opt_wm": source_optimizers["wm"].state_dict(), "opt_actor": source_optimizers["actor"].state_dict(),
            "opt_critic": critic_optimizer, "opt_discriminator": source_optimizers["amp_prior"].state_dict(),
            "it": 37, "updates": 15, "actor_updates": 14, "disc_updates": 30,
            "env_steps": 1184, "train_time": 9.5, "update_credit": 0.4,
            "initialization": {"type": "random", "reward_curriculum": [{"iteration": 20}]},
            "torch_rng": torch.get_rng_state(), "cuda_rng": [],
        }
        trainer = AMPTrainer.__new__(AMPTrainer)
        trainer.cfg = dataclasses.replace(self.cfg, command_low="0.3,0,0", command_high="0.8,0,0",
                                          standing_command_probability=0., allow_reward_change=True)
        trainer.device, trainer.motion_hash = "cpu", "command-test-library"
        trainer.wm, trainer.actor, trainer.critic, trainer.prior = (torch.nn.Linear(3, 2) for _ in range(4))
        trainer.opt_wm = torch.optim.Adam(trainer.wm.parameters(), lr=0.9)
        trainer.opt_actor = torch.optim.Adam(trainer.actor.parameters(), lr=0.9)
        trainer.opt_critic = torch.optim.Adam(trainer.critic.parameters(), lr=0.9)
        trainer.prior.optimizer = torch.optim.Adam(trainer.prior.parameters(), lr=0.9)
        with patch("supermimic.amp_train.torch.load", return_value=checkpoint), \
             patch("supermimic.amp_train.torch.set_rng_state"), \
             patch("supermimic.amp_train.torch.cuda.set_rng_state_all"), patch("builtins.print"):
            trainer.load("legacy-command-checkpoint.pt")
        for name, value in critic_weights.items():
            torch.testing.assert_close(trainer.critic.state_dict()[name], value, rtol=0, atol=0)
        actual_optimizer = trainer.opt_critic.state_dict()
        self.assertEqual(actual_optimizer["param_groups"], critic_optimizer["param_groups"])
        for parameter, values in critic_optimizer["state"].items():
            for key, expected in values.items():
                torch.testing.assert_close(actual_optimizer["state"][parameter][key], expected, rtol=0, atol=0)
        history = trainer.initialization["reward_curriculum"]
        self.assertEqual(history[0], {"iteration": 20})
        self.assertEqual(len(history), 2)
        entry = history[-1]
        self.assertEqual(entry["iteration"], 37)
        self.assertEqual(entry["changes"], {
            "command_low": {"old": [-0.5, -0.5, -1.], "new": [0.3, 0., 0.]},
            "command_high": {"old": [1., 0.5, 1.], "new": [0.8, 0., 0.]},
            "standing_command_probability": {"old": 0.05, "new": 0.},
        })
        self.assertEqual(entry["old_reward_config"]["command_low"], [-0.5, -0.5, -1.])
        self.assertEqual(entry["new_reward_config"]["command_low"], [0.3, 0., 0.])
        self.assertEqual(entry["old_reward_config"]["standing_command_probability"], 0.05)
        self.assertEqual(entry["new_reward_config"]["standing_command_probability"], 0.)
        self.assertFalse(entry["critic_reset"])


if __name__ == "__main__":
    unittest.main()
