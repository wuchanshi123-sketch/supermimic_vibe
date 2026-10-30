"""CPU regression checks for selecting AMP positive clips by name prefix.

Run with PYTHONPATH=code. No motion files, GPU or simulator are constructed;
constructor wiring checks stop at a mock AMPEnv before any CUDA allocation.
"""

from __future__ import annotations

import copy
import dataclasses
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from supermimic.amp_augmentation import TimeScaledExpertPairs, parse_clip_prefixes, select_expert_clips
from supermimic.amp_prior import ExpertFeaturePairs
from supermimic.amp_train import AMPTrainer, AMPTrainCfg, CHECKPOINT_TYPE, reward_config, validate_cfg, validate_resume_config


class FakeLibrary:
    def __init__(self):
        self.device, self.dt = "cpu", 0.02
        self.starts = torch.tensor([0, 3, 7, 14])
        self.lengths = torch.tensor([3, 4, 7, 3])
        self.names = ["walk_forward_short", "idle_turn", "walk_forward_long", "jog_forward"]
        self.num_frames, self.num_clips = 17, 4
        index = torch.arange(self.num_frames).float()
        self.data = {
            "pos": torch.stack((index, torch.zeros_like(index), torch.full_like(index, 0.8)), -1),
            "rot": torch.eye(3).expand(self.num_frames, 3, 3).clone(),
            "q": index[:, None].expand(-1, 29).clone() * 0.01,
            "v": torch.zeros(self.num_frames, 3),
            "w": torch.zeros(self.num_frames, 3),
            "qd": torch.full((self.num_frames, 29), 0.4),
        }
        self.data["v"][:3, 0] = 0.4
        self.data["v"][7:14, 0] = 0.6
        self.data["v"][14:, 0] = 1.0

    def frame(self, index):
        return {key: value[index] for key, value in self.data.items()}


def features(state):
    return torch.cat((state["pos"], state["v"], state["qd"][..., :3]), -1)


class ExpertClipSelectionTests(unittest.TestCase):
    def setUp(self):
        world = patch.dict(os.environ, {"WORLD_SIZE": "1"})
        world.start()
        self.addCleanup(world.stop)
        self.lib = FakeLibrary()
        self.cfg = AMPTrainCfg()
        self.saved = dataclasses.asdict(self.cfg)

    def test_prefix_parser_canonicalizes_and_rejects_empty_tokens_or_nonstring_prefixes(self):
        self.assertEqual(parse_clip_prefixes(""), ())
        self.assertEqual(parse_clip_prefixes(" walk_forward,arc_walk,walk_forward "), ("arc_walk", "walk_forward"))
        self.assertEqual(parse_clip_prefixes(("walk_forward", "arc_walk")), ("arc_walk", "walk_forward"))
        for value in (",walk", "walk,", "walk,,arc", "walk, ,arc", None, 3, ["walk", 2]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_clip_prefixes(value)

    def test_selection_matches_name_starts_and_retains_original_ids(self):
        self.assertEqual(select_expert_clips(self.lib.names, "walk_forward"), [0, 2])
        self.assertEqual(select_expert_clips(self.lib.names, "jog,walk_forward"), [0, 2, 3])
        self.assertEqual(select_expert_clips(self.lib.names, ()), [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            select_expert_clips(self.lib.names, "forward")
        with self.assertRaises(ValueError):
            select_expert_clips(self.lib.names, "does_not_exist")

    def test_selected_original_pairs_use_full_bank_offsets_and_uniform_clips(self):
        full = torch.stack((torch.arange(self.lib.num_frames).float(), torch.arange(self.lib.num_frames).float() + 100), -1)
        selected = select_expert_clips(self.lib.names, "walk_forward")
        sampler = ExpertFeaturePairs(full, self.lib.starts[selected], self.lib.lengths[selected])
        self.assertEqual(sampler.features.data_ptr(), full.data_ptr())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73)
            first, second = sampler.sample(30000)
        index = first[:, 0].long()
        self.assertTrue(bool(((index < 2) | ((index >= 7) & (index <= 12))).all()))
        self.assertFalse(bool(torch.isin(index, torch.tensor([2, 13])).any()))
        torch.testing.assert_close(second - first, torch.ones_like(first), rtol=0, atol=0)
        fraction_short_clip = float((index < 2).float().mean())
        self.assertTrue(0.46 < fraction_short_clip < 0.54, fraction_short_clip)

    def test_default_original_sampler_retains_rng_and_samples_exactly(self):
        full = features(self.lib.data)
        original = ExpertFeaturePairs(full, self.lib.starts, self.lib.lengths)
        with torch.random.fork_rng(devices=[]):
            before = torch.get_rng_state().clone()
            selected = select_expert_clips(self.lib.names, parse_clip_prefixes(""))
            equivalent = ExpertFeaturePairs(full, self.lib.starts[selected], self.lib.lengths[selected])
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
            torch.manual_seed(19)
            expected = original.sample(512)
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(19)
            actual = equivalent.sample(512)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)

    def test_augmented_selection_filters_before_eligibility_and_keeps_original_clip_ids(self):
        bank = TimeScaledExpertPairs(self.lib, features, (0.25, 0.5), clip_ids=[0, 2], batch_size=3)
        torch.testing.assert_close(bank.source_indices, torch.tensor([0, 1, 7, 8, 9, 10, 11, 12]))
        self.assertEqual(bank.eligible_clip_ids, (0, 2))
        self.assertEqual(set(bank.clip_ids.tolist()), {0, 2})
        torch.testing.assert_close(bank.counts, torch.tensor([2, 6]))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(89)
            scale, pair = bank.sample_indices(30000)
        self.assertTrue(bool(((scale >= 0) & (scale < 2)).all()))
        self.assertTrue(0.46 < float((bank.clip_ids[pair] == 0).float().mean()) < 0.54)

    def test_augmented_none_and_explicit_all_clips_have_identical_banks_and_rng(self):
        with torch.random.fork_rng(devices=[]):
            before = torch.get_rng_state().clone()
            default = TimeScaledExpertPairs(self.lib, features, (0.5,), clip_ids=None, batch_size=3)
            explicit = TimeScaledExpertPairs(self.lib, features, (0.5,), clip_ids=list(range(4)), batch_size=3)
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
            for field in ("first", "second", "source_indices", "clip_ids", "counts"):
                torch.testing.assert_close(getattr(default, field), getattr(explicit, field), rtol=0, atol=0)
            torch.manual_seed(91)
            expected = default.sample_indices(512)
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(91)
            actual = explicit.sample_indices(512)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)

    def test_selection_never_mutates_the_source_library(self):
        before = {key: value.clone() for key, value in self.lib.data.items()}
        starts, lengths, names = self.lib.starts.clone(), self.lib.lengths.clone(), list(self.lib.names)
        selected = select_expert_clips(self.lib.names, "walk_forward")
        TimeScaledExpertPairs(self.lib, features, (0.5,), clip_ids=selected)
        for key, value in self.lib.data.items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        torch.testing.assert_close(self.lib.starts, starts, rtol=0, atol=0)
        torch.testing.assert_close(self.lib.lengths, lengths, rtol=0, atol=0)
        self.assertEqual(self.lib.names, names)

    def test_legacy_config_defaults_to_all_experts_and_identical_reward_provenance(self):
        legacy = dict(self.saved)
        legacy.pop("expert_clip_prefixes")
        restored = AMPTrainCfg(**legacy)
        self.assertEqual(restored.expert_clip_prefixes, "")
        self.assertEqual(reward_config(legacy)["expert_clip_prefixes"], [])
        self.assertEqual(reward_config(legacy), reward_config(restored))
        self.assertEqual(validate_resume_config(legacy, restored), {})

    def test_changed_expert_selection_is_guarded_and_canonical_on_resume(self):
        selected = dataclasses.replace(self.cfg, expert_clip_prefixes="walk_forward")
        with self.assertRaises(ValueError):
            validate_resume_config(self.saved, selected)
        selected.allow_reward_change = True
        self.assertEqual(validate_resume_config(self.saved, selected), {"expert_clip_prefixes": {"old": [], "new": ["walk_forward"]}})
        old = dict(self.saved, expert_clip_prefixes="walk_forward,arc_walk")
        candidate = dataclasses.replace(self.cfg, expert_clip_prefixes=" arc_walk,walk_forward,arc_walk ")
        self.assertEqual(validate_resume_config(old, candidate), {})

    def test_validation_checks_syntax_without_requiring_library_access(self):
        validate_cfg(dataclasses.replace(self.cfg, expert_clip_prefixes="walk_forward"))
        validate_cfg(dataclasses.replace(self.cfg, expert_clip_prefixes="does_not_exist"))
        for text in (",walk", "walk,", "walk,,arc"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_cfg(dataclasses.replace(self.cfg, expert_clip_prefixes=text))

    def _constructor_probe(self, prefixes, no_match=False):
        class StopAtEnvironment(Exception):
            pass
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", old_tf32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            motions = root / "motions.npz"
            motions.write_bytes(b"only hashed by the constructor probe")
            cfg = dataclasses.replace(self.cfg, motions=str(motions), out=str(root / "out"),
                                      expert_clip_prefixes=prefixes, device="cuda")
            trainer = AMPTrainer.__new__(AMPTrainer)
            with patch("supermimic.amp_train.torch.cuda.is_available", return_value=True), \
                 patch("supermimic.amp_train.torch.manual_seed"), \
                 patch.dict("supermimic.amp_train.networks.AMP", {"enabled": False}), \
                 patch("supermimic.amp_train.MotionLibrary", return_value=self.lib) as library, \
                 patch("supermimic.amp_train.AMPEnv", side_effect=StopAtEnvironment) as environment:
                with self.assertRaises(ValueError if no_match else StopAtEnvironment):
                    trainer.__init__(cfg)
                library.assert_called_once_with(str(motions), "cuda")
                if no_match:
                    environment.assert_not_called()
                else:
                    environment.assert_called_once()
                    self.assertIs(environment.call_args.args[0], self.lib)
                    self.assertEqual(environment.call_args.kwargs["rsi_probability"], cfg.rsi_probability)
                    self.assertEqual(self.lib.num_clips, 4)
                    self.assertEqual(trainer.expert_clip_ids, [0, 2] if prefixes else None)

    def test_actual_trainer_passes_full_library_to_environment_for_selected_experts(self):
        self._constructor_probe("walk_forward")

    def test_actual_trainer_default_keeps_full_library_and_lazy_all_selection(self):
        self._constructor_probe("")

    def test_actual_trainer_rejects_no_match_before_environment_creation(self):
        self._constructor_probe("does_not_exist", no_match=True)

    def test_actual_load_records_selection_metadata_and_preserves_critic_adam(self):
        source = {name: torch.nn.Linear(3, 2) for name in ("wm", "actor", "critic", "amp_prior")}
        optimizers = {name: torch.optim.Adam(module.parameters(), lr=0.001) for name, module in source.items()}
        source["critic"](torch.tensor([[1., -2., 3.]])).sum().backward()
        optimizers["critic"].step()
        weights = copy.deepcopy(source["critic"].state_dict())
        optimizer = copy.deepcopy(optimizers["critic"].state_dict())
        legacy = dict(self.saved)
        legacy.pop("expert_clip_prefixes")
        checkpoint = {
            "checkpoint_type": CHECKPOINT_TYPE, "expert_library_sha256": "selection-test-library", "cfg": legacy,
            **{name: module.state_dict() for name, module in source.items()},
            "opt_wm": optimizers["wm"].state_dict(), "opt_actor": optimizers["actor"].state_dict(),
            "opt_critic": optimizer, "opt_discriminator": optimizers["amp_prior"].state_dict(),
            "it": 43, "updates": 17, "actor_updates": 16, "disc_updates": 34,
            "env_steps": 1376, "train_time": 12.5, "update_credit": 0.4,
            "initialization": {"type": "random", "reward_curriculum": [{"iteration": 20}]},
            "torch_rng": torch.get_rng_state(), "cuda_rng": [],
        }
        trainer = AMPTrainer.__new__(AMPTrainer)
        trainer.cfg = dataclasses.replace(self.cfg, expert_clip_prefixes="walk_forward", allow_reward_change=True)
        trainer.device, trainer.motion_hash = "cpu", "selection-test-library"
        trainer.wm, trainer.actor, trainer.critic, trainer.prior = (torch.nn.Linear(3, 2) for _ in range(4))
        trainer.opt_wm = torch.optim.Adam(trainer.wm.parameters(), lr=0.9)
        trainer.opt_actor = torch.optim.Adam(trainer.actor.parameters(), lr=0.9)
        trainer.opt_critic = torch.optim.Adam(trainer.critic.parameters(), lr=0.9)
        trainer.prior.optimizer = torch.optim.Adam(trainer.prior.parameters(), lr=0.9)
        trainer.expert_sampling_metadata = {
            "original_clips": 2, "source_motion_library_clips": 4, "selected_clip_ids": [0, 2],
            "selected_clip_names": [self.lib.names[0], self.lib.names[2]],
        }
        with patch("supermimic.amp_train.torch.load", return_value=checkpoint), \
             patch("supermimic.amp_train.torch.set_rng_state"), \
             patch("supermimic.amp_train.torch.cuda.set_rng_state_all"), patch("builtins.print"):
            trainer.load("legacy-expert-selection.pt")
        for name, expected in weights.items():
            torch.testing.assert_close(trainer.critic.state_dict()[name], expected, rtol=0, atol=0)
        actual_optimizer = trainer.opt_critic.state_dict()
        self.assertEqual(actual_optimizer["param_groups"], optimizer["param_groups"])
        for parameter, values in optimizer["state"].items():
            for key, expected in values.items():
                torch.testing.assert_close(actual_optimizer["state"][parameter][key], expected, rtol=0, atol=0)
        history = trainer.initialization["reward_curriculum"]
        self.assertEqual(history[0], {"iteration": 20})
        self.assertEqual(len(history), 2)
        entry = history[-1]
        self.assertEqual(entry["iteration"], 43)
        self.assertEqual(entry["changes"], {"expert_clip_prefixes": {"old": [], "new": ["walk_forward"]}})
        self.assertEqual(entry["old_reward_config"]["expert_clip_prefixes"], [])
        self.assertEqual(entry["new_reward_config"]["expert_clip_prefixes"], ["walk_forward"])
        self.assertEqual(entry["expert_positive_sampling"], trainer.expert_sampling_metadata)
        self.assertFalse(entry["critic_reset"])


if __name__ == "__main__":
    unittest.main()
