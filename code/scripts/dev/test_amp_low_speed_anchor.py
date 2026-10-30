"""CPU checks for training-only preservation of a resumed AMP actor at low speed."""
import copy
import dataclasses
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from supermimic.amp_networks import CommandActor
from supermimic.amp_train import (AMPTrainer, AMPTrainCfg, actor_state_hash,
    low_speed_anchor_penalty, validate_cfg, validate_resume_config)


def trainer(weight=1.):
    t=AMPTrainer.__new__(AMPTrainer)
    t.cfg=AMPTrainCfg(device='cpu',resume='own_start.pt',low_speed_anchor_weight=weight,allow_reward_change=True)
    t.device='cpu';t.motion_hash='test-library'
    t.actor=CommandActor(history=4,hidden=(8,))
    t.wm=nn.Linear(4,4);t.critic=nn.Linear(4,1);t.prior=nn.Linear(4,1)
    t.opt_wm=torch.optim.Adam(t.wm.parameters());t.opt_actor=torch.optim.Adam(t.actor.parameters());t.opt_critic=torch.optim.Adam(t.critic.parameters());t.prior.optimizer=torch.optim.Adam(t.prior.parameters())
    t.it=42000;t.updates=16799;t.actor_updates=16599;t.disc_updates=33598;t.env_steps=177278976;t.train_time=1.;t.update_credit=.4
    t.initialization={'type':'random','initial_weight_hashes':{'actor':'original-random-initialization-hash'}}
    t.anchor_actor=None;t.anchor_metadata=None
    return t


def source():
    return {'cfg':{},'it':42000,'initialization':{'type':'random','initial_weight_hashes':{'actor':'original-random-initialization-hash'}}}


class LowSpeedAnchorTests(unittest.TestCase):
    def test_default_and_legacy_configs(self):
        cfg=AMPTrainCfg();saved=dataclasses.asdict(cfg);saved.pop('low_speed_anchor_weight')
        self.assertEqual(validate_resume_config(saved,cfg),{})
        changed=dataclasses.replace(cfg,low_speed_anchor_weight=1.,resume='own.pt')
        with self.assertRaises(ValueError):validate_resume_config(saved,changed)
        changed.allow_reward_change=True
        self.assertEqual(validate_resume_config(saved,changed)['low_speed_anchor_weight'],{'old':0.,'new':1.})

    def test_invalid_or_fresh_anchor_rejected(self):
        for v in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):validate_cfg(AMPTrainCfg(low_speed_anchor_weight=v,resume='own.pt'))
        with self.assertRaises(ValueError):validate_cfg(AMPTrainCfg(low_speed_anchor_weight=1.))
        validate_cfg(AMPTrainCfg(low_speed_anchor_weight=1.,resume='own.pt'))

    def test_gate_and_loss_units(self):
        cfg=AMPTrainCfg(low_speed_anchor_weight=2.)
        a=torch.ones(3,29);ref=torch.zeros_like(a);command=torch.tensor([[.3,0,0],[.65,0,0],[.8,0,0]])
        loss,mse=low_speed_anchor_penalty(a,ref,command,cfg)
        torch.testing.assert_close(mse,torch.tensor(.5))
        torch.testing.assert_close(loss,torch.tensor(1.))

    def test_gradients_only_reach_current_low_speed_action(self):
        a=torch.ones(2,29,requires_grad=True);ref=torch.zeros(2,29,requires_grad=True)
        loss,_=low_speed_anchor_penalty(a,ref,torch.tensor([[.3,0,0],[.8,0,0]]),AMPTrainCfg(low_speed_anchor_weight=1.))
        loss.backward();self.assertIsNone(ref.grad)
        self.assertGreater(a.grad[0].abs().sum().item(),0)
        torch.testing.assert_close(a.grad[1],torch.zeros(29),rtol=0,atol=0)

    def test_disabled_restore_touches_no_actor_and_consumes_no_rng(self):
        t=AMPTrainer.__new__(AMPTrainer);t.cfg=AMPTrainCfg()
        before=torch.get_rng_state().clone()
        t._restore_low_speed_anchor(source(),'own.pt')
        self.assertIsNone(t.anchor_actor)
        torch.testing.assert_close(torch.get_rng_state(),before,rtol=0,atol=0)
        self.assertFalse(hasattr(t,'actor'))

    def test_clone_is_frozen_independent_and_preserves_normalizer(self):
        t=trainer();t.actor.obs_norm.update(torch.randn(16,384))
        expected=actor_state_hash(t.actor);before=torch.get_rng_state().clone()
        t._restore_low_speed_anchor(source(),'own.pt')
        self.assertEqual(actor_state_hash(t.anchor_actor),expected)
        torch.testing.assert_close(torch.get_rng_state(),before,rtol=0,atol=0)
        self.assertFalse(t.anchor_actor.training)
        self.assertTrue(all(not p.requires_grad for p in t.anchor_actor.parameters()))
        with torch.no_grad():next(t.actor.parameters()).add_(1.)
        t.actor.obs_norm.update(torch.ones(16,384)*10)
        self.assertEqual(actor_state_hash(t.anchor_actor),expected)
        self.assertNotEqual(actor_state_hash(t.actor),expected)
        self.assertEqual(t.anchor_metadata['source_initialization']['initial_weight_hashes']['actor'],'original-random-initialization-hash')

    def test_save_load_and_disable_reenable_keep_original_reference(self):
        t=trainer();t._restore_low_speed_anchor(source(),'own_start.pt')
        expected=actor_state_hash(t.anchor_actor)
        with torch.no_grad():next(t.actor.parameters()).add_(.2)
        t.actor.obs_norm.update(torch.ones(8,384))
        actor_hash=actor_state_hash(t.actor)
        with tempfile.TemporaryDirectory() as directory, patch('torch.cuda.get_rng_state_all',return_value=[]),patch('torch.cuda.set_rng_state_all'):
            first=Path(directory)/'first.pt';t.save(first)
            restored=trainer();restored.load(first)
            self.assertEqual(actor_state_hash(restored.anchor_actor),expected)
            self.assertEqual(actor_state_hash(restored.actor),actor_hash)
            self.assertEqual(restored.anchor_metadata['source_checkpoint'],'own_start.pt')
            disabled=trainer(weight=0.);disabled.load(first)
            second=Path(directory)/'disabled.pt';disabled.save(second)
            enabled=trainer();enabled.load(second)
            self.assertEqual(actor_state_hash(enabled.anchor_actor),expected)
            self.assertEqual(enabled.anchor_metadata['source_checkpoint'],'own_start.pt')
            self.assertNotEqual(actor_state_hash(enabled.actor),expected)

    def test_missing_or_tampered_anchor_is_rejected(self):
        t=trainer();bad=source();bad['cfg']={'low_speed_anchor_weight':1.}
        with self.assertRaises(ValueError):t._restore_low_speed_anchor(bad,'own.pt')
        t._restore_low_speed_anchor(source(),'own.pt')
        saved={'cfg':{'low_speed_anchor_weight':1.},'low_speed_anchor':copy.deepcopy(t.anchor_actor.state_dict()),'low_speed_anchor_metadata':dict(t.anchor_metadata)}
        key=next(iter(saved['low_speed_anchor']));saved['low_speed_anchor'][key].add_(1)
        with self.assertRaises(ValueError):t._restore_low_speed_anchor(saved,'own.pt')

if __name__=='__main__':unittest.main()
