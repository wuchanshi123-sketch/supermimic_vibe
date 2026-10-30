"""Integration checks for opt-in command-gated AMP contact/yaw refinement."""
import dataclasses
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from supermimic.amp_task import AMPTask, AMPTaskCfg
from supermimic.amp_train import AMPTrainCfg, task_config, validate_cfg, validate_resume_config
from test_amp_curriculum import FakeBase, state, kinematics

FIELDS=('yaw_rate_weight','foot_landing_weight','foot_penetration_weight','refinement_speed_start',
        'refinement_speed_full','landing_height','landing_width','penetration_scale','penetration_slop')

class HighSpeedRefinementTests(unittest.TestCase):
    def test_new_defaults_preserve_historical_terms(self):
        task=AMPTask(FakeBase())
        kin=kinematics(bodies=2);a=torch.zeros(2,29)
        _,terms=task.reward(state(),a,a,torch.tensor([[.8,0.,0.],[.3,0.,0.]]),kin=kin)
        self.assertEqual(set(terms),{'linear','yaw','upright','action_rate','joint_limit'})
        self.assertIsNone(task.contact_costs)

    def test_speed_gate_and_dt_weights(self):
        cfg=AMPTaskCfg(w_yaw_rate=-2.,w_foot_landing=-.1,w_foot_penetration=-.02)
        geometry=SimpleNamespace(costs=lambda kin,**kwargs: {'landing':torch.ones(3),'penetration':torch.full((3,),2.)})
        with patch('supermimic.amp_contact_reward.FootContactCosts.from_model',return_value=geometry):
            task=AMPTask(FakeBase(),cfg,mj_model=object())
        kin=list(kinematics(batch=3,bodies=2));kin[3][:,1,2]=.2
        a=torch.zeros(3,29);commands=torch.tensor([[.3,0.,0.],[.65,0.,0.],[.8,0.,0.]])
        _,terms=task.reward(state(3),a,a,commands,kin=tuple(kin))
        gate=torch.tensor([0.,.5,1.])
        torch.testing.assert_close(terms['yaw_rate_refinement'],.02*-2*.2**2*gate)
        torch.testing.assert_close(terms['foot_landing'],.02*-.1*gate)
        torch.testing.assert_close(terms['foot_penetration'],.02*-.02*2*gate)

    def test_yaw_gradient_reduces_oscillation_without_roll_pitch_cost(self):
        task=AMPTask(FakeBase(),AMPTaskCfg(w_yaw_rate=-2.))
        kin=list(kinematics(bodies=2));angular=torch.tensor([[[0.,0.,0.],[.4,-.5,.2]],[[0.,0.,0.],[.4,-.5,-.2]]],requires_grad=True);kin[3]=angular
        a=torch.zeros(2,29)
        _,terms=task.reward(state(),a,a,torch.tensor([[.8,0.,0.],[.8,0.,0.]]),kin=tuple(kin))
        terms['yaw_rate_refinement'].sum().backward()
        self.assertLess(angular.grad[0,1,2].item(),0)
        self.assertGreater(angular.grad[1,1,2].item(),0)
        torch.testing.assert_close(angular.grad[:,1,:2],torch.zeros(2,2),rtol=0,atol=0)

    def test_legacy_configs_and_resume_guard(self):
        cfg=AMPTrainCfg();saved=dataclasses.asdict(cfg)
        for name in FIELDS:saved.pop(name)
        self.assertEqual(validate_resume_config(saved,cfg),{})
        for name in ('yaw_rate_weight','foot_landing_weight','foot_penetration_weight'):
            changed=dataclasses.replace(cfg,**{name:-.1})
            with self.assertRaises(ValueError):validate_resume_config(saved,changed)
            changed.allow_reward_change=True
            self.assertIn(name,validate_resume_config(saved,changed))
        new=dataclasses.replace(cfg,yaw_rate_weight=-2.,foot_landing_weight=-.1,foot_penetration_weight=-.02)
        task=task_config(new)
        self.assertEqual((task.w_yaw_rate,task.w_foot_landing,task.w_foot_penetration),(-2.,-.1,-.02))

    def test_invalid_values_and_missing_geometry_rejected(self):
        for field in ('yaw_rate_weight','foot_landing_weight','foot_penetration_weight'):
            for value in (.1,float('nan'),-float('inf')):
                with self.assertRaises(ValueError):validate_cfg(dataclasses.replace(AMPTrainCfg(),**{field:value}))
        for change in ({'refinement_speed_start':.8},{'landing_width':0.},{'penetration_scale':0.},{'penetration_slop':-.001}):
            with self.assertRaises(ValueError):validate_cfg(dataclasses.replace(AMPTrainCfg(),**change))
        with self.assertRaises(ValueError):AMPTask(FakeBase(),AMPTaskCfg(w_foot_landing=-.1))

if __name__=='__main__':unittest.main()
