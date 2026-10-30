"""CPU geometry and gradient checks for differentiable AMP foot contact costs."""
from __future__ import annotations

import math
import unittest

import mujoco
import numpy as np
import torch

from supermimic.amp_contact_reward import FootContactCosts


def model(kind="sphere", floor=0., omit=None, ground_quat="1 0 0 0"):
    bodies=[]
    for side in ("left", "right"):
        geoms=[]
        for index in range(1,8):
            name=f"{side}_foot{index}_collision"
            if name == omit:continue
            if kind=="capsule":
                spec='type="capsule" fromto="-0.1 0 0 0.1 0 0" size="0.01"'
            elif kind=="box":
                spec='type="box" size="0.01 0.01 0.01"'
            else:spec='type="sphere" pos="0 0 0" size="0.01"'
            geoms.append(f'<geom name="{name}" {spec}/>')
        bodies.append(f'<body name="{side}_ankle_roll_link" pos="0 0 0.5">'+''.join(geoms)+'</body>')
    ground=f'<geom name="ground" type="plane" pos="0 0 {floor}" quat="{ground_quat}" size="1 1 .1"/>'
    return mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+ground+''.join(bodies)+'</worldbody></mujoco>')


NAMES=("left_ankle_roll_link","right_ankle_roll_link")


def kin(shape=(2,), height=.01, downward=0., dtype=torch.float64):
    p=torch.zeros(shape+(2,3),dtype=dtype);p[...,2]=height
    r=torch.eye(3,dtype=dtype).expand(shape+(2,3,3)).clone()
    v=torch.zeros_like(p);v[...,2]=-downward
    return p,r,v,torch.zeros_like(p)


class ContactCostTests(unittest.TestCase):
    def test_sphere_distance_plane_height_and_reduction(self):
        c=FootContactCosts.from_model(model(floor=.2),NAMES)
        data=kin((2,3),height=.207,downward=.5)
        s=c.samples(data)
        self.assertEqual(s['centers'].shape,(2,3,14,2,3))
        self.assertEqual(s['minimum_clearance'].shape,(2,3,2))
        torch.testing.assert_close(s['clearance'],torch.full((2,3,14,2),-.003,dtype=torch.float64))
        costs=c.costs(data,penetration_scale=.005,penetration_slop=.0005)
        self.assertEqual(costs['landing'].shape,(2,3))
        torch.testing.assert_close(costs['penetration'],torch.full((2,3),.5,dtype=torch.float64),atol=1e-7,rtol=1e-6)
        expected=2*.5**2*torch.sigmoid(torch.tensor((.02+.003)/.01,dtype=torch.float64))
        torch.testing.assert_close(costs['landing'],expected.expand(2,3))

    def test_capsule_endpoints_rotation_and_rigid_point_velocity(self):
        c=FootContactCosts.from_model(model('capsule'),NAMES)
        p,r,v,w=kin((1,),height=.3)
        r[:]=torch.tensor([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]],dtype=torch.float64)
        w[...,1]=2.
        s=c.samples((p,r,v,w))
        # A local x-aligned segment becomes vertical after a y rotation.
        offsets=s['centers']-p[:,[0]*7+[1]*7,None,:]
        torch.testing.assert_close(offsets[...,2].abs(),torch.full((1,14,2),.1,dtype=torch.float64))
        expected=torch.cross(w[:,[0]*7+[1]*7,None,:].expand_as(offsets),offsets,dim=-1)
        torch.testing.assert_close(s['velocities'],expected)
        torch.testing.assert_close(s['minimum_clearance'],torch.full((1,2),.19,dtype=torch.float64))

    def test_capsule_lowest_endpoint_matches_mujoco_geometry(self):
        m=model('capsule')
        c=FootContactCosts.from_model(m,NAMES)
        p,r,v,w=kin((1,),height=.01)
        theta=.25
        r[:]=torch.tensor([[math.cos(theta),0.,math.sin(theta)],[0.,1.,0.],[-math.sin(theta),0.,math.cos(theta)]],dtype=torch.float64)
        expected=.01-.1*math.sin(theta)-.01
        torch.testing.assert_close(c.samples((p,r,v,w))['minimum_clearance'],torch.full((1,2),expected,dtype=torch.float64))
        self.assertTrue(all(t=='capsule' for t in c.metadata['geometry_types']))

    def test_rotating_capsule_has_landing_cost_with_zero_body_linear_velocity(self):
        c=FootContactCosts.from_model(model('capsule'),NAMES)
        p,r,v,w=kin((1,),height=.01)
        w[...,1]=2.
        # Pitching a horizontal 20cm capsule lowers one endpoint at 0.2m/s.
        landing=c.costs((p,r,v,w))['landing']
        expected=.04*torch.sigmoid(torch.tensor(2.,dtype=torch.float64))
        torch.testing.assert_close(landing,expected.expand(1))

    def test_penetration_slop_and_upward_motion_have_zero_cost(self):
        c=FootContactCosts.from_model(model(),NAMES)
        data=kin(height=.00975,downward=-.2)
        cost=c.costs(data,penetration_slop=.0005)
        torch.testing.assert_close(cost['penetration'],torch.zeros(2,dtype=torch.float64),rtol=0,atol=0)
        torch.testing.assert_close(cost['landing'],torch.zeros(2,dtype=torch.float64),rtol=0,atol=0)

    def test_gradients_raise_penetrated_feet_and_reduce_descent(self):
        c=FootContactCosts.from_model(model(),NAMES)
        p,r,v,w=kin((1,),height=.008,downward=.5)
        p.requires_grad_();v.requires_grad_()
        cost=c.costs((p,r,v,w),penetration_scale=.005)
        grad_p,grad_v=torch.autograd.grad(cost['landing'].sum()+cost['penetration'].sum(),(p,v))
        self.assertTrue(bool((grad_p[...,2]<0).all()))
        self.assertTrue(bool((grad_v[...,2]<0).all()))
        torch.testing.assert_close(grad_p[...,:2],torch.zeros_like(grad_p[...,:2]))
        torch.testing.assert_close(grad_v[...,:2],torch.zeros_like(grad_v[...,:2]))
        # Gradient descent of the positive cost raises position and vertical velocity.
        self.assertTrue(torch.isfinite(grad_p).all() and torch.isfinite(grad_v).all())

    def test_height_gate_is_differentiable_and_matches_finite_difference(self):
        c=FootContactCosts.from_model(model(),NAMES)
        p,r,v,w=kin((1,),height=.025,downward=.3)
        p.requires_grad_()
        val=c.costs((p,r,v,w))['landing'].sum()
        gradient=torch.autograd.grad(val,p)[0][0,0,2]
        eps=1e-5
        plus=p.detach().clone();minus=plus.clone();plus[0,0,2]+=eps;minus[0,0,2]-=eps
        numerical=(c.costs((plus,r,v,w))['landing'].sum()-c.costs((minus,r,v,w))['landing'].sum())/(2*eps)
        torch.testing.assert_close(gradient,numerical,atol=1e-6,rtol=1e-6)
        self.assertLess(float(gradient),0.)

    def test_invalid_geometry_and_parameters_are_rejected(self):
        for m,names in [(model('box'),NAMES),(model(omit='left_foot7_collision'),NAMES),(model(),('missing','right_ankle_roll_link')),(model(),('x/left_ankle_roll_link','y/left_ankle_roll_link'))]:
            with self.assertRaises(ValueError):FootContactCosts.from_model(m,names)
        c=FootContactCosts.from_model(model(),NAMES)
        for key,bad in [('landing_height',-.01),('landing_width',0.),('penetration_scale',0.),('penetration_slop',-.01),('landing_width',float('nan')),('penetration_slop',float('inf'))]:
            with self.subTest(key=key,bad=bad),self.assertRaises(ValueError):c.costs(kin(),**{key:bad})
        m=model();m.geom_size[1,0]=float('nan')
        with self.assertRaises(ValueError):FootContactCosts.from_model(m,NAMES)

    def test_unsupported_ground_orientation_and_missing_plane_are_rejected(self):
        m=model(ground_quat=f'{math.cos(.1)} {math.sin(.1)} 0 0')
        with self.assertRaises(ValueError):FootContactCosts.from_model(m,NAMES)
        m=model();m.geom_type[0]=int(mujoco.mjtGeom.mjGEOM_SPHERE)
        with self.assertRaises(ValueError):FootContactCosts.from_model(m,NAMES)


if __name__=='__main__':unittest.main()
