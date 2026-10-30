"""Check read-only hook and official contact API parity on existing GPU rollout."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,mujoco,mujoco_warp as mw
from evaluate_amp_contacts import ContactRecorder
from supermimic.amp_train import load_amp_actor
from supermimic.motion_lib import MotionLibrary
from supermimic.amp_eval import make_amp_eval_env,evaluate_command

actor,cfg,_=load_amp_actor('runs/amp_walk_best/best.pt')
env=make_amp_eval_env(MotionLibrary('data/processed/amp_walk_50hz.npz','cuda'),cfg,2,'cuda',2)
ends=[];reports=[]
for hooked in (False,False,True,True):
    rec=ContactRecorder(env) if hooked else None
    torch.manual_seed(59)
    last={}
    def observer(step,state,info,active):
        last.update({k:v.clone() for k,v in state.items()})
        if rec:rec.observe(step,state,info,active)
    reports.append(evaluate_command(actor,env,(.3,0,0),50,observer=observer,reset_perturbation=1))
    ends.append(last)
    if rec:rec.close()
def errors(i,j):return {k:float((ends[i][k]-ends[j][k]).abs().max()) for k in ends[i]}
checks={f'{i}_vs_{j}':errors(i,j) for i,j in ((0,1),(0,2),(0,3),(2,3))}
m=env.mj_model;d=env.sim.wp_data;cpu=mujoco.MjData(m)
mw.get_data_into(cpu,m,d,world_id=0)
n=int(d.nacon.numpy()[0]);indices=np.flatnonzero(d.contact.worldid.numpy()[:n]==0)
forces=rec.forces.numpy()[indices]
force_errors=[];point_errors=[]
for i,idx in enumerate(indices):
    f=np.zeros(6);mujoco.mj_contactForce(m,cpu,i,f)
    force_errors.append(float(np.max(np.abs(f-forces[i]))))
    con=cpu.contact[i];p=con.pos
    for geom in con.geom:
        body=m.geom_bodyid[geom];cv=cpu.cvel[body];reference=cpu.subtree_com[m.body_rootid[body]]
        ours=cv[3:]+np.cross(cv[:3],p-reference)
        obj=np.zeros(6);mujoco.mj_objectVelocity(m,cpu,mujoco.mjtObj.mjOBJ_BODY,int(body),obj,0)
        check=obj[3:]+np.cross(obj[:3],p-cpu.xipos[body])
        point_errors.append(float(np.max(np.abs(ours-check))))
# Resolve floating-point solver variability rather than demanding bitwise equality.
tolerances={'pos':.001,'rot':.01,'q':.01,'v':.01,'w':.05,'qd':.05}
rollout_pass=all(v<tolerances[k] for row in checks.values() for k,v in row.items())
result=dict(state_max_abs_errors=checks,conditions=['without','without','with','with'],
    rollout_hook_parity_pass=rollout_pass,rollout_tolerances=tolerances,
    note='GPU solver contact accumulation is not bitwise deterministic; uninstrumented repeat errors recorded alongside hook comparisons',
    force_decode_max_abs_error=max(force_errors,default=0),force_decode_cpu_mjContactForce_parity_pass=bool(len(indices) and max(force_errors)<1e-4),
    contact_point_velocity_max_abs_error=max(point_errors,default=0),mjObjectVelocity_point_parity_pass=bool(len(indices) and max(point_errors)<1e-6),
    actual_contacts_checked=len(indices),physics_samples_with_hook=len(rec.records),
    integrator=mujoco.mjtIntegrator(m.opt.integrator).name,
    methodology='CPU comparison exports existing GPU contact/cvel/constraint data without calling CPU forward or rebuilding contacts')
Path('runs/amp_speed_ablation_20261030/measurement_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert rollout_pass and result['force_decode_cpu_mjContactForce_parity_pass'] and result['mjObjectVelocity_point_parity_pass']
