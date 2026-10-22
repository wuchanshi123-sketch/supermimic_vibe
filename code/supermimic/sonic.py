"""Score SONIC's released g1 policy with the benchmark protocol (App. F), in our simulator.

As in the paper, SONIC runs "with its own actuator gains and action scale, receiving
observations as its training environment builds them". Conventions follow the deployment
reference (GR00T-WholeBodyControl gear_sonic_deploy/src/g1/g1_deploy_onnx_ref):
  - policy joint order is IsaacLab's; q_il[j] = q_mj[MUJOCO_TO_ISAACLAB[j]];
  - target_mj[i] = default[i] + a_il[ISAACLAB_TO_MUJOCO[i]] * scale[i], scale = 0.25 effort / kp;
  - encoder (g1 mode): reference joint positions / velocities (absolute, IsaacLab order) and the
    reference root orientation relative to the robot pelvis at frames t, t+5, ..., t+45;
  - decoder: token + 10-frame histories, oldest first, of pelvis gyro, q - default, qd, the raw
    previous actions and gravity in the pelvis frame.
The episode, reward, termination and normalized return are exactly SuperMimic's: SONIC's joint
targets are converted to our action units, so the simulator applies SONIC's targets and the
action-rate term is measured in the same units for both policies.

usage: python -m supermimic.sonic --motions data/processed/bones_seed_50hz.npz --weights data/sonic/sonic_g1.pt --out results/sonic_bones.json [--motor gains|full] [--clips N]
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from .motion_lib import MotionLibrary
from .rotations import mtm, mtv
from .sonic_net import SonicNet

SONIC_JOINTS = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
  "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
  "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
ISAACLAB_TO_MUJOCO = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
MUJOCO_TO_ISAACLAB = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
DEFAULT_ANGLES = [
  -0.312, 0.0, 0.0, 0.669, -0.363, 0.0, -0.312, 0.0, 0.0, 0.669, -0.363, 0.0, 0.0, 0.0, 0.0,
  0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0, 0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
]
# Motor constants of policy_parameters.hpp: armature, effort limit.
MOTORS = {"5020": (0.003609725, 25.0), "7520_14": (0.010177520, 88.0), "7520_22": (0.025101925, 139.0), "4010": (0.00425, 5.0)}
# Per MuJoCo joint: motor type and gain multiplier (ankles and waist roll/pitch: 2 x 5020).
JOINT_MOTOR = [
  ("7520_22", 1), ("7520_22", 1), ("7520_14", 1), ("7520_22", 1), ("5020", 2), ("5020", 2),
  ("7520_22", 1), ("7520_22", 1), ("7520_14", 1), ("7520_22", 1), ("5020", 2), ("5020", 2),
  ("7520_14", 1), ("5020", 2), ("5020", 2),
  ("5020", 1), ("5020", 1), ("5020", 1), ("5020", 1), ("5020", 1), ("4010", 1), ("4010", 1),
  ("5020", 1), ("5020", 1), ("5020", 1), ("5020", 1), ("5020", 1), ("4010", 1), ("4010", 1),
]
OMEGA = 10 * 2.0 * 3.1415926535
ZETA = 2.0


def sonic_actuators():
  """kp, kd, action scale, effort limit, armature per MuJoCo joint (SONIC's values)."""
  kp, kd, scale, effort, arm = [], [], [], [], []
  for mot, mult in JOINT_MOTOR:
    a, e = MOTORS[mot]
    k = a * OMEGA**2
    kp.append(mult * k)
    kd.append(mult * 2.0 * ZETA * a * OMEGA)
    # Action scale uses the single-motor values (0.25 * effort / stiffness), as in the header.
    scale.append(0.25 * e / k)
    effort.append(mult * e)
    arm.append(mult * a)
  return [torch.tensor(x, dtype=torch.float32) for x in (kp, kd, scale, effort, arm)]


class SonicActor:
  def __init__(self, net: SonicNet, env, lib: MotionLibrary, motor: str = "gains"):
    self.net, self.env, self.lib = net, env, lib
    dev = self.device = env.device
    n = env.num_envs
    ours = list(env.joint_names)
    assert sorted(ours) == sorted(SONIC_JOINTS), ours
    # Index maps: SONIC MuJoCo order <-> our order; IsaacLab order from SONIC MuJoCo order.
    self.ours_of_mj = torch.tensor([ours.index(j) for j in SONIC_JOINTS], device=dev)
    self.mj_of_ours = torch.tensor([SONIC_JOINTS.index(j) for j in ours], device=dev)
    self.mj_of_il = torch.tensor(MUJOCO_TO_ISAACLAB, device=dev)
    self.il_of_mj = torch.tensor(ISAACLAB_TO_MUJOCO, device=dev)
    kp, kd, scale, effort, arm = (t.to(dev) for t in sonic_actuators())
    self.default_mj = torch.tensor(DEFAULT_ANGLES, device=dev)
    self.scale_mj = scale
    self._set_actuators(kp, kd, effort, arm, motor)
    self.hist = {
      "w": torch.zeros(n, 10, 3, device=dev), "q": torch.zeros(n, 10, 29, device=dev),
      "qd": torch.zeros(n, 10, 29, device=dev), "a": torch.zeros(n, 10, 29, device=dev),
      "g": torch.zeros(n, 10, 3, device=dev),
    }
    self.last_a = torch.zeros(n, 29, device=dev)
    self.offsets = torch.arange(0, 50, 5, device=dev)

  def _set_actuators(self, kp, kd, effort, arm, motor):
    """Writes SONIC's PD gains (and with motor='full' also its effort limits and armature)
    into the simulated model; mjlab's G1 uses MuJoCo position actuators."""
    env = self.env
    # Unrandomized model fields are one array broadcast over the worlds; give these their own
    # per-world copies first (mjlab re-captures its CUDA graphs when it does this).
    fields = ("actuator_gainprm", "actuator_biasprm") + (("actuator_forcerange", "dof_armature") if motor == "full" else ())
    env.sim.expand_model_fields(fields)
    m = env.sim.model
    act_joint = env.act_to_joint  # actuator -> our joint index
    mj = self.mj_of_ours[act_joint]  # actuator -> SONIC MuJoCo joint index
    gp, bp = m.actuator_gainprm, m.actuator_biasprm
    before = (gp[..., :, 0].reshape(-1)[: len(mj)].clone(), bp[..., :, 2].reshape(-1)[: len(mj)].clone())
    gp[..., :, 0] = kp[mj]
    bp[..., :, 1] = -kp[mj]
    bp[..., :, 2] = -kd[mj]
    # Keep the host-side model in step, in case derived constants are ever rebuilt from it.
    mjm = env.sim.mj_model
    mjm.actuator_gainprm[:, 0] = kp[mj].cpu().numpy()
    mjm.actuator_biasprm[:, 1] = -kp[mj].cpu().numpy()
    mjm.actuator_biasprm[:, 2] = -kd[mj].cpu().numpy()
    self.kp_act, self.kd_act = kp[mj], kd[mj]
    changed = (before[0] - kp[mj]).abs() > 1e-3
    names = [SONIC_JOINTS[i] for i in mj.tolist()]
    print("[sonic] kp changed for:", [nm for nm, c in zip(names, changed.tolist()) if c])
    if motor == "full":
      fr = m.actuator_forcerange
      fr[..., :, 0] = -effort[mj]
      fr[..., :, 1] = effort[mj]
      da = m.dof_armature
      da[..., env.qd_adr[act_joint]] = arm[mj]
      from mjlab.managers.event_manager import RecomputeLevel

      env.sim.recompute_constants(RecomputeLevel.set_const)
      print("[sonic] effort limits and armature set to SONIC's motor values")

  def _proprio(self):
    s = self.env.state
    R = s["rot"]
    w_b = mtv(R, s["w"])
    g_b = mtv(R, torch.tensor([0.0, 0.0, -1.0], device=self.device).expand_as(s["w"]))
    q_mj = s["q"][:, self.ours_of_mj]
    qd_mj = s["qd"][:, self.ours_of_mj]
    q_il = (q_mj - self.default_mj)[:, self.mj_of_il]
    qd_il = qd_mj[:, self.mj_of_il]
    return w_b, q_il, qd_il, g_b

  def reset(self, ids):
    w, q, qd, g = self._proprio()
    for k, v in (("w", w), ("q", q), ("qd", qd), ("g", g)):
      self.hist[k][ids] = v[ids, None].expand(-1, 10, -1)
    self.hist["a"][ids] = 0.0
    self.last_a[ids] = 0.0

  @torch.no_grad()
  def act(self):
    env, lib = self.env, self.lib
    n = env.num_envs
    # History: append the current state with the previous raw action.
    w, q, qd, g = self._proprio()
    for k, v in (("w", w), ("q", q), ("qd", qd), ("a", self.last_a), ("g", g)):
      self.hist[k] = torch.cat([self.hist[k][:, 1:], v[:, None]], dim=1)
    # Encoder inputs at reference frames t, t+5, ..., t+45 (held at the clip's last frame).
    frame = env.frame
    idx = lib.clamp_to_clip(frame[:, None] + self.offsets, frame)
    ref = lib.frame(idx)
    rq = ref["q"][..., self.ours_of_mj][..., self.mj_of_il]
    rqd = ref["qd"][..., self.ours_of_mj][..., self.mj_of_il]
    Rrel = mtm(env.state["rot"][:, None], ref["rot"])  # R_base^T R_ref
    anchor = Rrel[..., :, :2].reshape(n, 60)
    token = self.net.encode_g1(torch.cat([rq.reshape(n, 290), rqd.reshape(n, 290)], -1), anchor)
    h = self.hist
    dec_hist = torch.cat([h["w"].reshape(n, 30), h["q"].reshape(n, 290), h["qd"].reshape(n, 290),
                          h["a"].reshape(n, 290), h["g"].reshape(n, 30)], -1)
    a_il = self.net.decode(token, dec_hist)
    self.last_a = a_il
    target_mj = self.default_mj + a_il[:, self.il_of_mj] * self.scale_mj
    target_ours = target_mj[:, self.mj_of_ours]
    # Our action units: q_sp = action_scale * a + q0.
    return (target_ours - env.q0) / env.action_scale


@torch.no_grad()
def evaluate_sonic(actor: SonicActor, lib: MotionLibrary, env, clip_ids=None, verbose=True):
  device, n, tcfg = env.device, env.num_envs, env.tcfg
  if clip_ids is None:
    clip_ids = torch.arange(lib.num_clips, device=device)
  order = clip_ids[torch.argsort(lib.lengths[clip_ids], descending=True)]
  ret = torch.zeros(lib.num_clips, device=device)
  failed = torch.zeros(lib.num_clips, dtype=torch.bool, device=device)
  t0 = time.time()
  env.reset()
  for c0 in range(0, len(order), n):
    clips = order[c0 : c0 + n]
    k = len(clips)
    ids = torch.arange(k, device=device)
    env.reset_envs(ids, frames=lib.starts[clips])
    env._finish_resets(ids, force=True)
    actor.reset(torch.arange(n, device=device))
    active = torch.zeros(n, dtype=torch.bool, device=device)
    active[:k] = True
    acc = torch.zeros(n, device=device)
    while active.any():
      a = actor.act()
      _, info = env.step(a)
      acc += info["reward"] * active
      fin = info["done"] & active
      if fin.any():
        i = fin.nonzero().squeeze(-1)
        ret[clips[i]] = acc[i]
        failed[clips[i]] = info["terminated"][i]
        active[i] = False
    if verbose:
      print(f"  clips {c0 + k}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)
  sel = clip_ids
  max_ret = tcfg.max_step_reward * (lib.lengths[sel] - 1).float()
  norm = (ret[sel] / max_ret).clamp(max=1.0)
  return {
    "norm_return": norm.mean().item(),
    "norm_return_frame_weighted": (ret[sel].sum() / max_ret.sum()).item(),
    "failed": int(failed[sel].sum().item()),
    "num_clips": int(len(sel)),
    "per_clip_norm_return": norm.tolist(),
    "per_clip_failed": failed[sel].tolist(),
    "eval_seconds": time.time() - t0,
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--motions", required=True)
  ap.add_argument("--weights", default="data/sonic/sonic_g1.pt")
  ap.add_argument("--out", required=True)
  ap.add_argument("--motor", default="gains", choices=["gains", "full"])
  ap.add_argument("--clips", type=int, default=0, help="score a random subset of this many clips")
  ap.add_argument("--num-envs", type=int, default=4096)
  args = ap.parse_args()
  from .evaluate import make_eval_env

  device = "cuda"
  torch.backends.cuda.matmul.allow_tf32 = False
  lib = MotionLibrary(args.motions, device)
  net = SonicNet().to(device)
  net.load_state_dict(torch.load(args.weights, map_location=device))
  net.eval()
  clip_ids = None
  if args.clips and args.clips < lib.num_clips:
    g = torch.Generator().manual_seed(1234)
    clip_ids = torch.randperm(lib.num_clips, generator=g)[: args.clips].to(device)
  n = min(args.num_envs, lib.num_clips if clip_ids is None else len(clip_ids))
  env = make_eval_env(lib, n, device)
  actor = SonicActor(net, env, lib, motor=args.motor)
  res = evaluate_sonic(actor, lib, env, clip_ids)
  gp = env.sim.model.actuator_gainprm[..., :, 0].reshape(-1, len(actor.kp_act))
  assert torch.allclose(gp, actor.kp_act.expand_as(gp)), "actuator gains were reset during the run"
  summary = {k: v for k, v in res.items() if not k.startswith("per_clip")}
  summary.update(motions=args.motions, motor=args.motor, clips_subset=args.clips)
  print(json.dumps(summary), flush=True)
  ids = clip_ids.tolist() if clip_ids is not None else list(range(lib.num_clips))
  json.dump({**summary, "clip_ids": ids, "names": [lib.names[i] for i in ids],
             "per_clip_norm_return": res["per_clip_norm_return"], "per_clip_failed": res["per_clip_failed"]},
            open(args.out, "w"))


if __name__ == "__main__":
  main()
