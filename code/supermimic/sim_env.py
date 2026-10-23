"""Batched G1 motion-tracking simulator on MuJoCo Warp (via mjlab Scene/Simulation).

The env owns the simulator, domain randomization (sim context xi), reset sampling
and the noisy observation history the policy sees. Reward and termination come from
supermimic.task so the simulator and the imagined rollouts share one definition.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sim.sim import Simulation
from mjlab.terrains import TerrainEntityCfg

from .kinematics import FastTreeKinematics
from .motion_lib import MotionLibrary
from .rotations import quat_from_euler_xyz, quat_mul, quat_normalize, quat_to_mat
from .task import NJ, OBS_DIM, XI_DIM, Task, TaskCfg

FOOT_GEOM_RE = r"^(left|right)_foot[1-7]_collision$"


@dataclass
class EnvCfg:
  num_envs: int = 4096
  episode_length_s: float = 10.0
  domain_rand: bool = True
  obs_noise: bool = True
  reset_noise: bool = True
  eval_friction: float = 1.0
  nconmax: int | None = 35
  njmax: int | None = 250
  set_const_every: int = 50
  compile: bool = True
  # Reset sampling mixture (uniform, low-return softmax, past-softmax).
  p_uniform: float = 0.5
  p_softmax: float = 0.4
  p_past: float = 0.1
  softmax_temp: float = 25.0
  bin_ema: float = 0.1
  # How the per-bin return estimates G_b are updated: "start" = only the bin an episode starts
  # in, with the episode's return; "traversed" = every bin the episode passes through, with its
  # discounted return-to-go from there (the paper: "a running estimate of the discounted return
  # from it, updated on the policy's own rollouts"). Both normalize by the largest return
  # attainable over the same remaining horizon.
  reset_stat: str = "start"
  # "traversed": bins entered with fewer steps left than this are not updated (the first
  # traversed test run, runs/bones_fix_traversed, used 25).
  rtg_min_steps: int = 1
  # BeyondMimic reset randomization.
  pose_range: tuple = ((-0.05, 0.05), (-0.05, 0.05), (-0.01, 0.01), (-0.1, 0.1), (-0.1, 0.1), (-0.2, 0.2))
  vel_range: tuple = ((-0.5, 0.5), (-0.5, 0.5), (-0.2, 0.2), (-0.52, 0.52), (-0.52, 0.52), (-0.78, 0.78))
  joint_pos_range: tuple = (-0.1, 0.1)
  # DR ranges.
  friction_range: tuple = (0.3, 1.2)
  enc_bias_range: tuple = (-0.01, 0.01)
  com_range: tuple = ((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05))


def _ranges(r, device):
  r = torch.tensor(r, dtype=torch.float32, device=device)
  return r[..., 0], r[..., 1]


class TrackingSimEnv:
  def __init__(self, lib: MotionLibrary, cfg: EnvCfg, task_cfg: TaskCfg, device: str, gamma: float = 0.99):
    self.lib = lib
    self.cfg = cfg
    self.device = device
    self.num_envs = n = cfg.num_envs
    self.gamma = gamma
    scene_cfg = SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=n,
      entities={"robot": get_g1_robot_cfg()},
    )
    self.scene = Scene(scene_cfg, device=device)
    sim_cfg = SimulationCfg(
      nconmax=cfg.nconmax,
      njmax=cfg.njmax,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    )
    self.sim = Simulation(num_envs=n, cfg=sim_cfg, model=self.scene.compile(), device=device)
    self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
    self.robot = self.scene["robot"]
    m = self.sim.mj_model
    self.mj_model = m
    self.decimation = 4

    # Robot-only kinematic tree (the scene adds a prefix to names).
    root_body = self._find_body("pelvis")
    self.kin = FastTreeKinematics(m, device, root_body=root_body)
    self.kin_c = torch.compile(self.kin.forward, dynamic=False) if cfg.compile else self.kin.forward
    self.root_qadr = m.jnt_qposadr[m.body_jntadr[root_body]]
    self.root_vadr = m.jnt_dofadr[m.body_jntadr[root_body]]
    hinge_ids = [m.body_jntadr[root_body + 1 + i] for i in range(NJ)]
    self.q_adr = torch.tensor([m.jnt_qposadr[j] for j in hinge_ids], device=device)
    self.qd_adr = torch.tensor([m.jnt_dofadr[j] for j in hinge_ids], device=device)
    # Actuator -> joint index (in our joint order).
    jid_to_k = {j: k for k, j in enumerate(hinge_ids)}
    self.act_to_joint = torch.tensor([jid_to_k[m.actuator_trnid[a, 0]] for a in range(m.nu)], device=device)
    assert m.nu == NJ

    rcfg = self.robot
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge_ids]
    self.joint_names = [nm.split("/")[-1] for nm in names]
    q0 = torch.tensor(rcfg.data.default_joint_pos[0].cpu().numpy(), device=device)
    rob_joint_names = [nm.split("/")[-1] for nm in rcfg.joint_names]
    perm = [rob_joint_names.index(nm) for nm in self.joint_names]
    self.q0 = q0[perm]
    lim = rcfg.data.soft_joint_pos_limits[0].to(device)[perm]
    scale = torch.zeros(NJ)
    import re

    for k, nm in enumerate(self.joint_names):
      for pat, val in G1_ACTION_SCALE.items():
        if re.fullmatch(pat, nm):
          scale[k] = val
    assert (scale > 0).all()
    self.action_scale = scale.to(device)
    self.task = Task(self.kin, self.q0, lim, self.action_scale, task_cfg, device,
                     compile_sizes=(n,) if cfg.compile else ())
    self.tcfg = task_cfg

    # Domain randomization fields.
    self.foot_geoms = torch.tensor(
      [g for g in range(m.ngeom) if _match(FOOT_GEOM_RE, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g))],
      device=device,
    )
    assert len(self.foot_geoms) == 14, len(self.foot_geoms)
    self.torso_body = self._find_body("torso_link")
    if cfg.domain_rand:
      self.sim.expand_model_fields(("geom_friction", "body_ipos"))
    self.default_torso_ipos = torch.tensor(m.body_ipos[self.torso_body], dtype=torch.float32, device=device)
    self.friction = torch.ones(n, device=device)
    self.enc_bias = torch.zeros(n, NJ, device=device)
    self.com_offset = torch.zeros(n, 3, device=device)

    # Episode bookkeeping.
    self.frame = torch.zeros(n, dtype=torch.long, device=device)
    self.ep_len = torch.zeros(n, dtype=torch.long, device=device)
    self.ep_id = torch.zeros(n, dtype=torch.long, device=device)
    self._next_ep = n
    self.max_ep_len = int(round(cfg.episode_length_s / task_cfg.dt))
    self.prev_action = torch.zeros(n, NJ, device=device)
    self.obs_hist = torch.zeros(n, task_cfg.history, OBS_DIM, device=device)
    # Return-based reset sampling.
    nb = lib.num_bins
    self.bin_G = torch.zeros(nb, device=device)
    self.bin_visits = torch.zeros(nb, device=device)
    self.bin_past_w = torch.zeros(nb, device=device)
    self.ep_bin = torch.zeros(n, dtype=torch.long, device=device)
    self.ep_disc_ret = torch.zeros(n, device=device)
    self.ep_disc_max = torch.zeros(n, device=device)
    self.ep_ret = torch.zeros(n, device=device)
    if cfg.reset_stat == "traversed":
      self.rew_hist = torch.zeros(n, self.max_ep_len, device=device)
    self.eval_mode = False
    self.state = None
    self.goal_buf = None

  # Model helpers.

  def _find_body(self, name):
    m = self.sim.mj_model
    for b in range(m.nbody):
      if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b).split("/")[-1] == name:
        return b
    raise KeyError(name)

  # State IO.

  def read_state(self) -> dict:
    qpos = self.sim.data.qpos
    qvel = self.sim.data.qvel
    ra, va = self.root_qadr, self.root_vadr
    pos = qpos[:, ra : ra + 3].clone()
    rot = quat_to_mat(quat_normalize(qpos[:, ra + 3 : ra + 7].clone()))
    v = qvel[:, va : va + 3].clone()
    w_loc = qvel[:, va + 3 : va + 6].clone()
    w = (rot @ w_loc.unsqueeze(-1)).squeeze(-1)
    q = qpos[:, self.q_adr].clone()
    qd = qvel[:, self.qd_adr].clone()
    return {"pos": pos, "rot": rot, "q": q, "v": v, "w": w, "qd": qd}

  def write_state(self, env_ids, pos, quat, q, v, w_world, qd):
    ra, va = self.root_qadr, self.root_vadr
    rot = quat_to_mat(quat)
    w_loc = (rot.transpose(-1, -2) @ w_world.unsqueeze(-1)).squeeze(-1)
    qpos = self.sim.data.qpos[env_ids].clone()
    qvel = self.sim.data.qvel[env_ids].clone()
    qpos[:, ra : ra + 3] = pos
    qpos[:, ra + 3 : ra + 7] = quat
    qpos[:, self.q_adr] = q
    qvel[:, va : va + 3] = v
    qvel[:, va + 3 : va + 6] = w_loc
    qvel[:, self.qd_adr] = qd
    self.sim.data.qpos[env_ids] = qpos
    self.sim.data.qvel[env_ids] = qvel

  @property
  def xi(self) -> torch.Tensor:
    return torch.cat([self.friction[:, None], self.enc_bias, self.com_offset], dim=-1)

  # Domain randomization.

  def _randomize(self, env_ids):
    k = len(env_ids)
    c = self.cfg
    if c.domain_rand and not self.eval_mode:
      lo, hi = c.friction_range
      self.friction[env_ids] = torch.rand(k, device=self.device) * (hi - lo) + lo
      lo, hi = c.enc_bias_range
      self.enc_bias[env_ids] = torch.rand(k, NJ, device=self.device) * (hi - lo) + lo
      lo, hi = _ranges(c.com_range, self.device)
      self.com_offset[env_ids] = torch.rand(k, 3, device=self.device) * (hi - lo) + lo
    else:
      self.friction[env_ids] = c.eval_friction
      self.enc_bias[env_ids] = 0.0
      self.com_offset[env_ids] = 0.0
    if c.domain_rand:
      gf = self.sim.model.geom_friction
      vals = gf[env_ids][:, self.foot_geoms].clone()
      vals[..., 0] = self.friction[env_ids, None]
      idx = env_ids[:, None].expand(-1, len(self.foot_geoms))
      gf[idx, self.foot_geoms[None].expand(k, -1)] = vals
      ip = self.sim.model.body_ipos
      ip[env_ids, self.torso_body] = self.default_torso_ipos + self.com_offset[env_ids]
      self._need_set_const = True

  # Reset sampling.

  def _sample_start_frames(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    lib, c = self.lib, self.cfg
    nb = lib.num_bins
    G = self.bin_G
    soft = torch.softmax(-c.softmax_temp * G, dim=0)
    self.bin_past_w = torch.maximum(self.bin_past_w * 0.999, soft)
    probs = c.p_uniform / nb + c.p_softmax * soft + c.p_past * self.bin_past_w / self.bin_past_w.sum().clamp_min(1e-12)
    bins = torch.multinomial(probs, k, replacement=True)
    lo, hi = lib.bin_start[bins], lib.bin_end[bins]
    # Leave at least one transition before the clip end.
    clip_last = lib.ends[lib.bin_clip[bins]] - 1
    hi = torch.minimum(hi, clip_last).clamp_min(lo + 1)
    u = torch.rand(k, device=self.device)
    frames = (lo + (u * (hi - lo).float()).long()).clamp(max=clip_last - 1).clamp_min(lib.starts[lib.bin_clip[bins]])
    return frames, bins

  def reset_envs(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
    k = len(env_ids)
    if k == 0:
      return
    c = self.cfg
    if frames is None:
      frames, bins = self._sample_start_frames(k)
    else:
      bins = self.lib.bin_of_frame[frames]
    self.sim.reset(env_ids)
    self._randomize(env_ids)
    ref = self.lib.frame(frames)
    pos, quat, q = ref["pos"].clone(), ref["quat"].clone(), ref["q"].clone()
    v, w, qd = ref["v"].clone(), ref["w"].clone(), ref["qd"].clone()
    if c.reset_noise and not self.eval_mode:
      lo, hi = _ranges(c.pose_range, self.device)
      r = torch.rand(k, 6, device=self.device) * (hi - lo) + lo
      pos = pos + r[:, :3]
      quat = quat_normalize(quat_mul(quat_from_euler_xyz(r[:, 3], r[:, 4], r[:, 5]), quat))
      lo, hi = _ranges(c.vel_range, self.device)
      r = torch.rand(k, 6, device=self.device) * (hi - lo) + lo
      v = v + r[:, :3]
      w = w + r[:, 3:]
      lo, hi = c.joint_pos_range
      q = q + torch.rand(k, NJ, device=self.device) * (hi - lo) + lo
      q = torch.minimum(torch.maximum(q, self.task.qmin), self.task.qmax)
    self.write_state(env_ids, pos, quat, q, v, w, qd)
    self.frame[env_ids] = frames
    self.ep_len[env_ids] = 0
    self.ep_id[env_ids] = torch.arange(self._next_ep, self._next_ep + k, device=self.device)
    self._next_ep += k
    self.prev_action[env_ids] = 0.0
    self.ep_bin[env_ids] = bins
    self.ep_disc_ret[env_ids] = 0.0
    self.ep_disc_max[env_ids] = 0.0
    self.ep_ret[env_ids] = 0.0

  def reset(self):
    ids = torch.arange(self.num_envs, device=self.device)
    self._need_set_const = False
    self._steps_since_const = 0
    self.reset_envs(ids)
    self._finish_resets(ids, force=True)
    return self.state

  def _finish_resets(self, env_ids, force=False):
    # set_const costs ~0.1 s for 4096 worlds, so derived constants (inverse weights
    # used for constraint regularization) are refreshed every set_const_every steps;
    # the COM offset itself acts immediately through body_ipos.
    self._steps_since_const += 1
    if self.cfg.domain_rand and self._need_set_const and (force or self._steps_since_const >= self.cfg.set_const_every):
      self.sim.recompute_constants(RecomputeLevel.set_const)
      self._need_set_const = False
      self._steps_since_const = 0
    s_all = self.read_state()
    self.state = s_all
    if len(env_ids):
      s = {kk: vv[env_ids] for kk, vv in s_all.items()}
      o = self.task.obs_clean(s, self.prev_action[env_ids], self.enc_bias[env_ids])
      o = o + self._noise(len(env_ids))
      self.obs_hist[env_ids] = o.unsqueeze(1).expand(-1, self.tcfg.history, -1)
      g = self.task.goal(s, self.ref_future(self.frame[env_ids]))
      if self.goal_buf is None:
        self.goal_buf = g
      else:
        self.goal_buf[env_ids] = g

  def _noise(self, k):
    if self.cfg.obs_noise and not self.eval_mode:
      return self.task.sample_obs_noise((k,))
    return torch.zeros(k, OBS_DIM, device=self.device)

  # Policy inputs.

  def ref_future(self, frames: torch.Tensor) -> dict:
    K = self.tcfg.future
    idx = frames[:, None] + torch.arange(1, K + 1, device=self.device)
    idx = self.lib.clamp_to_clip(idx, frames)
    return self.lib.frame(idx)

  def policy_inputs(self):
    return self.obs_hist.flatten(1), self.goal_buf

  # Step.

  def step(self, action: torch.Tensor):
    """Executes `action` (already including exploration noise). Returns the
    pre-reset next state and transition info; the env then resets finished
    episodes so self.state / policy_inputs() refer to the new episodes."""
    q_sp = self.action_scale * action + self.q0
    target = q_sp - self.enc_bias
    self.sim.data.ctrl[:] = target[:, self.act_to_joint]
    for _ in range(self.decimation):
      self.sim.step()
    s_next = self.read_state()
    frames_next = self.frame + 1
    ref = self.lib.frame(frames_next)
    ref_kin = self.kin_c(ref["pos"], ref["rot"], ref["q"], ref["v"], ref["w"], ref["qd"])
    out = self.task.analytics_fast(
      s_next, ref, ref_kin, self.ref_future(frames_next), action, self.prev_action, self.enc_bias
    )
    r, hard = out["r"], out["hard"]
    self.ep_len += 1
    clip_end = self.lib.remaining(frames_next) <= 0
    timeout = self.ep_len >= self.max_ep_len
    truncated = (clip_end | timeout) & ~hard
    done = hard | truncated
    # Return bookkeeping for reset sampling.
    disc = self.gamma ** (self.ep_len - 1).float()
    self.ep_disc_ret += disc * r
    self.ep_disc_max += disc * self.tcfg.max_step_reward
    self.ep_ret += r
    if self.cfg.reset_stat == "traversed":
      self.rew_hist[torch.arange(self.num_envs, device=self.device), (self.ep_len - 1).clamp(max=self.max_ep_len - 1)] = r
    info = {
      "reward": r,
      "terms": out["terms"],
      "terminated": hard,
      "truncated": truncated,
      "frame": self.frame.clone(),
      "ep_id": self.ep_id.clone(),
      "ep_len": self.ep_len.clone(),
    }
    self.frame = frames_next
    self.prev_action = action.clone()
    # Noisy observation of the new state.
    o = out["obs"] + self._noise(self.num_envs)
    self.obs_hist = torch.cat([self.obs_hist[:, 1:], o.unsqueeze(1)], dim=1)
    self.goal_buf = out["goal"]
    done_ids = done.nonzero(as_tuple=False).squeeze(-1)
    if len(done_ids):
      # A terminated episode scores 0 for the remaining frames it could have run.
      b = self.ep_bin[done_ids]
      if self.cfg.reset_stat == "traversed":
        self._update_bins_traversed(done_ids)
      else:
        ratio = self.ep_disc_ret[done_ids] / self._max_disc_return(done_ids)
        a = self.cfg.bin_ema
        G_old = self.bin_G[b]
        new = torch.where(self.bin_visits[b] > 0, (1 - a) * G_old + a * ratio, ratio)
        self.bin_G[b] = new.clamp(0, 1)
        self.bin_visits[b] += 1
      info["done_ep_return"] = self.ep_ret[done_ids].clone()
      info["done_ep_len"] = self.ep_len[done_ids].clone()
      info["done_terminated"] = hard[done_ids].clone()
      info["done_bin"] = b
      # Normalized return as in the paper: reward sum / largest attainable sum over the
      # length the episode could have run (to the clip end or the time limit).
      start = self.frame[done_ids] - self.ep_len[done_ids]
      potential = torch.minimum(self.lib.remaining(start), torch.full_like(start, self.max_ep_len)).float()
      info["done_norm_return"] = self.ep_ret[done_ids] / (self.tcfg.max_step_reward * potential.clamp_min(1))
      self.reset_envs(done_ids)
    self._finish_resets(done_ids)
    info["done"] = done
    return s_next, info

  def _update_bins_traversed(self, ids):
    """Updates G_b of every bin the finished episodes entered, with the discounted return-to-go
    from the entry step divided by the largest return attainable from there (to the clip end or
    the episode time limit). A terminated episode earns 0 after the fall."""
    k, T, g, dev = len(ids), self.max_ep_len, self.gamma, self.device
    L = self.ep_len[ids]
    start = self.frame[ids] - L
    t = torch.arange(T, device=dev)
    valid = t[None] < L[:, None]
    pw = g ** t.float()
    r = self.rew_hist[ids] * valid
    rtg = torch.flip(torch.cumsum(torch.flip(r * pw, [1]), 1), [1]) / pw
    last = self.lib.ends[self.lib.clip_of_frame[start]] - 1
    frames = torch.minimum(start[:, None] + t[None], last[:, None])
    rem = torch.minimum(self.lib.remaining(frames), (T - t)[None].expand(k, -1))
    potential = self.tcfg.max_step_reward * (1 - g ** rem.float()) / (1 - g)
    ratio = (rtg / potential.clamp_min(1e-6)).clamp(0, 1)
    b = self.lib.bin_of_frame[frames]
    entered = torch.cat([torch.ones(k, 1, dtype=torch.bool, device=dev), b[:, 1:] != b[:, :-1]], dim=1)
    sel = valid & entered & (rem >= self.cfg.rtg_min_steps)
    bi, ri = b[sel], ratio[sel]
    a = self.cfg.bin_ema
    self.bin_G[bi] = torch.where(self.bin_visits[bi] > 0, (1 - a) * self.bin_G[bi] + a * ri, ri)
    self.bin_visits[bi] += 1

  def _max_disc_return(self, ids):
    """Largest discounted return an episode from its start frame could have earned."""
    start = self.frame[ids] - self.ep_len[ids]
    horizon = torch.minimum(self.lib.remaining(start), torch.full_like(start, self.max_ep_len)).float()
    g = self.gamma
    return self.tcfg.max_step_reward * (1 - g**horizon) / (1 - g)


def _match(pattern, name):
  import re

  return name is not None and re.fullmatch(pattern, name.split("/")[-1]) is not None
