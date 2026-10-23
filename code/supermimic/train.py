"""SuperMimic training loop (Algorithm A.1).

Each iteration collects one simulator step from every environment. World-model and
policy updates run at `utd` updates per iteration (0.4 in the paper), so two updates
happen every five iterations. One update is `wm_batches` world-model gradient steps on
2048 k-step replay windows, then one BPTT policy step on 4096 imagined rollouts of
H steps, then critic fitting on the same rollouts.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time

import torch

from .motion_lib import MotionLibrary
from . import networks
from .networks import Critic, Policy, WorldModel, apply_delta_c, run_mlp, state_delta_c, state_error, state_error_c
from .replay import ReplayBuffer
from .sim_env import EnvCfg, TrackingSimEnv
from .task import GOAL_FRAME_DIM, NJ, OBS_DIM, XI_DIM, TaskCfg, flat_to_state, state_to_flat


@dataclasses.dataclass
class TrainCfg:
  motions: str = "data/processed/lafan1_50hz.npz"
  clip_prefixes: str = ""  # e.g. "walk,run,sprint" for the Fig. 4b subset
  out: str = "runs/debug"
  seed: int = 0
  num_envs: int = 4096
  buffer_steps: int = 1000
  iterations: int = 320_000
  utd: float = 0.4
  warmup_steps: int = 64
  # World model.
  k: int = 8
  wm_batch: int = 2048
  wm_batches: int = 10
  # Policy / critic.
  horizon: int = 32
  pi_batch: int = 4096
  gamma: float = 0.99
  lam: float = 0.95
  lr: float = 2e-4
  lr_final_frac: float = 0.03
  lr_decay_iters: int = 250_000
  critic_epochs: int = 5
  critic_minibatches: int = 4
  action_noise: float = 0.1
  # Not in the paper (0 = paper): share of environments that collect without action noise,
  # so the replay buffer also holds states of the deterministic closed loop.
  det_frac: float = 0.0
  # Reset-sampler statistic, see sim_env.EnvCfg.reset_stat ("start" was used for all runs
  # before 2026-10-23; "traversed" follows the paper's wording).
  reset_stat: str = "start"
  use_critic: bool = True
  # Bookkeeping.
  log_every: int = 100
  ckpt_every: int = 4000
  eval_every: int = 4000
  eval_clips: int = 0  # 0 = all clips
  eval_envs: int = 4096
  wall_hours: float = 1e9
  tf32: bool = True
  amp: bool = True
  compile_wm: bool = True
  lr_sqrt_world: bool = True
  norm_update_every: int = 5
  resume: str = ""


class _Timer:
  def __init__(self, on):
    self.on = on
    self.t = {}
    if on:
      torch.cuda.synchronize()
    self.last = time.time()

  def __call__(self, name):
    if not self.on:
      return
    torch.cuda.synchronize()
    now = time.time()
    self.t[name] = self.t.get(name, 0.0) + now - self.last
    self.last = now


def lr_at(cfg: TrainCfg, it: int) -> float:
  if it >= cfg.lr_decay_iters:
    return cfg.lr * cfg.lr_final_frac
  c = 0.5 * (1 + math.cos(math.pi * it / cfg.lr_decay_iters))
  return cfg.lr * (cfg.lr_final_frac + (1 - cfg.lr_final_frac) * c)


class Trainer:
  def __init__(self, cfg: TrainCfg, device="cuda"):
    self.cfg = cfg
    self.world = int(os.environ.get("WORLD_SIZE", "1"))
    self.rank = int(os.environ.get("RANK", "0"))
    if self.world > 1:
      import datetime

      import torch.distributed as dist

      local = int(os.environ.get("LOCAL_RANK", "0"))
      torch.cuda.set_device(local)
      device = f"cuda:{local}"
      dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
      networks.DIST["world"] = self.world
      if cfg.lr_sqrt_world:
        # Paper: 2 GPUs use sqrt(2) x 2e-4.
        cfg.lr = cfg.lr * math.sqrt(self.world)
    self.device = device
    torch.manual_seed(cfg.seed)
    if cfg.tf32:
      torch.backends.cuda.matmul.allow_tf32 = True
      torch.backends.cudnn.allow_tf32 = True
    networks.AMP["enabled"] = cfg.amp
    self.lib = MotionLibrary(cfg.motions, device, prefixes=cfg.clip_prefixes)
    self.tcfg = TaskCfg()
    self.env = self._make_env()
    self.task = self.env.task
    self.task.compile_sizes |= {cfg.wm_batch, cfg.pi_batch}
    s = self.env.reset()
    self.feat_dim = self.task.state_features(s).shape[-1]
    self.obs_dim = OBS_DIM * self.tcfg.history
    self.goal_dim = GOAL_FRAME_DIM * self.tcfg.future
    self.wm = WorldModel(self.task, self.feat_dim, XI_DIM).to(device)
    self.policy = Policy(self.obs_dim, self.goal_dim).to(device)
    self.critic_in = self.feat_dim + self.goal_dim + XI_DIM + 1
    self.critic = Critic(self.critic_in).to(device)
    self.opt_wm = torch.optim.Adam(self.wm.net.parameters(), lr=cfg.lr, fused=True)
    self.opt_pi = torch.optim.Adam(
      list(self.policy.encoder.parameters()) + list(self.policy.decoder.parameters()), lr=cfg.lr, betas=(0.9, 0.95), fused=True
    )
    self.opt_v = torch.optim.Adam(self.critic.net.parameters(), lr=cfg.lr, fused=True)
    if self.world > 1:
      import torch.distributed as dist

      for mod in (self.wm, self.policy, self.critic):
        for t in list(mod.parameters()) + list(mod.buffers()):
          dist.broadcast(t.data, 0)
      torch.manual_seed(cfg.seed + 1000 * self.rank)
    self.buf = self._make_buffer()
    self._wm_loss = torch.compile(self.wm_window_loss, dynamic=False) if cfg.compile_wm else self.wm_window_loss
    self.it = 0
    self.updates = 0
    self.env_steps = 0
    self.train_time = 0.0
    self.update_credit = 0.0
    self.ep_returns = []
    self.ep_norm_returns = []
    self.ep_terms = []
    self.ep_lens = []
    self.ep_step_rewards = []
    self.profile = bool(int(os.environ.get("SMIMIC_PROFILE", "0")))
    self.prof = {}
    os.makedirs(cfg.out, exist_ok=True)
    self.log_f = open(os.path.join(cfg.out, "log.jsonl" if self.rank == 0 else f"log_rank{self.rank}.jsonl"), "a")
    if cfg.resume:
      self.load(cfg.resume)

  def _make_env(self):
    return TrackingSimEnv(self.lib, EnvCfg(num_envs=self.cfg.num_envs, reset_stat=self.cfg.reset_stat), self.tcfg, self.device, gamma=self.cfg.gamma)

  def _make_buffer(self):
    return ReplayBuffer(self.cfg.num_envs, self.cfg.buffer_steps, XI_DIM, self.device)

  def _extra_state(self):
    return {}

  def _load_extra(self, ck):
    pass

  def sync_grads(self, params):
    """Average gradients across data-parallel ranks (no-op on one GPU)."""
    if self.world == 1:
      return
    import torch.distributed as dist

    grads = [p.grad for p in params if p.grad is not None]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat)
    flat /= self.world
    i = 0
    for g in grads:
      n = g.numel()
      g.copy_(flat[i : i + n].view_as(g))
      i += n

  # Checkpoints.

  def save(self, path):
    torch.save(
      {
        "cfg": dataclasses.asdict(self.cfg),
        "it": self.it,
        "updates": self.updates,
        "env_steps": self.env_steps,
        "train_time": self.train_time,
        "wm": self.wm.state_dict(),
        "policy": self.policy.state_dict(),
        "critic": self.critic.state_dict(),
        "opt_wm": self.opt_wm.state_dict(),
        "opt_pi": self.opt_pi.state_dict(),
        "opt_v": self.opt_v.state_dict(),
        "bin_G": self.env.bin_G,
        "bin_visits": self.env.bin_visits,
        "bin_past_w": self.env.bin_past_w,
        **self._extra_state(),
      },
      path + ".tmp",
    )
    os.replace(path + ".tmp", path)

  def load(self, path):
    ck = torch.load(path, map_location=self.device)
    self.wm.load_state_dict(ck["wm"])
    self.policy.load_state_dict(ck["policy"])
    self.critic.load_state_dict(ck["critic"])
    self.opt_wm.load_state_dict(ck["opt_wm"])
    self.opt_pi.load_state_dict(ck["opt_pi"])
    self.opt_v.load_state_dict(ck["opt_v"])
    self.env.bin_G[:] = ck["bin_G"]
    self.env.bin_visits[:] = ck["bin_visits"]
    self.env.bin_past_w[:] = ck["bin_past_w"]
    self.it, self.updates = ck["it"], ck["updates"]
    self._load_extra(ck)
    self.env_steps, self.train_time = ck["env_steps"], ck["train_time"]
    print(f"resumed from {path} at iteration {self.it}")

  # Data collection.

  @torch.no_grad()
  def collect(self, random=False):
    env = self.env
    obs, goal = env.policy_inputs()
    self.policy.obs_norm.update(obs)
    self.policy.goal_norm.update(goal)
    a = self.policy(obs, goal)
    noise = self.cfg.action_noise * torch.randn_like(a)
    if self.cfg.det_frac > 0:
      noise[: int(round(self.cfg.det_frac * a.shape[0]))] = 0.0
    a = a + noise
    s_flat = state_to_flat(env.state)
    xi, frame, ep = env.xi.clone(), env.frame.clone(), env.ep_id.clone()
    s_next, info = env.step(a)
    self.buf.add(s_flat, a, state_to_flat(s_next), xi, frame, ep)
    self.env_steps += env.num_envs
    if "done_ep_return" in info:
      self.ep_returns.append(info["done_ep_return"].mean().item())
      L = info["done_ep_len"].float()
      self.ep_norm_returns.append(info["done_norm_return"].mean().item())
      self.ep_step_rewards.append((info["done_ep_return"] / (L * self.tcfg.max_step_reward)).mean().item())
      self.ep_terms.append(info["done_terminated"].float().mean().item())
      self.ep_lens.append(L.mean().item())

  # World model.

  def update_wm(self):
    cfg, wm, task = self.cfg, self.wm, self.task
    stats = {"wm_loss": 0.0, "wm_err1": 0.0, "wm_errk": 0.0}
    for bi in range(cfg.wm_batches):
      s0, A, S1, xi, mask = self.buf.sample_windows(cfg.wm_batch, cfg.k)
      s = flat_to_state(s0)
      m = mask.float()
      if bi == 0 and self.updates % cfg.norm_update_every == 0:
        # Normalizer statistics from the true transitions of valid windows.
        with torch.no_grad():
          true_prev = s
          xs, ds = [], []
          for j in range(cfg.k):
            t1 = flat_to_state(S1[:, j])
            xs.append(wm.inputs(true_prev, A[:, j], xi)[mask])
            ds.append(state_delta_c(true_prev, t1)[mask])
            true_prev = t1
          wm.in_norm.update(torch.cat(xs))
          wm.out_norm.update(torch.cat(ds))
      loss, err1, errk = self._wm_loss(s0, A, S1, xi, m)
      self.opt_wm.zero_grad(set_to_none=True)
      loss.backward()
      self.sync_grads(list(wm.net.parameters()))
      torch.nn.utils.clip_grad_norm_(wm.net.parameters(), 1.0)
      self.opt_wm.step()
      stats["wm_loss"] += loss.detach() / cfg.wm_batches
      stats["wm_err1"] += err1 / cfg.wm_batches
      stats["wm_errk"] += errk / cfg.wm_batches
    stats = {kk: float(vv) for kk, vv in stats.items()}
    return stats

  def wm_window_loss(self, s0, A, S1, xi, m):
    """k-step autoregressive loss on logged windows (Eq. 1, state term)."""
    wm = self.wm
    std = wm.out_norm.std
    s = flat_to_state(s0)
    loss = 0.0
    errs = []
    for j in range(self.cfg.k):
      s = wm.step(s, A[:, j], xi)
      e = (state_error(s, flat_to_state(S1[:, j])) / std).pow(2).mean(-1)
      loss = loss + e
      errs.append(e.detach())
    msum = m.sum().clamp_min(1)
    loss = (loss * m).sum() / msum
    return loss, (errs[0] * m).sum() / msum, (errs[-1] * m).sum() / msum

  # Policy and critic.

  def critic_input(self, feats, goal, xi, rem):
    return torch.cat([feats, goal, xi, rem[:, None]], dim=-1)

  def update_policy(self):
    cfg, task, lib, wm = self.cfg, self.task, self.lib, self.wm
    B, H, K = cfg.pi_batch, cfg.horizon, self.tcfg.future
    dev = self.device
    s0, xi, frame, obs_hist, a_prev = self.buf.sample_starts(B, self.tcfg.history, task)
    bias = xi[:, 1 : 1 + NJ]
    with torch.no_grad():
      steps = torch.arange(0, H + K + 1, device=dev)
      idx_raw = frame[:, None] + steps
      idx = lib.clamp_to_clip(idx_raw, frame)
      valid = (idx_raw == idx)[:, : H + 1].float()  # frame t still inside the clip
      ref = lib.frame(idx)
      ref_kin = self.env.kin_c(ref["pos"], ref["rot"], ref["q"], ref["v"], ref["w"], ref["qd"])
      rem_all = lib.remaining(idx[:, : H + 1]).float().clamp(max=100) / 100.0

    def ref_at(t):
      return {k: v[:, t] for k, v in ref.items()}

    def ref_future(t):
      return {k: v[:, t + 1 : t + 1 + K] for k, v in ref.items()}

    def ref_kin_at(t):
      return tuple(x[:, t] for x in ref_kin)

    for p in wm.net.parameters():
      p.requires_grad_(False)
    s = flat_to_state(s0)
    hist = obs_hist + task.sample_obs_noise((B, self.tcfg.history))
    an = task.analytics_fast(s, ref_at(0), ref_kin_at(0), ref_future(0), a_prev, a_prev, bias)
    c = torch.ones(B, device=dev)
    J = torch.zeros(B, device=dev)
    rec = {"x": [], "r": [], "d": [], "m": [], "c": []}
    rew_terms = {}
    std, mean = wm.out_norm.std, wm.out_norm.mean
    for t in range(H):
      goal, feats = an["goal"], an["feats"]
      rec["x"].append(self.critic_input(feats, goal, xi, rem_all[:, t]).detach())
      a = self.policy(hist.flatten(1), goal)
      dlt = run_mlp(wm.net, wm.in_norm(torch.cat([feats, a, xi], dim=-1))) * std + mean
      s1 = apply_delta_c(s, dlt)
      an = task.analytics_fast(s1, ref_at(t + 1), ref_kin_at(t + 1), ref_future(t + 1), a, a_prev, bias)
      r, d = an["r"], an["soft"]
      m = valid[:, t + 1]
      J = J + (cfg.gamma**t) * c * r * m
      rec["r"].append((r * m).detach())
      rec["d"].append(d.detach())
      rec["m"].append(m)
      rec["c"].append(c.detach())
      for kk, vv in an["terms"].items():
        rew_terms[kk] = rew_terms.get(kk, 0.0) + vv.detach().mean() / H
      c = c * (1 - d)
      o1 = an["obs"] + task.sample_obs_noise((B,))
      hist = torch.cat([hist[:, 1:], o1.unsqueeze(1)], dim=1)
      s, a_prev = s1, a
    xH = self.critic_input(an["feats"], an["goal"], xi, rem_all[:, H])
    if cfg.use_critic:
      vH = self.critic(xH) * valid[:, H]
      J = J + (cfg.gamma**H) * c * vH
    loss_pi = -J.mean() / H
    self.opt_pi.zero_grad(set_to_none=True)
    loss_pi.backward()
    params = [p for g in self.opt_pi.param_groups for p in g["params"]]
    self.sync_grads(params)
    gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    torch.nn.utils.clip_grad_value_(params, 0.1)
    self.opt_pi.step()
    for p in wm.net.parameters():
      p.requires_grad_(True)
    self.critic.zero_grad(set_to_none=True)
    stats = {"pi_J": J.mean().item(), "pi_loss": loss_pi.item(), "pi_gnorm": gnorm.item(), "img_survival": c.mean().item()}
    stats.update({f"img_r_{k}": v.item() for k, v in rew_terms.items()})
    if cfg.use_critic:
      stats.update(self.update_critic(rec, xH.detach(), valid))
    return stats

  def update_critic(self, rec, xH, valid):
    cfg = self.cfg
    H = cfg.horizon
    X = torch.stack(rec["x"] + [xH], dim=1)  # (B, H+1, D)
    self.critic.norm.update(X)
    with torch.no_grad():
      V = self.critic(X) * valid
      R = torch.zeros_like(V)
      R[:, H] = V[:, H]
      for t in range(H - 1, -1, -1):
        r, d, m = rec["r"][t], rec["d"][t], rec["m"][t]
        R[:, t] = r + cfg.gamma * (1 - d) * m * ((1 - cfg.lam) * V[:, t + 1] + cfg.lam * R[:, t + 1])
      w = torch.stack(rec["c"], dim=1) * valid[:, :H]
    Xf = X[:, :H].reshape(-1, X.shape[-1])
    Rf = R[:, :H].reshape(-1)
    wf = w.reshape(-1)
    n = Xf.shape[0]
    mb = n // cfg.critic_minibatches
    tot = 0.0
    for _ in range(cfg.critic_epochs):
      perm = torch.randperm(n, device=self.device)
      for i in range(cfg.critic_minibatches):
        j = perm[i * mb : (i + 1) * mb]
        err = (self.critic(Xf[j]) - Rf[j]) ** 2
        loss = (err * wf[j]).sum() / wf[j].sum().clamp_min(1)
        self.opt_v.zero_grad(set_to_none=True)
        loss.backward()
        self.sync_grads(list(self.critic.net.parameters()))
        torch.nn.utils.clip_grad_norm_(self.critic.net.parameters(), 1.0)
        self.opt_v.step()
        tot += loss.item()
    return {"v_loss": tot / (cfg.critic_epochs * cfg.critic_minibatches), "v_mean": V[:, 0].mean().item()}

  # Main loop.

  def run(self):
    cfg = self.cfg
    if self.it == 0:
      for _ in range(cfg.warmup_steps):
        self.collect()
    t_last = time.time()
    agg = {}
    n_agg = 0
    while self.it < cfg.iterations and not self._should_stop():
      t0 = time.time()
      lr = lr_at(cfg, self.it)
      for opt in (self.opt_wm, self.opt_pi, self.opt_v):
        for g in opt.param_groups:
          g["lr"] = lr
      tp = _Timer(self.profile)
      self.collect()
      tp("collect")
      self.update_credit += cfg.utd
      while self.update_credit >= 1.0:
        self.update_credit -= 1.0
        st = self.update_wm()
        tp("wm")
        st.update(self.update_policy())
        tp("policy+critic")
        self.updates += 1
        for kk, vv in st.items():
          agg[kk] = agg.get(kk, 0.0) + vv
        n_agg += 1
      torch.cuda.synchronize()
      self.train_time += time.time() - t0
      self.it += 1
      if self.profile:
        for kk, vv in tp.t.items():
          self.prof[kk] = self.prof.get(kk, 0.0) + vv
        if self.it % cfg.log_every == 0:
          print("profile (s per iter):", {kk: round(vv / cfg.log_every, 4) for kk, vv in self.prof.items()})
          self.prof = {}
      if self.it % cfg.log_every == 0 and self.rank != 0:
        agg, n_agg = {}, 0
        self.ep_returns, self.ep_norm_returns, self.ep_terms, self.ep_lens, self.ep_step_rewards = [], [], [], [], []
      if self.it % cfg.log_every == 0 and self.rank == 0:
        rec = {kk: vv / max(n_agg, 1) for kk, vv in agg.items()}
        rec.setdefault("imagined_steps", self.updates * cfg.pi_batch * cfg.horizon)
        rec.update(
          it=self.it,
          updates=self.updates,
          env_steps=self.env_steps * self.world,
          world=self.world,
          hours=self.train_time / 3600,
          lr=lr,
          iter_s=(time.time() - t_last) / cfg.log_every,
          ep_return=sum(self.ep_returns) / max(len(self.ep_returns), 1),
          ep_norm_return=sum(self.ep_norm_returns) / max(len(self.ep_norm_returns), 1),
          ep_term_frac=sum(self.ep_terms) / max(len(self.ep_terms), 1),
          ep_len=sum(self.ep_lens) / max(len(self.ep_lens), 1),
          ep_step_reward=sum(self.ep_step_rewards) / max(len(self.ep_step_rewards), 1),
          bin_G_mean=self.env.bin_G[self.env.bin_visits > 0].mean().item() if (self.env.bin_visits > 0).any() else 0.0,
          bins_visited=(self.env.bin_visits > 0).float().mean().item(),
          mem_gb=torch.cuda.max_memory_allocated() / 1e9,
        )
        self.log_f.write(json.dumps(rec) + "\n")
        self.log_f.flush()
        print(
          f"it {self.it} upd {self.updates} h {rec['hours']:.3f} iter {rec['iter_s']:.3f}s "
          f"ret {rec['ep_norm_return']:.3f} len {rec['ep_len']:.1f} term {rec['ep_term_frac']:.3f} "
          f"wm1 {rec.get('wm_err1', 0):.4f} wm8 {rec.get('wm_errk', 0):.4f} J {rec.get('pi_J', 0):.3f} "
          f"surv {rec.get('img_survival', 0):.3f} v {rec.get('v_loss', 0):.4f} g {rec.get('pi_gnorm', 0):.3f}",
          flush=True,
        )
        agg, n_agg = {}, 0
        self.ep_returns, self.ep_norm_returns, self.ep_terms, self.ep_lens, self.ep_step_rewards = [], [], [], [], []
        t_last = time.time()
      if self.it % cfg.ckpt_every == 0 and self.rank == 0:
        self.save(os.path.join(cfg.out, f"ckpt_{self.it:07d}.pt"))
        self.save(os.path.join(cfg.out, "last.pt"))
      if cfg.eval_every and self.it % cfg.eval_every == 0:
        if self.rank == 0:
          self.run_eval()
        self._barrier()
    if self.rank == 0:
      self.save(os.path.join(cfg.out, "last.pt"))
      if cfg.eval_every and self.it % cfg.eval_every != 0:
        self.run_eval()
      open(os.path.join(cfg.out, "DONE"), "w").close()
    self._barrier()

  def _barrier(self):
    if self.world > 1:
      import torch.distributed as dist

      dist.barrier()

  def _should_stop(self):
    stop = self.train_time / 3600 >= self.cfg.wall_hours
    if self.world > 1:
      import torch.distributed as dist

      t = torch.tensor([1.0 if stop else 0.0], device=self.device)
      dist.broadcast(t, 0)
      stop = bool(t.item() > 0)
    return stop

  def run_eval(self):
    """Benchmark protocol (not counted in train_time)."""
    from .evaluate import evaluate, make_eval_env

    cfg = self.cfg
    if not hasattr(self, "eval_env"):
      g = torch.Generator(device="cpu").manual_seed(1234)
      if cfg.eval_clips and cfg.eval_clips < self.lib.num_clips:
        self.eval_clip_ids = torch.randperm(self.lib.num_clips, generator=g)[: cfg.eval_clips].to(self.device)
      else:
        self.eval_clip_ids = torch.arange(self.lib.num_clips, device=self.device)
      n = min(cfg.eval_envs, len(self.eval_clip_ids))
      self.eval_env = make_eval_env(self.lib, n, self.device)
    t0 = time.time()
    self.policy.eval()
    res = evaluate(lambda o, g: self.policy(o, g), self.lib, self.eval_env, self.eval_clip_ids)
    self.policy.train()
    rec = {
      "it": self.it,
      "updates": self.updates,
      "hours": self.train_time / 3600,
      "env_steps": self.env_steps,
      **{k: v for k, v in res.items() if not k.startswith("per_clip")},
    }
    with open(os.path.join(cfg.out, "eval.jsonl"), "a") as f:
      f.write(json.dumps({**rec, "per_clip_norm_return": res["per_clip_norm_return"], "per_clip_failed": res["per_clip_failed"]}) + "\n")
    print(f"EVAL it {self.it} h {rec['hours']:.3f} norm_return {rec['norm_return']:.4f} failed {rec['failed']}/{rec['num_clips']} ({time.time() - t0:.0f}s)", flush=True)


def parse_args() -> TrainCfg:
  ap = argparse.ArgumentParser()
  for f in dataclasses.fields(TrainCfg):
    t = f.type if not isinstance(f.type, str) else eval(f.type)
    if t is bool:
      ap.add_argument(f"--{f.name.replace('_', '-')}", type=lambda x: x.lower() in ("1", "true", "yes"), default=f.default)
    else:
      ap.add_argument(f"--{f.name.replace('_', '-')}", type=t, default=f.default)
  return TrainCfg(**vars(ap.parse_args()))


if __name__ == "__main__":
  cfg = parse_args()
  print(cfg)
  Trainer(cfg).run()
