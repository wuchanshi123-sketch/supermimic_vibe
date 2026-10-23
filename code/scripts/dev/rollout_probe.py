"""Diagnostic: what goes wrong in deterministic benchmark rollouts of a checkpoint.

For a fixed random set of clips, plays each from frame 0 (benchmark protocol) with the
deterministic policy and with action noise 0.1, and reports per condition: return, failures,
which termination condition fired (height / tilt), when it fired, and the mean action change
per step |a_t - a_{t-1}|, joint speed and root height error over the first 10 s.

usage: rollout_probe.py motions.npz n_clips ckpt [ckpt ...]
"""

import sys

import torch

from supermimic import networks
from supermimic.evaluate import load_supermimic_policy, make_eval_env
from supermimic.motion_lib import MotionLibrary


@torch.no_grad()
def probe(pol, lib, env, clips, noise, steps=500):
  dev = env.device
  n = len(clips)
  env.reset()
  ids = torch.arange(n, device=dev)
  env.reset_envs(ids, frames=lib.starts[clips])
  env._finish_resets(ids, force=True)
  alive = torch.ones(n, dtype=torch.bool, device=dev)
  fail_t = torch.full((n,), -1, device=dev)
  why = torch.zeros(n, dtype=torch.long, device=dev)  # 1 height, 2 tilt, 3 both
  ret = torch.zeros(n, device=dev)
  da, qd, ez = [], [], []
  prev = None
  g = torch.Generator(device=dev).manual_seed(0)
  for t in range(steps):
    obs, goal = env.policy_inputs()
    a = pol(obs, goal)
    if noise:
      a = a + noise * torch.randn(a.shape, device=dev, generator=g)
    if prev is not None:
      da.append(((a - prev).abs().mean(-1) * alive).sum() / alive.sum().clamp_min(1))
    prev = a
    frame_ref = env.frame + 1
    s, info = env.step(a)
    ref = lib.frame(frame_ref)
    e_z = (s["pos"][:, 2] - ref["pos"][:, 2]).abs()
    e_t = (s["rot"][:, 2, 2] - ref["rot"][:, 2, 2]).abs()
    qd.append((s["qd"].abs().mean(-1) * alive).sum() / alive.sum().clamp_min(1))
    ez.append((e_z * alive).sum() / alive.sum().clamp_min(1))
    ret += info["reward"] * alive
    term = info["terminated"] & alive
    if term.any():
      i = term.nonzero().squeeze(-1)
      fail_t[i] = t
      why[i] = (e_z[i] > env.tcfg.term_z).long() + 2 * (e_t[i] > env.tcfg.term_tilt).long()
    alive &= ~info["done"]
    if not alive.any():
      break
  f = fail_t >= 0
  return {
    "failed": int(f.sum()),
    "fail by height / tilt / both": [int((why == k).sum()) for k in (1, 2, 3)],
    "median fail step": int(fail_t[f].median()) if f.any() else None,
    "mean |da| per step": float(torch.stack(da).mean()),
    "mean |qd|": float(torch.stack(qd).mean()),
    "mean root height error": float(torch.stack(ez).mean()),
  }


def main():
  motions, n_clips, ckpts = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
  dev = "cuda"
  networks.AMP["enabled"] = False
  lib = MotionLibrary(motions, dev)
  g = torch.Generator().manual_seed(4321)
  # Clips of at least 10 s, where the decline is largest.
  long = (lib.lengths >= 500).nonzero().squeeze(-1).cpu()
  clips = long[torch.randperm(len(long), generator=g)[:n_clips]].to(dev)
  env = make_eval_env(lib, n_clips, dev)
  for ck in ckpts:
    pol, meta = load_supermimic_policy(ck, dev)
    for noise in (0.0, 0.1):
      r = probe(pol, lib, env, clips, noise)
      print(f"{ck.split('/')[-2]}/{ck.split('/')[-1]} noise {noise}: {r}", flush=True)


if __name__ == "__main__":
  main()
