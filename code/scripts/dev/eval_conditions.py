"""Diagnostic: benchmark return of checkpoints under the benchmark protocol and under
training-like conditions (observation noise, action noise, domain randomization), on a fixed
random subset of clips, plus which termination condition ends the failed clips.

usage: eval_conditions.py motions.npz n_clips ckpt [ckpt ...]
"""

import json
import sys

import torch

from supermimic import networks
from supermimic.evaluate import evaluate, load_supermimic_policy, make_eval_env
from supermimic.motion_lib import MotionLibrary


def main():
  motions, n_clips, ckpts = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
  dev = "cuda"
  networks.AMP["enabled"] = False
  torch.backends.cuda.matmul.allow_tf32 = True
  lib = MotionLibrary(motions, dev)
  g = torch.Generator().manual_seed(1234)
  clips = torch.randperm(lib.num_clips, generator=g)[:n_clips].to(dev)
  env = make_eval_env(lib, min(4096, n_clips), dev)
  conds = {
    "benchmark": dict(obs_noise=False, act_noise=0.0, dr=False),
    "obs noise": dict(obs_noise=True, act_noise=0.0, dr=False),
    "action noise 0.1": dict(obs_noise=False, act_noise=0.1, dr=False),
    "domain rand.": dict(obs_noise=False, act_noise=0.0, dr=True),
    "all (training-like)": dict(obs_noise=True, act_noise=0.1, dr=True),
  }
  out = {}
  for ck in ckpts:
    pol, meta = load_supermimic_policy(ck, dev)
    for name, c in conds.items():
      env.eval_mode = not c["dr"]
      # The env adds observation noise only outside eval mode; set it independently of DR.
      zero = torch.zeros(1, device=dev)
      env._noise = (lambda k: env.task.sample_obs_noise((k,))) if c["obs_noise"] else (lambda k: zero.new_zeros(k, env.task.obs_noise.shape[0]))
      torch.manual_seed(0)
      an = c["act_noise"]
      act = (lambda o, gl: pol(o, gl) + an * torch.randn(o.shape[0], 29, device=dev)) if an else (lambda o, gl: pol(o, gl))
      res = evaluate(act, lib, env, clips)
      out[f"{ck} | {name}"] = (round(res["norm_return"], 4), res["failed"])
      print(f"{ck.split('/')[-1]} it {meta['it']:>6d} | {name:20s} | return {res['norm_return']:.4f} failed {res['failed']}/{n_clips}", flush=True)
  json.dump(out, open("/dev/stdout", "w"), indent=0)


if __name__ == "__main__":
  main()
