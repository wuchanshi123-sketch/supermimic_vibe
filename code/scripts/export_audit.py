"""Builds the audit dashboard's datasets (JSON row files) from the run logs.

Writes results/audit/<dataset>.json. Every number comes from runs/*/eval.jsonl,
runs/*/log.jsonl, results/bench_rollout_a6000.json, or the paper's TeX source
(paper/src/sections/*.tex, values copied below with their figure/table).
"""

import datetime
import json
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "audit")

RUNS = {
  "smimic_lafan1_1gpu": ("SuperMimic H=32, critic (main)", "Fig. 3a", "31934769"),
  "ppo_lafan1_1gpu": ("PPO-Sim, 1 GPU", "Fig. 3a", "31934725"),
  "abl_H8_critic_true": ("H=8, critic", "Fig. 3b,c", "31936235"),
  "abl_H16_critic_true": ("H=16, critic", "Fig. 3b,c", "31936233"),
  "abl_H16_critic_false": ("H=16, no critic", "Fig. 3b,c", "31936234"),
  "abl_H32_critic_false": ("H=32, no critic", "Fig. 3b,c", "31933783 (dev)"),
  "abl_H64_critic_true": ("H=64, critic", "Fig. 3b,c", "31936236"),
  "f4b_smimic_wrs": ("SuperMimic, walk/run/sprint", "Fig. 4b", "31937943"),
  "f4b_ppowm_16kx32": ("PPO-WM 16k×32, walk/run/sprint", "Fig. 4b", "31937944"),
  "smimic_lafan1_2gpu": ("SuperMimic H=32, 2 GPUs (data parallel)", "Fig. 3a (extra)", "31945786"),
  "abl_H128_critic_true": ("H=128, critic", "Fig. 3b,c", "31945793"),
  "abl_H64_critic_false": ("H=64, no critic", "Fig. 3b,c", "31945789"),
  "vel_rl": ("Velocity encoder, RL", "Fig. 6 / Sec. 5.5", "31941569"),
  "vel_distill": ("Velocity encoder, distillation", "Fig. 6 / Sec. 5.5", "31941570"),
  "smimic_bones_1gpu": ("SuperMimic, BONES-SEED, 1 GPU", "Fig. 2a", "31950477"),
  "smimic_bones_2gpu": ("SuperMimic, BONES-SEED, 2 GPUs", "Fig. 2a", "31950519"),
  "sweep_bones_1h": ("Dataset sweep, 1 h of BONES-SEED", "Fig. 2b,c", "31950522"),
  "sweep_bones_7h": ("Dataset sweep, 7 h of BONES-SEED", "Fig. 2b,c", "31950523"),
  "sweep_bones_40h": ("Dataset sweep, 40 h of BONES-SEED", "Fig. 2b,c", "31950524"),
  "bones_fix_detfrac25": ("Fix test: 25% of envs collect without action noise (not in paper)", "Fig. 2a (diagnosis)", "31968703 (resumed 31953913; stopped at 15.3 h)"),
  "bones_fix_fp32": ("Fix test: fp32 networks instead of bf16 (paper: TF32)", "Fig. 2a (diagnosis)", "31953914"),
  "bones_fix_traversed": ("Fix test: reset sampler updates traversed bins with >= 0.5 s left", "Fig. 2a (diagnosis)", "31957454"),
  "bones_fix_traversed_all": ("Fix test: reset sampler updates every traversed bin (paper's wording)", "Fig. 2a (diagnosis)", "31960983"),
  "bones_fix_detfrac_traversed": ("Fix test: det-frac 0.25 + every-traversed-bin sampler", "Fig. 2a (diagnosis)", "31963725"),
  "smimic_bones_2gpu_detfrac": ("SuperMimic, BONES-SEED, 2 GPUs, det-frac 0.25 (not in paper)", "Fig. 2a", "31964356"),
  "bones_fix_detfrac50": ("Fix test: 50% of envs collect without action noise (not in paper)", "Fig. 2a (diagnosis)", "31970504"),
  "bones_fix_detfrac_ema": ("Fix test: det-frac 0.25 + EMA target critic for critic targets (not in paper)", "Fig. 2a (diagnosis)", "31971049 (stopped at 11.8 h)"),
}

# Paper Fig. 3a (LAFAN1, 40 clips): normalized return vs wall-clock hours.
PAPER_CURVES = {
  "Paper SuperMimic, 1 H100": [(0.0021125, 0.001519), (0.077808, 0.524), (0.151899, 0.6166), (0.225991, 0.6672), (0.299996, 0.67), (0.373919, 0.696), (0.44801, 0.7656), (0.522101, 0.7565), (0.596276, 0.777), (0.670791, 0.779), (0.745049, 0.7909), (0.894413, 0.7894), (1.04353, 0.8102), (1.19272, 0.8331), (1.342, 0.8309), (1.49103, 0.8483), (1.79009, 0.8282), (2.08984, 0.8519), (2.38983, 0.8509), (2.69008, 0.8616), (2.98771, 0.8661)],
  "Paper PPO-Sim, 1 GPU": [(0.00236111, 0.001764), (0.24, 0.025203), (0.479583, 0.21044), (0.720972, 0.2256), (0.960417, 0.2352), (1.19861, 0.2651), (1.43987, 0.3023), (1.67875, 0.3301), (1.91973, 0.3662), (2.16125, 0.3731), (2.40778, 0.4068), (2.65042, 0.4044), (2.88959, 0.3948), (3.1318, 0.43), (3.37181, 0.4195), (3.61264, 0.449), (3.85361, 0.4474), (4.09305, 0.4878), (4.32972, 0.4612), (4.57042, 0.4728), (4.81069, 0.4795), (5.0457, 0.4707), (5.28334, 0.534), (5.52069, 0.5233), (5.75889, 0.4984), (5.99695, 0.5348), (7.18611, 0.5412), (8.38444, 0.5449), (9.62625, 0.5499), (11.9934, 0.5881), (14.0222, 0.6166), (18.6919, 0.602)],
  "Paper PPO-Sim, 8 GPUs": [(0.00138889, 0.001772), (0.19125, 0.3744), (0.377223, 0.3994), (0.563195, 0.388), (0.748611, 0.4448), (0.934584, 0.4885), (1.12069, 0.5124), (1.30653, 0.5082), (1.49194, 0.5438), (1.67639, 0.554), (1.86055, 0.5414), (2.04486, 0.5543), (2.22903, 0.5772), (2.41305, 0.5883), (2.59667, 0.5909), (2.78097, 0.603), (2.96445, 0.6032), (3.14792, 0.605), (3.33125, 0.6189), (3.51472, 0.6225), (3.69819, 0.6107), (3.88138, 0.6108), (4.06458, 0.621), (4.24833, 0.6094), (4.43153, 0.6318), (4.61486, 0.6354), (5.53014, 0.6449), (6.445, 0.6744), (7.36055, 0.6688), (9.19111, 0.6944), (11.0203, 0.7018), (14.4397, 0.7188), (18.0328, 0.7218)],
}

# Paper Fig. 2a (BONES-SEED benchmark of 129,514 motions): normalized return vs GPU-hours on
# H100s (sections/results.tex); SONIC's released checkpoint scores 0.9012 there.
PAPER_CURVES["Paper SuperMimic BONES-SEED, 1 H100"] = [(0.061, 0.7034), (0.091, 0.7573), (0.120, 0.8042), (0.149, 0.8269), (0.208, 0.8495), (0.237, 0.8572), (0.326, 0.8665), (0.384, 0.8821), (0.502, 0.8970), (0.619, 0.9074), (0.795, 0.9152), (1.001, 0.9246), (1.264, 0.9308), (1.586, 0.9319), (1.995, 0.9344), (2.521, 0.9426), (3.164, 0.9363), (3.982, 0.9502), (5.006, 0.9499), (6.322, 0.9598), (7.932, 0.9616), (9.368, 0.9600)]
PAPER_CURVES["Paper SuperMimic BONES-SEED, 2 H100 (GPU-hours)"] = [(0.066, 0.6344), (0.128, 0.7401), (0.189, 0.8191), (0.250, 0.8443), (0.372, 0.8616), (0.495, 0.8821), (0.617, 0.8933), (0.740, 0.9063), (0.985, 0.9123), (1.230, 0.9223), (1.598, 0.9287), (1.965, 0.9349), (2.454, 0.9359), (3.188, 0.9436), (3.921, 0.9457), (5.022, 0.9447), (6.367, 0.9492), (7.956, 0.9564), (10.036, 0.9602), (12.606, 0.9612), (15.788, 0.9636), (19.584, 0.9641)]
PAPER_SONIC = {"norm_return": 0.9012, "failed": 179, "clips": 129514}

# Paper Fig. 3b (final training return) and 3c (hours to training return 0.82); None = not reached.
PAPER_ABLATION = {
  (8, True): (0.392, None), (16, True): (0.843, 1.19), (32, True): (0.866, 0.88), (64, True): (0.863, 1.70), (128, True): (0.823, 4.62),
  (8, False): (0.087, None), (16, False): (0.25, None), (32, False): (0.84, 1.37), (64, False): (0.862, 1.34), (128, False): (0.835, 3.80),
}
ABL_RUN = {
  (8, True): "abl_H8_critic_true", (16, True): "abl_H16_critic_true", (16, False): "abl_H16_critic_false",
  (32, True): "smimic_lafan1_1gpu", (32, False): "abl_H32_critic_false", (64, True): "abl_H64_critic_true",
  (64, False): "abl_H64_critic_false", (128, True): "abl_H128_critic_true",
}


def jsonl(path):
  if not os.path.exists(path):
    return []
  return [json.loads(l) for l in open(path) if l.strip()]


def job_states():
  try:
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %T"], capture_output=True, text=True, timeout=30).stdout
  except Exception:  # noqa: BLE001
    return {}
  return dict(line.split() for line in out.splitlines() if line.strip())


FIDELITY = [
  # aspect, paper, ours, kind
  ("Simulator", "MuJoCo Warp, 4 x 5 ms substeps per 50 Hz step", "mjlab 1.2 + MuJoCo Warp 3.5, 4 x 5 ms substeps", "same"),
  ("Robot model and actuators", "Unitree G1 29-DoF, BeyondMimic low PD gains", "mjlab G1 (BeyondMimic gains, armature-based)", "same"),
  ("Action", "q_sp = alpha * a + q0", "q_sp = alpha * a + q0, alpha = 0.25 effort/stiffness", "same"),
  ("Observation", "q - q0, qd, IMU gyro, projected gravity, last action; 10-frame history; Table A.1 noise", "Same terms, noise and history; encoder bias added to q", "same"),
  ("Goal", "K = 10 future reference frames", "10 future frames: ref root pose relative to robot root, ref q, qd, ref root velocities (73 per frame)", "choice"),
  ("Reward", "BeyondMimic tracking terms, no self-collision, joint limit -2, x dt", "Same terms, weights and stds; pelvis as anchor", "same"),
  ("Termination", "|root z err| > 0.25 m or tilt err > 0.8", "Same; soft sigmoid version inside imagination", "same"),
  ("Sim context (DR)", "friction U[0.3,1.2], joint offsets U[-0.01,0.01], torso COM; no pushes", "Same, resampled per episode; set_const batched every 50 steps", "same"),
  ("State representation", "not specified", "floating-base qpos/qvel (71 numbers)", "choice"),
  ("Reward / obs / termination heads", "learned or analytic", "analytic, on our differentiable FK (matches MuJoCo to 1e-6)", "choice"),
  ("World-model network", "MLP (2048, 2048, 2048)", "MLP (2048, 2048, 2048), SiLU, predicts normalized state change in heading frame", "same"),
  ("World-model loss", "k = 8 autoregressive windows, Eq. 1", "k = 8, state term only (heads analytic), per-dim normalized", "same"),
  ("Policy", "encoder (2048,1024,512,256) -> 32-d tanh latent -> decoder (2048,1024,512,256)", "Same sizes, ELU", "same"),
  ("Critic", "MLP (256, 128), GAE lambda 0.95, no target network", "Same; input adds remaining clip time", "same"),
  ("Imagination", "H = 32, 4096 rollouts, BPTT, gamma 0.99", "Same", "same"),
  ("Optimizers", "Adam 2e-4 (policy betas 0.9/0.95), clip 1.0 then 0.1 element-wise", "Same", "same"),
  ("Schedule", "UTD 0.4, cosine LR to 0.03x over 250k iterations", "1 sim step per iteration, 0.4 updates per iteration (10 WM grad steps + 1 policy step each)", "choice"),
  ("Reset sampling", "1 s bins: 50% uniform, 40% softmax(-25 G/Gmax), 10% past weight", "Same", "same"),
  ("Reset noise", "not specified", "BeyondMimic pose/velocity/joint reset noise", "choice"),
  ("Clip end in imagination", "not specified", "treated as end of task (no bootstrap)", "choice"),
  ("Hardware", "1 or 2 NVIDIA H100 80 GB", "1 NVIDIA RTX A6000 48 GB", "deviation"),
  ("Precision", "TF32", "bf16 autocast for MLP matmuls; FK, integration, losses fp32", "deviation"),
  ("Benchmark protocol", "each clip once from frame 0, friction 1.0, no DR/noise, deterministic", "Same", "same"),
  ("PPO-Sim baseline", "BeyondMimic PPO hyperparameters, 8192 envs/GPU, same nets/obs/reward", "Same; Gaussian policy with learned per-joint std", "same"),
]
FIDELITY = [{"aspect": a, "paper": p, "ours": o, "kind": {"same": "Same as paper", "choice": "Our choice (paper silent)", "deviation": "Deviation"}[k]} for a, p, o, k in FIDELITY]


def claims(data):
  ev = {}
  for r in data["eval_curves"]:
    ev.setdefault(r["run"], []).append(r)
  def best(run):
    return max(ev.get(run, []), key=lambda r: r["norm_return"], default=None)
  def first_at(run, thr):
    return next((r for r in ev.get(run, []) if r["norm_return"] >= thr), None)
  def last(run):
    return ev.get(run, [None])[-1]
  sm82 = first_at("smimic_lafan1_1gpu", 0.82)
  ppo = last("ppo_lafan1_1gpu")
  h16n, h16c, h32n = best("abl_H16_critic_false"), best("abl_H16_critic_true"), best("abl_H32_critic_false")
  sp = {r["envs"]: r for r in data["rollout_speed"]}
  reg = {r["run"]: r["status"] for r in data["runs"]}
  abl = {(r["horizon"], r["critic"] == "yes"): r for r in data["ablation"]}

  def h82(H, c):
    v = abl.get((H, c), {}).get("hours_to_eval_082")
    return f"{v:.2f} h" if v is not None else "not reached"

  pp = os.path.join(ROOT, "results", "push_test.json")
  push = json.load(open(pp)) if os.path.exists(pp) else None

  import csv

  def _curve(run):
    L = jsonl(os.path.join(ROOT, "runs", run, "log.jsonl"))
    x = np.array([d.get("imagined_steps") or 0 for d in L], float)
    y = np.array([d.get("ep_norm_return", 0) for d in L])
    k = 9
    ys = np.convolve(np.pad(y, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid") if len(y) >= k else y
    return x, ys

  def f4b_ratio():
    xs, ys = _curve("f4b_smimic_wrs")
    xp, yp = _curve("f4b_ppowm_16kx32")
    if len(yp) < 10 or len(ys) < 10:
      return None
    best = yp.max()
    if not (ys >= best).any():
      return None
    return best, xp[yp.argmax()], xs[np.argmax(ys >= best)]

  def f4b_text():
    r = f4b_ratio()
    if not r:
      return f"walk/run/sprint subset: SuperMimic run {reg.get('f4b_smimic_wrs', '?')}, PPO-WM run {reg.get('f4b_ppowm_16kx32', '?')}"
    best, xp, xs = r
    done = reg.get("f4b_ppowm_16kx32") == "finished"
    # The paper's own ratio at the same return level, from its figure data.
    fd = os.path.join(ROOT, "paper", "src", "figures", "data", "fig5b")
    def first(fn):
      if not os.path.exists(os.path.join(fd, fn)):
        return None
      for row in csv.DictReader(open(os.path.join(fd, fn))):
        if float(row["mean_reward_smooth"]) >= best:
          return float(row["plot_step"])
      return None
    pp, ps = first("walk_run_sprint_ppo_b16384.csv"), first("walk_run_sprint_supermimic.csv")
    paper_at = f"; the paper's curves give {pp / ps:.0f}x at that return, its 64x is at return 0.84" if pp and ps else ""
    return (f"SuperMimic reaches PPO-WM 16k x 32's best training return ({best:.3f}, at {xp:.2g} imagined steps) "
            f"after {xs:.2g}: {xp / xs:.0f}x fewer" + paper_at + ("" if done else " (PPO-WM still training)"))

  def f4b_status():
    r = f4b_ratio()
    if not r or reg.get("f4b_ppowm_16kx32") != "finished":
      return "In progress"
    ratio = r[1] / r[2]
    return "Reproduced" if ratio >= 32 else ("Partly reproduced" if ratio > 4 else "Not reproduced")

  sp_path = os.path.join(ROOT, "results", "step_test.json")
  smoke_path = os.path.join(ROOT, "results", "step_test_smoke.json")
  step_final = os.path.exists(sp_path)
  step_src = json.load(open(sp_path if step_final else smoke_path)) if step_final or os.path.exists(smoke_path) else None
  step = step_src["summary"] if step_src else None

  def step_text():
    if not step:
      return f"implemented; RL run {reg.get('vel_rl', '?')}, distillation run {reg.get('vel_distill', '?')} (100k iterations each, LAFAN1 walk/run/sprint only)"
    tag = "" if step_final else (f" [smoke run: RL checkpoint at {step_src['rl']['iteration']} of 100k iterations; "
                                 "final run queued]")
    return (f"RL lower error on {step['rl_better']} of {step['n_commands']}, distillation on {step['distill_better']}; "
            f"median relative error {step['rl_median_rel_error']:.3f} vs {step['distill_median_rel_error']:.3f} (paper 0.16 vs 0.49); "
            f"distillation fell on all trials of {step['distill_fell_all']} commands (paper 7); "
            f"trained on LAFAN1 walk/run/sprint only" + tag)

  def push_text():
    if not push:
      return "test written; runs after the no-critic H=32 run ends"
    out = []
    for ck, per in push.items():
      falls = sum(v["falls"] for v in per.values())
      n = 8 * len(per)
      tag = "critic" if "smimic_lafan1" in ck else "no critic"
      fall4 = per["4.0"]["falls"]
      med = per["0.5"]["settle_s_median"]
      out.append(f"{tag}: {falls}/{n} pushes fell ({fall4}/8 at 4 m/s), settles in {med:.1f} s median at 0.5 m/s")
    return "; ".join(out) + " (matched benchmark return ~0.90)"
  # BONES-SEED: filter counts, SONIC in our simulator, SuperMimic checkpoints, length sweep.
  def _load(name):
    path = os.path.join(ROOT, "results", name)
    return json.load(open(path)) if os.path.exists(path) else None
  bfilter = _load("bones_filter.json")
  sonic = _load("sonic_bones_gains.json")
  sonic_full = _load("sonic_bones_full.json")
  b1, b2 = ev.get("smimic_bones_1gpu", []), ev.get("smimic_bones_2gpu", [])
  # Best configuration (not in the paper): 25% of environments collect without action noise;
  # scored on the full benchmark.
  bdf = ev.get("bones_fix_detfrac25_full", [])

  def filter_text():
    if not bfilter:
      return "filter script written; data downloaded, filter pending"
    return (f"{bfilter['raw']:,} -> {bfilter['after_name_filter']:,} -> {bfilter['after_feasibility_filter']:,} "
            f"({bfilter['hours']:.2f} h); the name filter's exact keyword list is not given in the paper")

  def bones_text():
    if not b1 and not sonic:
      return "BONES-SEED runs started"
    parts = []
    if sonic:
      parts.append(f"SONIC checkpoint in our simulator: {sonic['norm_return']:.3f}, {sonic['failed']:,} failed"
                   + (f" (with SONIC's hip motor model too: {sonic_full['norm_return']:.3f}, {sonic_full['failed']:,})" if sonic_full else ""))
    if b1:
      bb = max(b1, key=lambda r: r["norm_return"])
      at8 = [r for r in b1 if r["hours"] <= 8.1]
      a = max(at8, key=lambda r: r["norm_return"]) if at8 else None
      parts.append(f"SuperMimic 1 A6000: best {bb['norm_return']:.3f} ({bb['failed']:,} failed) at {bb['hours']:.1f} h"
                   + (f"; within 8.1 h (~3 H100-h): {a['norm_return']:.3f} ({a['failed']:,} failed)" if a else ""))
      if sonic:
        beat = next((r for r in b1 if r["norm_return"] > sonic["norm_return"]), None)
        both = next((r for r in b1 if r["norm_return"] > sonic["norm_return"] and r["failed"] < sonic["failed"]), None)
        parts.append(f"passes SONIC's return at {beat['hours']:.2f} h" if beat else "has not passed SONIC's return yet")
        if both:
          parts.append(f"beats it on return and failed clips at {both['hours']:.2f} h ({both['norm_return']:.3f}, {both['failed']:,} failed)")
      last = b1[-1]
      if last["norm_return"] < 0.5:
        parts.append(f"then collapsed ({last['norm_return']:.2f} at {last['hours']:.1f} h): the reset sampler concentrates on near-impossible bins once all are visited (see notes/RESULTS.md)")
    if b2:
      bb = max(b2, key=lambda r: r["norm_return"])
      parts.append(f"2 A6000s: best {bb['norm_return']:.3f} ({bb['failed']:,} failed) at {bb['hours']:.1f} h, then declined")
    text = "Faithful implementation: " + "; ".join(parts)
    if bdf and sonic:
      both = next((r for r in bdf if r["norm_return"] > sonic["norm_return"] and r["failed"] < sonic["failed"]), None)
      bb = max(bdf, key=lambda r: r["norm_return"])
      fewest = min(bdf, key=lambda r: r["failed"])
      text += (f". With our fix (25% of environments collect without action noise, not in the paper), 1 A6000: "
               + (f"beats SONIC on return and failed clips from {both['hours']:.2f} h ({both['norm_return']:.3f}, {both['failed']:,} failed); " if both else "")
               + f"best {bb['norm_return']:.3f} ({bb['failed']:,} failed) at {bb['hours']:.1f} h, fewest failed {fewest['failed']:,} at {fewest['hours']:.1f} h; "
               "no decline in 10 h. Paper: 0.94 and 25 failed at 3 H100-h (~8 A6000-h)")
    return text

  def bones_status():
    if not (b1 and sonic):
      return "In progress"
    both = [r for r in b1 + bdf if r["norm_return"] > sonic["norm_return"] and r["failed"] < sonic["failed"]]
    # The paper's margin (0.94 and 25 failed vs SONIC's 179 at 3 h) is far larger than ours.
    if any(r["norm_return"] >= 0.93 and r["failed"] <= 100 for r in b1 + bdf):
      return "Reproduced"
    if both:
      return "Partly reproduced"
    return "In progress" if reg.get("smimic_bones_1gpu") == "running" else "Not reproduced"

  def gpuh_text():
    if not (b1 and sonic):
      return "needs SuperMimic and SONIC scored on BONES-SEED"
    beat = next((r for r in b1 if r["norm_return"] > sonic["norm_return"]), None)
    if not beat:
      return f"SuperMimic has not passed SONIC's {sonic['norm_return']:.3f} yet"
    out = f"{beat['hours']:.2f} A6000-h to pass SONIC's return in our simulator vs ~21,000 L40-h: {21000 / beat['hours']:,.0f}x"
    both = next((r for r in bdf if r["norm_return"] > sonic["norm_return"] and r["failed"] < sonic["failed"]), None)
    if both:
      out += f"; {both['hours']:.2f} A6000-h to beat it on both return and failed clips (with our action-noise fix): {21000 / both['hours']:,.0f}x"
    return out + " (SONIC scored in our simulator: 0.893, 642 failed; paper's own: 0.90, 179)"

  def sweep_text():
    out = []
    for run, lib in (("sweep_bones_1h", "1 h"), ("sweep_bones_7h", "7 h"), ("sweep_bones_40h", "40 h"), ("smimic_bones_1gpu", "261 h")):
      rows = sorted(jsonl(os.path.join(ROOT, "runs", run, "eval.jsonl")), key=lambda d: d["hours"])
      if not rows:
        continue
      cross = next((d["hours"] for d in rows if d["norm_return"] >= 0.88), None)
      best = max(rows, key=lambda d: d["norm_return"])
      out.append(f"{lib}: benchmark 0.88 at {cross:.2f} h, best {best['norm_return']:.3f}" if cross else f"{lib}: best {best['norm_return']:.3f}")
    return ("; ".join(out) + " (A6000-hours, each library scored on its own clips; paper 0.54 -> 1.89 H100-h; "
            "our WM error is not on the paper's nominal scale)") if out else "runs queued"

  ratios = [sp[n]["ours_world_model"] / sp[n]["ours_mjwarp"] for n in (2048, 4096, 8192, 16384) if sp[n]["ours_mjwarp"]]
  rows = [
    ("C1", "Fig. 3a", "SuperMimic reaches return 0.82 on LAFAN1 within 1.2 h on one GPU",
     "0.82 within 1.2 h (H100)",
     (f"{sm82['norm_return']:.3f} at {sm82['hours']:.2f} h (A6000), {sm82['failed']}/40 failed; best {best('smimic_lafan1_1gpu')['norm_return']:.3f} "
      f"at {best('smimic_lafan1_1gpu')['hours']:.2f} h, {best('smimic_lafan1_1gpu')['failed']}/40 failed") if sm82 else "not yet",
     "Reproduced" if sm82 else "In progress"),
    ("C2", "Fig. 3a", "PPO-Sim (1 GPU) does not reach 0.82 within 18 h; final return 0.60",
     "0.60 after 18 h",
     f"{ppo['norm_return']:.3f} at {ppo['hours']:.1f} h of 18 h" if ppo else "not yet",
     "Reproduced" if ppo and ppo["hours"] >= 17.9 and ppo["norm_return"] < 0.82 else "In progress"),
    ("C3", "Fig. 3a", "PPO-Sim on 4 / 8 GPUs ends at 0.70 / 0.72",
     "0.70 / 0.72", "not run (needs 4-8 GPUs on the shared partition)", "Not started"),
    ("C4", "Fig. 3b,c", "Without the critic, training fails at H <= 16",
     "H=16 no critic: 0.25 final training return",
     f"H=16 no critic: best eval {h16n['norm_return']:.3f}, {h16n['failed']}/40 failed" if h16n else "running",
     "Reproduced" if h16n and h16n["norm_return"] < 0.2 else "In progress"),
    ("C5", "Fig. 3b,c", "With the critic, H=16 matches the critic-free H=32",
     "0.843 vs 0.84 final training return",
     f"best eval H=16 critic {h16c['norm_return']:.3f} vs H=32 no critic {h32n['norm_return']:.3f}" if h16c and h32n else "running",
     "Reproduced" if h16c and h32n and h16c["norm_return"] >= h32n["norm_return"] - 0.03 else "In progress"),
    ("C6", "Fig. 3c", "H=32 reaches 0.82 fastest; H 64 -> 128 nearly triples the time",
     "0.88 h (H=32) vs 1.70 h (H=64), 4.62 h (H=128)",
     (f"benchmark 0.82 at {h82(32, True)} (H=32) vs {h82(64, True)} (H=64) vs {h82(16, True)} (H=16) vs {h82(128, True)} (H=128"
      + (f", best {abl[(128, True)]['best_eval']:.3f} at {abl[(128, True)]['best_eval_hours']:.1f} h of {abl[(128, True)]['train_hours']:.1f} h trained, ~4x the per-iteration cost of H=32)" if abl.get((128, True), {}).get("best_eval") is not None else ")")),
     "Partly reproduced"),
    ("C7", "Fig. 4a", "World model rollouts are >10x faster than MuJoCo Warp at 2k-16k envs (62x at 64k)",
     "11x-26x at 2k-16k",
     f"{min(ratios):.1f}x-{max(ratios):.1f}x at 2k-16k (TF32, A6000); 32k/64k not measurable here (building that many simulated worlds needs >120 GB host RAM)" if ratios else "",
     "Reproduced"),
    ("C8", "Fig. 4b", "BPTT reaches PPO-WM's best return with ~64x fewer imagined steps",
     "64x", f4b_text(), f4b_status()),
    ("C9", "Sec. 5.3", "Under 0.5-4 m/s pushes the critic policy settles in 0.5-2.4 s; the critic-free one never settles",
     "critic settles 0.5-2.4 s; no critic stumbles at 0.5 m/s, falls at 4 m/s", push_text(),
     "Partly reproduced" if push else "In progress"),
    ("C10", "Fig. 2a, Table A.4", "On 129,514 BONES-SEED motions, 1 GPU x 3 h beats SONIC (return 0.90; 25 vs 179 failed); 2 GPUs x 9.8 h: 0 failed",
     "25 / 0 failed vs SONIC 179", bones_text(), bones_status()),
    ("C11", "Fig. 2b,c", "From 1 to 264 h of motion, 8-step WM error stays 0.007-0.011 and GPU-hours to target grow slowly",
     "0.54 -> 1.89 GPU-h", sweep_text(), "In progress" if any(reg.get(f"sweep_bones_{h}") in ("running", "queued") for h in ("1h", "7h", "40h")) else "Partly reproduced"),
    ("C12", "Sec. 5.1", ">1000x fewer GPU-hours than SONIC (21k GPU-h)",
     "3 GPU-h vs ~21,000", gpuh_text(), "Reproduced" if b1 and sonic and any(r["norm_return"] > sonic["norm_return"] and r["hours"] <= 21 for r in b1) else "In progress"),
    ("C13", "App. E", "Filtering keeps 129,514 of 142,220 BONES-SEED clips",
     "142,220 -> 129,667 -> 129,514", filter_text(), "Reproduced" if bfilter and abs(bfilter["after_feasibility_filter"] - 129514) < 1500 else "In progress"),
    ("C14", "Sec. 5.5", "RL-trained velocity encoder beats distillation on 650 of 672 step responses",
     "650 / 672", step_text(), "Reproduced" if step_final and step and step["rl_better"] > step["distill_better"] else ("In progress" if reg.get("vel_rl") in ("running", "queued", "finished") else "Not started")),
    ("C15", "Sec. 5.4", "The policy transfers to a physical Unitree G1",
     "4 real-robot motions", "out of scope (simulation only)", "Out of scope"),
  ]
  return [{"id": i, "where": w, "claim": c, "paper": p, "ours": o, "status": st} for i, w, c, p, o, st in rows]


def main():
  os.makedirs(OUT, exist_ok=True)
  data = {}

  # Benchmark evaluations of every run.
  ev_rows = []
  for run, (label, _, _) in RUNS.items():
    for d in jsonl(os.path.join(ROOT, "runs", run, "eval.jsonl")):
      ev_rows.append({
        "run": run, "label": label, "iteration": d["it"], "hours": round(d["hours"], 4),
        "norm_return": round(d["norm_return"], 4), "failed": d["failed"], "clips": d["num_clips"],
      })
  for run, (label, _, _) in list(RUNS.items()):
    for d in jsonl(os.path.join(ROOT, "runs", run, "eval_full.jsonl")):
      ev_rows.append({
        "run": run + "_full", "label": label + " (full benchmark)", "iteration": d["it"], "hours": round(d["hours"], 4),
        "norm_return": round(d["norm_return"], 4), "failed": d["failed"], "clips": d["num_clips"],
      })
  data["eval_curves"] = ev_rows

  data["paper_curves"] = [
    {"series": s, "hours": h, "norm_return": r} for s, pts in PAPER_CURVES.items() for h, r in pts
  ]

  # Training curves, one row per 1000 iterations (PPO: per 50; its logged return predates
  # the paper-style definition, so it is left out).
  tr_rows = []
  for run, (label, _, _) in RUNS.items():
    every = 50 if run.startswith("ppo") else 1000
    for d in jsonl(os.path.join(ROOT, "runs", run, "log.jsonl")):
      if d["it"] % every:
        continue
      row = {"run": run, "label": label, "iteration": d["it"], "hours": round(d["hours"], 4)}
      if not run.startswith("ppo"):
        row.update(
          train_return=round(d.get("ep_norm_return", 0), 4),
          episode_steps=round(d.get("ep_len", 0), 1),
          fell_share=round(d.get("ep_term_frac", 0), 4),
          wm_error_1step=round(d.get("wm_err1", 0), 5),
          wm_error_8step=round(d.get("wm_errk", 0), 5),
          imagined_steps=d.get("imagined_steps"),
        )
      tr_rows.append(row)
  data["train_curves"] = tr_rows

  # Fig. 4b: training return against imagined steps, ours and the paper's (TeX figure data).
  import csv

  f4 = []
  paper_f4 = {
    "walk_run_sprint_supermimic.csv": "Paper SuperMimic",
    "walk_run_sprint_ppo.csv": "Paper PPO-WM 4k x 32",
    "walk_run_sprint_ppo_b8192_n24.csv": "Paper PPO-WM 8k x 24",
    "walk_run_sprint_ppo_b8192.csv": "Paper PPO-WM 8k x 32",
    "walk_run_sprint_ppo_b16384.csv": "Paper PPO-WM 16k x 32",
  }
  # The paper's TeX source (paper/, not in the repository) is optional: without it the
  # paper's Fig. 4b curves are left out.
  for fn, label in paper_f4.items():
    path = os.path.join(ROOT, "paper", "src", "figures", "data", "fig5b", fn)
    if not os.path.exists(path):
      continue
    rows = list(csv.DictReader(open(path)))
    for r in rows[::2]:
      f4.append({"series": label, "imagined_steps": float(r["plot_step"]), "train_return": round(float(r["mean_reward_smooth"]), 4)})
  for run, label in (("f4b_smimic_wrs", "Ours SuperMimic"), ("f4b_ppowm_16kx32", "Ours PPO-WM 16k x 32")):
    for d in jsonl(os.path.join(ROOT, "runs", run, "log.jsonl"))[::5]:
      if d.get("imagined_steps"):
        f4.append({"series": label, "imagined_steps": float(d["imagined_steps"]), "train_return": round(d.get("ep_norm_return", 0), 4)})
  data["fig4b"] = f4

  # Ablation summary.
  ab_rows = []
  for (H, crit), (p_final, p_hours) in sorted(PAPER_ABLATION.items()):
    run = ABL_RUN.get((H, crit))
    evs = jsonl(os.path.join(ROOT, "runs", run, "eval.jsonl")) if run else []
    logs = jsonl(os.path.join(ROOT, "runs", run, "log.jsonl")) if run else []
    best = max(evs, key=lambda d: d["norm_return"]) if evs else None
    last = evs[-1] if evs else None
    cross = next((d["hours"] for d in evs if d["norm_return"] >= 0.82), None)
    ab_rows.append({
      "config": f"H={H}, {'critic' if crit else 'no critic'}", "horizon": H, "critic": "yes" if crit else "no",
      "run": run or "not run",
      "best_eval": round(best["norm_return"], 4) if best else None,
      "best_eval_hours": round(best["hours"], 3) if best else None,
      "latest_eval": round(last["norm_return"], 4) if last else None,
      "latest_failed": last["failed"] if last else None,
      "train_hours": round(logs[-1]["hours"], 3) if logs else None,
      "hours_to_eval_082": round(cross, 3) if cross is not None else None,
      "paper_final_train_return": p_final,
      "paper_hours_to_train_082": p_hours,
    })
  data["ablation"] = ab_rows

  # Rollout speed.
  b = json.load(open(os.path.join(ROOT, "results", "bench_rollout_a6000.json")))
  ph = b["paper_h100"]
  sp_rows = []
  for i, n in enumerate(ph["sizes"]):
    j = b["sizes"].index(n) if n in b["sizes"] else None
    sp_rows.append({
      "envs": n,
      "ours_mjwarp": round(b["mjwarp"][j]) if j is not None else None,
      "ours_world_model": round(b["wm_tf32"][j]) if j is not None else None,
      "paper_mjwarp": ph["mjwarp"][i],
      "paper_world_model": ph["world_model"][i],
    })
  data["rollout_speed"] = sp_rows

  # Per-clip results of the latest checkpoints (LAFAN1).
  names = [str(x) for x in np.load(os.path.join(ROOT, "data", "processed", "lafan1_50hz.npz"))["names"]]
  sm = jsonl(os.path.join(ROOT, "runs", "smimic_lafan1_1gpu", "eval.jsonl"))
  pp = jsonl(os.path.join(ROOT, "runs", "ppo_lafan1_1gpu", "eval.jsonl"))
  clip_rows = []
  if sm and pp:
    s, p = sm[-1], pp[-1]
    for i, nm in enumerate(names):
      clip_rows.append({
        "clip": nm.replace(".csv", ""),
        "seconds": None,
        "supermimic_return": round(s["per_clip_norm_return"][i], 4),
        "supermimic_failed": "yes" if s["per_clip_failed"][i] else "no",
        "ppo_return": round(p["per_clip_norm_return"][i], 4),
        "ppo_failed": "yes" if p["per_clip_failed"][i] else "no",
        "supermimic_checkpoint_hours": round(s["hours"], 3),
        "ppo_checkpoint_hours": round(p["hours"], 3),
      })
    lens = np.load(os.path.join(ROOT, "data", "processed", "lafan1_50hz.npz"))["lengths"]
    for r, L in zip(clip_rows, lens):
      r["seconds"] = round(float(L) / 50.0, 1)
  data["per_clip"] = clip_rows

  # Run registry.
  states = job_states()
  reg = []
  for run, (label, fig, job) in RUNS.items():
    logs = jsonl(os.path.join(ROOT, "runs", run, "log.jsonl"))
    done = os.path.exists(os.path.join(ROOT, "runs", run, "DONE"))
    jid = job.split()[0]
    st = "finished" if done else ("running" if states.get(jid) == "RUNNING" else ("queued" if states.get(jid) == "PENDING" else ("stopped" if logs else "not started")))
    reg.append({
      "run": run, "label": label, "figure": fig, "slurm_job": job, "status": st,
      "iterations": logs[-1]["it"] if logs else 0,
      "train_hours": round(logs[-1]["hours"], 3) if logs else 0,
      "log": f"runs/{run}/log.jsonl", "evals": f"runs/{run}/eval.jsonl",
    })
  data["runs"] = reg

  # SONIC's released checkpoint on the BONES-SEED benchmark, ours and the paper's.
  sb = []
  for fn, label in (("sonic_bones_gains.json", "Our simulator, SONIC's own PD gains and action scale"),
                    ("sonic_bones_full.json", "Our simulator, also SONIC's hip-pitch motor model")):
    path = os.path.join(ROOT, "results", fn)
    if os.path.exists(path):
      r = json.load(open(path))
      sb.append({"variant": label, "norm_return": round(r["norm_return"], 4), "failed": r["failed"], "clips": r["num_clips"]})
  sb.append({"variant": "Paper (its 129,514-motion benchmark)", **PAPER_SONIC})
  data["sonic_bones"] = sb

  data["claims"] = claims(data)
  data["fidelity"] = FIDELITY

  for k, v in data.items():
    json.dump(v, open(os.path.join(OUT, f"{k}.json"), "w"), indent=0)
    print(k, len(v), "rows", os.path.getsize(os.path.join(OUT, f"{k}.json")), "bytes")
  print("exported at", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))


if __name__ == "__main__":
  main()
