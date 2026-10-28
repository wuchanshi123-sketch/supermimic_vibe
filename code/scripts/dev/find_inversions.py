"""List clips whose reference motion actually turns upside down, with a checkpoint's benchmark score.

Inversion is measured on the reference root: R22 = 1 - 2(x^2 + y^2) of the root quaternion is the
world-z component of the pelvis z axis (1 upright, -1 upside down). Only clips whose name matches
the keyword pattern are read, so the full motion file is not decoded into memory per frame.

usage: find_inversions.py motions.npz eval_full.jsonl iteration "cartwheel|flip|handstand|roll"
"""

import json
import os
import re
import sys

import numpy as np


def main():
  motions, evf, it, pat = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
  d = np.load(motions)
  names = [os.path.basename(str(n)).removesuffix(".csv") for n in d["names"]]
  starts, lengths = d["starts"], d["lengths"]
  row = next(json.loads(l) for l in open(evf) if json.loads(l)["it"] == it)
  R, F = row["per_clip_norm_return"], row["per_clip_failed"]
  ids = [i for i, n in enumerate(names) if re.search(pat, n, re.I)]
  qpos = d["qpos"]
  out = []
  for i in ids:
    q = qpos[starts[i] : starts[i] + lengths[i], 3:7]
    r22 = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    out.append((names[i], lengths[i] / float(d["fps"]), float(r22.min()), float((r22 < 0).mean()), R[i], F[i]))
  out.sort(key=lambda x: (-(x[2] < -0.5 and not x[5]), -x[4]))
  inv = [o for o in out if o[2] < -0.5]
  print(f"{len(ids)} clips match; inverted (min R22 < -0.5): {len(inv)}, of which failed {sum(o[5] for o in inv)}, "
        f"mean return {np.mean([o[4] for o in inv]):.3f}; not inverted: {len(out) - len(inv)}, failed "
        f"{sum(o[5] for o in out if o[2] >= -0.5)}, mean return {np.mean([o[4] for o in out if o[2] >= -0.5]):.3f}")
  print(f"all {len(R)} clips: mean return {np.mean(R):.3f}, failed {sum(F)}")
  print("name | seconds | min R22 | frac upside down | return | failed")
  for o in out[:60]:
    print(f"{o[0]} | {o[1]:.1f} | {o[2]:.2f} | {o[3]:.2f} | {o[4]:.3f} | {o[5]}")


if __name__ == "__main__":
  main()
