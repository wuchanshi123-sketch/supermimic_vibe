"""Diagnostic: which motion categories fail in a run's in-training benchmark rows (the fixed
8,192-clip subset of train.py run_eval), checkpoint by checkpoint.

usage: subset_failures.py motions.npz run_dir [n_clips]
"""

import collections
import json
import re
import sys

import numpy as np
import torch


def main():
  motions, run = sys.argv[1], sys.argv[2]
  n = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
  names = [str(x).split("/")[-1].lower() for x in np.load(motions)["names"]]
  ids = torch.randperm(len(names), generator=torch.Generator(device="cpu").manual_seed(1234))[:n].numpy()
  word = lambda s: re.sub(r"(_r)?_\d{3}(__a\d+)?(_m)?\.csv$", "", s).split("_")[0]
  acro = {"handstand", "flip", "safety", "monkey", "tiger", "cartwheel", "inchworm"}
  cats = [word(names[i]) for i in ids]
  n_acro = sum(c in acro for c in cats)
  print(f"{n} clips, {n_acro} acrobatic ({sorted(collections.Counter(c for c in cats if c in acro).items())})")
  for line in open(f"{run}/eval.jsonl"):
    d = json.loads(line)
    f = np.array(d["per_clip_failed"], dtype=bool)
    fa = sum(1 for k in np.where(f)[0] if cats[k] in acro)
    top = collections.Counter(cats[k] for k in np.where(f)[0]).most_common(5)
    print(f"  it {d['it']:>6d}: failed {int(f.sum()):3d}, acrobatic {fa:3d}/{n_acro}, top {top}")


if __name__ == "__main__":
  main()
