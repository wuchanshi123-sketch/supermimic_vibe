"""Nested random subsets of a motion library for the dataset-length sweep (Fig. 2b,c).

Clips are taken in one fixed random order until each target length is reached, so the
1 h subset is contained in the 7 h subset, and so on.

usage: make_subsets.py library.npz out_prefix hours[,hours...] [seed]
  e.g. make_subsets.py data/processed/bones_seed_50hz.npz data/processed/bones_seed_sub 1,7,40
"""

import sys

import numpy as np


def main():
  src, prefix, hours = sys.argv[1], sys.argv[2], [float(h) for h in sys.argv[3].split(",")]
  seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
  d = np.load(src)
  starts, lengths, names, fps = d["starts"], d["lengths"], d["names"], float(d["fps"])
  order = np.random.default_rng(seed).permutation(len(lengths))
  cum = np.cumsum(lengths[order]) / fps / 3600
  qpos, qvel = d["qpos"], d["qvel"]
  for h in hours:
    k = int(np.searchsorted(cum, h)) + 1
    clips = np.sort(order[:k])
    sel = np.concatenate([np.arange(starts[c], starts[c] + lengths[c]) for c in clips])
    L = lengths[clips]
    out = f"{prefix}_{h:g}h.npz"
    np.savez(out, qpos=qpos[sel], qvel=qvel[sel], starts=np.concatenate([[0], np.cumsum(L)[:-1]]),
             lengths=L, names=names[clips], fps=d["fps"])
    print(f"{out}: {len(clips)} clips, {L.sum() / fps / 3600:.3f} h")


if __name__ == "__main__":
  main()
