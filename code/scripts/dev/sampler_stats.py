"""Diagnostic: where the return-based reset sampler puts its mass, from a checkpoint's bin
statistics (bin_G, bin_visits, bin_past_w) and the motion library's bin layout.

usage: sampler_stats.py motions.npz ckpt [ckpt ...]
"""

import collections
import re
import sys

import numpy as np
import torch


def main():
  d = np.load(sys.argv[1])
  lengths, names = d["lengths"], [str(n).split("/")[-1] for n in d["names"]]
  nb = (lengths + 49) // 50
  bin_clip = np.repeat(np.arange(len(lengths)), nb)
  for ck in sys.argv[2:]:
    c = torch.load(ck, map_location="cpu", weights_only=False)
    G, vis, past = c["bin_G"].numpy(), c["bin_visits"].numpy(), c["bin_past_w"].numpy()
    soft = np.exp(-25 * (G - G.min()))
    soft /= soft.sum()
    p = 0.5 / len(G) + 0.4 * soft + 0.1 * past / past.sum()
    order = np.argsort(-soft)
    top = order[: max(1, int(0.001 * len(G)))]
    print(f"{ck}: it {c['it']}, {len(G):,} bins, visited {np.mean(vis > 0):.3f}")
    for thr in (0.05, 0.1, 0.2, 0.4):
      m = G < thr
      print(f"  bins with G < {thr}: {m.sum():6d} ({m.mean():.4f}) hold softmax mass {soft[m].sum():.3f}, total reset prob {p[m].sum():.3f}")
    k = np.searchsorted(np.cumsum(soft[order]), 0.5) + 1
    print(f"  half of the softmax mass sits on {k} bins from {len(set(bin_clip[order[:k]]))} clips")
    stem = lambda n: re.sub(r"(_r)?_\d{3}(__a\d+)?(_m)?\.csv$", "", n)
    cnt = collections.Counter(stem(names[bin_clip[b]]) for b in order[:k])
    print("  their motions:", cnt.most_common(12))


if __name__ == "__main__":
  main()
