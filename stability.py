#!/usr/bin/env python3
"""
Subsample stability: does the headline pairwise conclusion survive resampling?

The reviewer's sharpest empirical demand. If the TimesFM/Chronos-Bolt DSC
contrast of -0.001 has a seed-to-seed SD of 0.03, it cannot be presented as a
meaningful number, and you want to find that out yourself.

    python stability.py --data forecasts --dataset m4_weekly \\
        --a timesfm --b chronos_bolt

Median order only, so this is fast: roughly 20 min for the default sweep.
"""

import argparse

import numpy as np

from idr_decomposition import decompose

SIZES = (1000, 2000, 5000)
SEEDS = 20


def load(path):
    d = np.load(path)
    return d["fc"], d["y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="forecasts")
    ap.add_argument("--dataset", default="m4_weekly")
    ap.add_argument("--a", default="timesfm")
    ap.add_argument("--b", default="chronos_bolt")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--thresholds", type=int, default=400)
    args = ap.parse_args()
    T = args.thresholds or None

    fa, ya = load(f"{args.data}/{args.dataset}__{args.a}.npz")
    fb, yb = load(f"{args.data}/{args.dataset}__{args.b}.npz")
    if not np.allclose(ya, yb):
        raise SystemExit("the two files do not share a case ordering")
    n_all = ya.size

    print(f"{args.dataset}: {args.a} minus {args.b}, median order, "
          f"{args.seeds} seeds per size\n")
    print(f"{'n':>7s}{'dDSC mean':>12s}{'dDSC sd':>10s}{'dMCB mean':>12s}"
          f"{'dMCB sd':>10s}{'sign flips':>12s}")

    for n in args.sizes:
        if n > n_all:
            continue
        dd, dm = [], []
        for s in range(args.seeds):
            rng = np.random.default_rng(s)
            idx = (np.sort(rng.choice(n_all, n, replace=False))
                   if n < n_all else np.arange(n_all))
            da = decompose(fa[idx], ya[idx], order="median", max_thresholds=T)
            db = decompose(fb[idx], yb[idx], order="median", max_thresholds=T)
            dd.append(da.dsc - db.dsc)
            dm.append(da.mcb - db.mcb)
        dd, dm = np.array(dd), np.array(dm)
        flips = int(min((dd > 0).sum(), (dd < 0).sum()))
        print(f"{n:7d}{dd.mean():12.4f}{dd.std():10.4f}"
              f"{dm.mean():12.4f}{dm.std():10.4f}"
              f"{flips:7d}/{args.seeds}")

    print("\nRead this as: if 'dDSC sd' is comparable to 'dDSC mean', the DSC")
    print("contrast is not resolvable at that subsample size and should not be")
    print("quoted to three decimals. Sign flips near half the seeds mean the")
    print("direction of the contrast is undetermined.")


if __name__ == "__main__":
    main()
