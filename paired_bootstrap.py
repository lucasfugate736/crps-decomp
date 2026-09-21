#!/usr/bin/env python3

import argparse

import numpy as np

from idr_decomposition import (_crps_of_idr, _idr_cdf, _marginal_cdf,
                               _thresholds, crps_ensemble)


def per_case(path, order, max_thresholds):
    d = np.load(path)
    fc, y, series = d["fc"], d["y"], d["series"]
    t = _thresholds(y, max_thresholds)
    iso = _crps_of_idr(_idr_cdf(fc, y, t, order), t, y)
    mg = _crps_of_idr(_marginal_cdf(y, t, y.size), t, y)
    return crps_ensemble(fc, y), iso, mg, series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="forecasts")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--order", default="median")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--thresholds", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    T = args.thresholds or None

    ca, ia, ma, sa = per_case(f"{args.data}/{args.dataset}__{args.a}.npz",
                              args.order, T)
    cb, ib, mb, sb = per_case(f"{args.data}/{args.dataset}__{args.b}.npz",
                              args.order, T)
    if not np.array_equal(sa, sb):
        raise SystemExit("the two files do not share a case ordering")

    uniq = np.unique(sa)
    where = {g: np.nonzero(sa == g)[0] for g in uniq}
    rng = np.random.default_rng(args.seed)

    d_dsc, d_mcb, d_crps = [], [], []
    for _ in range(args.boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([where[g] for g in pick])
        d_dsc.append((ma[idx].mean() - ia[idx].mean())
                     - (mb[idx].mean() - ib[idx].mean()))
        d_mcb.append((ca[idx].mean() - ia[idx].mean())
                     - (cb[idx].mean() - ib[idx].mean()))
        d_crps.append(ca[idx].mean() - cb[idx].mean())

    print(f"{args.dataset}: {args.a} minus {args.b}  (order={args.order}, "
          f"B={args.boot}, blocked by series)\n")
    for name, obs, draws in (
            ("DSC", (ma.mean() - ia.mean()) - (mb.mean() - ib.mean()), d_dsc),
            ("MCB", (ca.mean() - ia.mean()) - (cb.mean() - ib.mean()), d_mcb),
            ("CRPS", ca.mean() - cb.mean(), d_crps)):
        lo, hi = np.quantile(draws, [0.025, 0.975])
        covers = "contains 0" if lo <= 0 <= hi else "EXCLUDES 0"
        print(f"  {name:5s} diff {obs:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   {covers}")


if __name__ == "__main__":
    main()
