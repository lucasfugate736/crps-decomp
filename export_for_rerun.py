#!/usr/bin/env python3

import argparse
import json
import os
from glob import glob

import numpy as np

DEFAULT_MAX_CASES = 5000
SEED = 0
SKIP_MODELS = {"autoets"}          # numerically unstable, not used in the paper


def load_context(contexts, dataset):
    path = os.path.join(contexts, f"context__{dataset}__short.npz")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; extract contexts.tar.gz first")
    c = np.load(path, allow_pickle=True)
    ctx = c["context"]
    last = np.array([row[np.isfinite(row)][-1] for row in ctx], dtype=float)
    return last, int(c["seasonality"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecasts", default="forecasts")
    ap.add_argument("--contexts", default="contexts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-cases", type=int, default=DEFAULT_MAX_CASES)
    ap.add_argument("--level-included", action="store_true",
                    help="skip level removal (the Appendix A target)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    datasets = sorted({os.path.basename(p).split("__")[1]
                       for p in glob(os.path.join(args.forecasts, "*__*__short.npz"))
                       if not os.path.basename(p).startswith("context__")})

    for ds in datasets:
        last, season = load_context(args.contexts, ds)
        for fpath in sorted(glob(os.path.join(args.forecasts,
                                              f"*__{ds}__short.npz"))):
            model = os.path.basename(fpath).split("__")[0]
            if model.lower() in SKIP_MODELS or model == "context":
                continue

            d = np.load(fpath)
            Q, y, scale, item = d["Q"], d["y"], d["scale"], d["item"]
            W, H, K = Q.shape

            if not args.level_included:
                Q = Q - last[:, None, None]
                y = y - last[:, None]
            Q = Q / scale[:, None, None]
            y = y / scale[:, None]

            fc = Q.reshape(-1, K).astype(float)
            yy = y.reshape(-1).astype(float)
            series = np.repeat(item, H)
            step = np.tile(np.arange(1, H + 1), W)

            keep = np.isfinite(yy) & np.all(np.isfinite(fc), axis=1)
            fc, yy, series, step = fc[keep], yy[keep], series[keep], step[keep]

            # identical subsample rule to crpsdecomp.decompose_crps
            if len(yy) > args.max_cases:
                rng = np.random.default_rng(SEED)
                idx = np.sort(rng.choice(len(yy), args.max_cases, replace=False))
                fc, yy, series, step = fc[idx], yy[idx], series[idx], step[idx]

            tag = f"{ds}__{model}"
            np.savez_compressed(
                os.path.join(args.out, tag + ".npz"),
                fc=np.sort(fc, axis=1), y=yy,
                series=np.unique(series, return_inverse=True)[1].astype(np.int64),
                step=step.astype(np.int64))
            print(f"{tag:34s} n={len(yy):6d} series={len(np.unique(series)):5d} "
                  f"season={season}")

    meta = {"level_removed": not args.level_included,
            "max_cases": args.max_cases, "seed": SEED,
            "excluded": sorted(SKIP_MODELS)}
    json.dump(meta, open(os.path.join(args.out, "_export.json"), "w"), indent=2)
    print(f"\nwrote to {args.out}")


if __name__ == "__main__":
    main()
