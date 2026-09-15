#!/usr/bin/env python3
"""
Turn whatever you already have into the .npz files rerun_experiments.py expects,
and check them before you spend pod hours on bad input.

    python make_inputs.py --check forecasts/          # validate what's there
    python make_inputs.py --from-parquet raw/ --out forecasts/

The target format, one file per (dataset, model), named <dataset>__<model>.npz:

    fc      float (n, 9)   quantile forecasts, levels 0.1..0.9, ASCENDING
    y       float (n,)     realised target
    series  int   (n,)     series or window id -- the bootstrap block
    step    int   (n,)     horizon step, 1-based

fc and y must already be MASE-scaled and level-removed, exactly as you score
them now. Do the scaling upstream, not here, so the decomposition sees the same
numbers your Table 1 does.
"""

import argparse
import glob
import os

import numpy as np

QUANTILES = np.arange(1, 10) / 10.0


def check(path: str) -> list[str]:
    """Return a list of problems. Empty list means the file is usable."""
    bad = []
    d = np.load(path)
    for k in ("fc", "y", "series", "step"):
        if k not in d:
            bad.append(f"missing array '{k}'")
    if bad:
        return bad

    fc, y, series, step = d["fc"], d["y"], d["series"], d["step"]
    n = y.size
    if fc.ndim != 2 or fc.shape[0] != n:
        bad.append(f"fc is {fc.shape}, expected ({n}, 9)")
    if fc.shape[1] != 9:
        bad.append(f"fc has {fc.shape[1]} quantile levels, expected 9")
    for name, a in (("series", series), ("step", step)):
        if a.size != n:
            bad.append(f"{name} has {a.size} entries, expected {n}")
    if not np.isfinite(fc).all():
        bad.append(f"fc has {(~np.isfinite(fc)).sum()} non-finite values")
    if not np.isfinite(y).all():
        bad.append(f"y has {(~np.isfinite(y)).sum()} non-finite values")

    # quantile crossing: harmless for CRPS but a sign the columns are misordered
    if fc.ndim == 2 and fc.shape[0] == n:
        crossed = (np.diff(fc, axis=1) < 0).any(axis=1).sum()
        if crossed > 0.5 * n:
            bad.append(f"{crossed}/{n} rows have descending quantiles -- "
                       "columns are probably reversed")
        elif crossed:
            bad.append(f"note: {crossed}/{n} rows have crossing quantiles")

    # level removal check: after subtracting the last observed value the target
    # should be roughly centred. A large offset means it was not applied.
    if abs(float(np.median(y))) > 3 * float(np.std(y)) + 1e-9:
        bad.append(f"median(y)={np.median(y):.2f} vs sd={np.std(y):.2f} -- "
                   "level removal may not have been applied")

    n_series = np.unique(series).size
    if n_series < 20:
        bad.append(f"only {n_series} distinct series: the block bootstrap will "
                   "be very coarse")
    return bad


def from_parquet(src: str, out: str) -> None:
    """
    Adapt this to your own layout. Expects one parquet per (dataset, model) with
    columns: series_id, step, target, and q0.1 ... q0.9.
    """
    import pandas as pd

    os.makedirs(out, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(src, "*.parquet"))):
        tag = os.path.basename(path).rsplit(".", 1)[0]
        df = pd.read_parquet(path)
        qcols = [f"q{q:.1f}" for q in QUANTILES]
        missing = [c for c in qcols if c not in df.columns]
        if missing:
            raise SystemExit(f"{path}: missing columns {missing}")
        np.savez_compressed(
            os.path.join(out, f"{tag}.npz"),
            fc=df[qcols].to_numpy(float),
            y=df["target"].to_numpy(float),
            series=pd.factorize(df["series_id"])[0].astype(np.int64),
            step=df["step"].to_numpy(np.int64),
        )
        print(f"wrote {tag}.npz  n={len(df)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check")
    ap.add_argument("--from-parquet")
    ap.add_argument("--out", default="forecasts")
    args = ap.parse_args()

    if args.from_parquet:
        from_parquet(args.from_parquet, args.out)

    target = args.check or (args.out if args.from_parquet else None)
    if not target:
        ap.error("give --check or --from-parquet")

    ok = True
    for path in sorted(glob.glob(os.path.join(target, "*.npz"))):
        problems = check(path)
        name = os.path.basename(path)
        if not problems:
            d = np.load(path)
            print(f"OK   {name:40s} n={d['y'].size:6d} "
                  f"series={np.unique(d['series']).size}")
        else:
            ok = False
            print(f"BAD  {name}")
            for p in problems:
                print(f"       - {p}")

    if not glob.glob(os.path.join(target, "*.npz")):
        print(f"no .npz files in {target}")
    elif ok:
        print("\nall files usable -- safe to start the run")


if __name__ == "__main__":
    main()
