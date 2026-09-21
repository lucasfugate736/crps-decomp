#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from idr_decomposition import block_bootstrap, decompose, decompose_cross_fit

# n=2000 takes ~100 s for the exact stochastic order and scales ~n^2; 5000 is
# roughly 10 min per model-dataset. Subsample only the stochastic-order check.
STOCHASTIC_SUBSAMPLE = 5000
BOOTSTRAP_B = 1000


def load(path):
    d = np.load(path)
    return d["fc"], d["y"], d["series"], d.get("step")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=BOOTSTRAP_B)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel workers; each (dataset, model) is one task")
    ap.add_argument("--only", default="*", help="glob filter, e.g. 'm4_weekly__*'")
    ap.add_argument("--thresholds", type=int, default=0,
                    help="CRPS integration grid for point estimates; "
                         "0 = every distinct outcome (most accurate, slowest)")
    ap.add_argument("--boot-thresholds", type=int, default=400,
                    help="coarser grid for the bootstrap: grid bias is common "
                         "to every resample, so interval WIDTH is unaffected")
    ap.add_argument("--subsample", type=int, default=STOCHASTIC_SUBSAMPLE,
                    help="cases for the exact stochastic-order check; "
                         "cost grows as the square of this")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "parts"), exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.data, args.only + ".npz")))
    todo = [p for p in paths if not os.path.exists(_part(args.out, p))]
    print(f"{len(paths)} tasks, {len(paths) - len(todo)} already done, "
          f"{len(todo)} to run on {args.jobs} workers")

    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            list(ex.map(_run_one, [(p, vars(args)) for p in todo]))
    else:
        for p in todo:
            _run_one((p, vars(args)))

    rows = []
    for p in paths:
        part = _part(args.out, p)
        if os.path.exists(part):
            rows.extend(json.load(open(part)))
    _write(rows, vars(args))


def _part(out, path):
    return os.path.join(out, "parts", os.path.basename(path)[:-4] + ".json")


def _run_one(job):
    """One (dataset, model) task. Written to its own file so the run resumes
    after a spot-instance interruption instead of starting over."""
    path, args = job
    T = args["thresholds"] or None
    TB = args["boot_thresholds"]
    rows = []
    for path in [path]:
        tag = os.path.basename(path)[:-4]
        dataset, model = tag.split("__", 1)
        fc, y, series, step = load(path)
        rng = np.random.default_rng(args["seed"])
        print(f"[{dataset}/{model}] n={y.size}")

        med = decompose(fc, y, order="median", max_thresholds=T)
        cf = decompose_cross_fit(fc, y, order="median", k=args["folds"],
                                 seed=args["seed"], groups=series,
                                 max_thresholds=T)

        # exact stochastic order on a subsample -- this is the number that
        # defends (or refutes) the median-order approximation
        if y.size > args["subsample"]:
            sub = rng.choice(y.size, args["subsample"], replace=False)
        else:
            sub = np.arange(y.size)
        sto = decompose(fc[sub], y[sub], order="stochastic", max_thresholds=TB)
        med_sub = decompose(fc[sub], y[sub], order="median", max_thresholds=TB)

        ci = block_bootstrap(fc, y, series, b=args["boot"], order="median",
                             seed=args["seed"], max_thresholds=TB)

        rows.append(dict(
            dataset=dataset, model=model, n=y.size,
            crps=med.crps, unc=med.unc,
            mcb_median=med.mcb, dsc_median=med.dsc,
            mcb_stochastic=sto.mcb, dsc_stochastic=sto.dsc,
            mcb_median_sub=med_sub.mcb, dsc_median_sub=med_sub.dsc,
            mcb_gap=sto.mcb - med_sub.mcb, dsc_gap=sto.dsc - med_sub.dsc,
            mcb_crossfit=cf.mcb, dsc_crossfit=cf.dsc,
            mcb_lo=ci["mcb"][0], mcb_hi=ci["mcb"][1],
            dsc_lo=ci["dsc"][0], dsc_hi=ci["dsc"][1],
            crps_lo=ci["crps"][0], crps_hi=ci["crps"][1],
        ))

        # horizon thirds (blind spot 3), cross-fitted so the split is honest
        if step is not None:
            edges = np.quantile(step, [1 / 3, 2 / 3])
            for name, sel in (("early", step <= edges[0]),
                              ("late", step > edges[1])):
                h = decompose_cross_fit(fc[sel], y[sel], order="median",
                                        k=args["folds"], seed=args["seed"],
                                        groups=series[sel], max_thresholds=T)
                rows.append(dict(dataset=f"{dataset}:{name}", model=model,
                                 n=int(sel.sum()), crps=h.crps, unc=h.unc,
                                 mcb_crossfit=h.mcb, dsc_crossfit=h.dsc))

    json.dump(rows, open(_part(args["out"], path), "w"))
    print(f"  done {tag}")
    return rows


def _write(rows, args):
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(args["out"], "decomposition.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    neg = [r for r in rows
           if min(r.get("mcb_median", 0), r.get("dsc_median", 0)) < 0]
    print(f"\nwrote {args["out"]}/decomposition.csv")
    print(f"negative components under the median order: {len(neg)} of {len(rows)}")
    gaps = [abs(r["mcb_gap"]) for r in rows if "mcb_gap" in r]
    if gaps:
        print(f"max |MCB(stochastic) - MCB(median)| on the subsample: {max(gaps):.4f}")
        print("  -> this is the number that replaces the \\todo in Section 4")


if __name__ == "__main__":
    main()
