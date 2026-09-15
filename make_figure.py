#!/usr/bin/env python3
"""
Build Figure 1 from results/decomposition.csv.

    python make_figure.py --results results --out figures

Writes figures/fig1_mcb_dsc.pdf (and .png). Run analyze.py with
--level-removed first so decomposition.csv holds the residualised numbers the
caption describes.

Design choices, all deliberate:
  * equal x/y scaling, so the slope-one iso-CRPS diagonals actually render at
    45 degrees -- the caption claims this, so it has to be true;
  * only the models involved in a highlighted comparison are labelled, the rest
    are small dots, because a legible figure beats a complete one;
  * two callouts carrying the paper's two visual claims.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PANELS = ["electricity/H", "M4-hourly", "M4-weekly", "solar/H"]
ALIAS = {"electricity/H": ["electricity/H", "electricity-H"],
         "M4-hourly": ["m4_hourly", "M4-hourly"],
         "M4-weekly": ["m4_weekly", "M4-weekly"],
         "solar/H": ["solar/H", "solar-H"]}
PRETTY = {"tirex": "TiRex", "timesfm": "TimesFM-2.5", "chronos2": "Chronos-2",
          "moirai2": "Moirai-2.0", "chronos_bolt": "Chronos-Bolt",
          "SeasonalNaive": "Seas. naive", "seasonalnaive": "Seas. naive"}
SKIP = {"CLIM", "AutoETS", "autoets"}

# which models to label per panel, and the callout to draw
HIGHLIGHT = {
    "M4-weekly": (["timesfm", "chronos_bolt"], "same DSC,\ndifferent MCB"),
    "electricity/H": (["tirex", "chronos_bolt"], "similar CRPS,\nopposite trade-off"),
    "solar/H": (["tirex", "chronos_bolt"], None),
    "M4-hourly": (["tirex", "chronos_bolt"], None),
}


def load(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["model"] in SKIP:
                continue
            try:
                rows.append({"dataset": r["dataset"], "model": r["model"],
                             "crps": float(r["crps"]), "mcb": float(r["mcb"]),
                             "dsc": float(r["dsc"])})
            except (ValueError, KeyError):
                continue
    return rows


def panel(ax, rows, title, highlight, callout, baseline):
    """One configuration. Equal aspect, so the slope-one diagonals render at 45
    degrees; the baseline is excluded from the frame because it is a far outlier
    that would compress every TSFM into a single dot."""
    if not rows:
        ax.set_visible(False)
        return
    mcb = np.array([r["mcb"] for r in rows])
    dsc = np.array([r["dsc"] for r in rows])

    pad = 0.42
    span = max(np.ptp(mcb), np.ptp(dsc)) * (1 + 2 * pad) or 1.0
    cx = (mcb.min() + mcb.max()) / 2
    cy = (dsc.min() + dsc.max()) / 2
    x0, x1 = cx - span / 2, cx + span / 2
    y0, y1 = cy - span / 2, cy + span / 2

    step = span / 5
    c0 = x0 - y1
    for k in range(-1, 12):
        c = c0 + k * step
        ax.plot([x0, x1], [x0 - c, x1 - c], color="0.88", lw=0.6, zorder=0)

    order = sorted(rows, key=lambda r: r["mcb"])
    for i, r in enumerate(order):
        hot = r["model"] in highlight
        ax.scatter(r["mcb"], r["dsc"], s=46 if hot else 22,
                   color="#1f4e79" if hot else "#a8c2da",
                   zorder=3, edgecolors="white", linewidths=0.7)
        if hot:
            left = i < len(order) / 2
            ax.annotate(f"{PRETTY.get(r['model'], r['model'])} ({r['crps']:.3f})",
                        (r["mcb"], r["dsc"]), textcoords="offset points",
                        xytext=(-9 if left else 9, 8),
                        ha="right" if left else "left",
                        fontsize=7.5, zorder=4)

    if callout:
        hx = [r["mcb"] for r in rows if r["model"] in highlight]
        hy = [r["dsc"] for r in rows if r["model"] in highlight]
        if len(hx) == 2:
            ax.annotate("", xy=(hx[0], hy[0]), xytext=(hx[1], hy[1]),
                        arrowprops=dict(arrowstyle="<->", color="#1f4e79",
                                        lw=1.1, shrinkA=7, shrinkB=7), zorder=2)
            ax.text(np.mean(hx), min(hy) - span * 0.13, callout, fontsize=7.5,
                    ha="center", va="top", color="#1f4e79", style="italic")

    if baseline:
        ax.text(0.98, 0.03,
                f"seas. naive: MCB {baseline['mcb']:.3f}, DSC {baseline['dsc']:.3f}"
                " (off panel)", transform=ax.transAxes, fontsize=6.5,
                ha="right", va="bottom", color="0.45")

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"MCB $\rightarrow$ worse calibration", fontsize=7.5)
    ax.set_ylabel(r"DSC $\rightarrow$ more discrimination", fontsize=7.5)
    ax.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = load(os.path.join(args.results, "decomposition.csv"))
    if not rows:
        raise SystemExit("no usable rows in decomposition.csv")

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4))
    for ax, name in zip(axes.ravel(), PANELS):
        keys = ALIAS[name]
        sub = [r for r in rows if r["dataset"] in keys]
        base = next((r for r in sub if r["model"] in
                     ("SeasonalNaive", "seasonalnaive")), None)
        sub = [r for r in sub if r is not base]
        hl, callout = HIGHLIGHT.get(name, ([], None))
        panel(ax, sub, name, set(hl), callout, base)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(args.out, f"fig1_mcb_dsc.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print("wrote", path)


if __name__ == "__main__":
    main()
