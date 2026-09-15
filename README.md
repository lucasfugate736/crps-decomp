# crps-decomp — calibration/discrimination decomposition for forecast files

Computes the isotonicity-based CRPS decomposition

    CRPS = MCB - DSC + UNC

(Arnold, Walz, Ziegel & Gneiting, *Decompositions of the mean continuous ranked
probability score*, EJS 18(2), 2024) from any file of quantile forecasts, so a
leaderboard can report calibration and discrimination next to its aggregate
score. This is the code for the paper *Same Score, Different Reasons:
Calibration and Discrimination in Time-Series Foundation Models*.

Anonymised for double-blind review.

## What you get

`MCB` is score lost to miscalibration — an estimate of what post-hoc
recalibration could recover. `DSC` is score gained over climatology by the
recalibrated forecast. `UNC` is fixed by the outcomes. Two models with the same
CRPS can sit at very different `(MCB, DSC)` positions, and the difference tells
you whether to improve the model or just recalibrate its output.

## Install

    pip install "numpy>=1.24" "scipy>=1.12"

scipy 1.12 is the floor: `scipy.optimize.isotonic_regression` does not exist
before it.

## Quick start

```python
import numpy as np
from idr_decomposition import decompose, decompose_cross_fit, block_bootstrap

fc = ...      # (n, 9) quantile forecasts, levels 0.1 ... 0.9, ascending
y  = ...      # (n,)   realised outcomes
series = ...  # (n,)   series or window id, the bootstrap block

decompose(fc, y, order="median")                     # fast, total order
decompose(fc, y, order="stochastic")                 # exact, O(n^2)
decompose_cross_fit(fc, y, k=5, groups=series)       # removes in-sample optimism
block_bootstrap(fc, y, series, b=1000)               # 95% intervals
```

`order="stochastic"` is the construction Arnold et al. prove nonnegativity for.
It is quadratic in the number of comparable pairs: about 100 s at n=2,000 and
30 min at n=5,000, so it is practical as a check on a subsample rather than as
the default. `order="median"` fits under the total order induced by the forecast
median, which is fast but forfeits the guarantee — see the paper's Appendix on
what it discards.

## Batch use

```
python rerun_experiments.py --data forecasts/ --out results/ --jobs 4
```

Expects one `<dataset>__<model>.npz` per pair containing `fc (n,9)`, `y (n,)`,
`series (n,)`, `step (n,)`, already scaled as you intend to score them. Writes
one JSON per pair as it goes, so an interrupted run resumes rather than
restarting. `make_inputs.py --check` validates the files first and catches the
usual problems: reversed quantile columns, non-finite values, a `series` array
that doesn't line up with `y`.

`export_for_rerun.py` converts GIFT-Eval-style forecast dumps into that format.

## Things worth knowing before you trust the numbers

**In-sample IDR is optimistic.** Fit and evaluated on the same cases, `MCB`
carries a positive bias. `decompose_cross_fit` removes it, at the cost of the
nonnegativity guarantee, which is an in-sample property. On a forecaster that is
calibrated by construction, cross-fitting returns `MCB ≈ +0.003` against
`+0.018` in sample. Report both and read a negative cross-fitted `MCB` as a
forecaster at or below the resolution of the estimate.

**Discretisation is not free.** Representing a forecast by nine quantile atoms
while IDR returns a full CDF puts a positive, spread-dependent offset into every
`MCB`. Issue the marginal forecast for every case to measure it: its `DSC` is
exactly zero, and its `MCB` is the offset.

**The bootstrap holds the fit fixed.** Refitting IDR on each resample lets it
exploit the duplicates that sampling with replacement creates, depressing
`CRPS_iso` and inflating both components. Intervals therefore condition on the
fitted map. Block by series: cases are not independent across horizon steps
within a window.

**Prefer paired differences.** For head-to-head comparisons, bootstrap the
difference rather than comparing marginal intervals — both models are scored on
the same series, so the difference is far better determined than either level.

## Files

| file | purpose |
|---|---|
| `idr_decomposition.py` | CRPS, IDR under both orders, decomposition, cross-fitting, bootstrap |
| `rerun_experiments.py` | batch driver, parallel and resumable |
| `make_inputs.py` | format conversion and validation |
| `export_for_rerun.py` | GIFT-Eval forecast dumps to the expected format |
| `paired_bootstrap.py` | paired difference between two models |
| `stability.py` | subsample-size and seed stability sweep |
| `make_figure.py` | the MCB–DSC plane |

## License

MIT.
