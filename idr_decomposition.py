

from __future__ import annotations

import numpy as np
from scipy.optimize import isotonic_regression
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order

__all__ = [
    "crps_ensemble",
    "crps_discrete",
    "crps_marginal",
    "decompose",
    "decompose_cross_fit",
    "block_bootstrap",
    "Decomposition",
]

_CAP = 10**5  # integer capacity scale for the min-cut solver (int32-safe)


# --------------------------------------------------------------------------
# CRPS
# --------------------------------------------------------------------------
def crps_discrete(atoms: np.ndarray, weights: np.ndarray, y: float) -> float:
    """Exact CRPS of a discrete distribution (sorted atoms, weights summing to 1)."""
    order = np.argsort(atoms)
    v = np.asarray(atoms, float)[order]
    w = np.asarray(weights, float)[order]
    cdf = np.cumsum(w)

    # Integrate (F(z) - 1{y <= z})^2 over the real line.
    # Below v[0]: F = 0, indicator = 1{y <= z}; above v[-1]: F = 1.
    lo, hi = min(v[0], y), max(v[-1], y)
    grid = np.unique(np.concatenate([v, [y, lo, hi]]))
    widths = np.diff(grid)
    if widths.size == 0:
        return 0.0
    # CDF value on each interval [grid[k], grid[k+1])
    idx = np.searchsorted(v, grid[:-1], side="right") - 1
    F = np.where(idx >= 0, cdf[np.clip(idx, 0, None)], 0.0)
    ind = (y <= grid[:-1]).astype(float)
    return float(np.sum((F - ind) ** 2 * widths))


def crps_ensemble(fc: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    CRPS of equally weighted m-member ensembles, vectorised.
    Equivalent to the NRG estimator  mean|X-y| - 0.5*mean|X-X'|.
    """
    fc = np.sort(np.asarray(fc, float), axis=1)
    y = np.asarray(y, float)
    m = fc.shape[1]
    term1 = np.abs(fc - y[:, None]).mean(axis=1)
    diffs = np.abs(fc[:, :, None] - fc[:, None, :]).sum(axis=(1, 2))
    return term1 - diffs / (2.0 * m * m)


def crps_marginal(y: np.ndarray) -> np.ndarray:
    """
    CRPS of the empirical (climatological) CDF of all outcomes, evaluated at
    each outcome. O(n log n) rather than O(n^2).
    """
    y = np.asarray(y, float)
    n = y.size
    s = np.sort(y)
    csum = np.cumsum(s)
    total = csum[-1]
    rank = np.searchsorted(s, y, side="left")
    below = np.where(rank > 0, csum[np.clip(rank - 1, 0, None)], 0.0)
    # mean |s_i - y|
    term1 = (rank * y - below + (total - below) - (n - rank) * y) / n
    # 0.5 * mean |s_i - s_j|  (constant across y)
    i = np.arange(1, n + 1)
    gini = 2.0 * np.sum((2 * i - n - 1) * s) / (n * n)
    return term1 - 0.5 * gini


def _crps_of_idr(cdf: np.ndarray, thresholds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    CRPS of IDR-fitted CDFs. `cdf` is (n, T), non-decreasing along axis 1,
    evaluated at `thresholds` (length T, ascending). Integrates exactly on the
    threshold grid, with the tails handled by clamping F to {0,1} outside.
    """
    y = np.asarray(y, float)
    t = np.asarray(thresholds, float)
    lo, hi = min(t[0], y.min()), max(t[-1], y.max())
    grid = np.concatenate([[lo], t, [hi]])
    widths = np.diff(grid)                       # (T+1,)
    F = np.concatenate([np.zeros((cdf.shape[0], 1)), cdf], axis=1)  # (n, T+1)
    ind = (y[:, None] <= grid[:-1][None, :]).astype(float)
    return np.sum((F - ind) ** 2 * widths[None, :], axis=1)


# --------------------------------------------------------------------------
# Isotonic regression on a partial order, exactly, by min-cut D&C
# --------------------------------------------------------------------------
def _dominance_edges(fc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Edges (i, j) of the componentwise/stochastic order: forecast i <=_st j iff
    every quantile of i is <= the corresponding quantile of j.

    Returns the full comparability relation (transitive edges included -- they
    are redundant but harmless, and cheaper than computing a covering relation).
    """
    fc = np.sort(np.asarray(fc, float), axis=1)
    n = fc.shape[0]
    src, dst = [], []
    # chunked to keep the (n, n, m) comparison from blowing up memory
    step = max(1, int(2e7 // max(1, n * fc.shape[1])))
    for a in range(0, n, step):
        b = min(n, a + step)
        le = np.all(fc[a:b, None, :] <= fc[None, :, :], axis=2)  # (b-a, n)
        np.fill_diagonal(le[:, a:b], False)
        ii, jj = np.nonzero(le)
        src.append(ii + a)
        dst.append(jj)
    if not src:
        return np.array([], int), np.array([], int)
    return np.concatenate(src), np.concatenate(dst)


def _min_closure(weights: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Minimal maximum-weight closed set S: i in S and (i -> j) implies j in S,
    maximising sum(weights[S]). Solved as a min-cut; returns a boolean mask.
    """
    n = weights.size
    S, T = n, n + 1
    w = np.rint(weights * _CAP).astype(np.int64)
    pos, neg = np.nonzero(w > 0)[0], np.nonzero(w < 0)[0]
    if pos.size == 0:
        return np.zeros(n, bool)

    # An "infinite" capacity only has to exceed the total cuttable capacity.
    big = int(w[pos].sum()) + 1
    if big >= 2**31 - 1:                       # keep everything int32-safe
        scale = (2**31 - 2) / float(big)
        w = np.rint(w * scale).astype(np.int64)
        pos, neg = np.nonzero(w > 0)[0], np.nonzero(w < 0)[0]
        big = int(w[pos].sum()) + 1

    rows = np.concatenate([np.full(pos.size, S), neg, src])
    cols = np.concatenate([pos, np.full(neg.size, T), dst])
    vals = np.concatenate([w[pos], -w[neg], np.full(src.size, big)])

    graph = csr_matrix((vals.astype(np.int32),
                        (rows.astype(np.int32), cols.astype(np.int32))),
                       shape=(n + 2, n + 2))
    res = maximum_flow(graph, S, T)

    # Residual capacities. res.flow is antisymmetric, so (graph - flow) already
    # supplies the reverse-arc residuals; keep only strictly positive entries.
    residual = (graph - res.flow).tocoo()
    keep = residual.data > 0
    residual = csr_matrix((residual.data[keep].astype(np.int32),
                           (residual.row[keep], residual.col[keep])),
                          shape=graph.shape)
    reachable = breadth_first_order(residual, S, return_predecessors=False)
    mask = np.zeros(n + 2, bool)
    mask[reachable] = True
    return mask[:n]


def _isotonic_dag(a: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Exact L2 isotonic regression on a DAG: minimise sum (theta_i - a_i)^2
    subject to theta_i <= theta_j for every edge (i -> j).
    Classic maximum-closure divide and conquer; terminates exactly.
    """
    theta = np.empty_like(a, dtype=float)
    stack = [np.arange(a.size)]
    while stack:
        idx = stack.pop()
        lam = a[idx].mean()
        if idx.size == 1:
            theta[idx] = lam
            continue
        pos = np.full(a.size, -1, np.int64)
        pos[idx] = np.arange(idx.size)
        keep = (pos[src] >= 0) & (pos[dst] >= 0)
        S = _min_closure(a[idx] - lam, pos[src[keep]], pos[dst[keep]])
        if S.all() or not S.any():
            theta[idx] = lam
        else:
            stack.append(idx[S])
            stack.append(idx[~S])
    return theta


# --------------------------------------------------------------------------
# IDR
# --------------------------------------------------------------------------
def _idr_cdf(fc: np.ndarray, y: np.ndarray, thresholds: np.ndarray,
             order: str) -> np.ndarray:

    n = fc.shape[0]
    cdf = np.empty((n, thresholds.size), float)

    if order == "median":
        key = np.median(fc, axis=1)

        uniq, inv = np.unique(key, return_inverse=True)
        w = np.bincount(inv).astype(float)
        for t, z in enumerate(thresholds):
            b = (y <= z).astype(float)
            g = np.bincount(inv, weights=b) / w          # group means
            fitted = isotonic_regression(g[::-1], weights=w[::-1],
                                         increasing=True).x[::-1]
            cdf[:, t] = fitted[inv]
    elif order == "stochastic":
        src, dst = _dominance_edges(fc)
        # theta antitone w.r.t. (i <=_st j) is theta isotone w.r.t. reversed edges
        for t, z in enumerate(thresholds):
            b = (y <= z).astype(float)
            cdf[:, t] = _isotonic_dag(b, dst, src)
    else:
        raise ValueError(f"unknown order: {order!r}")

    # enforce monotonicity in z (holds in exact arithmetic; guards rounding)
    return np.maximum.accumulate(np.clip(cdf, 0.0, 1.0), axis=1)


def _predict_idr(fc_tr: np.ndarray, cdf_tr: np.ndarray, fc_te: np.ndarray,
                 order: str) -> np.ndarray:

    tr = np.sort(np.asarray(fc_tr, float), axis=1)
    te = np.sort(np.asarray(fc_te, float), axis=1)

    if order == "median":
        k_tr, k_te = np.median(tr, axis=1), np.median(te, axis=1)
        o = np.argsort(k_tr, kind="mergesort")
        k_tr, cdf = k_tr[o], cdf_tr[o]
        pos = np.searchsorted(k_tr, k_te)
        lo = np.clip(pos - 1, 0, len(k_tr) - 1)
        hi = np.clip(pos, 0, len(k_tr) - 1)
        span = k_tr[hi] - k_tr[lo]
        w = np.where(span > 0, (k_te - k_tr[lo]) / np.where(span > 0, span, 1.0), 0.0)
        w = np.clip(w, 0.0, 1.0)[:, None]
        out = (1.0 - w) * cdf[lo] + w * cdf[hi]
    else:
        out = np.empty((te.shape[0], cdf_tr.shape[1]), float)
        for r in range(te.shape[0]):
            below = np.all(tr <= te[r], axis=1)
            above = np.all(te[r] <= tr, axis=1)
            up = cdf_tr[below].min(axis=0) if below.any() else None
            lo_ = cdf_tr[above].max(axis=0) if above.any() else None
            if up is None and lo_ is None:
                out[r] = cdf_tr.mean(axis=0)
            elif up is None:
                out[r] = lo_
            elif lo_ is None:
                out[r] = up
            else:
                out[r] = 0.5 * (np.minimum(lo_, up) + np.maximum(lo_, up))

    return np.maximum.accumulate(np.clip(out, 0.0, 1.0), axis=1)


# --------------------------------------------------------------------------
# Decomposition
# --------------------------------------------------------------------------
class Decomposition(dict):
    """dict with attribute access: .crps .mcb .dsc .unc"""
    __getattr__ = dict.__getitem__

    def __repr__(self):
        return (f"CRPS={self['crps']:.4f}  MCB={self['mcb']:+.4f}  "
                f"DSC={self['dsc']:+.4f}  UNC={self['unc']:.4f}")


def _marginal_cdf(y: np.ndarray, thresholds: np.ndarray, n: int) -> np.ndarray:
    """Empirical marginal (climatological) CDF, evaluated on the shared grid."""
    ys = np.sort(np.asarray(y, float))
    row = np.searchsorted(ys, thresholds, side="right") / ys.size
    return np.tile(row, (n, 1))


def _thresholds(y: np.ndarray, max_thresholds: int | None) -> np.ndarray:
    u = np.unique(y)
    if max_thresholds is None or u.size <= max_thresholds:
        return u
    q = np.linspace(0, 1, max_thresholds)
    return np.unique(np.quantile(u, q))


def decompose(fc: np.ndarray, y: np.ndarray, order: str = "stochastic",
              max_thresholds: int | None = 400) -> Decomposition:

    fc = np.asarray(fc, float)
    y = np.asarray(y, float)
    t = _thresholds(y, max_thresholds)

    crps = crps_ensemble(fc, y).mean()
    cdf = _idr_cdf(fc, y, t, order)
    crps_iso = _crps_of_idr(cdf, t, y).mean()
    crps_mg = _crps_of_idr(_marginal_cdf(y, t, y.size), t, y).mean()

    return Decomposition(crps=crps, mcb=crps - crps_iso,
                         dsc=crps_mg - crps_iso, unc=crps_mg,
                         crps_iso=crps_iso, order=order)


def decompose_cross_fit(fc: np.ndarray, y: np.ndarray, order: str = "stochastic",
                        k: int = 5, seed: int = 0,
                        max_thresholds: int | None = 400,
                        groups: np.ndarray | None = None) -> Decomposition:

    fc = np.asarray(fc, float)
    y = np.asarray(y, float)
    n = y.size
    rng = np.random.default_rng(seed)

    if groups is None:
        fold = rng.integers(0, k, n)
    else:
        g = np.unique(groups)
        gf = dict(zip(g, rng.integers(0, k, g.size)))
        fold = np.array([gf[v] for v in groups])

    t = _thresholds(y, max_thresholds)
    iso_scores = np.empty(n, float)

    for f in range(k):
        te = fold == f
        tr = ~te
        if not te.any() or not tr.any():
            continue
        cdf_tr = _idr_cdf(fc[tr], y[tr], t, order)
        cdf_te = _predict_idr(fc[tr], cdf_tr, fc[te], order)
        iso_scores[te] = _crps_of_idr(cdf_te, t, y[te])

    crps = crps_ensemble(fc, y).mean()
    crps_mg = _crps_of_idr(_marginal_cdf(y, t, y.size), t, y).mean()
    crps_iso = iso_scores.mean()
    return Decomposition(crps=crps, mcb=crps - crps_iso,
                         dsc=crps_mg - crps_iso, unc=crps_mg,
                         crps_iso=crps_iso, order=order, k=k)


def block_bootstrap(fc: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    b: int = 1000, order: str = "stochastic", seed: int = 0,
                    cross_fit: bool = False, level: float = 0.95,
                    max_thresholds: int | None = 400) -> dict:

    fc = np.asarray(fc, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    t = _thresholds(y, max_thresholds)

    per_crps = crps_ensemble(fc, y)
    cdf = _idr_cdf(fc, y, t, order)
    per_iso = _crps_of_idr(cdf, t, y)
    per_mg = _crps_of_idr(_marginal_cdf(y, t, y.size), t, y)

    uniq = np.unique(groups)
    where = {g: np.nonzero(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)

    draws = {"crps": [], "mcb": [], "dsc": [], "unc": []}
    for _ in range(b):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([where[g] for g in pick])
        c, i_, m = per_crps[idx].mean(), per_iso[idx].mean(), per_mg[idx].mean()
        draws["crps"].append(c)
        draws["mcb"].append(c - i_)
        draws["dsc"].append(m - i_)
        draws["unc"].append(m)

    alpha = (1.0 - level) / 2.0
    return {k: (float(np.quantile(v, alpha)), float(np.quantile(v, 1 - alpha)))
            for k, v in draws.items()}
