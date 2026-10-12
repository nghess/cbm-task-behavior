"""Stratified two-group comparisons used throughout the analysis.

Every question here has the same shape: does outcome y differ between two kinds
of trial (flip_state 0 vs 1, or odor vs no-odor) once trials are compared only
within strata that share the nuisance variables (mouse, target cell, where the
mouse was at trial start, where the previous target was)? `stratified_diff` is
that estimate, `permutation_p` its trial-level test, and `per_mouse` the
across-animal summary.
"""
import numpy as np
import pandas as pd
from scipy import stats as sps


def _strata(d, keys):
    return d.groupby(keys, sort=False).ngroup().to_numpy()


def _diff(y, label, strata, n_strata):
    """Precision-weighted mean over strata of mean(y | label) - mean(y | ~label).
    Strata missing either kind of trial carry no information and drop out."""
    n1 = np.bincount(strata, weights=label, minlength=n_strata)
    n0 = np.bincount(strata, weights=~label, minlength=n_strata)
    s1 = np.bincount(strata, weights=y * label, minlength=n_strata)
    s0 = np.bincount(strata, weights=y * ~label, minlength=n_strata)
    ok = (n1 > 0) & (n0 > 0)
    if not ok.any():
        return np.nan
    w = 1 / (1 / n1[ok] + 1 / n0[ok])
    return float(((s1[ok] / n1[ok] - s0[ok] / n0[ok]) * w).sum() / w.sum())


def stratified_diff(d, y, label, keys):
    """d[label] is boolean; returns the stratified True-minus-False difference in d[y]."""
    d = d.dropna(subset=[y])
    strata = _strata(d, keys)
    return _diff(d[y].to_numpy(float), d[label].to_numpy(bool), strata, strata.max() + 1)


def permutation_p(d, y, label, keys, n=5000, seed=0):
    """Two-sided p for stratified_diff, shuffling the label within strata."""
    d = d.dropna(subset=[y])
    strata = _strata(d, keys)
    order = np.argsort(strata, kind='stable')
    strata, yv, lab = strata[order], d[y].to_numpy(float)[order], d[label].to_numpy(bool)[order]
    n_strata = strata.max() + 1
    obs = _diff(yv, lab, strata, n_strata)
    rng = np.random.default_rng(seed)
    null = np.empty(n)
    for i in range(n):
        shuffled = lab[np.lexsort((rng.random(len(lab)), strata))]
        null[i] = _diff(yv, shuffled, strata, n_strata)
    return (np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1)


def per_mouse(d, y, label, keys):
    """stratified_diff computed separately for each mouse."""
    return pd.Series({m: stratified_diff(g, y, label, keys) for m, g in d.groupby('mouse')}, name=y)


def summarize(d, y, label, keys, n_perm=5000):
    """One row: pooled stratified effect, permutation p, and the across-mouse picture."""
    mice = per_mouse(d, y, label, keys).dropna()
    lo, hi = (sps.t.interval(0.95, len(mice) - 1, mice.mean(), sps.sem(mice))
              if len(mice) > 1 else (np.nan, np.nan))
    effect = stratified_diff(d, y, label, ['mouse'] + keys)
    return dict(effect=effect, p_perm=permutation_p(d, y, label, ['mouse'] + keys, n_perm),
                mouse_mean=mice.mean(), ci_lo=lo, ci_hi=hi,
                mice_same_sign=f'{int((np.sign(mice) == np.sign(effect)).sum())}/{len(mice)}',
                n_trials=int(d[y].notna().sum()))


def hazard(duration, event, edges, marks=None):
    """Events per unit time at risk in each bin of elapsed time (life-table hazard).

    duration: how long each spell lasted; event: whether it ended in the event of
    interest (False = censored). With `marks` - one array of event times per spell -
    the rate of those marked events is returned instead, each spell being at risk
    for its whole duration.
    """
    duration = np.asarray(duration, float)
    exposure = np.clip(duration[:, None], edges[:-1], edges[1:]) - edges[:-1]
    if marks is None:
        times = duration[np.asarray(event, bool)]
    else:
        times = np.concatenate([np.asarray(m, float) for m in marks]) if len(marks) else np.array([])
    counts = np.histogram(times, bins=edges)[0]
    exposure = exposure.sum(axis=0)
    return np.where(exposure > 0, counts / np.maximum(exposure, 1e-12), np.nan)


def km_survival(duration, event, grid):
    """Kaplan-Meier P(spell still going at each time in grid), treating ~event as censoring."""
    duration, event = np.asarray(duration, float), np.asarray(event, bool)
    order = np.argsort(duration, kind='stable')
    duration, event = duration[order], event[order]
    at_risk = len(duration) - np.arange(len(duration))
    surv = np.cumprod(1 - event / at_risk)
    idx = np.searchsorted(duration, grid, side='right') - 1
    return np.where(idx >= 0, surv[np.maximum(idx, 0)], 1.0)


def km_median(duration, event):
    """Kaplan-Meier median duration (NaN if survival never reaches one half)."""
    duration, event = np.asarray(duration, float), np.asarray(event, bool)
    order = np.argsort(duration, kind='stable')
    surv = np.cumprod(1 - event[order] / (len(duration) - np.arange(len(duration))))
    below = np.flatnonzero(surv <= 0.5)
    return float(duration[order][below[0]]) if len(below) else np.nan


def logistic_fit(X, y, max_iter=100):
    """Unpenalized logistic regression by Newton's method. Returns (intercept, coefficients)."""
    X = np.c_[np.ones(len(X)), np.asarray(X, float)]
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        prob = 1 / (1 + np.exp(-X @ beta))
        step = np.linalg.solve(X.T @ (X * (prob * (1 - prob))[:, None]), X.T @ (y - prob))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta[0], beta[1:]
