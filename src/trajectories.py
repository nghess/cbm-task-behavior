"""Describe the shape of every search path, and how much ground each session covers.

trials.py and visits.py follow the point the task tracked online (the nose, as
Bonsai saw it), because that is what decides when a cell has been entered. That
point is coarse - it sticks and jumps - so the shape of a path is measured here on
the SLEAP body centroid instead, lightly smoothed. Cells checked are still counted
with the task's own point.

Two tables:
  trajectories - one row per searched trial and phase of the search (how long it has
                 been going, SEARCH_PHASES): distance, pauses, turning, straightness,
                 distance along the walls, new cells checked. Sum over phases for
                 the whole search.
  sessions     - one row per session: its length and the distance covered in all of
                 it, searching or not.

Turning and straightness are computed on the path resampled at equal steps of
distance, so they describe its shape and do not depend on how fast it was run.

Run as a script from the directory above: `python -m src.trajectories` for every
experiment, `python -m src.trajectories vision` for one.
"""
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from . import config as cfg
from .trials import USECOLS, analysed, discover_sessions
from .visits import cell_of

BODY = [f'{cfg.TRAJ_POS}_x', f'{cfg.TRAJ_POS}_y']


def shape(x, y, dist, elapsed, n_phases):
    """Turning and straightness of one search path, as per-phase sums and counts so
    that searches can be pooled. Points are taken every RESAMPLE_STEP px of distance
    travelled, so the result does not depend on speed."""
    grid = np.arange(0, dist[-1], cfg.RESAMPLE_STEP)
    px, py = np.interp(grid, dist, x), np.interp(grid, dist, y)
    phase = np.searchsorted(cfg.SEARCH_PHASES, np.interp(grid, dist, elapsed), side='right') - 1
    count = lambda at, weights=None: np.bincount(at, weights=weights, minlength=n_phases)[:n_phases]
    heading = np.arctan2(np.diff(py), np.diff(px))
    turn = np.abs((np.diff(heading) + np.pi) % (2 * np.pi) - np.pi)
    # straightness: how far the animal ends up from where it began, over stretches of path
    # STRAIGHT_SCALE long, as a fraction of that length (1 = a straight line)
    k = cfg.STRAIGHT_SCALE // cfg.RESAMPLE_STEP
    net = np.hypot(np.diff(px[::k]), np.diff(py[::k])) / cfg.STRAIGHT_SCALE
    return dict(turn_sum=count(phase[1:-1], turn), turn_n=count(phase[1:-1]),
                straight_sum=count(phase[::k][:-1], net), straight_n=count(phase[::k][:-1]))


def session_trajectories(path):
    df = pd.read_csv(path, usecols=USECOLS + BODY)
    ts = df['bonsai_ts'].to_numpy(float) / 1000
    x = gaussian_filter1d(df[BODY[0]].to_numpy(float), cfg.SMOOTH_SIGMA)
    y = gaussian_filter1d(df[BODY[1]].to_numpy(float), cfg.SMOOTH_SIGMA)
    step = np.r_[0, np.hypot(np.diff(x), np.diff(y))]
    dt = np.r_[0, np.diff(ts)]
    moving = step > cfg.MOVE_SPEED * dt
    # within one cell of a wall: the ring of the grid that never holds a target
    wall = ((x < cfg.CELL_W) | (x >= cfg.FRAME_W - cfg.CELL_W) |
            (y < cfg.CELL_H) | (y >= cfg.FRAME_H - cfg.CELL_H))
    cell = cell_of(df[f'{cfg.POS}_x'].to_numpy(float), df[f'{cfg.POS}_y'].to_numpy(float))
    no_target = df['target_cell'].isna().to_numpy()
    n_phases = len(cfg.SEARCH_PHASES)

    rows = []
    for trial, idx in df.groupby('trial_number').indices.items():
        i0, i1 = idx[0], idx[-1] + 1
        hits = np.flatnonzero(no_target[i0:i1])
        if trial == 0 or not len(hits) or no_target[i0]:        # same trials as trials.py keeps
            continue
        j = i0 + hits[0]                                        # frame the target was found
        elapsed = ts[i0:j + 1] - ts[i0]
        phase = np.searchsorted(cfg.SEARCH_PHASES, elapsed, side='right') - 1
        sl = slice(i0 + 1, j + 1)                               # steps taken during the search
        count = lambda weights, at=phase[1:]: np.bincount(at, weights=weights, minlength=n_phases)
        cells, first_seen = np.unique(cell[i0:j + 1], return_index=True)
        per_phase = pd.DataFrame(dict(
            search_s=count(dt[sl]), path_px=count(step[sl]),
            moving_s=count(dt[sl] * moving[sl]), wall_path_px=count(step[sl] * wall[sl]),
            cells_new=count(None, phase[first_seen[cells >= 0]]),
            **shape(x[i0:j + 1], y[i0:j + 1], np.cumsum(step[i0:j + 1]) - step[i0], elapsed, n_phases)))
        per_phase.insert(0, 'phase', range(n_phases))
        per_phase.insert(0, 'flip', int(df['flip_state'].iloc[i0]))
        per_phase.insert(0, 'trial', int(trial))
        rows.append(per_phase[per_phase.search_s > 0])
    whole = dict(duration_s=ts[-1] - ts[0], total_path_px=step.sum(), total_moving_s=dt[moving].sum())
    return pd.concat(rows, ignore_index=True), whole


def build_trajectories(experiment='odor', verbose=True):
    trials, sessions = [], []
    for s in discover_sessions(experiment):
        t, whole = session_trajectories(s['path'])
        keys = {k: s[k] for k in ('mouse', 'session', 'session_type', 'session_num')}
        for key in reversed(list(keys)):
            t.insert(0, key, keys[key])
        trials.append(t)
        sessions.append({**keys, **whole})
        if verbose:
            print(f"{s['mouse']}/{s['session']}: {t['trial'].nunique()} searches, {whole['total_path_px'] / 1e6:.2f} Mpx")
    trials, sessions = pd.concat(trials, ignore_index=True), pd.DataFrame(sessions)
    cfg.OUT_DIR.mkdir(exist_ok=True)
    trials.to_csv(cfg.TRAJECTORIES_CSV[experiment], index=False)
    sessions.to_csv(cfg.SESSIONS_CSV[experiment], index=False)
    return trials, sessions


def _load(table, experiment):
    d = pd.read_csv(table[experiment], dtype={'mouse': str})
    d = analysed(d, experiment)
    d.insert(0, 'experiment', experiment)
    return d


def load_trajectories(experiment='odor'):
    """One row per searched trial and search phase, for the analysed sessions."""
    return _load(cfg.TRAJECTORIES_CSV, experiment)


def load_sessions(experiment='odor'):
    """One row per analysed session."""
    return _load(cfg.SESSIONS_CSV, experiment)


if __name__ == '__main__':
    for experiment in sys.argv[1:] or cfg.EVENTS_DIRS:
        build_trajectories(experiment)
