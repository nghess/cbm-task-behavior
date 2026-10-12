"""Break every search into visits to the two ends of the arena ("patches").

A visit to an end starts on the first frame the nose is in that end zone and
lasts until the mouse either finds the target or visits the opposite end. Steps
out into the neutral middle row that come back to the same end are part of the
visit, and so are dips into the opposite end shallower than MIN_VISIT_DEPTH. A visit that ends with the mouse entering the other end is a patch
switch; its residence time runs to the last frame in the patch it gave up on,
and the crossing that follows is the travel.

What a patch has to offer is unchecked target cells: the task ends the search the
moment the nose enters the target cell, so every cell entered without that
happening is known to be empty. Each visit therefore also records when the nose
first entered each of the patch's 12 target cells.

Run as a script to rebuild the per-visit tables: from the directory above,
`python -m src.visits` for every experiment, `python -m src.visits vision`
for one.
"""
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from . import config as cfg
from .trials import USECOLS, analysed, discover_sessions, zone_of

OTHER = {'low': 'high', 'high': 'low'}
PATCH_CELLS = 4 * cfg.INNER_COLS     # target cells in each end


def cell_of(x, y):
    """target_cell id under each (x, y), -1 on the border ring."""
    col, row = x.astype(int) // cfg.CELL_W - 1, y.astype(int) // cfg.CELL_H - 1
    inside = (col >= 0) & (col < cfg.INNER_COLS) & (row >= 0) & (row < cfg.INNER_ROWS)
    return np.where(inside, row * cfg.INNER_COLS + col, -1)


def _visit_bounds(z):
    """(first, last) frame of each run of end-zone frames spent at one end, ignoring
    steps into the middle row in between."""
    ends = np.flatnonzero(z != 'mid')
    breaks = np.flatnonzero(z[ends][1:] != z[ends][:-1]) + 1
    return ends[np.r_[0, breaks]], ends[np.r_[breaks - 1, len(ends) - 1]]


def trial_visits(z, ts, step, y, cell):
    """Visits within one search. All arguments are per-frame arrays over the search."""
    last = len(z) - 1
    depth = np.where(z == 'low', cfg.LOW_MAX_Y - y, np.where(z == 'high', y - cfg.HIGH_MIN_Y, 0))
    # a dip into an end that never gets past its first row of cells is not a visit:
    # treat it as part of the middle row, so the visits either side of it join up.
    # (Unless the search ended there - the target was in that first row.)
    z = z.copy()
    if (z != 'mid').any():
        for a, b in zip(*_visit_bounds(z)):
            if depth[a:b + 1].max() < cfg.MIN_VISIT_DEPTH and b != last:
                z[a:b + 1] = 'mid'
    if not (z != 'mid').any():
        return []
    firsts, lasts = _visit_bounds(z)
    # first frame each target cell was entered in this search, and which end it belongs to
    cells, first_seen = np.unique(cell, return_index=True)
    cells, first_seen = cells[cells >= 0], first_seen[cells >= 0]
    row = cells // cfg.INNER_COLS
    cell_end = np.where(row < 4, 'low', np.where(row > 4, 'high', 'mid'))

    rows = []
    for k, (a, b) in enumerate(zip(firsts, lasts)):
        patch = z[a]
        switched = k < len(firsts) - 1
        end = b if switched else last            # the final visit runs to the find
        inside = z[a:end + 1] == patch
        here = cell_end == patch
        new = np.sort(first_seen[here & (first_seen >= a) & (first_seen <= end)])
        rows.append(dict(
            visit=k, patch=patch, from_trial_start=bool(a == 0),
            outcome='switch' if switched else 'found' if z[last] == patch else 'found in middle',
            enter_s=ts[a] - ts[0], residence_s=ts[end] - ts[a],
            in_patch_s=float(inside.sum()) / cfg.FPS, path_px=step[a + 1:end + 1].sum(),
            travel_s=ts[firsts[k + 1]] - ts[b] if switched else np.nan,
            max_depth_px=float(depth[a:end + 1][inside].max()),
            mid_excursions=int((np.diff(inside.astype(int)) == -1).sum()),
            cells_before=int((here & (first_seen < a)).sum()), cells_new=len(new),
            cells_entered=int(np.isin(cells[here], cell[a:end + 1]).sum()),
            new_cell_s=' '.join(f'{v:.2f}' for v in ts[new] - ts[a])))
    return rows


def session_visits(path):
    df = pd.read_csv(path, usecols=USECOLS)
    x = df[f'{cfg.POS}_x'].to_numpy(float)
    y = df[f'{cfg.POS}_y'].to_numpy(float)
    xs, ys = gaussian_filter1d(x, cfg.SMOOTH_SIGMA), gaussian_filter1d(y, cfg.SMOOTH_SIGMA)
    step = np.r_[0, np.hypot(np.diff(xs), np.diff(ys))]
    ts = df['bonsai_ts'].to_numpy(float) / 1000
    no_target = df['target_cell'].isna().to_numpy()
    zone, cell = zone_of(y), cell_of(x, y)

    rows = []
    for trial, idx in df.groupby('trial_number').indices.items():
        i0, i1 = idx[0], idx[-1] + 1
        target = df['target_cell'].iloc[i0:i1].dropna()
        hits = np.flatnonzero(no_target[i0:i1])
        if trial == 0 or not len(hits) or not len(target):     # same trials as trials.py keeps
            continue
        sl = slice(i0, i0 + hits[0] + 1)
        flip, target = int(df['flip_state'].iloc[i0]), int(target.iloc[0])
        target_zone = str(zone_of((target // cfg.INNER_COLS + 1.5) * cfg.CELL_H))
        for row in trial_visits(zone[sl], ts[sl], step[sl], y[sl], cell[sl]):
            rows.append(dict(trial=int(trial), flip=flip, target_cell=target, target_zone=target_zone, **row))
    return pd.DataFrame(rows)


def build_visits(experiment='odor', verbose=True):
    frames = []
    for s in discover_sessions(experiment):
        v = session_visits(s['path'])
        for key in ('session_num', 'session_type', 'session', 'mouse'):
            v.insert(0, key, s[key])
        frames.append(v)
        if verbose:
            print(f"{s['mouse']}/{s['session']}: {len(v)} visits")
    visits = pd.concat(frames, ignore_index=True)
    cfg.OUT_DIR.mkdir(exist_ok=True)
    visits.to_csv(cfg.VISITS_CSV[experiment], index=False)
    return visits


def load_visits(experiment='odor'):
    v = pd.read_csv(cfg.VISITS_CSV[experiment], dtype={'mouse': str}, keep_default_na=False,
                    na_values={'travel_s': ['']})
    v = analysed(v, experiment)
    v['experiment'] = experiment
    v['cue'] = v['flip'] != 2
    v['switched'] = v['outcome'] == 'switch'
    v['target_here'] = v['target_zone'] == v['patch']
    # what the cue says about this patch: it is the cued end, the other end, or there is no cue
    v['cue_here'] = np.where(~v['cue'], 'no cue',
                             np.where(v['flip'].map(cfg.CUE_ZONE) == v['patch'], 'cued end', 'uncued end'))
    v['new_cell_s'] = [np.array(s.split(), dtype=float) for s in v['new_cell_s']]
    return v


if __name__ == '__main__':
    for experiment in sys.argv[1:] or cfg.EVENTS_DIRS:
        build_visits(experiment)
