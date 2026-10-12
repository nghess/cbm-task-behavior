"""Reduce every session's events.csv to one row per trial.

A trial starts on the first frame of a new trial_number (end of the ITI: the
flip_state switches and, on flip_state 0/1, the cue comes on) and its search phase
ends on the frame target_cell goes NaN (the task consumed the active target:
click, and in cbm-odor the odor goes off). Everything after that - walking to a port, drinking, the ITI -
is not search. events.csv's reward_state is not used for this: it is rebuilt as
"click until the next poke onset", so whenever a poke goes unrecorded it stays
True into the following trial.

Run as a script to rebuild the per-trial tables: from the directory above,
`python -m src.trials` for every experiment, `python -m src.trials vision`
for one.
"""
import re
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from . import config as cfg

USECOLS = ['trial_number', 'bonsai_ts', f'{cfg.POS}_x', f'{cfg.POS}_y',
           'target_cell', 'flip_state']


def zone_of(y):
    """'low' / 'mid' / 'high' long-axis zone for y (scalar or array)."""
    y = np.asarray(y, dtype=float)
    return np.where(y < cfg.LOW_MAX_Y, 'low', np.where(y >= cfg.HIGH_MIN_Y, 'high', 'mid'))


def cell_rect(cell):
    """(x0, y0, x1, y1) of a target_cell id."""
    col, row = cell % cfg.INNER_COLS, cell // cfg.INNER_COLS
    x0, y0 = (col + 1) * cfg.CELL_W, (row + 1) * cfg.CELL_H
    return x0, y0, x0 + cfg.CELL_W, y0 + cfg.CELL_H


def dist_to_cell(x, y, cell):
    """Shortest distance (px) from a point to a target cell; 0 if inside it."""
    x0, y0, x1, y1 = cell_rect(cell)
    return float(np.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1)))


def discover_sessions(experiment='odor'):
    sessions = {}
    for events_dir in cfg.EVENTS_DIRS[experiment]:
        for path in sorted(events_dir.glob('*/*/events.csv')):
            mouse, session = path.parts[-3], path.parts[-2]
            # a trailing 'b' marks a session number that was run a second time
            m = re.fullmatch(r'([a-z]+)(\d+)b?', session)
            sessions.setdefault((mouse, session), dict(
                mouse=mouse, session=session, session_type=m.group(1),
                session_num=int(m.group(2)), path=path))
    return [sessions[k] for k in sorted(sessions)]


def session_trials(path):
    df = pd.read_csv(path, usecols=USECOLS)
    x = df[f'{cfg.POS}_x'].to_numpy(float)
    y = df[f'{cfg.POS}_y'].to_numpy(float)
    xs, ys = gaussian_filter1d(x, cfg.SMOOTH_SIGMA), gaussian_filter1d(y, cfg.SMOOTH_SIGMA)
    step = np.r_[0, np.hypot(np.diff(xs), np.diff(ys))]
    ts = df['bonsai_ts'].to_numpy(float) / 1000
    no_target = df['target_cell'].isna().to_numpy()
    zone = zone_of(y)

    rows = []
    for trial, idx in df.groupby('trial_number').indices.items():
        i0, i1 = idx[0], idx[-1] + 1
        target = df['target_cell'].iloc[i0:i1].dropna()
        row = dict(trial=int(trial), flip=int(df['flip_state'].iloc[i0]),
                   target_cell=int(target.iloc[0]) if len(target) else -1,
                   t_start_s=ts[i0] - ts[0], start_x=x[i0], start_y=y[i0],
                   start_zone=zone[i0])
        hits = np.flatnonzero(no_target[i0:i1])
        # trial 0 opens with a free reward, so it is not a search
        row['found'] = bool(len(hits)) and trial != 0 and row['target_cell'] >= 0
        if row['found']:
            j = i0 + hits[0]                              # frame the target was encountered
            sl = slice(i0, j + 1)
            z, t = zone[sl], ts[sl] - ts[i0]
            row.update(search_s=ts[j] - ts[i0], path_px=step[i0 + 1:j + 1].sum(),
                       straight_px=dist_to_cell(x[i0], y[i0], row['target_cell']),
                       found_err_px=dist_to_cell(x[j], y[j], row['target_cell']))
            # first end zone entered from outside it (so a mouse that starts in an
            # end zone has to leave it and come back, or reach the other one)
            entered = np.flatnonzero((z[1:] != 'mid') & (z[1:] != z[:-1])) + 1
            row['first_entry'] = z[entered[0]] if len(entered) else 'none'
            # split the search at the first frame in the target's end zone: the
            # approach (getting to the right end) and the local search once there.
            # Undefined for the few no-odor targets in the neutral middle row.
            tz = str(zone_of((row['target_cell'] // cfg.INNER_COLS + 1.5) * cfg.CELL_H))
            if tz != 'mid':
                r = int(np.flatnonzero(z == tz)[0])       # always exists: the target is in tz
                wrong = z != 'mid'
                wrong &= z != tz
                row.update(reach_s=t[r], reach_path_px=step[i0 + 1:i0 + r + 1].sum(),
                           local_s=t[-1] - t[r], local_path_px=step[i0 + r + 1:j + 1].sum(),
                           wrong_end_first=bool(wrong[:r].any()),
                           left_after_reach=bool(wrong[r:].any()))
        rows.append(row)

    out = pd.DataFrame(rows).sort_values('trial').reset_index(drop=True)
    out['target_zone'] = np.where(out['target_cell'] < 0, 'none',
                                  zone_of((out['target_cell'] // cfg.INNER_COLS + 1.5) * cfg.CELL_H))
    # previous trial's target, only where the previous trial really is trial-1
    consecutive = out['trial'].diff() == 1
    out['prev_target_zone'] = out['target_zone'].shift().where(consecutive, 'none')
    out['prev_flip'] = out['flip'].shift().where(consecutive, -1).astype(int)
    return out


def build_trials(experiment='odor', verbose=True):
    frames = []
    for s in discover_sessions(experiment):
        t = session_trials(s['path'])
        for key in ('session_num', 'session_type', 'session', 'mouse'):
            t.insert(0, key, s[key])
        frames.append(t)
        if verbose:
            print(f"{s['mouse']}/{s['session']}: {int(t['found'].sum())} searched trials")
    trials = pd.concat(frames, ignore_index=True)
    cfg.OUT_DIR.mkdir(exist_ok=True)
    trials.to_csv(cfg.TRIALS_CSV[experiment], index=False)
    return trials


def analysed(d, experiment):
    """Rows of d from the sessions that enter the analysis (see config), with
    session_num replaced by the session's position in that mouse's sequence."""
    d = d[d['session_type'].isin(cfg.SESSION_TYPES[experiment]) &
          (d['session_num'] >= cfg.FIRST_SESSION[experiment])]
    order = d[['mouse', 'session_num', 'session']].drop_duplicates().sort_values(['mouse', 'session_num', 'session'])
    order['position'] = order.groupby('mouse').cumcount() + 1
    d = d.merge(order[order['position'] <= cfg.SESSIONS_PER_MOUSE], on=['mouse', 'session_num', 'session'])
    return d.assign(session_num=d['position']).drop(columns='position')


def load_trials(experiment='odor'):
    """Searched trials of the analysed sessions, with the derived columns the analysis uses."""
    t = pd.read_csv(cfg.TRIALS_CSV[experiment], dtype={'mouse': str})
    t = analysed(t[t['found']], experiment)
    t['cue'] = t['flip'] != 2            # odor (cbm-odor) or visual stimulus (cbm-vision) present
    t['odor'] = t['cue']                 # name used by odor_search_strategy.ipynb
    # where the target is relative to where the mouse was when the trial began
    t['geometry'] = np.where(t['target_zone'] == 'mid', 'target in middle',
                    np.where(t['start_zone'] == 'mid', 'from middle',
                    np.where(t['start_zone'] == t['target_zone'], 'same end', 'opposite end')))
    # path lengths are heavy-tailed; compare them as log2 ratios
    for col in ('path_px', 'reach_path_px', 'local_path_px'):
        t['log2_' + col] = np.log2(t[col] + 50)
    for col in ('left_after_reach', 'wrong_end_first'):
        t[col] = t[col].astype(float)
    return t


if __name__ == '__main__':
    for experiment in sys.argv[1:] or cfg.EVENTS_DIRS:
        build_trials(experiment)
