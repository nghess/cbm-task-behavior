"""Where each mouse spends its time: one occupancy map per session.

The arena is cut into OCC_BINS bins (a quarter of a target cell on a side) and the
time the SLEAP body centroid spends in each is added up, separately for the frames
that belong to a search (trial start to target found, as in trials.py) and for all
other frames (walking to the port, drinking, the ITI).

Maps are in seconds, indexed [y bin, x bin]. Run as a script from the directory
above: `python -m src.occupancy` for every experiment, `python -m src.occupancy vision`
for one.
"""
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from . import config as cfg
from .trials import USECOLS, analysed, discover_sessions

BODY = [f'{cfg.TRAJ_POS}_x', f'{cfg.TRAJ_POS}_y']
KEYS = ['mouse', 'session', 'session_type', 'session_num']
X_EDGES = np.linspace(0, cfg.FRAME_W, cfg.OCC_BINS[0] + 1)
Y_EDGES = np.linspace(0, cfg.FRAME_H, cfg.OCC_BINS[1] + 1)


def session_occupancy(path):
    """(search map, other map) for one session, each in seconds."""
    df = pd.read_csv(path, usecols=USECOLS + BODY)
    ts = df['bonsai_ts'].to_numpy(float) / 1000
    x = gaussian_filter1d(df[BODY[0]].to_numpy(float), cfg.SMOOTH_SIGMA)
    y = gaussian_filter1d(df[BODY[1]].to_numpy(float), cfg.SMOOTH_SIGMA)
    dt = np.r_[np.diff(ts), 0]                      # time each frame stands for
    no_target = df['target_cell'].isna().to_numpy()

    searching = np.zeros(len(df), bool)
    for trial, idx in df.groupby('trial_number').indices.items():
        i0, i1 = idx[0], idx[-1] + 1
        hits = np.flatnonzero(no_target[i0:i1])
        if trial == 0 or not len(hits) or no_target[i0]:        # same trials as trials.py keeps
            continue
        searching[i0:i0 + hits[0]] = True

    def hist(keep):
        return np.histogram2d(y[keep], x[keep], bins=[Y_EDGES, X_EDGES], weights=dt[keep])[0]
    return hist(searching), hist(~searching)


def build_occupancy(experiment='odor', verbose=True):
    sessions = discover_sessions(experiment)
    search, other = [], []
    for s in sessions:
        a, b = session_occupancy(s['path'])
        search.append(a)
        other.append(b)
        if verbose:
            print(f"{s['mouse']}/{s['session']}: {a.sum() / 60:.1f} min searching, {b.sum() / 60:.1f} min not")
    cfg.OUT_DIR.mkdir(exist_ok=True)
    np.savez_compressed(cfg.OCCUPANCY_NPZ[experiment], search=np.array(search), other=np.array(other),
                        **{k: np.array([s[k] for s in sessions]) for k in KEYS})


def load_occupancy(experiment='odor'):
    """(sessions, search maps, other maps) for the analysed sessions. `sessions` has one
    row per session, in the same order as the first axis of the two map arrays."""
    z = np.load(cfg.OCCUPANCY_NPZ[experiment])
    index = pd.DataFrame({k: z[k] for k in KEYS}).astype({'mouse': str, 'session': str, 'session_type': str})
    index['row'] = np.arange(len(index))
    index = analysed(index, experiment).sort_values(['mouse', 'session_num']).reset_index(drop=True)
    rows = index.pop('row').to_numpy()
    index.insert(0, 'experiment', experiment)
    return index, z['search'][rows], z['other'][rows]


if __name__ == '__main__':
    for experiment in sys.argv[1:] or cfg.EVENTS_DIRS:
        build_occupancy(experiment)
