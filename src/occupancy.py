"""Where each mouse spends its time: one occupancy map per session.

The arena is cut into OCC_BINS bins (a quarter of a target cell on a side) and the
time a tracked point spends in each is added up, separately for each behavioral
epoch (appetitive, consummatory, drinking - see epochs.py). Maps are built for two
SLEAP points: the body centroid and the nose.

Maps are in seconds, indexed [y bin, x bin]. Run as a script from the directory
above: `python -m src.occupancy` for every experiment, `python -m src.occupancy vision`
for one.
"""
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from . import config as cfg
from .epochs import COLUMNS, EPOCHS, label_epochs
from .trials import analysed, discover_sessions

POINTS = list(cfg.OCCUPANCY_NPZ)                # 'centroid', 'nose'
KEYS = ['mouse', 'session', 'session_type', 'session_num']
X_EDGES = np.linspace(0, cfg.FRAME_W, cfg.OCC_BINS[0] + 1)
Y_EDGES = np.linspace(0, cfg.FRAME_H, cfg.OCC_BINS[1] + 1)


def session_occupancy(path):
    """{point: maps} for one session; maps is [epoch, y bin, x bin] in seconds, epochs in EPOCHS order."""
    df = pd.read_csv(path, usecols=COLUMNS + [f'{p}_{axis}' for p in POINTS for axis in 'xy'])
    ts = df['bonsai_ts'].to_numpy(float) / 1000
    dt = np.r_[np.diff(ts), 0]                      # time each frame stands for
    epoch = label_epochs(df)

    maps = {}
    for point in POINTS:
        x = gaussian_filter1d(df[f'{point}_x'].to_numpy(float), cfg.SMOOTH_SIGMA)
        y = gaussian_filter1d(df[f'{point}_y'].to_numpy(float), cfg.SMOOTH_SIGMA)
        maps[point] = np.array([np.histogram2d(y[keep], x[keep], bins=[Y_EDGES, X_EDGES], weights=dt[keep])[0]
                                for keep in (epoch == k for k in range(len(EPOCHS)))])
    return maps


def build_occupancy(experiment='odor', verbose=True):
    sessions = discover_sessions(experiment)
    maps = []
    for s in sessions:
        maps.append(session_occupancy(s['path']))
        if verbose:
            minutes = maps[-1][POINTS[0]].sum(axis=(1, 2)) / 60
            print(f"{s['mouse']}/{s['session']}: " + ', '.join(f'{name} {m:.1f} min' for name, m in zip(EPOCHS, minutes)))
    cfg.OUT_DIR.mkdir(exist_ok=True)
    for point in POINTS:
        np.savez_compressed(cfg.OCCUPANCY_NPZ[point][experiment], maps=np.array([m[point] for m in maps]),
                            **{k: np.array([s[k] for s in sessions]) for k in KEYS})


def load_occupancy(experiment='odor', point='centroid'):
    """(sessions, maps) for the analysed sessions and one tracked point ('centroid' or
    'nose'). `sessions` has one row per session; `maps` is {epoch: [session, y bin, x bin]
    in seconds}, sessions in the same order."""
    z = np.load(cfg.OCCUPANCY_NPZ[point][experiment])
    index = pd.DataFrame({k: z[k] for k in KEYS}).astype({'mouse': str, 'session': str, 'session_type': str})
    index['row'] = np.arange(len(index))
    index = analysed(index, experiment).sort_values(['mouse', 'session_num']).reset_index(drop=True)
    rows = index.pop('row').to_numpy()
    index.insert(0, 'experiment', experiment)
    return index, {name: z['maps'][rows, k] for k, name in enumerate(EPOCHS)}


if __name__ == '__main__':
    for experiment in sys.argv[1:] or cfg.EVENTS_DIRS:
        build_occupancy(experiment)
