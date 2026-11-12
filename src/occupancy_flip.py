"""Where the nose is, split by trial condition: the trial's flip state and the end its target is in.

occupancy.py builds one map per session and epoch. Here each map is split further by
the trial the time belongs to, so that trials with the cue at the low end (flip_state 0),
at the high end (1) and with no cue (2) can be compared, and the no-cue trials can be
matched to the cue trials by where the target was.

  search       - only while a trial is running: from its first frame (cue on for flip 0/1)
                 until its target is found. The ITI and the rest of the search epoch before
                 a trial starts are left out: those frames carry the previous trial's number
                 and flip state, and the next trial's cue is not on yet.
  retrieval,
  consumption  - as labelled by epochs.py, assigned to the trial just found (those frames
                 still carry its number).

Trial 0 (free reward) and trials whose target was never found are left out. Maps are in
seconds, indexed [epoch, flip state, target end, y bin, x bin], on the SLEAP nose with the
same smoothing and bins as occupancy.py. Only the analysed sessions are built.

Run as a script from the directory above: `python -m src.occupancy_flip` for every
experiment, `python -m src.occupancy_flip vision` for one.
"""
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from . import config as cfg
from .epochs import COLUMNS, EPOCHS, SEARCH, RETRIEVAL, CONSUMPTION, label_epochs
from .occupancy import KEYS, X_EDGES, Y_EDGES
from .trials import analysed, discover_sessions, zone_of

FLIPS = [0, 1, 2]
TARGET_ENDS = ['low', 'mid', 'high']
POINT = 'nose'


def session_flip_occupancy(path):
    """[epoch, flip, target end, y bin, x bin] seconds for one session."""
    df = pd.read_csv(path, usecols=COLUMNS + ['trial_number', 'flip_state', f'{POINT}_x', f'{POINT}_y'])
    ts = df['bonsai_ts'].to_numpy(float) / 1000
    dt = np.r_[np.diff(ts), 0]
    epoch = label_epochs(df)
    x = gaussian_filter1d(df[f'{POINT}_x'].to_numpy(float), cfg.SMOOTH_SIGMA)
    y = gaussian_filter1d(df[f'{POINT}_y'].to_numpy(float), cfg.SMOOTH_SIGMA)

    # each trial's flip state and target end, from its first frame; kept only if its target was found
    trial = df['trial_number'].to_numpy()
    first = df.groupby('trial_number').head(1).set_index('trial_number')
    found = df['target_cell'].isna().groupby(df['trial_number']).any()
    keep_trial = (first.index > 0) & first['target_cell'].notna() & found.reindex(first.index).to_numpy()
    first = first[keep_trial]
    flip = pd.Series(first['flip_state'].astype(int), index=first.index).reindex(trial).to_numpy(float)
    end = pd.Series([TARGET_ENDS.index(z) for z in zone_of((first['target_cell'] // cfg.INNER_COLS + 1.5) * cfg.CELL_H)],
                    index=first.index).reindex(trial).to_numpy(float)

    running = df['target_cell'].notna().to_numpy()
    windows = {SEARCH: (epoch == SEARCH) & running, RETRIEVAL: epoch == RETRIEVAL, CONSUMPTION: epoch == CONSUMPTION}
    out = np.zeros((len(EPOCHS), len(FLIPS), len(TARGET_ENDS), len(Y_EDGES) - 1, len(X_EDGES) - 1), dtype=np.float32)
    for e, window in windows.items():
        for f in FLIPS:
            for z in range(len(TARGET_ENDS)):
                keep = window & (flip == f) & (end == z)
                if keep.any():
                    out[e, f, z] = np.histogram2d(y[keep], x[keep], bins=[Y_EDGES, X_EDGES], weights=dt[keep])[0]
    return out


def build_flip_occupancy(experiment='odor', verbose=True):
    sessions = pd.DataFrame(discover_sessions(experiment))
    sessions = analysed(sessions, experiment).sort_values(['mouse', 'session_num']).reset_index(drop=True)
    maps = []
    for s in sessions.itertuples():
        maps.append(session_flip_occupancy(s.path))
        if verbose:
            minutes = maps[-1][EPOCHS.index('search')].sum(axis=(1, 2, 3)) / 60
            print(f"{s.mouse}/{s.session}: search by flip state " + ', '.join(f'{f}: {m:.1f} min' for f, m in zip(FLIPS, minutes)))
    cfg.OUT_DIR.mkdir(exist_ok=True)
    np.savez_compressed(cfg.OCCUPANCY_FLIP_NPZ[experiment], maps=np.array(maps),
                        **{k: sessions[k].to_numpy() for k in KEYS})


def load_flip_occupancy(experiment='odor'):
    """(sessions, maps): one row per analysed session, and maps as
    {epoch: [session, flip, target end, y bin, x bin] in seconds}, sessions in the same order."""
    z = np.load(cfg.OCCUPANCY_FLIP_NPZ[experiment], allow_pickle=True)
    index = pd.DataFrame({k: z[k] for k in KEYS}).astype({'mouse': str, 'session': str, 'session_type': str, 'session_num': int})
    index.insert(0, 'experiment', experiment)
    return index, {name: z['maps'][:, k].astype(float) for k, name in enumerate(EPOCHS)}


if __name__ == '__main__':
    for experiment in sys.argv[1:] or cfg.EVENTS_DIRS:
        build_flip_occupancy(experiment)
