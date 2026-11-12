"""Label every frame of a session with the behavioral epoch it belongs to.

The three epochs are the ones used in the lab's cluster-states-viewer:

  search       - searching: from leaving the port after one reward until the next
                 target is found. Includes the ITI.
  retrieval    - target found, reward available: until the mouse pokes a port.
  consumption  - at the port: from that poke until the ITI begins, keeping only
                 frames within CONSUMPTION_GAP_S of a poke pulse. The rest of that
                 window (the mouse has wandered off before the ITI starts) goes
                 back to search, as in the viewer.

The viewer starts the retrieval epoch on reward_state's rising edge. Here it
starts on the frame target_cell goes NaN, which is the same event, because in
these events.csv files reward_state stays True until the next recorded poke and
so misses an edge whenever a poke goes unrecorded (see trials.py).

For the same reason there is a fourth label. When a target is found and no poke is
recorded before the next ITI, the stretch in between holds an unlogged visit to the
port and cannot be split into retrieval and consumption. It is labelled UNSPLIT
and left out of all three epochs, instead of letting the retrieval epoch run on
into the next search.

(Until Oct 2026 these epochs were named appetitive, consummatory and drinking; only the
names changed.)
"""
import numpy as np

from . import config as cfg

SEARCH, RETRIEVAL, CONSUMPTION, UNSPLIT = range(4)
EPOCHS = ['search', 'retrieval', 'consumption']        # label i is EPOCHS[i]; UNSPLIT belongs to none
COLUMNS = ['bonsai_ts', 'target_cell', 'iti', 'poke_left', 'poke_right']


def _onsets(flag):
    """Indices where a boolean array turns True."""
    return np.flatnonzero(flag[1:] & ~flag[:-1]) + 1


def label_epochs(df):
    """Epoch label (SEARCH / RETRIEVAL / CONSUMPTION / UNSPLIT) for every row of an events table."""
    ts = df['bonsai_ts'].to_numpy(float) / 1000
    no_target = df['target_cell'].isna().to_numpy()
    poke = (df['poke_left'] | df['poke_right']).to_numpy(bool)
    iti = df['iti'].to_numpy(bool)

    # walk the three kinds of event in time order: target found, poke onset, ITI onset
    events = sorted([(i, 'found') for i in _onsets(no_target)] + [(i, 'poke') for i in _onsets(poke)] +
                    [(i, 'iti') for i in _onsets(iti)])
    epoch = np.full(len(df), SEARCH, dtype=np.int8)
    state, since = (RETRIEVAL if no_target[0] else SEARCH), 0      # a session opens with a free reward
    for i, kind in events:
        new = state
        if state == SEARCH and kind == 'found':
            new = RETRIEVAL
        elif state == RETRIEVAL and kind == 'poke':
            new = CONSUMPTION
        elif state == RETRIEVAL and kind == 'iti':
            state, new = UNSPLIT, SEARCH                             # no poke was recorded for this reward
        elif state == CONSUMPTION and kind == 'iti':
            new = SEARCH
        if new != state or state == UNSPLIT:
            epoch[since:i] = state
            state, since = new, i
    epoch[since:] = state

    # consumption is only the time actually spent at the port
    consumption = np.flatnonzero(epoch == CONSUMPTION)
    poke_t = ts[poke]
    if len(consumption) and len(poke_t):
        k = np.clip(np.searchsorted(poke_t, ts[consumption]), 1, len(poke_t) - 1)
        gap = np.minimum(np.abs(ts[consumption] - poke_t[k - 1]), np.abs(poke_t[k] - ts[consumption]))
        epoch[consumption[gap > cfg.CONSUMPTION_GAP_S]] = SEARCH
    return epoch
