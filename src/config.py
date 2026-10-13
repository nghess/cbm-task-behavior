"""Paths, arena geometry and inclusion thresholds for the cbm-odor / cbm-vision behavior analysis."""
from pathlib import Path

DATA_ROOT = Path('E:/')
OUT_DIR = Path(__file__).parent.parent / 'output'

# Per experiment: where the events live (<animal>/<session>/events.csv, one row per
# video frame) and where the per-trial table is written. Directories are searched in
# order and the first one holding a session wins. cbm-vision's events_temp holds the
# sessions not yet synced to ephys; the behavior columns used here are identical in both.
EVENTS_DIRS = {
    'odor':   [DATA_ROOT / 'cbm-odor' / 'events'],
    'vision': [DATA_ROOT / 'cbm-vision' / 'events', DATA_ROOT / 'cbm-vision' / 'events_temp'],
}
TRIALS_CSV = {
    'odor':   OUT_DIR / 'trials.csv',
    'vision': OUT_DIR / 'trials_vision.csv',
}
TRAJECTORIES_CSV = {
    'odor':   OUT_DIR / 'trajectories.csv',
    'vision': OUT_DIR / 'trajectories_vision.csv',
}
SESSIONS_CSV = {
    'odor':   OUT_DIR / 'sessions.csv',
    'vision': OUT_DIR / 'sessions_vision.csv',
}

OCCUPANCY_NPZ = {
    'odor':   OUT_DIR / 'occupancy.npz',
    'vision': OUT_DIR / 'occupancy_vision.npz',
}

# Session types that enter the analysis. cbm-odor's 'o' and 't' sessions were shaping
# sessions; they are still reduced to trials and visits but dropped on load.
SESSION_TYPES = {'odor': ['x'], 'vision': ['v']}

# Which of those sessions: each mouse's first SESSIONS_PER_MOUSE sessions of the
# experiment, in session-number order, starting from FIRST_SESSION. Odor starts at x1
# because only three mice ran an x0. Loaders renumber the kept sessions 1, 2, ... per
# mouse (`session_num`), so the number is how many sessions the mouse has had; the
# `session` column keeps the original name.
FIRST_SESSION = {'odor': 1, 'vision': 0}
SESSIONS_PER_MOUSE = 10

VISITS_CSV = {
    'odor':   OUT_DIR / 'visits.csv',
    'vision': OUT_DIR / 'visits_vision.csv',
}

# Camera frame (px). events.csv positions are already in this full-resolution space.
FRAME_W, FRAME_H = 888, 1968
FPS = 30

# Target grid, from cbm_odor_task.py (and cbm-viewer/grid.py): a 5 x 11 grid over the
# whole frame whose outer ring is never a target, leaving 3 x 9 addressable cells
# numbered row-major from the low-y end.
GRID_COLS, GRID_ROWS = 5, 11
INNER_COLS, INNER_ROWS = GRID_COLS - 2, GRID_ROWS - 2
CELL_W, CELL_H = FRAME_W // GRID_COLS, FRAME_H // GRID_ROWS

# Long-axis zones. flip_state 0 targets (and the odor / visual cue) are confined to inner rows 0-3,
# flip_state 1 to inner rows 5-8; inner row 4 is the neutral band holding both water
# ports. Zones are named by y only, so the border ring belongs to its nearest end.
LOW_MAX_Y = (1 + 4) * CELL_H     # y below this  -> 'low' end  (flip_state 0 side)
HIGH_MIN_Y = (1 + 5) * CELL_H    # y at/above    -> 'high' end (flip_state 1 side)
MID_Y = FRAME_H / 2

CUE_ZONE = {0: 'low', 1: 'high'}   # flip_state 2 = no cue

# Position used for all zone logic: the point the task itself tracked online
# (named "centroid" by Bonsai but actually the nose - see cbm-viewer/config.py).
POS = 'bonsai_centroid'
SMOOTH_SIGMA = 2                 # frames; Gaussian smoothing before path length (as in trial_distance_flip_state.ipynb)

# Trials whose search lasts longer than this are treated as disengaged for the
# search-cost comparisons (choice measures keep them).
MAX_SEARCH_S = 120

# An entry into an end zone counts as a visit to that end only if the nose gets at
# least this far in (one cell: past the first row of targets). Shallower dips are the
# mouse hovering at the middle row, where the ports are (see marginal_value.ipynb).
MIN_VISIT_DEPTH = CELL_H

# Path shape (trajectories.py) is measured on the SLEAP body centroid, which tracks
# smoothly; the task's own point sticks and jumps.
TRAJ_POS = 'centroid'
MOVE_SPEED = 50                  # px/s; slower than this counts as a pause (the nose is ~165 px ahead of the centroid)
RESAMPLE_STEP = 40               # px of path between the points turning is measured at
STRAIGHT_SCALE = 480             # px of path straightness is measured over (about 2.7 cells)
SEARCH_PHASES = [0, 4, 10, 25]   # s into a search; paths are described separately for each stretch (the last is open-ended)

# Occupancy maps (occupancy.py): bins across x and along y, a quarter of a target cell on a side.
OCC_BINS = (4 * GRID_COLS, 4 * GRID_ROWS)
