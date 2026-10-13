# task-behavior

Behavioral analysis of the cbm-odor and cbm-vision search task: do mice use the cue to find the target, and what search strategy do they use instead?

## The task

Each trial hides one target cell in a 3 x 9 grid on the arena floor. The mouse finds it by putting its nose in the cell, then drinks at a port in the middle row, waits out the ITI, and the next trial starts.

| `flip_state` | Target | Cue |
|---|---|---|
| 0 | one of the 12 cells at the low-y end | at the low-y end |
| 1 | one of the 12 cells at the high-y end | at the high-y end |
| 2 | any of the 27 cells | none |

The cue is an odor (1% pinene) in cbm-odor and a stimulus on a monitor in cbm-vision. The odor goes off when the target is found; the visual stimulus stays on until the next trial starts.

The arena is divided along its long axis into three zones: `low` (y < 890 px), `high` (y >= 1068 px) and `mid`, the row between them that holds the ports.

## Data

| Experiment | Sessions analysed | Mice | Location |
|---|---|---|---|
| cbm-odor | each mouse's first ten `x` sessions from `x1` | 9 | `E:/cbm-odor/events` |
| cbm-vision | each mouse's first ten `v` sessions | 9 (the same mice) | `E:/cbm-vision/events`, falling back to `events_temp` |

- The input is each session's `events.csv` (one row per video frame). The Bonsai columns are used throughout: trial number, timestamp, nose position, target cell and flip state. The trajectory notebook also uses the SLEAP body centroid.
- The odor `o` and `t` sessions were shaping sessions and are not analysed. They are still reduced to trial and visit tables but dropped on load (`SESSION_TYPES` in `src/config.py`).
- Every mouse ran its vision sessions after its odor sessions.
- **Ten sessions per mouse per experiment.** For most mice that is x1-x10 and v0-v9. `x0` is left out because only three mice ran it. The exceptions:
  - 9007 has no `x7`, so its odor sessions are x1-x6 and x8-x11.
  - 9007 ran vision sessions 0 and 2 twice (`v0b`, `v2b`); the repeats count as sessions, so its ten run to v7.
  - 9009 has no `v6`, so its vision sessions run to v10.
  - 9000 ran only six vision sessions (v0-v5).
- In the loaded tables `session_num` is the session's position in the mouse's own sequence (1-10); `session` keeps the original name. The selection is set by `FIRST_SESSION` and `SESSIONS_PER_MOUSE` in `src/config.py`.

## Notebooks

Read them in this order. Each ends with a summary of its findings and caveats.

### 1. `odor_search_strategy.ipynb`

Do mice use the odor? Two behaviors are tested separately:

- **Direction**: from the middle row, which end does the mouse enter first?
- **Persistence and search cost**: odor trials against no-odor trials with the same target cell and start zone.

Result: the odor has no influence on which end mice go to. Odor trials are slightly cheaper than matched no-odor trials (path about 8% shorter; mice leave the correct end 3.4 percentage points less often).

### 2. `vision_search_strategy.ipynb`

The same analysis on cbm-vision.

Result: the visual stimulus has no detectable effect on direction, persistence or search cost.

### 3. `odor_vs_vision_search_strategy.ipynb`

The two experiments side by side, and compared within mouse.

Result: neither cue steers direction. The odor's small effect on persistence is larger than the stimulus's in 8 of 9 mice. Mice search about 17% more efficiently in the vision sessions even without a cue, most simply from more practice, so each cue is only ever compared with the no-cue trials of its own sessions.

### 4. `marginal_value.ipynb`

Treats each end of the arena as a patch and asks when mice abandon one for the other, and whether that follows the marginal value theorem.

Result: there is a typical persistence period of about 4 s, which is one run to the far wall and back; after that, leaving is roughly memoryless. That is about when a fresh patch stops paying, but mice do not leave sooner when the patch is already depleted, and they do not avoid cells they have already checked.

### 5. `search_trajectories.ipynb`

Describes the shape of the search paths (pausing, speed, turning, straightness, wall-following, cells checked per distance) and asks how it changes with experience and what distinguishes a mouse's good sessions.

Result: with experience mice pause less, cover more ground and leave the walls; their paths get only slightly straighter, and only between the odor and vision experiments. A good session is one in which the mouse moves more vigorously (faster, straighter, fewer pauses), but a given distance does not check more cells. Searches get slower and more tortuous the longer they last, so measures are compared at the same point in a search.

### 6. `odor_search_strategy_30min.ipynb`

Notebook 1 repeated on the trials that began in the first 30 minutes of each session, when mice are most engaged, followed by a comparison of the first 30 minutes with the rest of the session.

Result: mice are clearly less engaged after 30 minutes (fewer trials per minute, longer searches), but the odor effect is not stronger in the first 30 minutes. It is slightly weaker there than over the whole session, and nominally largest after 30 minutes, though that difference is not reliable.

### 7. `occupancy_centroid.ipynb`

Heat maps of where each mouse's body is in each behavioral epoch (appetitive, consummatory, drinking), and for the search (appetitive) epoch: every session of every mouse, each mouse against the average mouse, the last three sessions minus the first three, and the best sessions minus the worst.

Result: each mouse has its own map and keeps it from the odor sessions to the vision sessions. Occupancy shifts from the walls to the interior over the whole history (wall lane 51% of search time to 42%) and in a mouse's best sessions compared with its worst, but changes little within either experiment.

### 8. `occupancy_nose.ipynb`

Notebook 7 with the SLEAP nose in place of the body centroid: where the mouse is checking, not where its body is.

Result: the same story as the body maps. The nose is outside the target grid, where no target can be, for 42% of search time in odor and 36% in vision; that share falls over the whole history (46% to 35%) and is lower in a mouse's best sessions than in its worst.

## Code

The notebooks are the only code in the base directory. Everything they share is in the `src/` package, and the tables they load are in `output/`.

| File | What it does |
|---|---|
| `src/config.py` | Data paths, arena and grid geometry, zone boundaries, which sessions to analyse, thresholds. |
| `src/trials.py` | Reduces each session to one row per trial: search time, path length, zones entered, whether the mouse left the correct end. |
| `src/visits.py` | Breaks each search into visits to the two ends: residence time, outcome (switch or find), and when each target cell was first entered. |
| `src/trajectories.py` | Describes each search path on the SLEAP body centroid, by how long the search has been going, and totals the ground covered in each session. |
| `src/epochs.py` | Labels every frame as appetitive (searching), consummatory (target found, heading to the port) or drinking, as in cluster-states-viewer. |
| `src/occupancy.py` | Builds occupancy maps per session (time per bin) for the body centroid and for the nose, one per epoch. |
| `src/stats.py` | Stratified two-group comparisons with permutation tests, per-mouse summaries, hazard and Kaplan-Meier estimates. |
| `output/` | The tables the notebooks load: `trials`, `visits`, `trajectories` and `sessions`, each as `<name>.csv` for odor and `<name>_vision.csv` for vision, plus the occupancy maps as `occupancy*.npz` (body centroid) and `occupancy_nose*.npz` (nose). |

## Rebuilding

The notebooks read the tables in `output/`, so they run without the raw data. To rebuild the tables from `events.csv` (a few minutes each), from the base directory:

```
python -m src.trials          # both experiments; add `vision` or `odor` for one
python -m src.visits
python -m src.trajectories
python -m src.occupancy
```

Then re-run the notebooks. The matched comparisons use 5,000-shuffle permutation tests and take a minute or two per notebook.

Requires numpy, pandas, scipy and matplotlib.

## Conventions used throughout

- **A search** runs from the first frame of a trial to the frame the target is found. Walking to the port, drinking and the ITI are not part of it. The two occupancy notebooks instead use the lab's appetitive epoch, which starts when the mouse leaves the port and so includes the ITI.
- **Position** is the nose point the task tracked online (`bonsai_centroid`). It sticks and jumps, so path shape in notebook 5 is measured on the SLEAP body centroid instead, and distances there are about 60% of the nose-path distances in notebooks 1-4.
- **Dropped everywhere**: trial 0 of each session (free reward) and trials where the target was never found.
- **Dropped from cost and patch analyses**: searches longer than 120 s (`MAX_SEARCH_S`), and trials with the target in the middle row.
- **Comparisons are stratified**: effects are computed within mouse and within the nuisance variables that matter (target cell, start zone, previous target's end), then summarised across mice. Position-based measures such as occupancy are avoided because a search ends at the target, which biases them toward the target's end.
- **A visit** to an end counts only if the nose gets at least one cell deep (`MIN_VISIT_DEPTH`); shallower entries are treated as hovering at the middle row.
# cbm-task-behavior
