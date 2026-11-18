# Data

The recordings show identifiable people, so **no raw video and no keypoint data are stored in this repository.**

## Dataset A (primary)

One CSV row per retained frame.

| Column | Description |
|---|---|
| `f0` ... `f284` | 285 features: x, y, z of 95 keypoints (33 pose, 21 + 21 hand slots, 20 face), flattened per frame |
| `label` | gesture class (6 classes) |
| `signer` | anonymised signer code `S1` ... `S7` |
| `video` | video identifier (the signer code, gesture and take number) |
| `old_row` | row index in the original 6,300-row extraction before filtering; used for ordering, grouping and the positional signer rule |

5,379 rows, 210 videos (35 per class, 5 per signer and class), 7 signers. The released signer column must use the codes `S1`-`S7`, because the checks in `A_experiments.py` compare the column with `SIGNERS = ['S1', ..., 'S7']`.

## Files needed by `Figures_3_6.ipynb`

`data/dataset_keypoints.csv` (the 6,300-row keypoint file before filtering) is needed only for Figures 3 and 4. It is not included.

## Access

The keypoint data are available on reasonable request from the corresponding author listed in the manuscript, for non-commercial research use, with an agreement not to attempt to identify participants. Raw videos are not shared.
