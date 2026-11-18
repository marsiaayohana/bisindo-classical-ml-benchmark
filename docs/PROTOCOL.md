# Experiment protocol (Dataset A)

`A_experiments.ipynb` (same content as `A_experiments.py`) runs all primary experiments. Each experiment cell reloads and re-validates the data itself and writes only to `A_outputs/`; Cells 8, 9 and 10 also read files written by earlier cells.

## Input

A CSV with the 285 `f*` features, `label`, `signer`, `video` and `old_row` (289 columns), 5,379 rows. `video_no` is optional: it is checked against the positional rule only if present and is never created. `load_A()` prints the schema it found. Set `DATA_CSV` (top of Cell 0's module text) or the environment variable `A_DATA_CSV` if your file name differs. The `signer` column must use the codes S1-S7.

Checks that need only `old_row`:

- signer = `SIGNERS[((old_row % 1050) // 30) // 5]`
- label = class block `old_row // 1050` (alphabetical class order)
- every video lies in exactly one 30-row block and every block holds exactly one video
- each video has one signer and one label, and at most 30 frames

## Execution order and dependencies

Recommended order: **Cell 0 -> Cell 1 -> Cells 2-5 -> Cells 6-7 -> Cells 8-9 -> Cell 10.**

| Cell | Needs (files in `A_outputs/` written earlier) | Writes |
|---|---|---|
| 0 | none (writes `a_common.py` next to the notebook; rerun after any edit of the module text) | `a_common.py` |
| 1 | Cell 0 | `A_environment.json` (stop if any anchor prints FAIL) |
| 2 | Cell 0 | `A_frame_cv_*` |
| 3 | Cell 0 | `A_delta_cv_*`, `A_delta_gap_stats.csv` |
| 4 | Cell 0 | `A_split_*`, `A_cm_*` |
| 5 | Cell 0 | `A_grouped_cv_*` |
| 6 | Cell 0 (TensorFlow/Keras) | `A_dl_cv_*`, `A_dl_sequence_lengths.csv` |
| 7 | Cell 0 (TensorFlow/Keras) | `A_dl_split_*` |
| 8 | Cell 2 (`A_frame_cv_folds.csv`), Cell 3 (`A_delta_cv_folds.csv`), Cell 6 (`A_dl_cv_folds.csv`) | `A_stats_vs_RF_frame.csv` |
| 9 | Cell 2 (`A_frame_cv_folds.csv`, for the default scores) | `A_nested_tuning_folds.csv`, `A_tuning_comparison.csv` |
| 10 | outputs of Cells 2-9 | `A_manifest_cv_summary.csv` |

Cells 2-7 do not depend on each other. Cell 9 (nested grid search, MLP included) is the slowest classical cell. Do not rerun a cell partially: rerunning Cell 2, 3 or 6 changes the inputs of Cells 8-10, so rerun Cells 8-10 afterwards. Cell 10 skips missing inputs silently, so run it only after Cells 2-9.

## Design rules and where they are implemented

| Rule | Implementation |
|---|---|
| Deltas only within a video, ordered by `old_row`, first retained frame = 0, never across videos | `delta_features()` in Cell 0; delta-delta sets the first two frames to 0 |
| Gaps caused by filtered frames are documented | Cell 3 saves `A_delta_gap_stats.csv` (share of deltas that span a removed frame) |
| Deep models: one sequence per video, ordered by `old_row`, at most 30 frames, zero-padded | `build_sequences()` and `scale_and_pad()` (Cells 6-7); 210 sequences asserted |
| Scaling fitted inside each training fold | `StandardScaler` is fitted on the real frames of the training subset only; padding is added after scaling |
| Random state 42; folds, predictions and summaries saved | every cell: `*_folds.csv`, `*_oof_predictions.csv`, `*_summary.csv` |
| Frame-level and video-grouped evaluation kept separate | Cells 2-4 frame level; Cell 5 `StratifiedGroupKFold` (asserts no video on both sides); every output has an `eval_unit` column |
| Deep models compared only with video-level classical results | Cell 8 adds `same_unit_as_baseline`; Cell 10 prints one block per evaluation unit |

## Implementation notes

- Deep-model early stopping uses a **stratified random 15%** of the training videos as validation set (not the last 15% of the array).
- Deep-model seeds are `set_random_seed(42 + fold)`; TensorFlow runs on different hardware can still differ slightly.
- Grouped CV uses `StratifiedGroupKFold(5, shuffle=True, random_state=42)`.
- Wilcoxon signed-rank tests with n = 5 folds have a minimum two-sided exact p-value of 0.0625, so significance at alpha = 0.05 cannot be reached; Cohen's d is unstable with five pairs. Bootstrap intervals and paired exact McNemar tests on the 210 out-of-fold video predictions complement them.

## Built-in reproducibility anchors (Cell 1)

RF 5-fold 97.95 ± 0.41; MLP 5-fold 97.53; RF 70:30 = 1,572 of 1,614 correct with the saved confusion matrix. If an anchor prints FAIL, stop: the data file or pipeline differs from the one behind the saved outputs.

## Limitation: zero padding is not masked

Padding is added after scaling, so a padded step is the all-zero vector, which in scaled space is the training-mean frame; nothing marks it as missing.

- **1D-CNN:** `Conv1D` windows at the sequence end mix real and padded steps, and `MaxPooling1D` and `GlobalAveragePooling1D` include the padded positions.
- **LSTM:** the recurrence keeps running over the trailing zero steps and the prediction is read after them.
- Video length after filtering is not random, so the number of real frames can correlate with signer, class or recording quality. On Dataset A about 15% of all timesteps are padding (mean 25.6 real frames of 30). `A_dl_sequence_lengths.csv` lists real and padded steps per video.
- A length-only diagnostic (`scripts/verify_mcnemar_and_lengthonly.py`) gives 17.14-21.90% accuracy against a 16.67% majority-class rate, so length alone does not explain the deep-model results. This does not prove that no length shortcut exists.

Possible remedy, not run in this study: append a binary real-frame indicator channel to every timestep (input width 285 -> 286) so both architectures treat padding alike.
