# Skeleton-Based Recognition of Indonesian Sign Language (BISINDO): A Classical Learning Benchmark

Code, saved experiment outputs, and figures for the manuscript submitted to the *IAES International Journal of Artificial Intelligence* (IJ-AI).

The study benchmarks four classical classifiers (Random Forest, MLP, SVM, KNN) on skeleton keypoints extracted with MediaPipe Holistic from six BISINDO gesture classes, with a 1D-CNN and an LSTM as video-level deep baselines. The primary dataset ("Dataset A") has 5,379 quality-filtered frames from 210 videos and 7 signers, with 285 features per frame (95 keypoints x 3 coordinates).

**Gesture classes:** Halo (Hello), Apa Kabar (How Are You?), Maaf (Sorry), Terima Kasih (Thank You), Sama-sama (You're Welcome), Tolong (Help).

## Headline results (Dataset A)

| Evaluation | RF | MLP | SVM | KNN |
|---|---|---|---|---|
| Random stratified 5-fold, frame level (%) | 97.95 ± 0.41 | 97.53 ± 0.49 | 95.56 ± 0.62 | 96.10 ± 0.60 |
| Video-grouped 5-fold, frame level (%) | 94.02 ± 1.63 | 94.16 ± 2.05 | 93.19 ± 1.71 | 92.58 ± 1.66 |
| Video level (majority vote, grouped folds, %) | 98.10 | 98.10 | 97.62 | 97.62 |

Video-level deep baselines (5-fold, 210 videos): 1D-CNN 99.52 ± 0.95, LSTM 98.57 ± 1.17. Standard deviations are population SDs (ddof = 0). Random folds overstate accuracy relative to video-grouped folds, so the two protocols must not be compared as if they measured the same thing; see `docs/PROTOCOL.md`.

## Repository structure

```
.
├── A_experiments.ipynb        # main experiments on Dataset A (same content as A_experiments.py)
├── A_experiments.py           # script version, "# %%" cell blocks
├── Figures_3_6.ipynb          # regenerates Figures 3-6
├── notebooks/
│   └── Indonesia_sign_language_exploratory.ipynb   # keypoint extraction and additional analyses (Tables 5, 7, 8, 9, 11)
├── scripts/
│   └── verify_mcnemar_and_lengthonly.py   # McNemar pairing check and length-only diagnostic
├── docs/
│   └── PROTOCOL.md            # protocol, execution order, limitations
├── results/
│   ├── A_outputs/             # saved outputs of A_experiments (folds, predictions, summaries, confusion matrices)
│   └── supp/                  # outputs of the supplementary checks
├── figures/                   # Figures 3-6 (PNG and PDF)
├── data/
│   └── README.md              # data schema and access (the data are not included)
├── requirements.txt
└── LICENSE
```

## Setup

Tested with Python 3.11.9. TensorFlow is only needed for the 1D-CNN and LSTM cells.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## How to reproduce

1. Obtain the Dataset A file (see `data/README.md`) and place it where the notebook expects it. The file name is set by `DATA_CSV` at the top of the module text in Cell 0, or by the environment variable `A_DATA_CSV`.
2. Start Jupyter **from the repository root** and open `A_experiments.ipynb`. Run Cell 0, then Cell 1 (it checks the data against built-in anchors, such as RF 5-fold 97.95 ± 0.41, and stops if any prints FAIL), then Cells 2-10 in the order given in `docs/PROTOCOL.md`. Outputs are written to `A_outputs/`.
3. Optional checks: `python scripts/verify_mcnemar_and_lengthonly.py results/A_outputs/ results/supp/` (note the trailing slashes).
4. Figures: open `Figures_3_6.ipynb`. Figures 5 and 6 need only the saved CSVs in `results/A_outputs/`. Figures 3 and 4 also need the keypoint file (`data/dataset_keypoints.csv`, not included).

All random states are 42 (`StratifiedKFold` and `StratifiedGroupKFold`, 5 folds, shuffled). TensorFlow runs on different hardware can differ slightly. `results/A_outputs/A_environment.json` records the exact package versions and data checksum used for the saved outputs.

## Manuscript tables and figures to files

| Manuscript item | Files |
|---|---|
| Table 2, Figure 5 (cross-validation accuracy) | `A_frame_cv_*`, `A_delta_cv_*`, `A_grouped_cv_*`, `Aout_video_level_classical_grouped.csv`, `A_dl_cv_*` |
| Table 3 (train-test splits) | `A_split_results.csv`, `A_dl_split_results.csv`, `A_split_predictions.csv` |
| Table 4 (Wilcoxon, Cohen's d) | `A_stats_vs_RF_frame.csv` |
| Table 5 (bootstrap CIs) | `Aout_bootstrap_ci_7030.csv` |
| Table 6, Figure 6 (errors, confusion matrix) | `A_cm_7030_*.csv` |
| Table 10 (default, grid-search, nested CV) | `A_nested_tuning_folds.csv`, `A_tuning_comparison.csv` |
| McNemar tests at video level | `Aout_paired_video_level.csv`, `results/supp/A_supp_mcnemar_pairing_check.csv` |
| Length-only diagnostic | `results/supp/A_supp_length_only_check.csv`, `A_dl_sequence_lengths.csv` |
| Figures 3 and 4 | `Figures_3_6.ipynb` (needs the keypoint file) |
| Tables 7, 8, 9, 11 (facial ablation, runtime, LOSO, threshold ablation) | `notebooks/Indonesia_sign_language_exploratory.ipynb` (writes the `c*_results.csv` files listed below) |

### Exploratory analyses (Tables 5, 7, 8, 9 and 11)

`notebooks/Indonesia_sign_language_exploratory.ipynb` contains the video-to-keypoint extraction (MediaPipe Holistic) and the additional analyses reported in the manuscript: facial-feature ablation, runtime, leave-one-signer-out (LOSO), grid-search tuning, bootstrap intervals, and the missing-keypoint threshold ablation. These cells write `c1_loso_results.csv`, `c3_facial_feature_ablation.csv`, `c4_hyperparameter_tuning_results.csv`, `c5_runtime_results.csv`, `c6_bootstrap_ci_results.csv` and `c8_ablation_results.csv`. Notes:

- The notebook is stored without outputs. Some comments and print messages are in Indonesian.
- Signer names are replaced by the codes S1-S7, and the local video path is replaced by `path/to/Dataset Isyarat`. The raw videos are not distributed, so the extraction cells document the pipeline but cannot be rerun without them. The raw keypoint file must use the same signer codes in its `video` column.
- The exploratory notebook was developed before `A_experiments.ipynb`. For all results that appear in both, `A_experiments.ipynb` and the saved files in `results/A_outputs/` are the reference.

## Data and privacy

The recordings show identifiable people, so the raw videos are not distributed. Signers are anonymised as S1-S7. See `data/README.md` for the schema and for how to request the keypoint data.

## Limitations

Seven signers in one controlled recording environment (one of them a native Deaf signer), six isolated gesture classes, and zero-padding of the video sequences that is not masked in the deep baselines. The results do not extend to larger vocabularies, continuous signing, or unconstrained conditions. See the manuscript and `docs/PROTOCOL.md`.

## Citation

If you use this code or the saved outputs, please cite the manuscript (full reference to be added after publication).

## License

Code is released under the MIT License (see `LICENSE`). The license covers the code and saved outputs, not the recordings or keypoint data, which are available on request under the conditions in `data/README.md`.
