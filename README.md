# Reproduction of PrivateKT: Differentially Private Knowledge Transfer for Federated Learning

## Overview
- **Paper:** T. Qi, F. Wu, C. Wu, L. He, Y. Huang, X. Xie, "Differentially private knowledge transfer for federated learning," *Nature Communications*, 14, 3785 (2023).
- **DOI:** [10.1038/s41467-023-38794-x](https://doi.org/10.1038/s41467-023-38794-x)
- **Original code:** https://github.com/taoqi98/PrivateKT (MIT licence, kept in `LICENSE`)
- **Reproduced result:** Fig. 2, MNIST, Class Non-IID, with local differential privacy (ε = 5). Paper value for PrivateKT: **94.06 %** accuracy.
- **Outcome:** `<Reproduced / Partly reproduced / Not reproduced>`: `<one-line reason>`

This repository is a fork of the authors' code. The authors' files in `Code/` were ported to TensorFlow 2 so they run on current Google Colab. I added seeded runs, logs, results and a figure script.

## Setup
- Google Colab with a **T4 GPU** (Runtime → Change runtime type → T4 GPU). Exact versions are in `SETUP.md`.
- Colab already has every package needed (TensorFlow 2, NumPy, pandas, Matplotlib). `requirements.txt` holds the `pip freeze` of the session I used. The authors' original pins are kept in `requirements_original.txt`. Those pins do **not** install on current Python; see `NOTES.md`.
- The data (MNIST) is included in `demo_data/`, so nothing needs to be downloaded.

## How to run
1. Open `reproduce.ipynb` in Colab (File → Open notebook → GitHub → paste this repo URL).
2. In the first cell, replace `<your-username>` with the account that hosts this repo.
3. Run all cells in order. The notebook:
   - records the environment;
   - re-runs the authors' original instructions and logs the failures;
   - applies the TensorFlow 2 port;
   - times one round;
   - runs 3 seeds;
   - writes the results and the comparison table;
   - draws the figure.
4. To redraw only the figure from the saved results, run `python make_figure.py` from the repository root.

## Expected output
- `results/seed{0,1,2}.csv`: test accuracy for every round of each seed.
- `results/my_results.csv`: the three seeds combined.
- `results/comparison_table.csv`: paper value vs. mine.
- `fig1.pdf`: accuracy vs. round, with the authors' log and the paper value for comparison.
- `logs/`: full output of every attempt and run.
- Runtime: about `<X>` s per round on a T4, so `<T>` rounds × 3 seeds ≈ `<Y>` min.

| Metric | Paper | Mine (mean ± SD, 3 seeds) | Difference (%) |
|---|---|---|---|
| Accuracy at round `<T>` (authors' log) | `<value>` | `<value>` | `<value>` |
| Final accuracy (Fig. 2) | 94.06 | `<value>` | `<value>` |

## Differences from the paper
| Item | Paper / original repo | This reproduction | Reason |
|---|---|---|---|
| Framework | TensorFlow 1.13, Keras 2.2.4 | TensorFlow `<2.x>` (`tf.keras`) | old versions do not install on Colab |
| Training rounds T | 500 (~24 h per README) | `<T>` | free Colab time limit (~1 h) |
| Repetitions | 5 | 3 seeds (0, 1, 2) | time limit |
| Random seed | none set | fixed per run | assignment requirement |
| `CUDA_VISIBLE_DEVICES="2"` | set in notebook | removed | hides Colab's only GPU |
| Other hyper-parameters | ε=5, K=2, B=25, lr=0.05, public ratio 0.2, α=0.5 | unchanged | |

`results/authors_log.csv` is copied from the saved output of cell 7 of the authors' `Code/main.ipynb`. It has 496 of the 500 rounds, because a few lines are missing from that saved output.

## Citation
If you use this reproduction, please cite the original paper:
```bibtex
@article{qi2023privatekt,
  title   = {Differentially private knowledge transfer for federated learning},
  author  = {Qi, Tao and Wu, Fangzhao and Wu, Chuhan and He, Liang and Huang, Yongfeng and Xie, Xing},
  journal = {Nature Communications},
  volume  = {14},
  pages   = {3785},
  year    = {2023},
  doi     = {10.1038/s41467-023-38794-x}
}
```
Citation details for this repository are in `CITATION.cff`.
