"""Fig. 1: reproduced PrivateKT accuracy vs. the paper. Run from repo root: python make_figure.py"""
from pathlib import Path
import matplotlib.pyplot as plt, pandas as pd
ROOT = Path(__file__).resolve().parent
PAPER_FINAL = 94.06  # Fig. 2, MNIST, Class Non-IID, with LDP (eps=5), 500 rounds
mine = pd.read_csv(ROOT / "results/my_results.csv"); auth = pd.read_csv(ROOT / "results/authors_log.csv")
T = int(mine["round"].max()) + 1; n = mine["seed"].nunique()
st = mine.groupby("round")["accuracy"].agg(["mean", "std"]) * 100; auth = auth[auth["round"] < T]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#555555"   # Okabe-Ito (colour-blind safe)
fig, ax = plt.subplots(figsize=(3.5, 2.3)); x = st.index + 1
ax.plot(x, st["mean"], color=BLUE, lw=1.4, marker="o", ms=2.5, label=f"Ours (mean of {n} seeds)")
ax.fill_between(x, st["mean"] - st["std"], st["mean"] + st["std"], color=BLUE, alpha=0.2, lw=0, hatch="////", label="Ours ±1 SD")
ax.plot(auth["round"] + 1, auth["accuracy"] * 100, color=ORANGE, lw=1.2, ls="--", marker="s", ms=2, label="Authors' published log", zorder=3)
ax.axhline(PAPER_FINAL, color=GREY, lw=1.0, ls=":", label=f"Paper, 500 rounds ({PAPER_FINAL:.2f}%)")
ax.set_xlabel("Communication round"); ax.set_ylabel("Test accuracy (%)")
ax.set_xlim(1, T); ax.set_ylim(0, 100); ax.grid(True, lw=0.3, alpha=0.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.savefig(ROOT / "fig1.pdf", bbox_inches="tight"); fig.savefig(ROOT / "fig1_preview.png", dpi=200, bbox_inches="tight")
from PIL import Image; Image.open(ROOT / "fig1_preview.png").convert("L").save(ROOT / "fig1_grayscale_check.png")
print(f"Saved fig1.pdf ({T} rounds, {n} seeds)")
