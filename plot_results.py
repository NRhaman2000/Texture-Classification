"""
plot_results.py

Reads results.json (produced by run_experiments.py) and writes report figures:
  data_efficiency.png  - classical vs CNN accuracy against training-set size
  seed_variance.png    - CNN top-1 across seeds, with mean and std band

Run after run_experiments.py finishes (or partway through; it plots what exists).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#16283b"; TEAL = "#0f766e"; AMBER = "#b45309"; GRID = "#dbe3ec"
SEEDS = (1, 2, 3)
FRACTIONS = (0.25, 0.50, 1.00)
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})

r = json.load(open("results.json"))


def get(key, field="top1"):
    return r[key][field] * 100 if key in r else None


# ---------- data-efficiency curve ----------
xs, c_vals, n_vals = [], [], []
for f in FRACTIONS:
    c = get(f"classical_frac{int(f*100)}")
    n = get(f"cnn_frac{int(f*100)}_seed{SEEDS[0]}")
    if c is not None or n is not None:
        xs.append(int(f * 100)); c_vals.append(c); n_vals.append(n)

if xs:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    cx = [x for x, v in zip(xs, c_vals) if v is not None]
    cy = [v for v in c_vals if v is not None]
    nx = [x for x, v in zip(xs, n_vals) if v is not None]
    ny = [v for v in n_vals if v is not None]
    if cy:
        ax.plot(cx, cy, "o-", color=TEAL, lw=2.6, ms=8,
                label="Classical (Gabor + Fourier + LBP)")
        for x, v in zip(cx, cy):
            ax.annotate(f"{v:.1f}%", (x, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9,
                        weight="bold", color=TEAL)
    if ny:
        ax.plot(nx, ny, "s-", color=AMBER, lw=2.6, ms=8,
                label="CNN from scratch")
        for x, v in zip(nx, ny):
            ax.annotate(f"{v:.1f}%", (x, v), textcoords="offset points",
                        xytext=(0, -16), ha="center", fontsize=9,
                        weight="bold", color=AMBER)
    ax.set_xlabel("Training data used (% of the 1,880 training images)")
    ax.set_ylabel("Test top-1 accuracy (%)")
    ax.set_xticks(xs)
    ax.grid(True, color=GRID, lw=0.9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=True, edgecolor=GRID, fontsize=9.5, loc="upper left")
    ax.set_title("Data efficiency: how much each approach needs",
                 fontsize=12.5, weight="bold", loc="left", pad=10)
    plt.tight_layout()
    plt.savefig("data_efficiency.png", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote data_efficiency.png")

# ---------- seed variance ----------
accs = [get(f"cnn_frac100_seed{s}") for s in SEEDS]
accs = [a for a in accs if a is not None]
if len(accs) >= 2:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    labels = [f"seed {s}" for s in SEEDS[:len(accs)]]
    ax.bar(labels, accs, color=AMBER, width=0.55)
    m, sd = float(np.mean(accs)), float(np.std(accs))
    ax.axhline(m, color=INK, ls="--", lw=1.4)
    ax.fill_between([-0.5, len(accs) - 0.5], m - sd, m + sd,
                    color=INK, alpha=0.10, zorder=0)
    ax.set_xlim(-0.5, len(accs) - 0.5)
    for i, a in enumerate(accs):
        ax.text(i, a + 0.25, f"{a:.2f}%", ha="center",
                fontsize=10, weight="bold", color=INK)
    ax.text(len(accs) - 0.55, m + sd + 0.35,
            f"mean {m:.2f} $\\pm$ {sd:.2f}%", ha="right",
            fontsize=10, weight="bold", color=INK)
    ax.set_ylabel("Test top-1 accuracy (%)")
    ax.set_ylim(0, max(accs) * 1.25)
    ax.grid(True, axis="y", color=GRID, lw=0.9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("CNN run-to-run variance", fontsize=12.5,
                 weight="bold", loc="left", pad=10)
    plt.tight_layout()
    plt.savefig("seed_variance.png", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote seed_variance.png")

# ---------- text summary for the report ----------
print("\n--- paste into your report ---")
for f in FRACTIONS:
    k = f"classical_frac{int(f*100)}"
    if k in r:
        print(f"Classical {int(f*100):3d}%: top-1 {r[k]['top1']*100:.2f}%  "
              f"macro-F1 {r[k]['macro_f1']*100:.2f}%  params {r[k].get('best_params')}")
if accs:
    f1s = [r[f"cnn_frac100_seed{s}"]["macro_f1"] * 100
           for s in SEEDS if f"cnn_frac100_seed{s}" in r]
    print(f"CNN 100% over {len(accs)} seeds: top-1 {np.mean(accs):.2f} +/- {np.std(accs):.2f}%  "
          f"macro-F1 {np.mean(f1s):.2f} +/- {np.std(f1s):.2f}%")
