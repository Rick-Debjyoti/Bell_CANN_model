"""Generate publication-quality figures for the ZI-Bell CANN paper."""

import pickle
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BASE = "D:/Projects/FINAL REPOS/Bell_CANN_model"
HIST_DIR = f"{BASE}/src2/python/training_histories/training_histories"
FIG_DIR = f"{BASE}/Reports/temp/ZI_Bell_CANN_Paper_v2/figures"
CSV_PATH = f"{BASE}/src2/python/comparison_results_tuned.csv"

os.makedirs(FIG_DIR, exist_ok=True)

# ─── Figure 1: Training curves (2x3) ─────────────────────────────────────────

models_curve = [
    ("Bell_CANN", "Bell CANN"),
    ("ZI_Bell_CANN", "ZI-Bell CANN"),
    ("Poisson_CANN", "Poisson CANN"),
    ("NegBin_CANN", "NegBin CANN"),
    ("Bell_NN", "Bell NN"),
    ("ZI_Bell_NN", "ZI-Bell NN"),
]

fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=300)
axes = axes.flatten()

for idx, (fname, label) in enumerate(models_curve):
    pkl_path = os.path.join(HIST_DIR, f"{fname}_history.pkl")
    with open(pkl_path, "rb") as f:
        hist = pickle.load(f)

    epochs = range(1, len(hist["loss"]) + 1)
    ax = axes[idx]
    ax.plot(epochs, hist["loss"], color="#1f77b4", linewidth=1.2, label="Training")
    ax.plot(epochs, hist["val_loss"], color="#ff7f0e", linewidth=1.2, label="Validation")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    # Let y-axis auto-scale per subplot for clarity
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))

fig.suptitle("Training and Validation Loss Curves", fontsize=13, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out1 = os.path.join(FIG_DIR, "training_curves.png")
fig.savefig(out1, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out1}")

# ─── Figure 2: Grouped bar chart of test deviance ────────────────────────────

df = pd.read_csv(CSV_PATH)

# Normalize model names for parsing
name_map = {
    "Poisson GLM": ("Poisson", "GLM", False),
    "Bell GLM": ("Bell", "GLM", False),
    "NegBin GLM": ("NegBin", "GLM", False),
    "ZIP GLM": ("ZIP", "GLM", True),
    "ZINB GLM": ("ZINB", "GLM", True),
    "ZI-Bell GLM": ("ZI-Bell", "GLM", True),
    "Poisson NN": ("Poisson", "NN", False),
    "Bell NN": ("Bell", "NN", False),
    "NegBin NN": ("NegBin", "NN", False),
    "Poisson CANN": ("Poisson", "CANN", False),
    "Bell CANN": ("Bell", "CANN", False),
    "NegBin CANN": ("NegBin", "CANN", False),
    "ZIP-NN": ("ZIP", "NN", True),
    "ZINB-NN": ("ZINB", "NN", True),
    "ZI-Bell-NN": ("ZI-Bell", "NN", True),
    "ZIP-CANN": ("ZIP", "CANN", True),
    "ZINB-CANN": ("ZINB", "CANN", True),
    "ZI-Bell-CANN": ("ZI-Bell", "CANN", True),
}

# Map base distribution family
dist_family = {
    "Poisson": "Poisson", "ZIP": "Poisson",
    "Bell": "Bell", "ZI-Bell": "Bell",
    "NegBin": "NegBin", "ZINB": "NegBin",
}

arch_colors = {"GLM": "#888888", "NN": "#1f77b4", "CANN": "#d62728"}

# Build structured data
rows = []
for _, row in df.iterrows():
    model = row["Model"]
    if model in name_map:
        dist, arch, zi = name_map[model]
        family = dist_family[dist]
        rows.append({
            "Model": model, "Distribution": dist, "Family": family,
            "Architecture": arch, "ZI": zi,
            "Deviance": row["Test Deviance"],
        })

data = pd.DataFrame(rows)

# Order: within each family, show GLM/NN/CANN, non-ZI first then ZI
family_order = ["Poisson", "Bell", "NegBin"]
arch_order = ["GLM", "NN", "CANN"]

ordered = []
for fam in family_order:
    for arch in arch_order:
        sub = data[(data["Family"] == fam) & (data["Architecture"] == arch)]
        # non-ZI first
        for _, r in sub[~sub["ZI"]].iterrows():
            ordered.append(r)
        for _, r in sub[sub["ZI"]].iterrows():
            ordered.append(r)

ordered_df = pd.DataFrame(ordered).reset_index(drop=True)

fig2, ax2 = plt.subplots(figsize=(14, 5.5), dpi=300)

x = np.arange(len(ordered_df))
bar_colors = [arch_colors[a] for a in ordered_df["Architecture"]]
hatches = ["///" if zi else "" for zi in ordered_df["ZI"]]

bars = ax2.bar(x, ordered_df["Deviance"], color=bar_colors, edgecolor="black",
               linewidth=0.6, width=0.7)
for bar, h in zip(bars, hatches):
    bar.set_hatch(h)

ax2.set_xticks(x)
ax2.set_xticklabels(ordered_df["Model"], rotation=45, ha="right", fontsize=8)
ax2.set_ylabel("Test Deviance", fontsize=11)
ax2.set_title("Test Deviance by Model", fontsize=13, fontweight="bold")
ax2.tick_params(axis="y", labelsize=9)

# Set y-axis to zoom into the relevant range
ymin = ordered_df["Deviance"].min() * 0.96
ymax = ordered_df["Deviance"].max() * 1.02
ax2.set_ylim(ymin, ymax)

# Add family group separators
# Count models per family
family_counts = []
for fam in family_order:
    n = len(ordered_df[ordered_df["Family"] == fam])
    family_counts.append(n)

cumsum = 0
for i, (fam, cnt) in enumerate(zip(family_order, family_counts)):
    mid = cumsum + cnt / 2 - 0.5
    ax2.text(mid, ymax * 0.999, fam, ha="center", va="top", fontsize=10,
             fontweight="bold", fontstyle="italic")
    if i < len(family_order) - 1:
        sep = cumsum + cnt - 0.5
        ax2.axvline(sep, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    cumsum += cnt

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=arch_colors["GLM"], edgecolor="black", label="GLM"),
    Patch(facecolor=arch_colors["NN"], edgecolor="black", label="NN"),
    Patch(facecolor=arch_colors["CANN"], edgecolor="black", label="CANN"),
    Patch(facecolor="white", edgecolor="black", hatch="///", label="Zero-Inflated"),
]
ax2.legend(handles=legend_elements, fontsize=9, frameon=True, loc="upper right")

fig2.tight_layout()
out2 = os.path.join(FIG_DIR, "model_comparison_barchart.png")
fig2.savefig(out2, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"Saved: {out2}")

print("Done.")
