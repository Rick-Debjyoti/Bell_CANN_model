"""Generate all plots for the Bell-CANN paper.

Creates publication-quality figures for:
1. Training/validation loss curves
2. Embedding visualizations (2D/3D)
3. Model comparison bar chart
4. Zero-inflation probability analysis
5. Predicted vs observed by risk factor
6. Residual analysis (Pearson residuals)
7. ClaimNb distribution
8. Optuna convergence & parameter importance

Usage:
    cd src2/python
    python generate_plots.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import lambertw as W_scipy

# Style
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from config import SEED
from data_loader import load_data, get_model_inputs
from train import set_seeds, train_model, predict_standard, predict_zi, bias_regularize
from models import build_bell_nn, build_bell_cann
from zi_models import build_zibell_cann
from losses import bell_deviance, zibell_nll
from metrics import bell_deviance_metric

# Output directory
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def save_fig(name):
    """Save figure and close."""
    path = os.path.join(PLOT_DIR, f"{name}.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ===================================================================
# 1. ClaimNb Distribution
# ===================================================================

def plot_claim_distribution(data):
    """Bar chart of ClaimNb frequency distribution."""
    print("\n[1] Plotting ClaimNb distribution...")
    y = np.concatenate([data["ylearn"], data["ytest"]])

    counts = {}
    for v in range(5):
        if v < 4:
            counts[str(v)] = np.sum(y == v)
        else:
            counts["4+"] = np.sum(y >= 4)

    labels = list(counts.keys())
    values = list(counts.values())
    total = sum(values)
    pcts = [v / total * 100 for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=sns.color_palette("muted", len(labels)),
                  edgecolor="black", linewidth=0.5)

    for bar, pct, val in zip(bars, pcts, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.005,
                f"{pct:.1f}%\n(n={val:,})", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Number of Claims (ClaimNb)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Claim Counts — freMTPL2freq Dataset")
    ax.grid(axis="y", alpha=0.3)
    save_fig("claim_distribution")


# ===================================================================
# 2. Training / Validation Loss Curves
# ===================================================================

def plot_training_curves(histories, names):
    """Plot training and validation loss curves for multiple models."""
    print("\n[2] Plotting training curves...")
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 4.5), squeeze=False)

    colors = sns.color_palette("muted", 2)
    for i, (hist, name) in enumerate(zip(histories, names)):
        ax = axes[0, i]
        epochs = range(1, len(hist.history["loss"]) + 1)
        ax.plot(epochs, hist.history["loss"], color=colors[0],
                label="Training", linewidth=1.5, alpha=0.8)
        ax.plot(epochs, hist.history["val_loss"], color=colors[1],
                label="Validation", linewidth=1.5, alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(name)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_fig("training_validation_loss")


# ===================================================================
# 3. Embedding Visualizations
# ===================================================================

def plot_embeddings(model, data):
    """Visualize learned VehBrand and Region embeddings."""
    print("\n[3] Plotting embeddings...")

    brand_emb = model.get_layer("BrandEmb").get_weights()[0]
    region_emb = model.get_layer("RegionEmb").get_weights()[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # VehBrand embeddings
    ax = axes[0]
    for i in range(brand_emb.shape[0]):
        ax.scatter(brand_emb[i, 0], brand_emb[i, 1], s=60, zorder=3)
        ax.annotate(f"B{i}", (brand_emb[i, 0], brand_emb[i, 1]),
                    fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Embedding Dim 1")
    ax.set_ylabel("Embedding Dim 2")
    ax.set_title("VehBrand Embeddings")
    ax.grid(alpha=0.3)

    # Region embeddings
    ax = axes[1]
    for i in range(region_emb.shape[0]):
        ax.scatter(region_emb[i, 0], region_emb[i, 1], s=60, zorder=3)
        ax.annotate(f"R{i}", (region_emb[i, 0], region_emb[i, 1]),
                    fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Embedding Dim 1")
    ax.set_ylabel("Embedding Dim 2")
    ax.set_title("Region Embeddings")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_fig("embedding_visualization")


# ===================================================================
# 4. Model Comparison Bar Chart
# ===================================================================

def plot_model_comparison():
    """Bar chart comparing test deviance across all models."""
    print("\n[4] Plotting model comparison...")
    tuned_path = os.path.join(os.path.dirname(__file__), "comparison_results_tuned.csv")
    default_path = os.path.join(os.path.dirname(__file__), "comparison_results.csv")
    csv_path = tuned_path if os.path.exists(tuned_path) else default_path
    if not os.path.exists(csv_path):
        print("  Skipped: no comparison_results CSV found")
        return
    print(f"  Using: {os.path.basename(csv_path)}")

    df = pd.read_csv(csv_path)

    # Color by model family
    colors = []
    for m in df["Model"]:
        if "GLM" in m:
            colors.append("#4C72B0")
        elif "CANN" in m:
            colors.append("#DD8452")
        else:
            colors.append("#55A868")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(df)), df["Test Deviance"], color=colors,
                   edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Model"], fontsize=9)
    ax.set_xlabel("Test Deviance")
    ax.set_title("Model Comparison — Test Deviance (lower is better)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, df["Test Deviance"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="GLM"),
        Patch(facecolor="#55A868", label="NN"),
        Patch(facecolor="#DD8452", label="CANN"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    save_fig("model_comparison_deviance")


# ===================================================================
# 5. Zero-Inflation Analysis
# ===================================================================

def plot_zero_inflation(model, data, inputs_test):
    """Analyze and plot zero-inflation probability pi(x)."""
    print("\n[5] Plotting zero-inflation analysis...")
    raw = model.predict(inputs_test, verbose=0)
    mu_pred = raw[:, 0]
    pi_pred = raw[:, 1]

    test_df = data["test"].copy()
    test_df["pi_pred"] = pi_pred
    test_df["mu_pred"] = mu_pred

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Distribution of pi
    ax = axes[0, 0]
    ax.hist(pi_pred, bins=50, color="#4C72B0", edgecolor="black",
            linewidth=0.5, alpha=0.8)
    ax.set_xlabel(r"$\hat{\pi}$ (Zero-Inflation Probability)")
    ax.set_ylabel("Frequency")
    ax.set_title(r"Distribution of $\hat{\pi}$")
    ax.axvline(np.mean(pi_pred), color="red", linestyle="--",
               label=f"Mean={np.mean(pi_pred):.4f}")
    ax.legend()
    ax.grid(alpha=0.3)

    # Pi vs DrivAge
    ax = axes[0, 1]
    if "DrivAge" in test_df.columns:
        age_col = "DrivAge"
    else:
        age_col = "DrivAgeX"
    bins = pd.qcut(test_df[age_col], 10, duplicates="drop")
    grouped = test_df.groupby(bins)["pi_pred"].mean()
    ax.plot(range(len(grouped)), grouped.values, "o-", color="#DD8452")
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels([str(x) for x in grouped.index], rotation=45, fontsize=7)
    ax.set_xlabel("Driver Age (deciles)")
    ax.set_ylabel(r"Mean $\hat{\pi}$")
    ax.set_title(r"Zero-Inflation Probability vs Driver Age")
    ax.grid(alpha=0.3)

    # Pi vs BonusMalus
    ax = axes[1, 0]
    if "BonusMalus" in test_df.columns:
        bm_col = "BonusMalus"
    else:
        bm_col = "BonusMalusX"
    bins_bm = pd.qcut(test_df[bm_col], 10, duplicates="drop")
    grouped_bm = test_df.groupby(bins_bm)["pi_pred"].mean()
    ax.plot(range(len(grouped_bm)), grouped_bm.values, "o-", color="#55A868")
    ax.set_xticks(range(len(grouped_bm)))
    ax.set_xticklabels([str(x) for x in grouped_bm.index], rotation=45, fontsize=7)
    ax.set_xlabel("BonusMalus (deciles)")
    ax.set_ylabel(r"Mean $\hat{\pi}$")
    ax.set_title(r"Zero-Inflation Probability vs BonusMalus")
    ax.grid(alpha=0.3)

    # Pi vs Density
    ax = axes[1, 1]
    if "Density" in test_df.columns:
        den_col = "Density"
    else:
        den_col = "DensityX"
    bins_d = pd.qcut(test_df[den_col], 10, duplicates="drop")
    grouped_d = test_df.groupby(bins_d)["pi_pred"].mean()
    ax.plot(range(len(grouped_d)), grouped_d.values, "o-", color="#C44E52")
    ax.set_xticks(range(len(grouped_d)))
    ax.set_xticklabels([str(x) for x in grouped_d.index], rotation=45, fontsize=7)
    ax.set_xlabel("Density (deciles)")
    ax.set_ylabel(r"Mean $\hat{\pi}$")
    ax.set_title(r"Zero-Inflation Probability vs Density")
    ax.grid(alpha=0.3)

    plt.suptitle("ZI-Bell CANN: Zero-Inflation Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig("zero_inflation_analysis")


# ===================================================================
# 6. Predicted vs Observed by Risk Factor
# ===================================================================

def plot_predicted_vs_observed(y_pred, data, model_name="Bell CANN"):
    """Plot predicted vs observed claim frequency by risk factors."""
    print("\n[6] Plotting predicted vs observed...")
    test_df = data["test"].copy()
    test_df["y_pred"] = y_pred
    test_df["y_true"] = data["ytest"]

    risk_factors = []
    for col in ["DrivAge", "VehAge", "VehPower", "BonusMalus", "Density"]:
        if col in test_df.columns:
            risk_factors.append(col)

    if not risk_factors:
        risk_factors = ["DrivAgeX", "VehAgeX", "VehPowerX", "BonusMalusX", "DensityX"]

    n = len(risk_factors)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), squeeze=False)

    for i, col in enumerate(risk_factors):
        ax = axes[0, i]
        bins = pd.qcut(test_df[col], 10, duplicates="drop")
        grouped = test_df.groupby(bins).agg(
            obs=("y_true", "mean"),
            pred=("y_pred", "mean"),
        )
        x = range(len(grouped))
        ax.plot(x, grouped["obs"], "o-", color="#4C72B0", label="Observed", markersize=5)
        ax.plot(x, grouped["pred"], "s--", color="#DD8452", label="Predicted", markersize=5)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in grouped.index], rotation=45, fontsize=6)
        ax.set_xlabel(col)
        ax.set_ylabel("Mean Claim Frequency")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle(f"{model_name}: Predicted vs Observed by Risk Factor", fontsize=13)
    plt.tight_layout()
    save_fig("predicted_vs_observed")


# ===================================================================
# 7. Residual Analysis
# ===================================================================

def plot_residuals(y_pred, data, model_name="Bell CANN"):
    """Pearson residuals analysis."""
    print("\n[7] Plotting residual analysis...")
    y_true = data["ytest"]
    y_pred = np.maximum(y_pred, 1e-10)

    # Bell variance: Var(Y) = theta * (1 + W(theta)) where theta = mu
    w = np.real(W_scipy(y_pred))
    var_y = y_pred * (1 + w)
    pearson_r = (y_true - y_pred) / np.sqrt(var_y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Residuals vs fitted
    ax = axes[0]
    ax.scatter(y_pred, pearson_r, alpha=0.05, s=3, color="#4C72B0")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Pearson Residuals")
    ax.set_title("Residuals vs Fitted")
    ax.set_ylim(-5, 15)
    ax.grid(alpha=0.3)

    # QQ plot
    ax = axes[1]
    from scipy import stats
    sorted_r = np.sort(pearson_r)
    n = len(sorted_r)
    theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, n))
    ax.scatter(theoretical, sorted_r, alpha=0.05, s=3, color="#55A868")
    ax.plot([-4, 4], [-4, 4], "r--", linewidth=1)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("Q-Q Plot of Pearson Residuals")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-5, 15)
    ax.grid(alpha=0.3)

    # Histogram
    ax = axes[2]
    ax.hist(pearson_r, bins=100, density=True, color="#DD8452",
            edgecolor="black", linewidth=0.3, alpha=0.7)
    x_range = np.linspace(-4, 8, 200)
    ax.plot(x_range, stats.norm.pdf(x_range), "r-", linewidth=1.5,
            label="N(0,1)")
    ax.set_xlabel("Pearson Residuals")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Pearson Residuals")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f"{model_name}: Residual Analysis", fontsize=13)
    plt.tight_layout()
    save_fig("residual_analysis")


# ===================================================================
# 8. Optuna Visualization
# ===================================================================

def plot_optuna_results():
    """Plot Optuna optimization history and parameter importance."""
    print("\n[8] Plotting Optuna results...")
    results_dir = os.path.join(os.path.dirname(__file__), "optuna_results")
    summary_path = os.path.join(results_dir, "optuna_summary.csv")

    if not os.path.exists(summary_path):
        print("  Skipped: optuna_summary.csv not found")
        return

    summary = pd.read_csv(summary_path)

    # Bar chart of tuned deviances
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = []
    for m in summary["model"]:
        if "CANN" in m:
            colors.append("#DD8452")
        else:
            colors.append("#55A868")

    bars = ax.barh(range(len(summary)), summary["test_deviance"],
                   color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary["model"], fontsize=9)
    ax.set_xlabel("Test Deviance (Optuna-tuned)")
    ax.set_title("Optuna-Tuned Model Comparison")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, summary["test_deviance"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    save_fig("optuna_comparison")

    # Plot convergence for individual models
    trial_files = [f for f in os.listdir(results_dir) if f.endswith("_trials.csv")]
    if not trial_files:
        return

    # Pick key models for convergence plot
    key_models = ["Bell_CANN", "ZIBell_CANN", "Poisson_CANN", "NegBin_CANN"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for idx, model_name in enumerate(key_models):
        ax = axes[idx // 2, idx % 2]
        trial_file = os.path.join(results_dir, f"optuna_{model_name}_trials.csv")
        if not os.path.exists(trial_file):
            ax.set_visible(False)
            continue

        df = pd.read_csv(trial_file)
        if "value" not in df.columns:
            ax.set_visible(False)
            continue

        values = df["value"].values
        best_so_far = np.minimum.accumulate(values)
        ax.plot(range(1, len(values)+1), values, "o", color="#4C72B0",
                alpha=0.5, markersize=4, label="Trial value")
        ax.plot(range(1, len(best_so_far)+1), best_so_far, "-",
                color="#DD8452", linewidth=2, label="Best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Test Deviance")
        ax.set_title(model_name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Optuna Optimization Convergence", fontsize=13)
    plt.tight_layout()
    save_fig("optuna_convergence")


# ===================================================================
# 9. Activation Functions (for paper illustration)
# ===================================================================

def plot_activation_functions():
    """Plot activation functions used in the models."""
    print("\n[9] Plotting activation functions...")
    x = np.linspace(-5, 5, 500)

    activations = {
        "tanh": np.tanh(x),
        "ReLU": np.maximum(0, x),
        "SELU": np.where(x > 0, 1.0507 * x,
                         1.0507 * 1.6733 * (np.exp(x) - 1)),
        "Exponential": np.exp(np.clip(x, -5, 3)),
        "Sigmoid": 1 / (1 + np.exp(-x)),
    }

    fig, axes = plt.subplots(1, 5, figsize=(20, 3.5))
    colors = sns.color_palette("muted", 5)

    for i, (name, y) in enumerate(activations.items()):
        ax = axes[i]
        ax.plot(x, y, color=colors[i], linewidth=2)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.grid(alpha=0.3)
        if name == "Exponential":
            ax.set_ylim(-1, 25)

    plt.tight_layout()
    save_fig("activation_functions")


# ===================================================================
# Main
# ===================================================================

def main():
    set_seeds(SEED)
    data = load_data()

    print("=" * 70)
    print("GENERATING ALL PLOTS FOR PAPER")
    print(f"Output directory: {PLOT_DIR}")
    print("=" * 70)

    # --- Plots that don't need model training ---
    plot_claim_distribution(data)
    plot_model_comparison()
    plot_optuna_results()
    plot_activation_functions()

    # --- Load or train key models for remaining plots ---
    q0 = data["q0"]
    n_br = data["n_brands"]
    n_re = data["n_regions"]
    ylearn = data["ylearn"]
    ytest = data["ytest"]
    Elearn = data["Elearn"]

    lambda_hom = np.sum(ylearn) / np.sum(Elearn)

    glm_learn = data["glm_preds_learn"]
    glm_test = data["glm_preds_test"]

    # Bell CANN
    lv_learn = np.log(np.maximum(glm_learn["bellGLM"], 1e-10))
    lv_test = np.log(np.maximum(glm_test["bellGLM"], 1e-10))
    lambda_cann = np.sum(ylearn) / np.sum(glm_learn["bellGLM"])

    bell_cann = build_bell_cann(q0=q0, n_brands=n_br, n_regions=n_re,
                                lambda_hom=lambda_cann)
    inputs_learn = get_model_inputs(
        data["Xlearn"], data["BrandLearn"], data["RegionLearn"], lv_learn)
    inputs_test = get_model_inputs(
        data["Xtest"], data["BrandTest"], data["RegionTest"], lv_test)

    # Try tuned weights first, then default weights, then train from scratch
    saved_bell_tuned = os.path.join(os.path.dirname(__file__), "saved_models_tuned", "Bell_CANN.h5")
    saved_bell = os.path.join(os.path.dirname(__file__), "saved_models", "Bell_CANN.h5")
    hist_bell_path = os.path.join(os.path.dirname(__file__), "training_histories", "Bell_CANN_history.pkl")

    if os.path.exists(saved_bell_tuned):
        print("\n  Loading tuned Bell CANN weights...")
        bell_cann.load_weights(saved_bell_tuned)
    elif os.path.exists(saved_bell):
        print("\n  Loading saved Bell CANN weights...")
        bell_cann.load_weights(saved_bell)
    else:
        print("\n  Training Bell CANN for plots (no saved weights found)...")
        set_seeds(SEED)
        bell_cann, _ = train_model(bell_cann, inputs_learn, ylearn)
        bell_cann = bias_regularize(bell_cann, inputs_learn, ylearn)

    # Load training history if available
    bell_hist = None
    if os.path.exists(hist_bell_path):
        import pickle
        with open(hist_bell_path, "rb") as f:
            bell_hist_dict = pickle.load(f)
        # Wrap in a simple object with .history attribute
        class HistoryWrapper:
            def __init__(self, d): self.history = d
        bell_hist = HistoryWrapper(bell_hist_dict)

    bell_pred = predict_standard(bell_cann, inputs_test)

    # ZI-Bell CANN
    lv_learn_zi = np.log(np.maximum(glm_learn["mu_zibell"], 1e-10))
    lv_test_zi = np.log(np.maximum(glm_test["mu_zibell"], 1e-10))
    lambda_zi = np.sum(ylearn) / np.sum(glm_learn["mu_zibell"])
    p_zero = np.mean(ylearn == 0)
    pi_logit_init = np.log(p_zero / (1 - p_zero))

    zibell_cann = build_zibell_cann(q0=q0, n_brands=n_br, n_regions=n_re,
                                     lambda_hom=lambda_zi, pi_init=pi_logit_init)
    inputs_learn_zi = get_model_inputs(
        data["Xlearn"], data["BrandLearn"], data["RegionLearn"], lv_learn_zi)
    inputs_test_zi = get_model_inputs(
        data["Xtest"], data["BrandTest"], data["RegionTest"], lv_test_zi)

    saved_zi_tuned = os.path.join(os.path.dirname(__file__), "saved_models_tuned", "ZI_Bell_CANN.h5")
    saved_zibell = os.path.join(os.path.dirname(__file__), "saved_models", "ZI_Bell_CANN.h5")
    hist_zi_path = os.path.join(os.path.dirname(__file__), "training_histories", "ZI_Bell_CANN_history.pkl")

    if os.path.exists(saved_zi_tuned):
        print("  Loading tuned ZI-Bell CANN weights...")
        zibell_cann.load_weights(saved_zi_tuned)
    elif os.path.exists(saved_zibell):
        print("  Loading saved ZI-Bell CANN weights...")
        zibell_cann.load_weights(saved_zibell)
    else:
        print("  Training ZI-Bell CANN for plots (no saved weights found)...")
        set_seeds(SEED)
        zibell_cann, _ = train_model(zibell_cann, inputs_learn_zi, ylearn)

    # Load training history if available
    zi_hist = None
    if os.path.exists(hist_zi_path):
        import pickle
        with open(hist_zi_path, "rb") as f:
            zi_hist_dict = pickle.load(f)
        zi_hist = HistoryWrapper(zi_hist_dict)

    # --- Generate model-dependent plots ---
    if bell_hist is not None and zi_hist is not None:
        plot_training_curves(
            [bell_hist, zi_hist],
            ["Bell CANN", "ZI-Bell CANN"]
        )
    else:
        print("\n  Skipping training curves (loaded from saved weights)")

    plot_embeddings(bell_cann, data)
    plot_zero_inflation(zibell_cann, data, inputs_test_zi)
    plot_predicted_vs_observed(bell_pred, data, "Bell CANN")
    plot_residuals(bell_pred, data, "Bell CANN")

    print(f"\n{'=' * 70}")
    print(f"All plots saved to {PLOT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
