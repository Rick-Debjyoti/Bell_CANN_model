"""LocalGLMnet: Interpretable Bell CANN and ZI-Bell CANN.

Implements the LocalGLMnet architecture (Richman & Wüthrich, 2023) for
Bell and ZI-Bell distributions. Instead of a black-box prediction:

    ŷ = f_NN(x)

LocalGLMnet produces:

    ŷ = exp(β₀ + Σ βⱼ(x)·xⱼ + LogVol)

where β(x) are learned, feature-dependent attention weights.
Each prediction decomposes into additive feature contributions,
enabling local interpretability while maintaining NN flexibility.

This is a NOVEL contribution: LocalGLMnet has not been applied to
Bell distribution before. Applying it to ZI-Bell (both mu and pi
components) extends Jose et al. (2024) who only used ZIP.

Usage:
    cd src2/python
    python localglmnet.py
"""

import os
import pickle
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, Model

from config import SEED, EPOCHS, PATIENCE, LEARNING_RATE, BATCH_SIZE, VALIDATION_SPLIT
from data_loader import load_data, get_model_inputs, NN_FEATURES
from train import set_seeds
from losses import bell_deviance, zibell_nll
from metrics import bell_deviance_metric

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 11,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})

PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved_models_tuned", "saved_models_tuned")
os.makedirs(PLOT_DIR, exist_ok=True)


# ===================================================================
# LocalGLMnet Bell CANN
# ===================================================================

def build_localglmnet_bell_cann(q0, n_brands, n_regions, d=2,
                                 hidden_layers=None, lr=LEARNING_RATE,
                                 lambda_hom=0.0, activation="tanh"):
    """Build LocalGLMnet Bell CANN.

    Architecture:
        Input features x → NN → β(x)  [attention weights]
        Prediction: exp(β₀ + Σ βⱼ(x)·xⱼ + LogVol)

    The key difference from standard CANN: instead of Dense(1) at the end,
    the network outputs q_total attention weights, one per input feature.
    The prediction is the dot product β(x)·x, preserving interpretability.

    Parameters
    ----------
    q0 : int — number of continuous features
    n_brands, n_regions : int — embedding categories
    d : int — embedding dimension
    hidden_layers : list of int
    lr : float
    lambda_hom : float — bias for intercept
    activation : str

    Returns
    -------
    model : keras.Model
        Outputs: prediction (shape n,1)
    attention_model : keras.Model
        Outputs: attention weights β(x) (shape n, q_total)
    """
    if hidden_layers is None:
        hidden_layers = [100, 80, 60, 40]

    q_total = q0 + 2 * d  # continuous + brand_emb + region_emb

    # Inputs
    Design   = layers.Input(shape=(q0,), name="Design")
    VehBrand = layers.Input(shape=(1,), name="VehBrand")
    Region   = layers.Input(shape=(1,), name="Region")
    LogVol   = layers.Input(shape=(1,), name="LogVol")

    # Embeddings
    brand_emb  = layers.Embedding(n_brands, d, input_length=1, name="BrandEmb")(VehBrand)
    brand_flat = layers.Flatten(name="BrandFlat")(brand_emb)
    region_emb = layers.Embedding(n_regions, d, input_length=1, name="RegionEmb")(Region)
    region_flat = layers.Flatten(name="RegionFlat")(region_emb)

    # All features concatenated (this is x)
    features = layers.Concatenate(name="all_features")(
        [Design, brand_flat, region_flat])

    # Attention network: learns β(x)
    x = features
    for i, units in enumerate(hidden_layers):
        x = layers.Dense(units, activation=activation, name=f"attn_hidden_{i}")(x)

    # Output: q_total attention weights (one per feature)
    beta = layers.Dense(
        q_total, activation="linear", name="attention_weights"
    )(x)

    # Intercept (learnable bias)
    beta_0 = layers.Dense(
        1, activation="linear", use_bias=True,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(lambda_hom),
        name="intercept"
    )(layers.Lambda(lambda t: t[:, :1] * 0, name="zeros_input")(features))

    # LocalGLM prediction: β₀ + Σ βⱼ(x)·xⱼ
    contributions = layers.Multiply(name="contributions")([beta, features])
    linear_pred = layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=-1, keepdims=True),
        name="sum_contributions"
    )(contributions)

    network_out = layers.Add(name="network_out")([linear_pred, beta_0])

    # Add skip connection (LogVol) and exponentiate
    add_out = layers.Add(name="add_logvol")([network_out, LogVol])
    response = layers.Dense(
        1, activation=tf.keras.activations.exponential,
        dtype="float64", trainable=False,
        kernel_initializer="ones", bias_initializer="zeros",
        name="response"
    )(add_out)

    model = Model(inputs=[Design, VehBrand, Region, LogVol], outputs=response)
    model.compile(optimizer=tf.keras.optimizers.Nadam(lr), loss=bell_deviance)

    # Separate model for extracting attention weights
    attention_model = Model(
        inputs=[Design, VehBrand, Region, LogVol],
        outputs=[beta, contributions]
    )

    return model, attention_model


# ===================================================================
# LocalGLMnet ZI-Bell CANN (Novel: both mu and pi interpretable)
# ===================================================================

def build_localglmnet_zibell_cann(q0, n_brands, n_regions, d=2,
                                   hidden_layers=None, lr=LEARNING_RATE,
                                   lambda_hom=0.0, pi_init=0.0,
                                   activation="tanh"):
    """Build LocalGLMnet ZI-Bell CANN with interpretable mu AND pi.

    Two sets of attention weights:
        β_μ(x) → explains what drives claim frequency
        β_π(x) → explains what drives excess zeros

    This extends Jose et al. (2024) from ZIP to ZI-Bell.
    """
    if hidden_layers is None:
        hidden_layers = [100, 80, 60, 40]

    q_total = q0 + 2 * d

    Design   = layers.Input(shape=(q0,), name="Design")
    VehBrand = layers.Input(shape=(1,), name="VehBrand")
    Region   = layers.Input(shape=(1,), name="Region")
    LogVol   = layers.Input(shape=(1,), name="LogVol")

    brand_emb  = layers.Embedding(n_brands, d, input_length=1, name="BrandEmb")(VehBrand)
    brand_flat = layers.Flatten(name="BrandFlat")(brand_emb)
    region_emb = layers.Embedding(n_regions, d, input_length=1, name="RegionEmb")(Region)
    region_flat = layers.Flatten(name="RegionFlat")(region_emb)

    features = layers.Concatenate(name="all_features")(
        [Design, brand_flat, region_flat])

    # Shared hidden layers
    x = features
    for i, units in enumerate(hidden_layers):
        x = layers.Dense(units, activation=activation, name=f"shared_hidden_{i}")(x)

    # Mu branch: attention weights for count rate
    beta_mu = layers.Dense(q_total, activation="linear", name="beta_mu")(x)
    beta_mu_0 = layers.Dense(
        1, activation="linear", use_bias=True,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(lambda_hom),
        name="mu_intercept"
    )(layers.Lambda(lambda t: t[:, :1] * 0)(features))

    contrib_mu = layers.Multiply(name="mu_contributions")([beta_mu, features])
    linear_mu = layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=-1, keepdims=True),
        name="sum_mu"
    )(contrib_mu)
    raw_mu = layers.Add(name="raw_mu")([linear_mu, beta_mu_0])

    # Pi branch: attention weights for zero-inflation
    beta_pi = layers.Dense(q_total, activation="linear", name="beta_pi")(x)
    beta_pi_0 = layers.Dense(
        1, activation="linear", use_bias=True,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(pi_init),
        name="pi_intercept"
    )(layers.Lambda(lambda t: t[:, :1] * 0)(features))

    contrib_pi = layers.Multiply(name="pi_contributions")([beta_pi, features])
    linear_pi = layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=-1, keepdims=True),
        name="sum_pi"
    )(contrib_pi)
    raw_pi = layers.Add(name="raw_pi_out")([linear_pi, beta_pi_0])

    # Mu: add LogVol then exp
    mu_plus_vol = layers.Add(name="mu_add_logvol")([raw_mu, LogVol])
    mu_out = layers.Lambda(
        lambda t: tf.exp(tf.cast(t, tf.float64)), name="mu_exp", dtype="float64"
    )(mu_plus_vol)

    # Pi: sigmoid
    pi_out = layers.Lambda(
        lambda t: tf.sigmoid(tf.cast(t, tf.float64)), name="pi_sigmoid", dtype="float64"
    )(raw_pi)

    output = layers.Concatenate(name="mu_pi_output", dtype="float64")([mu_out, pi_out])

    model = Model(inputs=[Design, VehBrand, Region, LogVol], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Nadam(lr), loss=zibell_nll)

    # Attention models
    attention_model = Model(
        inputs=[Design, VehBrand, Region, LogVol],
        outputs=[beta_mu, contrib_mu, beta_pi, contrib_pi]
    )

    return model, attention_model


# ===================================================================
# Feature contribution plots
# ===================================================================

def get_feature_names(q0, d):
    """Get full feature names including embedding dimensions."""
    names = list(NN_FEATURES)
    for i in range(d):
        names.append(f"BrandEmb_{i+1}")
    for i in range(d):
        names.append(f"RegionEmb_{i+1}")
    return names


def plot_feature_contributions(contributions, feature_names, model_name,
                               n_samples=5000):
    """Plot feature contributions β_j(x) * x_j for each feature."""
    n = min(n_samples, contributions.shape[0])
    idx = np.random.choice(contributions.shape[0], n, replace=False)
    contrib_sample = contributions[idx]

    n_features = len(feature_names)
    cols = min(4, n_features)
    rows = (n_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = np.atleast_2d(axes)

    for i, fname in enumerate(feature_names):
        ax = axes[i // cols, i % cols]
        ax.scatter(range(n), contrib_sample[:, i], alpha=0.1, s=1,
                   color="#4C72B0")
        ax.axhline(0, color="red", linewidth=0.5, linestyle="--")
        ax.set_title(fname, fontsize=9)
        ax.set_ylabel(r"$\beta_j(x) \cdot x_j$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    # Hide unused axes
    for i in range(n_features, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    plt.suptitle(f"{model_name}: Feature Contributions", fontsize=13)
    plt.tight_layout()


def plot_attention_weights(betas, feature_names, model_name,
                           n_samples=5000):
    """Plot attention weight distributions β_j(x) per feature."""
    n = min(n_samples, betas.shape[0])
    idx = np.random.choice(betas.shape[0], n, replace=False)
    beta_sample = betas[idx]

    # Variable importance: mean |β_j|
    importance = np.mean(np.abs(beta_sample), axis=0)
    sorted_idx = np.argsort(importance)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of variable importance
    ax = axes[0]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_imp = importance[sorted_idx]
    bars = ax.barh(range(len(sorted_names)), sorted_imp,
                   color=sns.color_palette("muted", len(sorted_names)),
                   edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel(r"Mean $|\beta_j(x)|$")
    ax.set_title("Variable Importance")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Box plot of attention weights
    ax = axes[1]
    # Show top 7 most important features
    top_k = min(7, len(feature_names))
    top_idx = sorted_idx[:top_k]
    top_names = [feature_names[i] for i in top_idx]
    top_betas = beta_sample[:, top_idx]
    ax.boxplot(top_betas, labels=top_names, vert=True, showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.5))
    ax.axhline(0, color="red", linewidth=0.5, linestyle="--")
    ax.set_ylabel(r"$\beta_j(x)$")
    ax.set_title("Attention Weight Distributions (Top Features)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3)

    plt.suptitle(f"{model_name}: LocalGLMnet Interpretation", fontsize=13)
    plt.tight_layout()


def plot_zi_comparison(contrib_mu, contrib_pi, feature_names, n_samples=3000):
    """Compare mu vs pi feature contributions for ZI-Bell LocalGLMnet."""
    n = min(n_samples, contrib_mu.shape[0])
    idx = np.random.choice(contrib_mu.shape[0], n, replace=False)

    imp_mu = np.mean(np.abs(contrib_mu[idx]), axis=0)
    imp_pi = np.mean(np.abs(contrib_pi[idx]), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Side-by-side importance
    x_pos = np.arange(len(feature_names))
    width = 0.35

    ax = axes[0]
    ax.barh(x_pos - width/2, imp_mu, width, label=r"$\mu$ (count rate)",
            color="#4C72B0", edgecolor="black", linewidth=0.3)
    ax.barh(x_pos + width/2, imp_pi, width, label=r"$\pi$ (zero-inflation)",
            color="#DD8452", edgecolor="black", linewidth=0.3)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_xlabel("Mean |Contribution|")
    ax.set_title("Feature Importance: Count vs Zero-Inflation")
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Ratio plot: which features drive zeros vs counts
    ax = axes[1]
    ratio = imp_pi / (imp_mu + 1e-10)
    sorted_ratio_idx = np.argsort(ratio)[::-1]
    sorted_names = [feature_names[i] for i in sorted_ratio_idx]
    sorted_ratios = ratio[sorted_ratio_idx]
    colors = ["#DD8452" if r > 1 else "#4C72B0" for r in sorted_ratios]
    ax.barh(range(len(sorted_names)), sorted_ratios, color=colors,
            edgecolor="black", linewidth=0.3)
    ax.axvline(1.0, color="red", linewidth=1, linestyle="--",
               label="Equal importance")
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel(r"$|\pi$-contribution| / |$\mu$-contribution|")
    ax.set_title("Zero-Inflation vs Count Rate Dominance")
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.suptitle("ZI-Bell LocalGLMnet: Dual Interpretability", fontsize=13)
    plt.tight_layout()


# ===================================================================
# Main
# ===================================================================

def main():
    set_seeds(SEED)
    data = load_data()

    q0 = data["q0"]
    n_br = data["n_brands"]
    n_re = data["n_regions"]
    d = 2  # embedding dim (per Richman & Wüthrich 2023)
    ylearn = data["ylearn"]
    ytest = data["ytest"]
    glm_learn = data["glm_preds_learn"]
    glm_test = data["glm_preds_test"]

    feature_names = get_feature_names(q0, d)

    Elearn = data["Elearn"]
    Etest = data["Etest"]

    print("=" * 70)
    print("LocalGLMnet: INTERPRETABLE BELL MODELS")
    print("  Architecture per Richman & Wüthrich (2023):")
    print("  hidden=[100,80,60,40], tanh, d=2, no dropout")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. LocalGLMnet Bell NN (pure NN — total feature effects)
    #    Per Richman & Wüthrich (2023) and Jose et al. (2024):
    #    offset = log(Exposure), β(x)·x = total effect of each feature
    # ------------------------------------------------------------------
    print("\n[1] Training LocalGLMnet Bell NN...")
    print("    (offset=log(Exposure) — gives TOTAL feature effects)")

    lv_nn_learn = np.log(Elearn)
    lv_nn_test = np.log(Etest)
    lambda_nn = np.sum(ylearn) / np.sum(Elearn)

    inputs_nn_learn = get_model_inputs(
        data["Xlearn"], data["BrandLearn"], data["RegionLearn"], lv_nn_learn)
    inputs_nn_test = get_model_inputs(
        data["Xtest"], data["BrandTest"], data["RegionTest"], lv_nn_test)

    nn_model, nn_attn_model = build_localglmnet_bell_cann(
        q0, n_br, n_re, d=d, lambda_hom=lambda_nn)

    cbs = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=PATIENCE,
        restore_best_weights=True)]

    nn_model.fit(
        inputs_nn_learn, ylearn.reshape(-1),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=cbs, verbose=2,
    )

    # Bias regularization
    preds_nn_learn = nn_model.predict(inputs_nn_learn, verbose=0).ravel()
    ratio_nn = np.mean(ylearn) / np.mean(preds_nn_learn)
    intercept_nn = nn_model.get_layer("intercept")
    iw_nn = intercept_nn.get_weights()
    iw_nn[1] = iw_nn[1] + np.log(ratio_nn)
    intercept_nn.set_weights(iw_nn)

    preds_nn = nn_model.predict(inputs_nn_test, verbose=0).ravel()
    dev_nn = bell_deviance_metric(ytest, preds_nn)
    print(f"  LocalGLMnet Bell NN test deviance: {dev_nn:.6f}")

    os.makedirs(SAVED_DIR, exist_ok=True)
    nn_model.save_weights(os.path.join(SAVED_DIR, "LocalGLMnet_Bell_NN.weights.h5"))

    betas_nn, contribs_nn = nn_attn_model.predict(inputs_nn_test, verbose=0)

    plot_attention_weights(betas_nn, feature_names, "LocalGLMnet Bell NN (Total Effects)")
    plt.savefig(os.path.join(PLOT_DIR, "localglmnet_bell_nn_importance.png"))
    plt.close()
    print("  Saved: localglmnet_bell_nn_importance.png")

    plot_feature_contributions(contribs_nn, feature_names, "LocalGLMnet Bell NN")
    plt.savefig(os.path.join(PLOT_DIR, "localglmnet_bell_nn_contributions.png"))
    plt.close()
    print("  Saved: localglmnet_bell_nn_contributions.png")

    # ------------------------------------------------------------------
    # 2. LocalGLMnet Bell CANN (residual effects beyond GLM)
    #    offset = log(bellGLM), β(x)·x = NN correction to GLM
    # ------------------------------------------------------------------
    set_seeds(SEED)
    print(f"\n{'=' * 70}")
    print("[2] Training LocalGLMnet Bell CANN...")
    print("    (offset=log(bellGLM) — gives RESIDUAL effects beyond GLM)")

    lv_learn = np.log(np.maximum(glm_learn["bellGLM"], 1e-10))
    lv_test = np.log(np.maximum(glm_test["bellGLM"], 1e-10))
    lambda_cann = np.sum(ylearn) / np.sum(glm_learn["bellGLM"])

    inputs_learn = get_model_inputs(
        data["Xlearn"], data["BrandLearn"], data["RegionLearn"], lv_learn)
    inputs_test = get_model_inputs(
        data["Xtest"], data["BrandTest"], data["RegionTest"], lv_test)

    model, attn_model = build_localglmnet_bell_cann(
        q0, n_br, n_re, d=d, lambda_hom=lambda_cann)

    cbs = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=PATIENCE,
        restore_best_weights=True)]

    model.fit(
        inputs_learn, ylearn.reshape(-1),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=cbs, verbose=2,
    )

    # Bias regularization: adjust intercept bias for portfolio balance
    preds_learn = model.predict(inputs_learn, verbose=0).ravel()
    ratio = np.mean(ylearn) / np.mean(preds_learn)
    intercept_layer = model.get_layer("intercept")
    iw = intercept_layer.get_weights()
    iw[1] = iw[1] + np.log(ratio)
    intercept_layer.set_weights(iw)

    preds = model.predict(inputs_test, verbose=0).ravel()
    dev = bell_deviance_metric(ytest, preds)
    print(f"  LocalGLMnet Bell CANN test deviance: {dev:.6f}")

    # Save weights
    os.makedirs(SAVED_DIR, exist_ok=True)
    model.save_weights(os.path.join(SAVED_DIR, "LocalGLMnet_Bell_CANN.weights.h5"))

    # Extract attention weights and contributions
    betas, contribs = attn_model.predict(inputs_test, verbose=0)

    # Plot
    plot_attention_weights(betas, feature_names, "LocalGLMnet Bell CANN")
    plt.savefig(os.path.join(PLOT_DIR, "localglmnet_bell_importance.png"))
    plt.close()
    print("  Saved: localglmnet_bell_importance.png")

    plot_feature_contributions(contribs, feature_names, "LocalGLMnet Bell CANN")
    plt.savefig(os.path.join(PLOT_DIR, "localglmnet_bell_contributions.png"))
    plt.close()
    print("  Saved: localglmnet_bell_contributions.png")

    # ------------------------------------------------------------------
    # 3. LocalGLMnet ZI-Bell NN (Novel: total effects, both mu and pi)
    #    Per Jose et al. (2024) extension from ZIP to ZI-Bell
    # ------------------------------------------------------------------
    set_seeds(SEED)
    print(f"\n{'=' * 70}")
    print("[3] Training LocalGLMnet ZI-Bell NN (NOVEL)...")
    print("    (offset=log(Exposure) — total effects for mu and pi)")

    p_zero = np.mean(ylearn == 0)
    pi_logit_init = np.log(p_zero / (1 - p_zero))

    zi_nn_model, zi_nn_attn = build_localglmnet_zibell_cann(
        q0, n_br, n_re, d=d,
        lambda_hom=lambda_nn, pi_init=pi_logit_init)

    cbs = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=PATIENCE,
        restore_best_weights=True)]

    zi_nn_model.fit(
        inputs_nn_learn, ylearn.reshape(-1),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=cbs, verbose=2,
    )

    raw_zi_nn = zi_nn_model.predict(inputs_nn_test, verbose=0)
    mu_zi_nn = raw_zi_nn[:, 0]
    pi_zi_nn = raw_zi_nn[:, 1]
    y_zi_nn = (1.0 - pi_zi_nn) * mu_zi_nn
    dev_zi_nn = bell_deviance_metric(ytest, y_zi_nn)
    print(f"  LocalGLMnet ZI-Bell NN test deviance: {dev_zi_nn:.6f}")
    print(f"  Mean pi: {np.mean(pi_zi_nn):.6f}")

    zi_nn_model.save_weights(os.path.join(SAVED_DIR, "LocalGLMnet_ZIBell_NN.weights.h5"))

    beta_mu_nn, contrib_mu_nn, beta_pi_nn, contrib_pi_nn = zi_nn_attn.predict(
        inputs_nn_test, verbose=0)

    plot_attention_weights(beta_mu_nn, feature_names,
                           r"ZI-Bell NN LocalGLMnet: $\mu$ (Total Effects)")
    plt.savefig(os.path.join(PLOT_DIR, "localglmnet_zibell_nn_mu_importance.png"))
    plt.close()
    print("  Saved: localglmnet_zibell_nn_mu_importance.png")

    plot_attention_weights(beta_pi_nn, feature_names,
                           r"ZI-Bell NN LocalGLMnet: $\pi$ (Total Effects)")
    plt.savefig(os.path.join(PLOT_DIR, "localglmnet_zibell_nn_pi_importance.png"))
    plt.close()
    print("  Saved: localglmnet_zibell_nn_pi_importance.png")

    plot_zi_comparison(contrib_mu_nn, contrib_pi_nn, feature_names)
    plt.savefig(os.path.join(PLOT_DIR, "localglmnet_zibell_nn_mu_vs_pi.png"))
    plt.close()
    print("  Saved: localglmnet_zibell_nn_mu_vs_pi.png")

    # ------------------------------------------------------------------
    # 4. LocalGLMnet ZI-Bell CANN (Novel: residual effects beyond GLM)
    # ------------------------------------------------------------------
    set_seeds(SEED)
    print(f"\n{'=' * 70}")
    print("[4] Training LocalGLMnet ZI-Bell CANN...")
    print("    (offset=log(zibellGLM_mu) — residual effects beyond GLM)")

    lv_learn_zi = np.log(np.maximum(glm_learn["mu_zibell"], 1e-10))
    lv_test_zi = np.log(np.maximum(glm_test["mu_zibell"], 1e-10))
    lambda_zi = np.sum(ylearn) / np.sum(glm_learn["mu_zibell"])

    inputs_learn_zi = get_model_inputs(
        data["Xlearn"], data["BrandLearn"], data["RegionLearn"], lv_learn_zi)
    inputs_test_zi = get_model_inputs(
        data["Xtest"], data["BrandTest"], data["RegionTest"], lv_test_zi)

    zi_model, zi_attn_model = build_localglmnet_zibell_cann(
        q0, n_br, n_re, d=d,
        lambda_hom=lambda_zi, pi_init=pi_logit_init)

    cbs = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=PATIENCE,
        restore_best_weights=True)]

    zi_model.fit(
        inputs_learn_zi, ylearn.reshape(-1),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=cbs, verbose=2,
    )

    raw = zi_model.predict(inputs_test_zi, verbose=0)
    mu_pred = raw[:, 0]
    pi_pred = raw[:, 1]
    y_pred = (1.0 - pi_pred) * mu_pred
    dev_zi = bell_deviance_metric(ytest, y_pred)
    print(f"  LocalGLMnet ZI-Bell CANN test deviance: {dev_zi:.6f}")
    print(f"  Mean pi: {np.mean(pi_pred):.6f}")

    zi_model.save_weights(os.path.join(SAVED_DIR, "LocalGLMnet_ZIBell_CANN.weights.h5"))

    # Extract dual attention weights
    beta_mu, contrib_mu, beta_pi, contrib_pi = zi_attn_model.predict(
        inputs_test_zi, verbose=0)

    # Plot mu attention
    plot_attention_weights(beta_mu, feature_names,
                           r"ZI-Bell LocalGLMnet: $\mu$ Attention")
    plt.savefig(os.path.join(PLOT_DIR, "localglmnet_zibell_mu_importance.png"))
    plt.close()
    print("  Saved: localglmnet_zibell_mu_importance.png")

    # Plot pi attention
    plot_attention_weights(beta_pi, feature_names,
                           r"ZI-Bell LocalGLMnet: $\pi$ Attention")
    plt.savefig(os.path.join(PLOT_DIR, "localglmnet_zibell_pi_importance.png"))
    plt.close()
    print("  Saved: localglmnet_zibell_pi_importance.png")

    # Plot mu vs pi comparison
    plot_zi_comparison(contrib_mu, contrib_pi, feature_names)
    plt.savefig(os.path.join(PLOT_DIR, "localglmnet_zibell_mu_vs_pi.png"))
    plt.close()
    print("  Saved: localglmnet_zibell_mu_vs_pi.png")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("LocalGLMnet SUMMARY")
    print(f"{'=' * 70}")
    print(f"  NN models (total feature effects, per Richman & Wüthrich 2023):")
    print(f"    Bell NN test deviance:      {dev_nn:.6f}")
    print(f"    ZI-Bell NN test deviance:   {dev_zi_nn:.6f}")
    print(f"  CANN models (residual effects beyond GLM):")
    print(f"    Bell CANN test deviance:    {dev:.6f}")
    print(f"    ZI-Bell CANN test deviance: {dev_zi:.6f}")
    print(f"\n  Plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
