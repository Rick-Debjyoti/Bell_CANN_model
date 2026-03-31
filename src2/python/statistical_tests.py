"""Statistical tests and empirical model comparisons.

Part 1 — Formal GLM-level tests (valid under MLE asymptotics):
  1. Likelihood Ratio Test (LR): ZI vs non-ZI
  2. Vuong Test: non-nested model comparison
  3. Clarke Test: sign-based non-nested test

Part 2 — NN/CANN empirical comparison (standard actuarial practice):
  Out-of-sample deviance, log-likelihood, and portfolio average.
  Following Richman & Wüthrich (2023), Jose et al. (2024), Wüthrich & Merz (2019)
  who compare NNs via loss tables, NOT formal hypothesis tests.

Part 3 — Appendix: Vuong/Clarke applied to NN predictions (heuristic).
  NOTE: These are descriptive, not formal tests, because NNs fitted with
  early stopping do not produce MLEs (Richman & Wüthrich, 2023, p.80).

Usage:
    cd src2/python
    python statistical_tests.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import lambertw as W_scipy, gammaln

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, Model

from config import SEED, EPOCHS, PATIENCE, VALIDATION_SPLIT
from data_loader import load_data, get_model_inputs
from train import set_seeds, bias_regularize
from losses import (
    bell_deviance, poisson_deviance, negbin_nll,
    zip_nll, zinb_nll, zibell_nll,
)
from metrics import (
    poisson_loglik, bell_loglik, negbin_loglik,
    zip_loglik, zinb_loglik, zibell_loglik,
)
from optuna_best_params_all import BEST_PARAMS


# ===================================================================
# Per-observation log-likelihoods (needed for Vuong/Clarke)
# ===================================================================

def _poisson_loglik_obs(y, mu):
    """Per-observation Poisson log-likelihood."""
    from scipy.stats import poisson as pdist
    return pdist.logpmf(y.astype(int), mu)


def _bell_loglik_obs(y, mu):
    """Per-observation Bell log-likelihood."""
    mu = np.maximum(mu, 1e-10)
    w = np.real(W_scipy(mu))
    y_int = np.minimum(y.astype(int), 4)

    BELL = np.array([1, 1, 2, 5, 15], dtype=np.float64)
    LFACT = np.array([0, 0, np.log(2), np.log(6), np.log(24)])

    return (1.0 - np.exp(w)) + y * np.log(np.maximum(w, 1e-30)) + \
           np.log(BELL[y_int]) - LFACT[y_int]


def _negbin_loglik_obs(y, mu, alpha):
    """Per-observation NegBin log-likelihood."""
    mu = np.maximum(mu, 1e-10)
    inv_a = 1.0 / alpha
    return (gammaln(y + inv_a) - gammaln(inv_a) - gammaln(y + 1)
            + inv_a * np.log(1.0 / (1.0 + alpha * mu))
            + y * np.log(alpha * mu / (1.0 + alpha * mu)))


def _zip_loglik_obs(y, mu, pi):
    """Per-observation ZIP log-likelihood."""
    mu = np.maximum(mu, 1e-10)
    pi = np.clip(pi, 1e-7, 1 - 1e-7)
    ll = np.empty_like(y, dtype=np.float64)
    zero = y == 0
    ll[zero] = np.log(pi[zero] + (1 - pi[zero]) * np.exp(-mu[zero]))
    nz = ~zero
    ll[nz] = np.log(1 - pi[nz]) - mu[nz] + y[nz] * np.log(mu[nz]) - gammaln(y[nz] + 1)
    return ll


def _zibell_loglik_obs(y, mu, pi):
    """Per-observation ZI-Bell log-likelihood."""
    mu = np.maximum(mu, 1e-10)
    pi = np.clip(pi, 1e-7, 1 - 1e-7)
    w = np.real(W_scipy(mu))
    y_int = np.minimum(y.astype(int), 4)

    BELL = np.array([1, 1, 2, 5, 15], dtype=np.float64)
    LFACT = np.array([0, 0, np.log(2), np.log(6), np.log(24)])

    log_bell = (1.0 - np.exp(w)) + y * np.log(np.maximum(w, 1e-30)) + \
               np.log(BELL[y_int]) - LFACT[y_int]

    ll = np.empty_like(y, dtype=np.float64)
    zero = y == 0
    bell_0 = np.exp(log_bell[zero])
    ll[zero] = np.log(pi[zero] + (1 - pi[zero]) * bell_0)
    nz = ~zero
    ll[nz] = np.log(1 - pi[nz]) + log_bell[nz]
    return ll


def _zinb_loglik_obs(y, mu, pi, alpha):
    """Per-observation ZINB log-likelihood."""
    mu = np.maximum(mu, 1e-10)
    pi = np.clip(pi, 1e-7, 1 - 1e-7)
    inv_a = 1.0 / alpha
    ll = np.empty_like(y, dtype=np.float64)
    zero = y == 0
    nb_zero = (1.0 / (1.0 + alpha * mu[zero])) ** inv_a
    ll[zero] = np.log(pi[zero] + (1 - pi[zero]) * nb_zero)
    nz = ~zero
    log_nb = (gammaln(y[nz] + inv_a) - gammaln(inv_a) - gammaln(y[nz] + 1)
              + inv_a * np.log(1.0 / (1.0 + alpha * mu[nz]))
              + y[nz] * np.log(alpha * mu[nz] / (1.0 + alpha * mu[nz])))
    ll[nz] = np.log(1 - pi[nz]) + log_nb
    return ll


# ===================================================================
# Model builders for loading saved weights
# ===================================================================

SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved_models_tuned", "saved_models_tuned")


def _hidden_from_params(params):
    """Derive hidden layer sizes from Optuna params dict."""
    hidden = []
    n = params["first_neurons"]
    for _ in range(params["n_layers"]):
        hidden.append(max(int(n), 16))
        n *= params["shrink_factor"]
    return hidden


def _build_standard_from_params(params, loss_fn, q0, n_brands, n_regions, lambda_hom):
    """Build a standard (non-ZI) model from Optuna params dict.

    NOTE: Layer names are intentionally left as defaults to match the
    generic naming convention used by the saved weights (from newer TF).
    """
    hidden = _hidden_from_params(params)
    d = params["embedding_dim"]
    lr = params["learning_rate"]
    activation = params["activation"]
    dropout_rate = params.get("dropout_rate", 0.0)

    Design   = layers.Input(shape=(q0,))
    VehBrand = layers.Input(shape=(1,))
    Region   = layers.Input(shape=(1,))
    LogVol   = layers.Input(shape=(1,))

    brand_emb  = layers.Embedding(n_brands, d, input_length=1)(VehBrand)
    brand_flat = layers.Flatten()(brand_emb)
    region_emb = layers.Embedding(n_regions, d, input_length=1)(Region)
    region_flat = layers.Flatten()(region_emb)

    x = layers.Concatenate()([Design, brand_flat, region_flat])
    for i, units in enumerate(hidden):
        x = layers.Dense(units, activation=activation)(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

    network_out = layers.Dense(
        1, activation="linear", use_bias=True,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(lambda_hom),
    )(x)
    add_out = layers.Add()([network_out, LogVol])
    response = layers.Dense(
        1, activation=tf.keras.activations.exponential,
        dtype="float64", trainable=False,
        kernel_initializer="ones", bias_initializer="zeros",
    )(add_out)

    model = Model(inputs=[Design, VehBrand, Region, LogVol], outputs=response)
    model.compile(optimizer=tf.keras.optimizers.Nadam(lr), loss=loss_fn)
    return model


def _build_zi_from_params(params, loss_fn, q0, n_brands, n_regions,
                          lambda_hom, pi_init):
    """Build a ZI model from Optuna params dict.

    NOTE: Layer names are intentionally left as defaults to match saved weights.
    """
    hidden = _hidden_from_params(params)
    d = params["embedding_dim"]
    lr = params["learning_rate"]
    activation = params["activation"]
    dropout_rate = params.get("dropout_rate", 0.0)

    Design   = layers.Input(shape=(q0,))
    VehBrand = layers.Input(shape=(1,))
    Region   = layers.Input(shape=(1,))
    LogVol   = layers.Input(shape=(1,))

    brand_emb  = layers.Embedding(n_brands, d, input_length=1)(VehBrand)
    brand_flat = layers.Flatten()(brand_emb)
    region_emb = layers.Embedding(n_regions, d, input_length=1)(Region)
    region_flat = layers.Flatten()(region_emb)

    x = layers.Concatenate()([Design, brand_flat, region_flat])
    for i, units in enumerate(hidden):
        x = layers.Dense(units, activation=activation)(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

    raw_mu = layers.Dense(
        1, activation="linear", use_bias=True,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(lambda_hom),
    )(x)
    raw_pi = layers.Dense(
        1, activation="linear", use_bias=True,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(pi_init),
    )(x)

    mu_plus_vol = layers.Add()([raw_mu, LogVol])
    mu_out = layers.Lambda(
        lambda t: tf.exp(tf.cast(t, tf.float64)), dtype="float64"
    )(mu_plus_vol)
    pi_out = layers.Lambda(
        lambda t: tf.sigmoid(tf.cast(t, tf.float64)), dtype="float64"
    )(raw_pi)

    output = layers.Concatenate(dtype="float64")([mu_out, pi_out])
    model = Model(inputs=[Design, VehBrand, Region, LogVol], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Nadam(lr), loss=loss_fn)
    return model


def _load_weights_keras3(model, wpath):
    """Load weights from Keras 3 h5 format into a TF 2.10 model.

    Keras 3 stores weights as layers/<name>/vars/<index>, while TF 2.10
    expects the older format. This function maps by layer order.
    """
    import h5py

    with h5py.File(wpath, "r") as f:
        if "layers" not in f:
            # Old format — use standard loader
            model.load_weights(wpath)
            return

        # Collect all weight arrays from h5 in order
        h5_weights = {}
        for layer_name in sorted(f["layers"].keys()):
            grp = f["layers"][layer_name]
            if "vars" in grp and len(grp["vars"]) > 0:
                arrays = []
                for idx in sorted(grp["vars"].keys(), key=int):
                    arrays.append(np.array(grp["vars"][idx]))
                h5_weights[layer_name] = arrays

        # Match model layers (that have weights) to h5 layer groups by type
        # Strategy: embeddings first, then dense layers, in order
        h5_embeddings = sorted(
            [k for k in h5_weights if k.startswith("embedding")],
            key=lambda x: (len(x), x))
        h5_dense = sorted(
            [k for k in h5_weights if k.startswith("dense")],
            key=lambda x: (len(x), x))

        emb_idx = 0
        dense_idx = 0

        for layer in model.layers:
            if len(layer.get_weights()) == 0:
                continue
            ltype = layer.__class__.__name__

            if ltype == "Embedding" and emb_idx < len(h5_embeddings):
                key = h5_embeddings[emb_idx]
                layer.set_weights(h5_weights[key])
                emb_idx += 1
            elif ltype == "Dense" and dense_idx < len(h5_dense):
                key = h5_dense[dense_idx]
                layer.set_weights(h5_weights[key])
                dense_idx += 1


def _load_and_predict(model, weight_name, inputs_test, is_zi=False):
    """Load weights and return predictions. Returns (mu, pi) for ZI, (mu, None) otherwise."""
    wpath = os.path.join(SAVED_DIR, f"{weight_name}.weights.h5")
    if not os.path.exists(wpath):
        print(f"  WARNING: {wpath} not found, skipping.")
        return None, None
    _load_weights_keras3(model, wpath)
    if is_zi:
        raw = model.predict(inputs_test, verbose=0)
        return raw[:, 0], raw[:, 1]
    else:
        return model.predict(inputs_test, verbose=0).ravel(), None


# ===================================================================
# Statistical Tests
# ===================================================================

def likelihood_ratio_test(ll_restricted, ll_full, df_diff):
    """Likelihood Ratio Test.

    H0: The restricted model is adequate.
    H1: The full model provides significantly better fit.

    Parameters
    ----------
    ll_restricted : float — log-likelihood of the restricted model
    ll_full : float — log-likelihood of the full (zero-inflated) model
    df_diff : int — difference in number of parameters

    Returns
    -------
    dict with LR statistic, p-value, and decision
    """
    LR = -2 * (ll_restricted - ll_full)
    p_value = stats.chi2.sf(LR, df=df_diff)
    return {
        "LR_statistic": LR,
        "df": df_diff,
        "p_value": p_value,
        "significant_5pct": p_value < 0.05,
        "significant_1pct": p_value < 0.01,
    }


def vuong_test(ll_obs_1, ll_obs_2, n_params_1=None, n_params_2=None, n=None):
    """Vuong (1989) test for non-nested model comparison.

    Compares two models based on per-observation log-likelihood differences.

    H0: Both models are equally close to the true model.
    H1: One model is closer.

    Parameters
    ----------
    ll_obs_1 : ndarray — per-obs log-likelihood of model 1
    ll_obs_2 : ndarray — per-obs log-likelihood of model 2
    n_params_1, n_params_2 : int — number of params (for BIC correction)
    n : int — sample size

    Returns
    -------
    dict with Vuong statistic, p-value, and preferred model
    """
    m = ll_obs_1 - ll_obs_2

    # BIC-corrected version (Vuong 1989, Section 5)
    if n_params_1 is not None and n_params_2 is not None and n is not None:
        correction = (n_params_1 - n_params_2) * np.log(n) / (2 * n)
        m = m - correction

    m_bar = np.mean(m)
    s_m = np.std(m, ddof=1)
    n_obs = len(m)

    V = np.sqrt(n_obs) * m_bar / s_m if s_m > 0 else 0.0
    p_value = 2 * stats.norm.sf(abs(V))  # two-sided

    if V > 1.96:
        preferred = "Model 1"
    elif V < -1.96:
        preferred = "Model 2"
    else:
        preferred = "Neither (indistinguishable)"

    return {
        "vuong_statistic": V,
        "p_value": p_value,
        "preferred": preferred,
        "mean_diff": m_bar,
        "std_diff": s_m,
    }


def clarke_test(ll_obs_1, ll_obs_2):
    """Clarke (2007) sign-based test for non-nested models.

    Counts observations where model 1 fits better vs model 2.
    Under H0 of equivalent fit, the count follows Binomial(n, 0.5).

    Returns
    -------
    dict with test statistic, p-value, and counts
    """
    m = ll_obs_1 - ll_obs_2
    n_pos = np.sum(m > 0)
    n_neg = np.sum(m < 0)
    n_total = n_pos + n_neg  # exclude ties

    # Two-sided binomial test
    p_value = stats.binomtest(n_pos, n_total, 0.5).pvalue if n_total > 0 else 1.0

    if n_pos > n_neg and p_value < 0.05:
        preferred = "Model 1"
    elif n_neg > n_pos and p_value < 0.05:
        preferred = "Model 2"
    else:
        preferred = "Neither"

    return {
        "n_model1_better": int(n_pos),
        "n_model2_better": int(n_neg),
        "n_ties": int(len(m) - n_total),
        "p_value": p_value,
        "preferred": preferred,
    }


# ===================================================================
# Main
# ===================================================================

def _run_vuong_clarke(ll_obs_1, ll_obs_2, name1, name2, level, results):
    """Run both Vuong and Clarke tests and append results."""
    v = vuong_test(ll_obs_1, ll_obs_2)
    print(f"\n  {name1} vs {name2} ({level}):")
    print(f"    Vuong V = {v['vuong_statistic']:.4f}, p = {v['p_value']:.2e}, "
          f"Preferred: {v['preferred']}")
    results.append({"Test": f"Vuong: {name1} vs {name2} ({level})", **v})

    c = clarke_test(ll_obs_1, ll_obs_2)
    print(f"    Clarke: {name1} better={c['n_model1_better']}, "
          f"{name2} better={c['n_model2_better']}, p={c['p_value']:.2e}, "
          f"Preferred: {c['preferred']}")
    results.append({"Test": f"Clarke: {name1} vs {name2} ({level})", **c})


# ===================================================================
# NN/CANN empirical metrics (standard actuarial practice)
# ===================================================================

def _compute_nn_metrics(name, mu, pi, ytest, dist, alpha=None):
    """Compute out-of-sample deviance, log-lik, portfolio avg for an NN model."""
    from metrics import (
        bell_deviance_metric, poisson_deviance_metric, negbin_deviance_metric,
        poisson_loglik as _pll, bell_loglik as _bll, negbin_loglik as _nll,
        zip_loglik as _zll, zinb_loglik as _znll, zibell_loglik as _zbll,
        portfolio_average,
    )

    y_pred = (1.0 - pi) * mu if pi is not None else mu

    if dist == "poisson":
        dev = poisson_deviance_metric(ytest, y_pred)
        ll = _pll(ytest, mu) if pi is None else _zll(ytest, mu, pi)
    elif dist == "bell":
        dev = bell_deviance_metric(ytest, y_pred)
        ll = _bll(ytest, mu) if pi is None else _zbll(ytest, mu, pi)
    elif dist == "negbin":
        dev = negbin_deviance_metric(ytest, y_pred, alpha)
        ll = _nll(ytest, mu, alpha) if pi is None else _znll(ytest, mu, pi, alpha)

    return {
        "Model": name,
        "Test Deviance": dev,
        "Log-Lik": ll,
        "Portfolio Avg": portfolio_average(y_pred),
    }


def main():
    set_seeds(SEED)
    data = load_data()

    q0 = data["q0"]
    n_br = data["n_brands"]
    n_re = data["n_regions"]
    ytest = data["ytest"].astype(np.float64)
    ylearn = data["ylearn"]
    glm_learn = data["glm_preds_learn"]
    glm_test = data["glm_preds_test"]
    nb_alpha = data["negbin_alpha"]
    zinb_alpha = data["zinb_alpha"]

    p_zero = np.mean(ylearn == 0)
    pi_logit_init = np.log(p_zero / (1 - p_zero))
    obs_mean = np.mean(ytest)

    print("=" * 70)
    print("STATISTICAL TESTS FOR MODEL COMPARISON")
    print("=" * 70)

    formal_results = []

    # ==================================================================
    # PART 1: FORMAL GLM-level tests (valid under MLE asymptotics)
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 1: FORMAL GLM TESTS (MLE-based, asymptotically valid)")
    print("=" * 70)

    # --- Likelihood Ratio Tests ---
    print("\n--- Likelihood Ratio Tests (ZI vs non-ZI) ---")
    print("H0: Zero-inflation component is not needed")

    ll_bell = bell_loglik(ytest, glm_test["bellGLM"])
    ll_zibell = zibell_loglik(ytest, glm_test["mu_zibell"], glm_test["pi_zibell"])
    lr = likelihood_ratio_test(ll_bell, ll_zibell, df_diff=9+1)
    print(f"\n  Bell GLM vs ZI-Bell GLM: LR={lr['LR_statistic']:.4f}, "
          f"p={lr['p_value']:.2e}, sig={lr['significant_1pct']}")
    formal_results.append({"Test": "LR: Bell vs ZI-Bell (GLM)", **lr})

    ll_pois = poisson_loglik(ytest, glm_test["poissonGLM"])
    ll_zip = zip_loglik(ytest, glm_test["mu_zip"], glm_test["pi_zip"])
    lr2 = likelihood_ratio_test(ll_pois, ll_zip, df_diff=9+1)
    print(f"  Poisson GLM vs ZIP GLM: LR={lr2['LR_statistic']:.4f}, "
          f"p={lr2['p_value']:.2e}, sig={lr2['significant_1pct']}")
    formal_results.append({"Test": "LR: Poisson vs ZIP (GLM)", **lr2})

    ll_nb = negbin_loglik(ytest, glm_test["negbinGLM"], nb_alpha)
    ll_zinb = zinb_loglik(ytest, glm_test["mu_zinb"], glm_test["pi_zinb"], zinb_alpha)
    lr3 = likelihood_ratio_test(ll_nb, ll_zinb, df_diff=9+1)
    print(f"  NegBin GLM vs ZINB GLM: LR={lr3['LR_statistic']:.4f}, "
          f"p={lr3['p_value']:.2e}, sig={lr3['significant_1pct']}")
    formal_results.append({"Test": "LR: NegBin vs ZINB (GLM)", **lr3})

    # --- Vuong & Clarke Tests ---
    print("\n--- Vuong & Clarke Tests (non-nested, GLM only) ---")
    print("H0: Both models equally close to the true DGP")

    ll_bell_obs = _bell_loglik_obs(ytest, glm_test["bellGLM"])
    ll_pois_obs = _poisson_loglik_obs(ytest, glm_test["poissonGLM"])
    ll_nb_obs = _negbin_loglik_obs(ytest, glm_test["negbinGLM"], nb_alpha)
    ll_zibell_obs = _zibell_loglik_obs(ytest, glm_test["mu_zibell"],
                                        glm_test["pi_zibell"])
    ll_zip_obs = _zip_loglik_obs(ytest, glm_test["mu_zip"],
                                  glm_test["pi_zip"])
    ll_zinb_obs = _zinb_loglik_obs(ytest, glm_test["mu_zinb"],
                                    glm_test["pi_zinb"], zinb_alpha)

    _run_vuong_clarke(ll_bell_obs, ll_pois_obs, "Bell", "Poisson", "GLM", formal_results)
    _run_vuong_clarke(ll_bell_obs, ll_nb_obs, "Bell", "NegBin", "GLM", formal_results)
    _run_vuong_clarke(ll_zibell_obs, ll_zip_obs, "ZI-Bell", "ZIP", "GLM", formal_results)
    _run_vuong_clarke(ll_zibell_obs, ll_zinb_obs, "ZI-Bell", "ZINB", "GLM", formal_results)

    # Save formal results
    df_formal = pd.DataFrame(formal_results)
    df_formal.to_csv("statistical_tests_formal.csv", index=False)
    print(f"\n  Formal GLM tests saved to statistical_tests_formal.csv "
          f"({len(formal_results)} tests)")

    # ==================================================================
    # PART 2: NN/CANN EMPIRICAL COMPARISON
    #   Standard actuarial practice: deviance, log-lik, portfolio avg
    #   Per Richman & Wüthrich (2023) Table 2, Jose et al. (2024) Table 5
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 2: NN/CANN EMPIRICAL COMPARISON (out-of-sample metrics)")
    print("  Following actuarial NN literature — loss tables, not formal tests")
    print("=" * 70)

    # Prepare inputs
    lv_nn_test = np.log(data["Etest"])
    inputs_nn_test = get_model_inputs(
        data["Xtest"], data["BrandTest"], data["RegionTest"], lv_nn_test)

    def cann_inputs(glm_key):
        lv = np.log(np.maximum(glm_test[glm_key], 1e-10))
        return get_model_inputs(data["Xtest"], data["BrandTest"], data["RegionTest"], lv)

    # Load all 12 models and collect predictions
    nn_preds = {}
    nn_meta = {}  # {name: (dist, alpha)}

    for name, optuna_key, loss_fn, glm_key, alpha, dist, is_zi in [
        ("Poisson_NN",    "Poisson_NN",   poisson_deviance,     None,          None,       "poisson", False),
        ("Bell_NN",       "Bell_NN",      bell_deviance,        None,          None,       "bell",    False),
        ("NegBin_NN",     "NegBin_NN",    negbin_nll(nb_alpha), None,          nb_alpha,   "negbin",  False),
        ("Poisson_CANN",  "Poisson_CANN", poisson_deviance,     "poissonGLM",  None,       "poisson", False),
        ("Bell_CANN",     "Bell_CANN",    bell_deviance,        "bellGLM",     None,       "bell",    False),
        ("NegBin_CANN",   "NegBin_CANN",  negbin_nll(nb_alpha), "negbinGLM",   nb_alpha,   "negbin",  False),
        ("ZIP_NN",        "ZIP_NN",       zip_nll,              None,          None,       "poisson", True),
        ("ZINB_NN",       "ZINB_NN",      zinb_nll(zinb_alpha), None,          zinb_alpha, "negbin",  True),
        ("ZI_Bell_NN",    "ZIBell_NN",    zibell_nll,           None,          None,       "bell",    True),
        ("ZIP_CANN",      "ZIP_CANN",     zip_nll,              "mu_zip",      None,       "poisson", True),
        ("ZINB_CANN",     "ZINB_CANN",    zinb_nll(zinb_alpha), "mu_zinb",     zinb_alpha, "negbin",  True),
        ("ZI_Bell_CANN",  "ZIBell_CANN",  zibell_nll,           "mu_zibell",   None,       "bell",    True),
    ]:
        if glm_key is not None:
            lam = np.sum(ylearn) / np.sum(glm_learn[glm_key])
            inp = cann_inputs(glm_key)
        else:
            lam = np.sum(ylearn) / np.sum(data["Elearn"])
            inp = inputs_nn_test

        if is_zi:
            m = _build_zi_from_params(
                BEST_PARAMS[optuna_key], loss_fn, q0, n_br, n_re, lam, pi_logit_init)
        else:
            m = _build_standard_from_params(
                BEST_PARAMS[optuna_key], loss_fn, q0, n_br, n_re, lam)

        mu, pi = _load_and_predict(m, name, inp, is_zi=is_zi)
        if mu is not None:
            nn_preds[name] = (mu, pi)
            nn_meta[name] = (dist, alpha)

    print(f"\n  Loaded {len(nn_preds)} models")

    # Compute metrics for each model
    nn_metrics = []
    for name in nn_preds:
        mu, pi = nn_preds[name]
        dist, alpha = nn_meta[name]
        row = _compute_nn_metrics(name, mu, pi, ytest, dist, alpha)
        nn_metrics.append(row)

    df_nn = pd.DataFrame(nn_metrics).sort_values("Test Deviance")
    print(f"\n  Observed portfolio mean: {obs_mean:.6f}")
    print()
    print(df_nn.to_string(index=False))

    df_nn.to_csv("nn_model_comparison.csv", index=False)
    print(f"\n  NN empirical comparison saved to nn_model_comparison.csv")

    # ==================================================================
    # PART 3: APPENDIX — Vuong/Clarke for NN (heuristic only)
    #   NOTE: NN models with early stopping do NOT produce MLEs.
    #   Richman & Wüthrich (2023, p.80): "we cannot rely on an
    #   asymptotic theory for MLEs because early stopping implies
    #   that we do not consider the MLE."
    #   These results are DESCRIPTIVE, not formal tests.
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 3: APPENDIX — Heuristic Vuong/Clarke for NN/CANN")
    print("  CAVEAT: NNs with early stopping are NOT MLEs.")
    print("  P-values are descriptive, not formally valid.")
    print("=" * 70)

    heuristic_results = []

    def get_ll_obs(model_name, dist):
        mu, pi = nn_preds[model_name]
        if dist == "poisson" and pi is None:
            return _poisson_loglik_obs(ytest, mu)
        elif dist == "bell" and pi is None:
            return _bell_loglik_obs(ytest, mu)
        elif dist == "negbin" and pi is None:
            return _negbin_loglik_obs(ytest, mu, nb_alpha)
        elif dist == "zip":
            return _zip_loglik_obs(ytest, mu, pi)
        elif dist == "zibell":
            return _zibell_loglik_obs(ytest, mu, pi)
        elif dist == "zinb":
            return _zinb_loglik_obs(ytest, mu, pi, zinb_alpha)
        raise ValueError(f"Unknown dist={dist} for model={model_name}")

    comparisons = [
        # NN level
        ("Bell_NN", "bell", "Poisson_NN", "poisson", "NN"),
        ("Bell_NN", "bell", "NegBin_NN", "negbin", "NN"),
        ("ZI_Bell_NN", "zibell", "ZIP_NN", "zip", "NN"),
        ("ZI_Bell_NN", "zibell", "ZINB_NN", "zinb", "NN"),
        # CANN level
        ("Bell_CANN", "bell", "Poisson_CANN", "poisson", "CANN"),
        ("Bell_CANN", "bell", "NegBin_CANN", "negbin", "CANN"),
        ("ZI_Bell_CANN", "zibell", "ZIP_CANN", "zip", "CANN"),
        ("ZI_Bell_CANN", "zibell", "ZINB_CANN", "zinb", "CANN"),
        # Cross: CANN vs NN
        ("Bell_CANN", "bell", "Bell_NN", "bell", "CANN vs NN"),
        ("NegBin_CANN", "negbin", "NegBin_NN", "negbin", "CANN vs NN"),
        ("ZI_Bell_CANN", "zibell", "ZI_Bell_NN", "zibell", "CANN vs NN"),
    ]

    for m1, d1, m2, d2, level in comparisons:
        if m1 in nn_preds and m2 in nn_preds:
            ll1 = get_ll_obs(m1, d1)
            ll2 = get_ll_obs(m2, d2)
            n1 = m1.replace("_", " ").replace("ZI Bell", "ZI-Bell")
            n2 = m2.replace("_", " ").replace("ZI Bell", "ZI-Bell")
            _run_vuong_clarke(ll1, ll2, n1, n2, level, heuristic_results)

    df_heur = pd.DataFrame(heuristic_results)
    df_heur.to_csv("statistical_tests_heuristic.csv", index=False)
    print(f"\n  Heuristic NN tests saved to statistical_tests_heuristic.csv "
          f"({len(heuristic_results)} tests)")

    # ==================================================================
    # Combined summary
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("OUTPUT FILES")
    print(f"{'=' * 70}")
    print(f"  statistical_tests_formal.csv    — {len(formal_results)} formal GLM tests")
    print(f"  nn_model_comparison.csv         — {len(nn_metrics)} NN/CANN empirical metrics")
    print(f"  statistical_tests_heuristic.csv — {len(heuristic_results)} heuristic NN tests (appendix)")

    # Also save combined for backward compatibility
    all_results = formal_results + heuristic_results
    pd.DataFrame(all_results).to_csv("statistical_tests.csv", index=False)


if __name__ == "__main__":
    main()
