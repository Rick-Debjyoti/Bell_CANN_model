"""Run all 18 models using Optuna-tuned architectures and output comparison table.

For the 12 NN/CANN models: loads best hyperparameters from optuna_best_params_all.py,
rebuilds each model with its optimal architecture, trains with full epochs, evaluates,
and saves weights + training history.

For the 6 GLM models: evaluates R predictions (same as run_comparison.py).

Prerequisites:
    Run run_optuna_all.py first to generate optuna_best_params_all.py

Usage:
    cd src2/python
    python run_comparison_tuned.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks

from config import SEED, EPOCHS, PATIENCE, VALIDATION_SPLIT
from data_loader import load_data, get_model_inputs
from train import (
    set_seeds, predict_standard, predict_zi,
    evaluate_glm, bias_regularize,
)
from losses import (
    bell_deviance, poisson_deviance, negbin_nll,
    zip_nll, zinb_nll, zibell_nll,
)
from metrics import (
    bell_deviance_metric, poisson_deviance_metric, negbin_deviance_metric,
    poisson_loglik, bell_loglik, negbin_loglik,
    zip_loglik, zinb_loglik, zibell_loglik,
    aic, bic, portfolio_average,
)

# Output directories
SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved_models_tuned", "saved_models_tuned")
HIST_DIR = os.path.join(os.path.dirname(__file__), "training_histories", "training_histories")
os.makedirs(SAVED_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)


# ===================================================================
# Model builders (same as run_optuna_all.py but with dict params)
# ===================================================================

def _build_standard_from_params(params, loss_fn, q0, n_brands, n_regions, lambda_hom):
    """Build a standard model from a params dict."""
    n_layers = params["n_layers"]
    first_neurons = params["first_neurons"]
    shrink = params["shrink_factor"]
    d = params["embedding_dim"]
    lr = params["learning_rate"]
    activation = params["activation"]
    dropout_rate = params.get("dropout_rate", 0.0)
    use_l2 = params.get("use_l2", False)

    hidden = []
    n = first_neurons
    for _ in range(n_layers):
        hidden.append(max(int(n), 16))
        n *= shrink

    Design   = layers.Input(shape=(q0,), name="Design")
    VehBrand = layers.Input(shape=(1,), name="VehBrand")
    Region   = layers.Input(shape=(1,), name="Region")
    LogVol   = layers.Input(shape=(1,), name="LogVol")

    brand_emb  = layers.Embedding(n_brands, d, input_length=1, name="BrandEmb")(VehBrand)
    brand_flat = layers.Flatten(name="BrandFlat")(brand_emb)
    region_emb = layers.Embedding(n_regions, d, input_length=1, name="RegionEmb")(Region)
    region_flat = layers.Flatten(name="RegionFlat")(region_emb)

    x = layers.Concatenate()([Design, brand_flat, region_flat])
    for i, units in enumerate(hidden):
        reg = tf.keras.regularizers.l2(1e-4) if use_l2 else None
        x = layers.Dense(units, activation=activation,
                         kernel_regularizer=reg, name=f"hidden_{i}")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"drop_{i}")(x)

    network_out = layers.Dense(
        1, activation="linear", use_bias=True,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(lambda_hom),
        name="network_out"
    )(x)
    add_out = layers.Add(name="add_logvol")([network_out, LogVol])
    response = layers.Dense(
        1, activation=tf.keras.activations.exponential,
        dtype="float64", trainable=False,
        kernel_initializer="ones", bias_initializer="zeros",
        name="response"
    )(add_out)

    model = Model(inputs=[Design, VehBrand, Region, LogVol], outputs=response)
    model.compile(optimizer=tf.keras.optimizers.Nadam(lr), loss=loss_fn)
    return model, hidden


def _build_zi_from_params(params, loss_fn, q0, n_brands, n_regions,
                          lambda_hom, pi_init):
    """Build a ZI model from a params dict."""
    n_layers = params["n_layers"]
    first_neurons = params["first_neurons"]
    shrink = params["shrink_factor"]
    d = params["embedding_dim"]
    lr = params["learning_rate"]
    activation = params["activation"]
    dropout_rate = params.get("dropout_rate", 0.0)
    use_l2 = params.get("use_l2", False)

    hidden = []
    n = first_neurons
    for _ in range(n_layers):
        hidden.append(max(int(n), 16))
        n *= shrink

    Design   = layers.Input(shape=(q0,), name="Design")
    VehBrand = layers.Input(shape=(1,), name="VehBrand")
    Region   = layers.Input(shape=(1,), name="Region")
    LogVol   = layers.Input(shape=(1,), name="LogVol")

    brand_emb  = layers.Embedding(n_brands, d, input_length=1, name="BrandEmb")(VehBrand)
    brand_flat = layers.Flatten(name="BrandFlat")(brand_emb)
    region_emb = layers.Embedding(n_regions, d, input_length=1, name="RegionEmb")(Region)
    region_flat = layers.Flatten(name="RegionFlat")(region_emb)

    x = layers.Concatenate()([Design, brand_flat, region_flat])
    for i, units in enumerate(hidden):
        reg = tf.keras.regularizers.l2(1e-4) if use_l2 else None
        x = layers.Dense(units, activation=activation,
                         kernel_regularizer=reg, name=f"hidden_{i}")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"drop_{i}")(x)

    raw_mu = layers.Dense(
        1, activation="linear", use_bias=True,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(lambda_hom),
        name="raw_mu"
    )(x)
    raw_pi = layers.Dense(
        1, activation="linear", use_bias=True,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(pi_init),
        name="raw_pi"
    )(x)

    mu_plus_vol = layers.Add(name="mu_add_logvol")([raw_mu, LogVol])
    mu_out = layers.Lambda(
        lambda t: tf.exp(tf.cast(t, tf.float64)), name="mu_exp", dtype="float64"
    )(mu_plus_vol)
    pi_out = layers.Lambda(
        lambda t: tf.sigmoid(tf.cast(t, tf.float64)), name="pi_sigmoid", dtype="float64"
    )(raw_pi)

    output = layers.Concatenate(name="mu_pi_output", dtype="float64")([mu_out, pi_out])
    model = Model(inputs=[Design, VehBrand, Region, LogVol], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Nadam(lr), loss=loss_fn)
    return model, hidden


# ===================================================================
# Full evaluation (with all metrics)
# ===================================================================

def evaluate_full(name, model, inputs_test, y_true, dist, alpha=None, is_zi=False):
    """Evaluate model and return full metrics dict."""
    n = len(y_true)
    n_params = model.count_params()

    if is_zi:
        raw = model.predict(inputs_test, verbose=0)
        mu_pred = raw[:, 0]
        pi_pred = raw[:, 1]
        y_pred = (1.0 - pi_pred) * mu_pred
    else:
        y_pred = model.predict(inputs_test, verbose=0).ravel()
        mu_pred = y_pred
        pi_pred = None

    # Deviance
    if dist == "poisson":
        dev = poisson_deviance_metric(y_true, y_pred)
        ll = poisson_loglik(y_true, mu_pred) if not is_zi else zip_loglik(y_true, mu_pred, pi_pred)
    elif dist == "bell":
        dev = bell_deviance_metric(y_true, y_pred)
        ll = bell_loglik(y_true, mu_pred) if not is_zi else zibell_loglik(y_true, mu_pred, pi_pred)
    elif dist == "negbin":
        dev = negbin_deviance_metric(y_true, y_pred, alpha)
        ll = negbin_loglik(y_true, mu_pred, alpha) if not is_zi else zinb_loglik(y_true, mu_pred, pi_pred, alpha)

    return {
        "model": name,
        "deviance": dev,
        "loglik": ll,
        "aic": aic(ll, n_params),
        "bic": bic(ll, n_params, n),
        "portfolio_avg": portfolio_average(y_pred),
        "n_params": n_params,
    }


# ===================================================================
# Main
# ===================================================================

def main():
    set_seeds(SEED)

    # Load tuned params
    params_file = os.path.join(os.path.dirname(__file__), "optuna_best_params_all.py")
    if not os.path.exists(params_file):
        print("ERROR: optuna_best_params_all.py not found.")
        print("Run run_optuna_all.py first!")
        return

    # Import the best params
    import importlib.util
    spec = importlib.util.spec_from_file_location("best_params", params_file)
    bp_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bp_mod)
    BEST_PARAMS = bp_mod.BEST_PARAMS

    print("=" * 70)
    print("BELL-CANN COMPARISON — OPTUNA-TUNED ARCHITECTURES")
    print("=" * 70)

    data = load_data()
    q0 = data["q0"]
    n_br = data["n_brands"]
    n_re = data["n_regions"]
    nb_alpha = data["negbin_alpha"]
    zinb_alpha = data["zinb_alpha"]
    ylearn = data["ylearn"]
    ytest = data["ytest"]
    Elearn = data["Elearn"]
    glm_learn = data["glm_preds_learn"]
    glm_test = data["glm_preds_test"]

    p_zero = np.mean(ylearn == 0)
    pi_logit_init = np.log(p_zero / (1 - p_zero))
    p_glm = 9

    results = []

    # ==================================================================
    # PART 1: GLM models (same as before — predictions from R)
    # ==================================================================
    print("\n--- Evaluating GLM models ---")

    results.append(evaluate_glm("Poisson GLM", ytest, glm_test["poissonGLM"], p_glm, dist="poisson"))
    results.append(evaluate_glm("Bell GLM", ytest, glm_test["bellGLM"], p_glm, dist="bell"))
    results.append(evaluate_glm("NegBin GLM", ytest, glm_test["negbinGLM"], p_glm + 1, dist="negbin", alpha=nb_alpha))
    results.append(evaluate_glm("ZIP GLM", ytest, glm_test["zipGLM"], 2*p_glm+2, dist="zip", mu_pred=glm_test["mu_zip"], pi_pred=glm_test["pi_zip"]))
    results.append(evaluate_glm("ZINB GLM", ytest, glm_test["zinbGLM"], 2*p_glm+3, dist="zinb", alpha=zinb_alpha, mu_pred=glm_test["mu_zinb"], pi_pred=glm_test["pi_zinb"]))
    results.append(evaluate_glm("ZI-Bell GLM", ytest, glm_test["zibellGLM"], 2*p_glm+1, dist="zibell", mu_pred=glm_test["mu_zibell"], pi_pred=glm_test["pi_zibell"]))

    for r in results:
        print(f"  {r['model']:20s}  dev={r['deviance']:.6f}")

    # ==================================================================
    # PART 2: NN/CANN models with tuned architectures
    # ==================================================================

    # Model configs: (display_name, optuna_key, loss_fn, dist, alpha,
    #                 glm_key_or_None, mu_glm_key_or_None, is_zi, is_cann)
    nn_configs = [
        ("Poisson NN",    "Poisson_NN",   poisson_deviance,     "poisson", None,       None,          None,         False, False),
        ("Bell NN",       "Bell_NN",      bell_deviance,        "bell",    None,       None,          None,         False, False),
        ("NegBin NN",     "NegBin_NN",    negbin_nll(nb_alpha), "negbin",  nb_alpha,   None,          None,         False, False),
        ("Poisson CANN",  "Poisson_CANN", poisson_deviance,     "poisson", None,       "poissonGLM",  None,         False, True),
        ("Bell CANN",     "Bell_CANN",    bell_deviance,        "bell",    None,       "bellGLM",     None,         False, True),
        ("NegBin CANN",   "NegBin_CANN",  negbin_nll(nb_alpha), "negbin",  nb_alpha,   "negbinGLM",   None,         False, True),
        ("ZIP-NN",        "ZIP_NN",       zip_nll,              "poisson", None,       None,          None,         True,  False),
        ("ZINB-NN",       "ZINB_NN",      zinb_nll(zinb_alpha), "negbin",  zinb_alpha, None,          None,         True,  False),
        ("ZI-Bell-NN",    "ZIBell_NN",    zibell_nll,           "bell",    None,       None,          None,         True,  False),
        ("ZIP-CANN",      "ZIP_CANN",     zip_nll,              "poisson", None,       "mu_zip",      "mu_zip",     True,  True),
        ("ZINB-CANN",     "ZINB_CANN",    zinb_nll(zinb_alpha), "negbin",  zinb_alpha, "mu_zinb",     "mu_zinb",    True,  True),
        ("ZI-Bell-CANN",  "ZIBell_CANN",  zibell_nll,           "bell",    None,       "mu_zibell",   "mu_zibell",  True,  True),
    ]

    for display_name, optuna_key, loss_fn, dist, alpha, glm_key, mu_glm_key, is_zi, is_cann in nn_configs:
        params = BEST_PARAMS[optuna_key]
        print(f"\n  Training {display_name} (tuned: {params['n_layers']}L, "
              f"{params['first_neurons']}n, {params['activation']})...")
        set_seeds(SEED)

        # Log volumes
        if glm_key is not None:
            key = mu_glm_key or glm_key
            lv_learn = np.log(np.maximum(glm_learn[key], 1e-10))
            lv_test = np.log(np.maximum(glm_test[key], 1e-10))
            lambda_cann = np.sum(ylearn) / np.sum(glm_learn[key])
        else:
            lv_learn = np.log(Elearn)
            lv_test = np.log(data["Etest"])
            lambda_cann = np.sum(ylearn) / np.sum(Elearn)

        inputs_learn = get_model_inputs(data["Xlearn"], data["BrandLearn"], data["RegionLearn"], lv_learn)
        inputs_test = get_model_inputs(data["Xtest"], data["BrandTest"], data["RegionTest"], lv_test)

        # Build with tuned architecture
        if is_zi:
            model, hl = _build_zi_from_params(
                params, loss_fn, q0, n_br, n_re, lambda_cann, pi_logit_init)
        else:
            model, hl = _build_standard_from_params(
                params, loss_fn, q0, n_br, n_re, lambda_cann)

        # Train
        cbs = [callbacks.EarlyStopping(
            monitor="val_loss", mode="min",
            patience=PATIENCE, restore_best_weights=True)]

        history = model.fit(
            inputs_learn, ylearn.reshape(-1),
            epochs=EPOCHS, batch_size=params["batch_size"],
            validation_split=VALIDATION_SPLIT,
            callbacks=cbs, verbose=0,
        )

        # Bias regularization for CANN (non-ZI only)
        if is_cann and not is_zi:
            model = bias_regularize(model, inputs_learn, ylearn)

        # Save weights
        safe_name = display_name.replace(" ", "_").replace("-", "_")
        wpath = os.path.join(SAVED_DIR, f"{safe_name}.weights.h5")
        model.save_weights(wpath)

        # Save training history
        hist_path = os.path.join(HIST_DIR, f"{safe_name}_history.pkl")
        with open(hist_path, "wb") as f:
            pickle.dump(history.history, f)

        # Evaluate
        result = evaluate_full(display_name, model, inputs_test, ytest,
                               dist=dist, alpha=alpha, is_zi=is_zi)
        results.append(result)
        print(f"  {display_name:20s}  dev={result['deviance']:.6f}  "
              f"LL={result['loglik']:.1f}  avg={result['portfolio_avg']:.6f}  "
              f"arch={hl}")

    # ==================================================================
    # Output comparison table
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("COMPARISON TABLE (OPTUNA-TUNED)")
    print(f"{'=' * 70}")

    df = pd.DataFrame(results)
    cols = ["model", "deviance", "loglik", "aic", "bic", "portfolio_avg"]
    if "n_params" in df.columns:
        cols.append("n_params")
    df = df[cols]
    df.columns = ["Model", "Test Deviance", "Log-Lik", "AIC", "BIC", "Portfolio Avg"] + \
                 (["Params"] if "n_params" in cols else [])

    print(df.to_string(index=False, float_format="%.4f"))

    df.to_csv("comparison_results_tuned.csv", index=False)
    print("\nResults saved to comparison_results_tuned.csv")
    print(f"Model weights saved to {SAVED_DIR}/")
    print(f"Training histories saved to {HIST_DIR}/")


if __name__ == "__main__":
    main()
