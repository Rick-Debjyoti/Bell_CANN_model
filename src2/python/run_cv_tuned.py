"""Run 5-fold cross-validation using Optuna-tuned architectures.

Loads best hyperparameters from optuna_best_params_all.py and uses
per-model optimal architectures for fair CV comparison.

Prerequisites:
    Run run_optuna_improved.py (or run_optuna_all.py) first.

Usage:
    cd src2/python
    python run_cv_tuned.py

    # Monitor from another terminal:
    python monitor_progress.py --watch
"""

import os
import json
import time
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks

from config import SEED, N_FOLDS, EPOCHS, PATIENCE, BATCH_SIZE, VALIDATION_SPLIT
from data_loader import load_data, get_model_inputs, get_kfold_splits
from train import set_seeds, predict_standard, predict_zi
from losses import (
    bell_deviance, poisson_deviance, negbin_nll,
    zip_nll, zinb_nll, zibell_nll,
)
from metrics import (
    bell_deviance_metric, poisson_deviance_metric, negbin_deviance_metric,
)

CV_PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "cv_progress.json")


def update_cv_progress(model_name, fold, n_folds, mean_dev=None, status="running"):
    """Write CV progress to JSON file for external monitoring."""
    progress = {}
    if os.path.exists(CV_PROGRESS_FILE):
        try:
            with open(CV_PROGRESS_FILE, "r") as f:
                progress = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    progress[model_name] = {
        "fold": f"{fold}/{n_folds}",
        "mean_dev": float(mean_dev) if mean_dev is not None else None,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    progress["_last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(CV_PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


# ===================================================================
# Model builders from params (reused from run_comparison_tuned.py)
# ===================================================================

def _build_standard_from_params(params, loss_fn, q0, n_brands, n_regions, lambda_hom):
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
    return model


def _build_zi_from_params(params, loss_fn, q0, n_brands, n_regions,
                          lambda_hom, pi_init):
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
    return model


# ===================================================================
# CV with tuned params
# ===================================================================

def run_tuned_kfold_cv(params, builder_fn, data, dist, alpha=None,
                       is_zi=False, log_vol_key=None, n_folds=N_FOLDS,
                       seed=SEED, verbose=0):
    """Run K-Fold CV using tuned architecture."""
    Xlearn    = data["Xlearn"]
    BrandLrn  = data["BrandLearn"]
    RegionLrn = data["RegionLearn"]
    ylearn    = data["ylearn"]
    Elearn    = data["Elearn"]

    Xtest     = data["Xtest"]
    BrandTst  = data["BrandTest"]
    RegionTst = data["RegionTest"]
    ytest     = data["ytest"]
    Etest     = data["Etest"]

    if log_vol_key is None:
        lv_learn = np.log(Elearn)
        lv_test  = np.log(Etest)
    else:
        lv_learn = np.log(np.maximum(data["glm_preds_learn"][log_vol_key], 1e-10))
        lv_test  = np.log(np.maximum(data["glm_preds_test"][log_vol_key], 1e-10))

    batch_size = params["batch_size"]

    folds = get_kfold_splits(ylearn, n_folds=n_folds, seed=seed)
    fold_deviances = []

    dev_fn = {
        "poisson": poisson_deviance_metric,
        "bell": bell_deviance_metric,
        "negbin": lambda y, p: negbin_deviance_metric(y, p, alpha),
    }[dist]

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        set_seeds(seed + fold_i)

        X_tr, br_tr, rg_tr = Xlearn[train_idx], BrandLrn[train_idx], RegionLrn[train_idx]
        lv_tr, y_tr = lv_learn[train_idx], ylearn[train_idx]

        X_va, br_va, rg_va = Xlearn[val_idx], BrandLrn[val_idx], RegionLrn[val_idx]
        lv_va, y_va = lv_learn[val_idx], ylearn[val_idx]

        inputs_tr = get_model_inputs(X_tr, br_tr, rg_tr, lv_tr)
        inputs_va = get_model_inputs(X_va, br_va, rg_va, lv_va)

        model = builder_fn(params)

        cbs = [callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=PATIENCE,
            restore_best_weights=True)]

        model.fit(
            inputs_tr, y_tr.reshape(-1),
            validation_data=(inputs_va, y_va.reshape(-1)),
            epochs=EPOCHS, batch_size=batch_size,
            callbacks=cbs, verbose=verbose,
        )

        if is_zi:
            raw = model.predict(inputs_va, verbose=0)
            y_pred_va = (1.0 - raw[:, 1]) * raw[:, 0]
        else:
            y_pred_va = model.predict(inputs_va, verbose=0).ravel()

        dev = dev_fn(y_va, y_pred_va)
        fold_deviances.append(dev)
        if verbose:
            print(f"  Fold {fold_i+1}/{n_folds}: deviance = {dev:.6f}")
            # Log per-fold progress (builder_fn name not available here,
            # so we log generically — the main loop also logs)
            try:
                mean_so_far = np.mean(fold_deviances)
                # Write a generic fold marker to progress
                progress = {}
                if os.path.exists(CV_PROGRESS_FILE):
                    with open(CV_PROGRESS_FILE, "r") as f:
                        progress = json.load(f)
                # Find which model is currently running
                for k, v in progress.items():
                    if k != "_last_update" and v.get("status") == "starting":
                        progress[k]["fold"] = f"{fold_i+1}/{n_folds}"
                        progress[k]["mean_dev"] = float(mean_so_far)
                        progress[k]["status"] = "running"
                progress["_last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(CV_PROGRESS_FILE, "w") as f:
                    json.dump(progress, f, indent=2)
            except Exception:
                pass  # monitoring is best-effort

    # Retrain on full learning set
    set_seeds(seed)
    model_full = builder_fn(params)
    inputs_full = get_model_inputs(Xlearn, BrandLrn, RegionLrn, lv_learn)
    inputs_test = get_model_inputs(Xtest, BrandTst, RegionTst, lv_test)

    cbs = [callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=PATIENCE,
        restore_best_weights=True)]

    model_full.fit(
        inputs_full, ylearn.reshape(-1),
        epochs=EPOCHS, batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        callbacks=cbs, verbose=verbose,
    )

    if is_zi:
        raw = model_full.predict(inputs_test, verbose=0)
        y_pred_test = (1.0 - raw[:, 1]) * raw[:, 0]
    else:
        y_pred_test = model_full.predict(inputs_test, verbose=0).ravel()

    test_dev = dev_fn(ytest, y_pred_test)

    return {
        "fold_deviances": fold_deviances,
        "mean_deviance": np.mean(fold_deviances),
        "std_deviance": np.std(fold_deviances),
        "test_deviance": test_dev,
    }


# ===================================================================
# Main
# ===================================================================

def main():
    set_seeds(SEED)

    # Load best params
    params_file = os.path.join(os.path.dirname(__file__), "optuna_best_params_all.py")
    if not os.path.exists(params_file):
        print("ERROR: optuna_best_params_all.py not found. Run run_optuna_all.py first!")
        return

    import importlib.util
    spec = importlib.util.spec_from_file_location("bp", params_file)
    bp_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bp_mod)
    BEST = bp_mod.BEST_PARAMS

    print("=" * 70)
    print(f"BELL-CANN — {N_FOLDS}-Fold CV (OPTUNA-TUNED ARCHITECTURES)")
    print("=" * 70)

    data = load_data()
    q0 = data["q0"]
    n_br = data["n_brands"]
    n_re = data["n_regions"]
    nb_alpha = data["negbin_alpha"]
    zinb_alpha = data["zinb_alpha"]
    ylearn = data["ylearn"]
    Elearn = data["Elearn"]
    glm_learn = data["glm_preds_learn"]

    lambda_hom = np.sum(ylearn) / np.sum(Elearn)
    p_zero = np.mean(ylearn == 0)
    pi_logit_init = np.log(p_zero / (1 - p_zero))

    # Model configs: (display_name, optuna_key, loss_fn, dist, alpha,
    #                 glm_key, is_zi)
    configs = [
        ("Poisson NN",    "Poisson_NN",   poisson_deviance,     "poisson", None,       None,         False),
        ("Bell NN",       "Bell_NN",      bell_deviance,        "bell",    None,       None,         False),
        ("NegBin NN",     "NegBin_NN",    negbin_nll(nb_alpha), "negbin",  nb_alpha,   None,         False),
        ("Poisson CANN",  "Poisson_CANN", poisson_deviance,     "poisson", None,       "poissonGLM", False),
        ("Bell CANN",     "Bell_CANN",    bell_deviance,        "bell",    None,       "bellGLM",    False),
        ("NegBin CANN",   "NegBin_CANN",  negbin_nll(nb_alpha), "negbin",  nb_alpha,   "negbinGLM",  False),
        ("ZIP-NN",        "ZIP_NN",       zip_nll,              "poisson", None,       None,         True),
        ("ZINB-NN",       "ZINB_NN",      zinb_nll(zinb_alpha), "negbin",  zinb_alpha, None,         True),
        ("ZI-Bell-NN",    "ZIBell_NN",    zibell_nll,           "bell",    None,       None,         True),
        ("ZIP-CANN",      "ZIP_CANN",     zip_nll,              "poisson", None,       "mu_zip",     True),
        ("ZINB-CANN",     "ZINB_CANN",    zinb_nll(zinb_alpha), "negbin",  zinb_alpha, "mu_zinb",    True),
        ("ZI-Bell-CANN",  "ZIBell_CANN",  zibell_nll,           "bell",    None,       "mu_zibell",  True),
    ]

    results = []

    for display_name, optuna_key, loss_fn, dist, alpha, glm_key, is_zi in configs:
        params = BEST[optuna_key]
        print(f"\n--- CV for {display_name} "
              f"({params['n_layers']}L, {params['first_neurons']}n, {params['activation']}) ---")

        # Compute lambda for this model
        if glm_key is not None:
            lam = np.sum(ylearn) / np.sum(glm_learn[glm_key])
        else:
            lam = lambda_hom

        # Create a builder closure that captures the right params
        def make_builder(loss_fn=loss_fn, is_zi=is_zi, lam=lam):
            def builder(p):
                if is_zi:
                    return _build_zi_from_params(
                        p, loss_fn, q0, n_br, n_re, lam, pi_logit_init)
                else:
                    return _build_standard_from_params(
                        p, loss_fn, q0, n_br, n_re, lam)
            return builder

        update_cv_progress(optuna_key, 0, N_FOLDS, status="starting")

        cv = run_tuned_kfold_cv(
            params, make_builder(), data,
            dist=dist, alpha=alpha, is_zi=is_zi,
            log_vol_key=glm_key, verbose=1,
        )

        results.append({
            "Model": display_name,
            "Mean CV Dev": cv["mean_deviance"],
            "Std CV Dev": cv["std_deviance"],
            "Test Dev": cv["test_deviance"],
        })
        print(f"  => {cv['mean_deviance']:.6f} +/- {cv['std_deviance']:.6f}"
              f"  (test: {cv['test_deviance']:.6f})")

        update_cv_progress(optuna_key, N_FOLDS, N_FOLDS,
                           cv["mean_deviance"], "completed")

    # Output
    print(f"\n{'=' * 70}")
    print("CROSS-VALIDATION RESULTS (TUNED)")
    print(f"{'=' * 70}")

    df = pd.DataFrame(results)
    print(df.to_string(index=False, float_format="%.6f"))

    df.to_csv("cv_results_tuned.csv", index=False)
    print("\nResults saved to cv_results_tuned.csv")


if __name__ == "__main__":
    main()
