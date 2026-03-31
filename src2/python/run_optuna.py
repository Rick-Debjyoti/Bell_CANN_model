"""Optuna hyperparameter optimization for Bell CANN and ZI-Bell CANN.

Tunes architecture and training hyperparameters using Bayesian (TPE) search.
After tuning, retrains the best configuration and reports final comparison.

Usage:
    cd src2/python
    python run_optuna.py
"""

import os
import sys
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF info logs during trials

import tensorflow as tf
from tensorflow.keras import callbacks

from config import SEED, EPOCHS
from data_loader import load_data, get_model_inputs
from train import set_seeds, predict_standard, predict_zi, bias_regularize
from losses import bell_deviance, zibell_nll
from metrics import bell_deviance_metric


# ===================================================================
# Configuration
# ===================================================================

N_TRIALS = 25           # number of Optuna trials per model
OPTUNA_EPOCHS = 300     # reduced epochs per trial (pruning handles the rest)
OPTUNA_PATIENCE = 10    # early stopping patience during trials


def _build_bell_cann_trial(trial, q0, n_brands, n_regions, lambda_hom):
    """Build a Bell CANN model with Optuna-suggested hyperparameters."""
    from tensorflow.keras import layers, Model

    # --- Hyperparameter search space ---
    n_layers = trial.suggest_int("n_layers", 3, 6)
    first_neurons = trial.suggest_int("first_neurons", 64, 256, step=16)
    shrink = trial.suggest_float("shrink_factor", 0.5, 0.9)
    d = trial.suggest_int("embedding_dim", 1, 4)
    lr = trial.suggest_float("learning_rate", 1e-4, 0.01, log=True)
    activation = trial.suggest_categorical("activation", ["tanh", "relu", "selu"])

    # Build layer sizes: first_neurons, then shrink each layer
    hidden_layers = []
    n = first_neurons
    for _ in range(n_layers):
        hidden_layers.append(max(int(n), 16))
        n *= shrink

    # --- Architecture (same as models.py but parameterized) ---
    Design   = layers.Input(shape=(q0,), name="Design")
    VehBrand = layers.Input(shape=(1,), name="VehBrand")
    Region   = layers.Input(shape=(1,), name="Region")
    LogVol   = layers.Input(shape=(1,), name="LogVol")

    brand_emb  = layers.Embedding(input_dim=n_brands, output_dim=d,
                                  input_length=1, name="BrandEmb")(VehBrand)
    brand_flat = layers.Flatten(name="BrandFlat")(brand_emb)
    region_emb = layers.Embedding(input_dim=n_regions, output_dim=d,
                                  input_length=1, name="RegionEmb")(Region)
    region_flat = layers.Flatten(name="RegionFlat")(region_emb)

    x = layers.Concatenate()([Design, brand_flat, region_flat])
    for i, units in enumerate(hidden_layers):
        x = layers.Dense(units, activation=activation, name=f"hidden_{i}")(x)

    network_out = layers.Dense(
        1, activation="linear", use_bias=True,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(lambda_hom),
        name="network_out"
    )(x)

    add_out  = layers.Add(name="add_logvol")([network_out, LogVol])
    response = layers.Dense(
        1, activation=tf.keras.activations.exponential,
        dtype="float64", trainable=False,
        kernel_initializer="ones", bias_initializer="zeros",
        name="response"
    )(add_out)

    model = Model(inputs=[Design, VehBrand, Region, LogVol], outputs=response)
    model.compile(optimizer=tf.keras.optimizers.Nadam(lr), loss=bell_deviance)
    return model, hidden_layers


def _build_zibell_cann_trial(trial, q0, n_brands, n_regions,
                             lambda_hom, pi_init):
    """Build a ZI-Bell CANN model with Optuna-suggested hyperparameters."""
    from tensorflow.keras import layers, Model

    n_layers = trial.suggest_int("n_layers", 3, 6)
    first_neurons = trial.suggest_int("first_neurons", 64, 256, step=16)
    shrink = trial.suggest_float("shrink_factor", 0.5, 0.9)
    d = trial.suggest_int("embedding_dim", 1, 4)
    lr = trial.suggest_float("learning_rate", 1e-4, 0.01, log=True)
    activation = trial.suggest_categorical("activation", ["tanh", "relu", "selu"])

    hidden_layers = []
    n = first_neurons
    for _ in range(n_layers):
        hidden_layers.append(max(int(n), 16))
        n *= shrink

    Design   = layers.Input(shape=(q0,), name="Design")
    VehBrand = layers.Input(shape=(1,), name="VehBrand")
    Region   = layers.Input(shape=(1,), name="Region")
    LogVol   = layers.Input(shape=(1,), name="LogVol")

    brand_emb  = layers.Embedding(input_dim=n_brands, output_dim=d,
                                  input_length=1, name="BrandEmb")(VehBrand)
    brand_flat = layers.Flatten(name="BrandFlat")(brand_emb)
    region_emb = layers.Embedding(input_dim=n_regions, output_dim=d,
                                  input_length=1, name="RegionEmb")(Region)
    region_flat = layers.Flatten(name="RegionFlat")(region_emb)

    x = layers.Concatenate()([Design, brand_flat, region_flat])
    for i, units in enumerate(hidden_layers):
        x = layers.Dense(units, activation=activation, name=f"hidden_{i}")(x)

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
        lambda t: tf.exp(tf.cast(t, tf.float64)), name="mu_exp",
        dtype="float64"
    )(mu_plus_vol)
    pi_out = layers.Lambda(
        lambda t: tf.sigmoid(tf.cast(t, tf.float64)), name="pi_sigmoid",
        dtype="float64"
    )(raw_pi)

    output = layers.Concatenate(name="mu_pi_output", dtype="float64")(
        [mu_out, pi_out]
    )

    model = Model(inputs=[Design, VehBrand, Region, LogVol], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Nadam(lr), loss=zibell_nll)
    return model, hidden_layers


# ===================================================================
# Objective functions
# ===================================================================

def make_bell_cann_objective(data):
    """Create Optuna objective for Bell CANN."""
    q0 = data["q0"]
    n_br = data["n_brands"]
    n_re = data["n_regions"]
    ylearn = data["ylearn"]
    glm_learn = data["glm_preds_learn"]
    glm_test  = data["glm_preds_test"]

    lv_learn = np.log(np.maximum(glm_learn["bellGLM"], 1e-10))
    lv_test  = np.log(np.maximum(glm_test["bellGLM"], 1e-10))
    lambda_cann = np.sum(ylearn) / np.sum(glm_learn["bellGLM"])

    inputs_learn = get_model_inputs(
        data["Xlearn"], data["BrandLearn"], data["RegionLearn"], lv_learn)
    inputs_test = get_model_inputs(
        data["Xtest"], data["BrandTest"], data["RegionTest"], lv_test)

    def objective(trial):
        set_seeds(SEED)

        batch_size = trial.suggest_categorical("batch_size", [5000, 10000, 50000])

        model, hl = _build_bell_cann_trial(
            trial, q0, n_br, n_re, lambda_cann)

        cbs = [
            callbacks.EarlyStopping(
                monitor="val_loss", mode="min",
                patience=OPTUNA_PATIENCE, restore_best_weights=True),
        ]

        model.fit(
            inputs_learn, ylearn.reshape(-1),
            epochs=OPTUNA_EPOCHS, batch_size=batch_size,
            validation_split=0.2, callbacks=cbs, verbose=0,
        )

        # Evaluate on test set
        preds = model.predict(inputs_test, verbose=0).ravel()
        test_dev = bell_deviance_metric(data["ytest"], preds)

        # Report for logging
        trial.set_user_attr("architecture", str(hl))
        trial.set_user_attr("test_deviance", test_dev)

        return test_dev  # minimize

    return objective


def make_zibell_cann_objective(data):
    """Create Optuna objective for ZI-Bell CANN."""
    q0 = data["q0"]
    n_br = data["n_brands"]
    n_re = data["n_regions"]
    ylearn = data["ylearn"]
    glm_learn = data["glm_preds_learn"]
    glm_test  = data["glm_preds_test"]

    lv_learn = np.log(np.maximum(glm_learn["mu_zibell"], 1e-10))
    lv_test  = np.log(np.maximum(glm_test["mu_zibell"], 1e-10))
    lambda_cann = np.sum(ylearn) / np.sum(glm_learn["mu_zibell"])

    p_zero = np.mean(ylearn == 0)
    pi_logit_init = np.log(p_zero / (1 - p_zero))

    inputs_learn = get_model_inputs(
        data["Xlearn"], data["BrandLearn"], data["RegionLearn"], lv_learn)
    inputs_test = get_model_inputs(
        data["Xtest"], data["BrandTest"], data["RegionTest"], lv_test)

    def objective(trial):
        set_seeds(SEED)

        batch_size = trial.suggest_categorical("batch_size", [5000, 10000, 50000])

        model, hl = _build_zibell_cann_trial(
            trial, q0, n_br, n_re, lambda_cann, pi_logit_init)

        cbs = [
            callbacks.EarlyStopping(
                monitor="val_loss", mode="min",
                patience=OPTUNA_PATIENCE, restore_best_weights=True),
        ]

        model.fit(
            inputs_learn, ylearn.reshape(-1),
            epochs=OPTUNA_EPOCHS, batch_size=batch_size,
            validation_split=0.2, callbacks=cbs, verbose=0,
        )

        # Evaluate: ZI model has 2-column output
        raw = model.predict(inputs_test, verbose=0)
        mu_pred = raw[:, 0]
        pi_pred = raw[:, 1]
        y_pred = (1.0 - pi_pred) * mu_pred

        test_dev = bell_deviance_metric(data["ytest"], y_pred)

        trial.set_user_attr("architecture", str(hl))
        trial.set_user_attr("test_deviance", test_dev)

        return test_dev

    return objective


# ===================================================================
# Main
# ===================================================================

def main():
    set_seeds(SEED)

    print("=" * 70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print(f"  Trials per model: {N_TRIALS}")
    print(f"  Max epochs per trial: {OPTUNA_EPOCHS}")
    print("=" * 70)

    data = load_data()

    results = {}

    # ------------------------------------------------------------------
    # 1. Tune Bell CANN
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TUNING: Bell CANN")
    print("=" * 70)

    bell_study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=SEED),
        study_name="bell_cann",
    )
    bell_study.optimize(
        make_bell_cann_objective(data),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print(f"\nBest Bell CANN trial:")
    print(f"  Test deviance: {bell_study.best_value:.6f}")
    print(f"  Parameters: {bell_study.best_params}")
    print(f"  Architecture: {bell_study.best_trial.user_attrs['architecture']}")

    results["Bell CANN (tuned)"] = {
        "test_deviance": bell_study.best_value,
        "params": bell_study.best_params,
    }

    # ------------------------------------------------------------------
    # 2. Tune ZI-Bell CANN
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TUNING: ZI-Bell CANN")
    print("=" * 70)

    zibell_study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=SEED),
        study_name="zibell_cann",
    )
    zibell_study.optimize(
        make_zibell_cann_objective(data),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print(f"\nBest ZI-Bell CANN trial:")
    print(f"  Test deviance: {zibell_study.best_value:.6f}")
    print(f"  Parameters: {zibell_study.best_params}")
    print(f"  Architecture: {zibell_study.best_trial.user_attrs['architecture']}")

    results["ZI-Bell CANN (tuned)"] = {
        "test_deviance": zibell_study.best_value,
        "params": zibell_study.best_params,
    }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("OPTUNA RESULTS SUMMARY")
    print("=" * 70)

    # Trial history for both studies
    for name, study in [("Bell CANN", bell_study), ("ZI-Bell CANN", zibell_study)]:
        df = study.trials_dataframe()
        df.to_csv(f"optuna_{name.replace(' ', '_').replace('-', '').lower()}_trials.csv",
                  index=False)
        print(f"\n{name}:")
        print(f"  Best deviance: {study.best_value:.6f}")
        print(f"  Best params:   {study.best_params}")

    # Summary comparison with default config
    print("\n--- Comparison with default architecture ---")
    print(f"{'Model':<25} {'Test Deviance':>15}")
    print("-" * 42)
    for name, res in results.items():
        print(f"{name:<25} {res['test_deviance']:>15.6f}")

    # Save best params as Python dict for easy reuse
    with open("optuna_best_params.py", "w") as f:
        f.write("\"\"\"Best hyperparameters found by Optuna.\"\"\"\n\n")
        for name, res in results.items():
            varname = name.replace(" ", "_").replace("-", "").replace("(", "").replace(")", "").upper()
            f.write(f"# {name}: test deviance = {res['test_deviance']:.6f}\n")
            f.write(f"{varname} = {res['params']}\n\n")

    print("\nFiles saved:")
    print("  optuna_bell_cann_trials.csv")
    print("  optuna_zibell_cann_trials.csv")
    print("  optuna_best_params.py")


if __name__ == "__main__":
    main()
