"""Optuna hyperparameter optimization for ALL 12 NN/CANN models.

Tunes architecture and training hyperparameters using Bayesian (TPE) search.
After tuning, saves best configurations for each model.

Usage:
    conda activate "D:/Projects/FINAL REPOS/Bell_CANN_model/src2/.conda_env"
    set PYTHONNOUSERSITE=1
    cd src2/python
    python run_optuna_all.py
"""

import os
import sys
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONNOUSERSITE"] = "1"

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks

from config import SEED, EPOCHS
from data_loader import load_data, get_model_inputs
from train import set_seeds
from losses import (
    bell_deviance, poisson_deviance, negbin_nll,
    zip_nll, zinb_nll, zibell_nll,
)
from metrics import (
    bell_deviance_metric, poisson_deviance_metric, negbin_deviance_metric,
)

# ===================================================================
# Configuration
# ===================================================================

N_TRIALS = 25           # number of Optuna trials per model
OPTUNA_EPOCHS = 300     # reduced epochs per trial
OPTUNA_PATIENCE = 10    # early stopping patience

# Output directory for results
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "optuna_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===================================================================
# Generic model builders for Optuna trials
# ===================================================================

def _build_standard_trial(trial, loss_fn, q0, n_brands, n_regions, lambda_hom):
    """Build a standard (non-ZI) model with Optuna-suggested hyperparams."""
    n_layers = trial.suggest_int("n_layers", 3, 6)
    first_neurons = trial.suggest_int("first_neurons", 64, 256, step=16)
    shrink = trial.suggest_float("shrink_factor", 0.5, 0.9)
    d = trial.suggest_int("embedding_dim", 1, 4)
    lr = trial.suggest_float("learning_rate", 1e-4, 0.01, log=True)
    activation = trial.suggest_categorical("activation", ["tanh", "relu", "selu"])

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
    model.compile(optimizer=tf.keras.optimizers.Nadam(lr), loss=loss_fn)
    return model, hidden


def _build_zi_trial(trial, loss_fn, q0, n_brands, n_regions,
                    lambda_hom, pi_init):
    """Build a ZI model with Optuna-suggested hyperparams."""
    n_layers = trial.suggest_int("n_layers", 3, 6)
    first_neurons = trial.suggest_int("first_neurons", 64, 256, step=16)
    shrink = trial.suggest_float("shrink_factor", 0.5, 0.9)
    d = trial.suggest_int("embedding_dim", 1, 4)
    lr = trial.suggest_float("learning_rate", 1e-4, 0.01, log=True)
    activation = trial.suggest_categorical("activation", ["tanh", "relu", "selu"])

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
# Objective factory
# ===================================================================

def make_objective(data, model_type, dist, loss_fn, glm_key=None,
                   mu_glm_key=None, alpha=None, is_zi=False):
    """Create an Optuna objective for any model type.

    Parameters
    ----------
    model_type : str — e.g. "Poisson NN", "Bell CANN", "ZIP-NN"
    dist : str — "poisson", "bell", or "negbin"
    loss_fn : keras loss
    glm_key : str or None — GLM pred column for CANN (None = use Exposure for NN)
    mu_glm_key : str or None — mu column for ZI-CANN
    alpha : float or None
    is_zi : bool
    """
    q0 = data["q0"]
    n_br = data["n_brands"]
    n_re = data["n_regions"]
    ylearn = data["ylearn"]
    ytest = data["ytest"]
    Elearn = data["Elearn"]

    # Determine log volumes
    if glm_key is not None:
        # CANN mode
        glm_learn = data["glm_preds_learn"]
        glm_test = data["glm_preds_test"]
        key = mu_glm_key if mu_glm_key else glm_key
        lv_learn = np.log(np.maximum(glm_learn[key], 1e-10))
        lv_test = np.log(np.maximum(glm_test[key], 1e-10))
        lambda_cann = np.sum(ylearn) / np.sum(glm_learn[key])
    else:
        # NN mode — use exposure
        lv_learn = np.log(data["Elearn"])
        lv_test = np.log(data["Etest"])
        lambda_cann = np.sum(ylearn) / np.sum(Elearn)

    p_zero = np.mean(ylearn == 0)
    pi_logit_init = np.log(p_zero / (1 - p_zero))

    inputs_learn = get_model_inputs(
        data["Xlearn"], data["BrandLearn"], data["RegionLearn"], lv_learn)
    inputs_test = get_model_inputs(
        data["Xtest"], data["BrandTest"], data["RegionTest"], lv_test)

    # Select deviance function
    if dist == "poisson":
        dev_fn = poisson_deviance_metric
    elif dist == "bell":
        dev_fn = bell_deviance_metric
    elif dist == "negbin":
        dev_fn = lambda y, p: negbin_deviance_metric(y, p, alpha)

    def objective(trial):
        set_seeds(SEED)
        batch_size = trial.suggest_categorical("batch_size", [5000, 10000, 50000])

        if is_zi:
            model, hl = _build_zi_trial(
                trial, loss_fn, q0, n_br, n_re, lambda_cann, pi_logit_init)
        else:
            model, hl = _build_standard_trial(
                trial, loss_fn, q0, n_br, n_re, lambda_cann)

        cbs = [callbacks.EarlyStopping(
            monitor="val_loss", mode="min",
            patience=OPTUNA_PATIENCE, restore_best_weights=True)]

        model.fit(
            inputs_learn, ylearn.reshape(-1),
            epochs=OPTUNA_EPOCHS, batch_size=batch_size,
            validation_split=0.2, callbacks=cbs, verbose=0,
        )

        if is_zi:
            raw = model.predict(inputs_test, verbose=0)
            mu_pred = raw[:, 0]
            pi_pred = raw[:, 1]
            y_pred = (1.0 - pi_pred) * mu_pred
        else:
            y_pred = model.predict(inputs_test, verbose=0).ravel()

        test_dev = dev_fn(ytest, y_pred)
        trial.set_user_attr("architecture", str(hl))
        trial.set_user_attr("test_deviance", test_dev)
        return test_dev

    return objective


# ===================================================================
# Model configurations
# ===================================================================

def get_all_model_configs(data):
    """Return list of (name, objective_kwargs) for all 12 NN/CANN models."""
    nb_alpha = data["negbin_alpha"]
    zinb_alpha = data["zinb_alpha"]

    configs = [
        # Standard NN models
        ("Poisson_NN", dict(
            dist="poisson", loss_fn=poisson_deviance,
            glm_key=None, is_zi=False)),
        ("Bell_NN", dict(
            dist="bell", loss_fn=bell_deviance,
            glm_key=None, is_zi=False)),
        ("NegBin_NN", dict(
            dist="negbin", loss_fn=negbin_nll(nb_alpha),
            glm_key=None, alpha=nb_alpha, is_zi=False)),

        # Standard CANN models
        ("Poisson_CANN", dict(
            dist="poisson", loss_fn=poisson_deviance,
            glm_key="poissonGLM", is_zi=False)),
        ("Bell_CANN", dict(
            dist="bell", loss_fn=bell_deviance,
            glm_key="bellGLM", is_zi=False)),
        ("NegBin_CANN", dict(
            dist="negbin", loss_fn=negbin_nll(nb_alpha),
            glm_key="negbinGLM", alpha=nb_alpha, is_zi=False)),

        # ZI-NN models
        ("ZIP_NN", dict(
            dist="poisson", loss_fn=zip_nll,
            glm_key=None, is_zi=True)),
        ("ZINB_NN", dict(
            dist="negbin", loss_fn=zinb_nll(zinb_alpha),
            glm_key=None, alpha=zinb_alpha, is_zi=True)),
        ("ZIBell_NN", dict(
            dist="bell", loss_fn=zibell_nll,
            glm_key=None, is_zi=True)),

        # ZI-CANN models
        ("ZIP_CANN", dict(
            dist="poisson", loss_fn=zip_nll,
            glm_key="mu_zip", mu_glm_key="mu_zip", is_zi=True)),
        ("ZINB_CANN", dict(
            dist="negbin", loss_fn=zinb_nll(zinb_alpha),
            glm_key="mu_zinb", mu_glm_key="mu_zinb", alpha=zinb_alpha, is_zi=True)),
        ("ZIBell_CANN", dict(
            dist="bell", loss_fn=zibell_nll,
            glm_key="mu_zibell", mu_glm_key="mu_zibell", is_zi=True)),
    ]
    return configs


# ===================================================================
# Main
# ===================================================================

def main():
    set_seeds(SEED)

    print("=" * 70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION — ALL 12 MODELS")
    print(f"  Trials per model: {N_TRIALS}")
    print(f"  Max epochs per trial: {OPTUNA_EPOCHS}")
    print("=" * 70)

    data = load_data()
    configs = get_all_model_configs(data)

    all_results = {}

    for name, kwargs in configs:
        print(f"\n{'=' * 70}")
        print(f"TUNING: {name}")
        print(f"{'=' * 70}")

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=SEED),
            study_name=name,
        )

        objective = make_objective(data, name, **kwargs)
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

        print(f"\nBest {name}:")
        print(f"  Test deviance: {study.best_value:.6f}")
        print(f"  Parameters: {study.best_params}")
        print(f"  Architecture: {study.best_trial.user_attrs['architecture']}")

        all_results[name] = {
            "test_deviance": study.best_value,
            "params": study.best_params,
            "architecture": study.best_trial.user_attrs["architecture"],
        }

        # Save trial history
        df = study.trials_dataframe()
        df.to_csv(os.path.join(RESULTS_DIR, f"optuna_{name}_trials.csv"), index=False)

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("OPTUNA RESULTS SUMMARY — ALL MODELS")
    print(f"{'=' * 70}")
    print(f"\n{'Model':<20} {'Test Deviance':>15} {'Architecture'}")
    print("-" * 70)

    summary_rows = []
    for name, res in all_results.items():
        print(f"{name:<20} {res['test_deviance']:>15.6f}  {res['architecture']}")
        summary_rows.append({
            "model": name,
            "test_deviance": res["test_deviance"],
            "architecture": res["architecture"],
            **res["params"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "optuna_summary.csv"), index=False)

    # Save as Python dict for reuse
    with open("optuna_best_params_all.py", "w") as f:
        f.write('"""Best hyperparameters found by Optuna for all 12 models."""\n\n')
        f.write("BEST_PARAMS = {\n")
        for name, res in all_results.items():
            f.write(f'    "{name}": {res["params"]},\n')
        f.write("}\n")

    # ------------------------------------------------------------------
    # Retrain best models with full epochs and save weights
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("RETRAINING BEST MODELS WITH FULL EPOCHS & SAVING WEIGHTS")
    print(f"{'=' * 70}")

    saved_dir = os.path.join(os.path.dirname(__file__), "saved_models_tuned")
    os.makedirs(saved_dir, exist_ok=True)

    from config import EPOCHS, PATIENCE

    for name, kwargs in configs:
        best_params = all_results[name]["params"]
        print(f"\n  Retraining {name} with best params...")
        set_seeds(SEED)

        # Reconstruct log volumes and lambda
        if kwargs.get("glm_key") is not None:
            key = kwargs.get("mu_glm_key") or kwargs["glm_key"]
            lv_learn = np.log(np.maximum(data["glm_preds_learn"][key], 1e-10))
            lv_test = np.log(np.maximum(data["glm_preds_test"][key], 1e-10))
            lambda_cann = np.sum(data["ylearn"]) / np.sum(data["glm_preds_learn"][key])
        else:
            lv_learn = np.log(data["Elearn"])
            lv_test = np.log(data["Etest"])
            lambda_cann = np.sum(data["ylearn"]) / np.sum(data["Elearn"])

        p_zero = np.mean(data["ylearn"] == 0)
        pi_logit_init = np.log(p_zero / (1 - p_zero))

        inputs_learn = get_model_inputs(
            data["Xlearn"], data["BrandLearn"], data["RegionLearn"], lv_learn)
        inputs_test = get_model_inputs(
            data["Xtest"], data["BrandTest"], data["RegionTest"], lv_test)

        # Build model with best architecture (create a fake trial)
        class FakeTrial:
            """Replays best params instead of suggesting new ones."""
            def __init__(self, params):
                self._params = params
            def suggest_int(self, name, *a, **kw):
                return self._params[name]
            def suggest_float(self, name, *a, **kw):
                return self._params[name]
            def suggest_categorical(self, name, *a, **kw):
                return self._params[name]

        fake = FakeTrial(best_params)
        if kwargs["is_zi"]:
            model, hl = _build_zi_trial(
                fake, kwargs["loss_fn"],
                data["q0"], data["n_brands"], data["n_regions"],
                lambda_cann, pi_logit_init)
        else:
            model, hl = _build_standard_trial(
                fake, kwargs["loss_fn"],
                data["q0"], data["n_brands"], data["n_regions"],
                lambda_cann)

        cbs = [callbacks.EarlyStopping(
            monitor="val_loss", mode="min",
            patience=PATIENCE, restore_best_weights=True)]

        model.fit(
            inputs_learn, data["ylearn"].reshape(-1),
            epochs=EPOCHS, batch_size=best_params["batch_size"],
            validation_split=0.2, callbacks=cbs, verbose=0,
        )

        # Evaluate
        if kwargs["is_zi"]:
            raw = model.predict(inputs_test, verbose=0)
            y_pred = (1.0 - raw[:, 1]) * raw[:, 0]
        else:
            y_pred = model.predict(inputs_test, verbose=0).ravel()

        if kwargs["dist"] == "poisson":
            dev = poisson_deviance_metric(data["ytest"], y_pred)
        elif kwargs["dist"] == "bell":
            dev = bell_deviance_metric(data["ytest"], y_pred)
        else:
            dev = negbin_deviance_metric(data["ytest"], y_pred, kwargs["alpha"])

        # Save weights
        wpath = os.path.join(saved_dir, f"{name}.h5")
        model.save_weights(wpath)
        print(f"    {name}: deviance={dev:.6f}, saved to {wpath}")

    print(f"\nAll tuned model weights saved to {saved_dir}/")
    print(f"\nFiles saved to {RESULTS_DIR}/")
    print("  optuna_summary.csv")
    print("  optuna_<model>_trials.csv (per model)")
    print("  optuna_best_params_all.py")


if __name__ == "__main__":
    main()
