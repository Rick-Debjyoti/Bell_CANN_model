"""5-Fold cross-validation logic for Bell-CANN model comparison."""

import numpy as np
import tensorflow as tf

from config import N_FOLDS, SEED, BATCH_SIZE, EPOCHS, PATIENCE, VALIDATION_SPLIT
from data_loader import get_model_inputs, get_kfold_splits
from train import set_seeds, train_model, predict_standard, predict_zi
from metrics import (
    bell_deviance_metric, poisson_deviance_metric, negbin_deviance_metric,
)


def _deviance_for_dist(y_true, y_pred, dist, alpha=None):
    """Compute deviance given distribution type."""
    if dist == "poisson":
        return poisson_deviance_metric(y_true, y_pred)
    elif dist == "bell":
        return bell_deviance_metric(y_true, y_pred)
    elif dist == "negbin":
        return negbin_deviance_metric(y_true, y_pred, alpha)
    else:
        raise ValueError(f"Unknown dist: {dist}")


def run_kfold_cv(model_builder, builder_kwargs, data, dist,
                 alpha=None, is_zi=False, log_vol_key=None,
                 n_folds=N_FOLDS, seed=SEED, verbose=0):
    """Run stratified K-Fold cross-validation on the learning set.

    Parameters
    ----------
    model_builder : callable that returns a compiled Keras model
    builder_kwargs : dict passed to model_builder
    data : dict from load_data()
    dist : str in {"poisson", "bell", "negbin"} — base distribution
    alpha : float (for negbin)
    is_zi : bool — whether the model has 2-column ZI output
    log_vol_key : str or None — key in glm_preds for CANN volume,
                  None means use log(Exposure) (NN mode)
    n_folds : int
    seed : int
    verbose : int

    Returns
    -------
    dict with:
        fold_deviances : list of float (per-fold val deviance)
        mean_deviance : float
        std_deviance : float
        test_deviance : float (evaluated on held-out test set)
    """
    Xlearn    = data["Xlearn"]
    BrandLrn  = data["BrandLearn"]
    RegionLrn = data["RegionLearn"]
    ylearn    = data["ylearn"]
    Elearn    = data["Elearn"]

    Xtest    = data["Xtest"]
    BrandTst = data["BrandTest"]
    RegionTst = data["RegionTest"]
    ytest    = data["ytest"]
    Etest    = data["Etest"]

    # Determine log volume for each observation
    if log_vol_key is None:
        lv_learn = np.log(Elearn)
        lv_test  = np.log(Etest)
    else:
        lv_learn = np.log(np.maximum(data["glm_preds_learn"][log_vol_key], 1e-10))
        lv_test  = np.log(np.maximum(data["glm_preds_test"][log_vol_key], 1e-10))

    folds = get_kfold_splits(ylearn, n_folds=n_folds, seed=seed)
    fold_deviances = []

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        set_seeds(seed + fold_i)

        # Split
        X_tr  = Xlearn[train_idx]
        br_tr = BrandLrn[train_idx]
        rg_tr = RegionLrn[train_idx]
        lv_tr = lv_learn[train_idx]
        y_tr  = ylearn[train_idx]

        X_va  = Xlearn[val_idx]
        br_va = BrandLrn[val_idx]
        rg_va = RegionLrn[val_idx]
        lv_va = lv_learn[val_idx]
        y_va  = ylearn[val_idx]

        inputs_tr = get_model_inputs(X_tr, br_tr, rg_tr, lv_tr)
        inputs_va = get_model_inputs(X_va, br_va, rg_va, lv_va)

        # Build fresh model and train with explicit validation data
        model = model_builder(**builder_kwargs)
        cbs = [tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=PATIENCE,
            restore_best_weights=True
        )]
        model.fit(
            inputs_tr, y_tr.reshape(-1),
            validation_data=(inputs_va, y_va.reshape(-1)),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=cbs, verbose=verbose,
        )

        # Evaluate on validation fold
        if is_zi:
            preds = predict_zi(model, inputs_va)
            y_pred_va = preds["expected"]
        else:
            y_pred_va = predict_standard(model, inputs_va)

        dev = _deviance_for_dist(y_va, y_pred_va, dist, alpha)
        fold_deviances.append(dev)

        if verbose:
            print(f"  Fold {fold_i+1}/{n_folds}: deviance = {dev:.6f}")

    # Retrain on full learning set for test evaluation
    set_seeds(seed)
    model_full = model_builder(**builder_kwargs)
    inputs_full = get_model_inputs(Xlearn, BrandLrn, RegionLrn, lv_learn)
    inputs_test = get_model_inputs(Xtest, BrandTst, RegionTst, lv_test)

    model_full, _ = train_model(
        model_full, inputs_full, ylearn,
        batch_size=BATCH_SIZE, epochs=EPOCHS, patience=PATIENCE,
        val_split=VALIDATION_SPLIT, verbose=verbose,
    )

    if is_zi:
        preds_test = predict_zi(model_full, inputs_test)
        y_pred_test = preds_test["expected"]
    else:
        y_pred_test = predict_standard(model_full, inputs_test)

    test_dev = _deviance_for_dist(ytest, y_pred_test, dist, alpha)

    return {
        "fold_deviances": fold_deviances,
        "mean_deviance": np.mean(fold_deviances),
        "std_deviance": np.std(fold_deviances),
        "test_deviance": test_dev,
    }
