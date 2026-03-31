"""Training and evaluation logic for Bell-CANN model comparison."""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks


def setup_gpu():
    """Configure GPU: enable memory growth, log device info."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU enabled: {[g.name for g in gpus]}")
    else:
        print("No GPU found — running on CPU")


setup_gpu()

from config import SEED, BATCH_SIZE, EPOCHS, PATIENCE, VALIDATION_SPLIT
from data_loader import get_model_inputs
from metrics import (
    bell_deviance_metric, poisson_deviance_metric, negbin_deviance_metric,
    poisson_loglik, bell_loglik, negbin_loglik,
    zip_loglik, zinb_loglik, zibell_loglik,
    aic, bic, portfolio_average,
)


def set_seeds(seed=SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_model(model, X_train, y_train, batch_size=BATCH_SIZE,
                epochs=EPOCHS, patience=PATIENCE, val_split=VALIDATION_SPLIT,
                verbose=2):
    """Train a Keras model with early stopping.

    Parameters
    ----------
    model : keras.Model (compiled)
    X_train : list of arrays [Design, Brand, Region, LogVol]
    y_train : ndarray
    batch_size, epochs, patience, val_split : training params
    verbose : int

    Returns
    -------
    model : trained model
    history : keras History object
    """
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss", mode="min",
            patience=patience, restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train, y_train.reshape(-1),
        epochs=epochs, batch_size=batch_size,
        validation_split=val_split, callbacks=cbs,
        verbose=verbose,
    )
    return model, history


def bias_regularize(model, X_data, y_data):
    """Apply bias regularization to ensure portfolio balance.

    Rescales the network output bias so that mean(pred) == mean(obs).
    """
    preds = model.predict(X_data, verbose=0).ravel()
    ratio = np.mean(y_data) / np.mean(preds)

    # Adjust the 'network_out' layer bias
    net_layer = model.get_layer("network_out")
    weights = net_layer.get_weights()
    weights[1] = weights[1] + np.log(ratio)  # add log(ratio) to bias
    net_layer.set_weights(weights)

    return model


def predict_standard(model, X_inputs):
    """Get predictions from a standard (non-ZI) model.

    Returns
    -------
    ndarray of shape (n,)
    """
    return model.predict(X_inputs, verbose=0).ravel()


def predict_zi(model, X_inputs):
    """Get predictions from a ZI model (2-column output).

    Returns
    -------
    dict with keys 'expected', 'mu', 'pi'
    """
    raw = model.predict(X_inputs, verbose=0)
    mu = raw[:, 0]
    pi = raw[:, 1]
    expected = (1.0 - pi) * mu
    return {"expected": expected, "mu": mu, "pi": pi}


def evaluate_glm(name, y_true, y_pred, n_params, dist="poisson",
                 alpha=None, mu_pred=None, pi_pred=None):
    """Evaluate a GLM model (predictions already computed in R).

    Parameters
    ----------
    name : str
    y_true : ndarray
    y_pred : ndarray — E[Y] = (1-pi)*mu for ZI, mu for standard
    n_params : int
    dist : str in {"poisson", "bell", "negbin", "zip", "zinb", "zibell"}
    alpha : float (for negbin/zinb)
    mu_pred : ndarray (count mean for ZI models)
    pi_pred : ndarray (zero-inflation prob for ZI models)

    Returns
    -------
    dict with metrics
    """
    n = len(y_true)

    # Deviance
    if dist == "poisson":
        dev = poisson_deviance_metric(y_true, y_pred)
        ll  = poisson_loglik(y_true, y_pred)
    elif dist == "bell":
        dev = bell_deviance_metric(y_true, y_pred)
        ll  = bell_loglik(y_true, y_pred)
    elif dist == "negbin":
        dev = negbin_deviance_metric(y_true, y_pred, alpha)
        ll  = negbin_loglik(y_true, y_pred, alpha)
    elif dist == "zip":
        dev = poisson_deviance_metric(y_true, y_pred)
        ll  = zip_loglik(y_true, mu_pred, pi_pred)
    elif dist == "zinb":
        dev = negbin_deviance_metric(y_true, y_pred, alpha)
        ll  = zinb_loglik(y_true, mu_pred, pi_pred, alpha)
    elif dist == "zibell":
        dev = bell_deviance_metric(y_true, y_pred)
        ll  = zibell_loglik(y_true, mu_pred, pi_pred)
    else:
        raise ValueError(f"Unknown dist: {dist}")

    return {
        "model": name,
        "deviance": dev,
        "loglik": ll,
        "aic": aic(ll, n_params),
        "bic": bic(ll, n_params, n),
        "portfolio_avg": portfolio_average(y_pred),
    }


def evaluate_nn(name, model, X_inputs, y_true, dist="poisson", alpha=None,
                is_zi=False):
    """Evaluate a trained NN/CANN model.

    Parameters
    ----------
    name : str
    model : keras.Model
    X_inputs : list of arrays
    y_true : ndarray
    dist : str
    alpha : float
    is_zi : bool — whether this is a ZI model with 2-column output

    Returns
    -------
    dict with metrics
    """
    n = len(y_true)
    n_params = model.count_params()

    if is_zi:
        preds = predict_zi(model, X_inputs)
        y_pred = preds["expected"]
        mu_pred = preds["mu"]
        pi_pred = preds["pi"]
    else:
        y_pred = predict_standard(model, X_inputs)
        mu_pred = y_pred
        pi_pred = None

    if dist == "poisson":
        dev = poisson_deviance_metric(y_true, y_pred)
        ll  = poisson_loglik(y_true, mu_pred) if not is_zi else zip_loglik(y_true, mu_pred, pi_pred)
    elif dist == "bell":
        dev = bell_deviance_metric(y_true, y_pred)
        ll  = bell_loglik(y_true, mu_pred) if not is_zi else zibell_loglik(y_true, mu_pred, pi_pred)
    elif dist == "negbin":
        dev = negbin_deviance_metric(y_true, y_pred, alpha)
        if is_zi:
            ll = zinb_loglik(y_true, mu_pred, pi_pred, alpha)
        else:
            ll = negbin_loglik(y_true, mu_pred, alpha)
    else:
        raise ValueError(f"Unknown dist: {dist}")

    return {
        "model": name,
        "deviance": dev,
        "loglik": ll,
        "aic": aic(ll, n_params),
        "bic": bic(ll, n_params, n),
        "portfolio_avg": portfolio_average(y_pred),
    }
