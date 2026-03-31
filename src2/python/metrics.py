"""Numpy-based evaluation metrics for Bell-CANN model comparison."""

import numpy as np
from scipy.special import lambertw as W_scipy


# Precomputed Bell numbers B_0..B_4
_BELL_NUMBERS = np.array([1, 1, 2, 5, 15], dtype=np.float64)
_LOG_BELL = np.log(_BELL_NUMBERS)
_LOG_FACTORIALS = np.array([0.0, 0.0, np.log(2), np.log(6), np.log(24)])


# ===================================================================
# Deviance Metrics
# ===================================================================

def bell_deviance_metric(y_true, y_pred):
    """Bell deviance (numpy)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_pred = np.maximum(y_pred, 1e-10)

    w_pred = np.real(W_scipy(y_pred))

    loss = np.empty_like(y_true)
    zero = y_true == 0
    loss[zero] = 2.0 * (-1.0 + np.exp(w_pred[zero]))

    nz = ~zero
    w_obs = np.real(W_scipy(np.maximum(y_true[nz], 1e-10)))
    loss[nz] = 2.0 * (
        np.exp(w_pred[nz]) - np.exp(w_obs) +
        y_true[nz] * np.log(w_obs / np.maximum(w_pred[nz], 1e-30))
    )
    return np.mean(loss)


def poisson_deviance_metric(y_true, y_pred):
    """Poisson deviance (numpy)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_pred = np.maximum(y_pred, 1e-10)

    loss = np.empty_like(y_true)
    zero = y_true == 0
    loss[zero] = 2.0 * y_pred[zero]

    nz = ~zero
    loss[nz] = 2.0 * (
        y_pred[nz] - y_true[nz] +
        y_true[nz] * np.log(y_true[nz] / y_pred[nz])
    )
    return np.mean(loss)


def negbin_deviance_metric(y_true, y_pred, alpha):
    """NegBin deviance (numpy, NB2 parameterization)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_pred = np.maximum(y_pred, 1e-10)

    loss = np.empty_like(y_true)
    zero = y_true == 0
    loss[zero] = 2.0 * (1.0 / alpha) * np.log(1.0 + alpha * y_pred[zero])

    nz = ~zero
    y = y_true[nz]
    mu = y_pred[nz]
    loss[nz] = 2.0 * (
        y * np.log(y / mu) -
        (y + 1.0 / alpha) * np.log((1.0 + alpha * y) / (1.0 + alpha * mu))
    )
    return np.mean(loss)


# ===================================================================
# Log-Likelihood Functions
# ===================================================================

def poisson_loglik(y_true, y_pred):
    """Poisson log-likelihood."""
    from scipy.stats import poisson as poisson_dist
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return np.sum(poisson_dist.logpmf(y_true.astype(int), y_pred))


def bell_loglik(y_true, y_pred):
    """Bell log-likelihood."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_pred = np.maximum(y_pred, 1e-10)

    w = np.real(W_scipy(y_pred))
    y_int = np.minimum(y_true.astype(int), 4)

    ll = (1.0 - np.exp(w)) + y_true * np.log(np.maximum(w, 1e-30))
    ll += _LOG_BELL[y_int] - _LOG_FACTORIALS[y_int]
    return np.sum(ll)


def negbin_loglik(y_true, y_pred, alpha):
    """NegBin log-likelihood (NB2)."""
    from scipy.special import gammaln
    y = np.asarray(y_true, dtype=np.float64).ravel()
    mu = np.maximum(np.asarray(y_pred, dtype=np.float64).ravel(), 1e-10)
    inv_a = 1.0 / alpha

    ll = (
        gammaln(y + inv_a)
        - gammaln(inv_a)
        - gammaln(y + 1.0)
        + inv_a * np.log(1.0 / (1.0 + alpha * mu))
        + y * np.log(alpha * mu / (1.0 + alpha * mu))
    )
    return np.sum(ll)


def zip_loglik(y_true, mu_pred, pi_pred):
    """ZIP log-likelihood."""
    y  = np.asarray(y_true, dtype=np.float64).ravel()
    mu = np.maximum(np.asarray(mu_pred, dtype=np.float64).ravel(), 1e-10)
    pi = np.clip(np.asarray(pi_pred, dtype=np.float64).ravel(), 1e-7, 1 - 1e-7)
    from scipy.special import gammaln

    ll = np.empty_like(y)
    zero = y == 0
    ll[zero] = np.log(pi[zero] + (1 - pi[zero]) * np.exp(-mu[zero]))

    nz = ~zero
    ll[nz] = (
        np.log(1 - pi[nz])
        - mu[nz] + y[nz] * np.log(mu[nz]) - gammaln(y[nz] + 1)
    )
    return np.sum(ll)


def zinb_loglik(y_true, mu_pred, pi_pred, alpha):
    """ZINB log-likelihood."""
    from scipy.special import gammaln
    y  = np.asarray(y_true, dtype=np.float64).ravel()
    mu = np.maximum(np.asarray(mu_pred, dtype=np.float64).ravel(), 1e-10)
    pi = np.clip(np.asarray(pi_pred, dtype=np.float64).ravel(), 1e-7, 1 - 1e-7)
    inv_a = 1.0 / alpha

    ll = np.empty_like(y)
    zero = y == 0
    nb_zero = (1.0 / (1.0 + alpha * mu[zero])) ** inv_a
    ll[zero] = np.log(pi[zero] + (1 - pi[zero]) * nb_zero)

    nz = ~zero
    log_nb = (
        gammaln(y[nz] + inv_a) - gammaln(inv_a) - gammaln(y[nz] + 1)
        + inv_a * np.log(1.0 / (1.0 + alpha * mu[nz]))
        + y[nz] * np.log(alpha * mu[nz] / (1.0 + alpha * mu[nz]))
    )
    ll[nz] = np.log(1 - pi[nz]) + log_nb
    return np.sum(ll)


def zibell_loglik(y_true, mu_pred, pi_pred):
    """ZI-Bell log-likelihood."""
    y  = np.asarray(y_true, dtype=np.float64).ravel()
    mu = np.maximum(np.asarray(mu_pred, dtype=np.float64).ravel(), 1e-10)
    pi = np.clip(np.asarray(pi_pred, dtype=np.float64).ravel(), 1e-7, 1 - 1e-7)

    w = np.real(W_scipy(mu))
    y_int = np.minimum(y.astype(int), 4)

    log_bell = (1.0 - np.exp(w)) + y * np.log(np.maximum(w, 1e-30))
    log_bell += _LOG_BELL[y_int] - _LOG_FACTORIALS[y_int]

    ll = np.empty_like(y)
    zero = y == 0
    bell_0 = np.exp(log_bell[zero])
    ll[zero] = np.log(pi[zero] + (1 - pi[zero]) * bell_0)

    nz = ~zero
    ll[nz] = np.log(1 - pi[nz]) + log_bell[nz]
    return np.sum(ll)


# ===================================================================
# AIC / BIC
# ===================================================================

def aic(loglik, k):
    """Akaike Information Criterion. loglik should be the log-likelihood value."""
    return -2.0 * loglik + 2.0 * k


def bic(loglik, k, n):
    """Bayesian Information Criterion."""
    return -2.0 * loglik + k * np.log(n)


# ===================================================================
# Portfolio Average
# ===================================================================

def portfolio_average(y_pred):
    """Mean predicted value (should be close to observed mean ~0.053)."""
    return float(np.mean(y_pred))
