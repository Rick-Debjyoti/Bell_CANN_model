"""Keras-compatible loss functions for Bell-CANN model comparison.

All losses operate on TensorFlow tensors and are compatible with model.compile().
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


# ---------------------------------------------------------------------------
# Precomputed Bell numbers B_0 .. B_4 (ClaimNb capped at 4)
# ---------------------------------------------------------------------------
# B_0=1, B_1=1, B_2=2, B_3=5, B_4=15
_BELL_NUMBERS = tf.constant([1.0, 1.0, 2.0, 5.0, 15.0], dtype=tf.float64)
_LOG_BELL = tf.math.log(_BELL_NUMBERS)
_LOG_FACTORIALS = tf.constant(
    [0.0, 0.0, np.log(2), np.log(6), np.log(24)], dtype=tf.float64
)


# ===================================================================
# 1. Bell Deviance Loss
# ===================================================================

def bell_deviance(y_true, y_pred):
    """Bell deviance loss (Keras-compatible).

    D_Bell = 2 * mean[
        if y==0:  -1 + exp(W(mu))
        if y>0:   exp(W(mu)) - exp(W(y)) + y*log(W(y)/W(mu))
    ]
    """
    y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.float64)
    y_pred = tf.cast(tf.reshape(y_pred, (-1,)), tf.float64)
    y_pred = tf.maximum(y_pred, 1e-10)

    w_pred = tfp.math.lambertw(y_pred)

    is_zero = tf.equal(y_true, 0.0)

    # Zero terms
    loss_zero = 2.0 * (-1.0 + tf.exp(w_pred))

    # Non-zero terms
    y_safe = tf.maximum(y_true, 1e-10)
    w_obs  = tfp.math.lambertw(y_safe)
    loss_nonzero = 2.0 * (
        tf.exp(w_pred) - tf.exp(w_obs) +
        y_true * tf.math.log(w_obs / tf.maximum(w_pred, 1e-30))
    )

    loss = tf.where(is_zero, loss_zero, loss_nonzero)
    return tf.reduce_mean(loss)


# ===================================================================
# 2. Poisson Deviance Loss
# ===================================================================

def poisson_deviance(y_true, y_pred):
    """Poisson deviance loss.

    D_Poi = 2 * mean[ mu - y + y*log(y/mu) ]
    """
    y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.float64)
    y_pred = tf.cast(tf.reshape(y_pred, (-1,)), tf.float64)
    y_pred = tf.maximum(y_pred, 1e-10)

    is_zero = tf.equal(y_true, 0.0)
    y_safe  = tf.maximum(y_true, 1e-10)

    loss_zero    = 2.0 * y_pred
    loss_nonzero = 2.0 * (y_pred - y_true + y_true * tf.math.log(y_safe / y_pred))

    loss = tf.where(is_zero, loss_zero, loss_nonzero)
    return tf.reduce_mean(loss)


# ===================================================================
# 3. NegBin NLL Loss
# ===================================================================

def negbin_nll(alpha):
    """Return a NegBin NLL loss function parameterized by dispersion alpha.

    NB2 parameterization: Var(Y) = mu + alpha * mu^2.
    NLL = -[lgamma(y+1/a) - lgamma(1/a) - lgamma(y+1)
            + (1/a)*log(1/(1+a*mu)) + y*log(a*mu/(1+a*mu))]
    """
    a = tf.constant(alpha, dtype=tf.float64)
    inv_a = 1.0 / a

    def loss_fn(y_true, y_pred):
        y = tf.cast(tf.reshape(y_true, (-1,)), tf.float64)
        mu = tf.cast(tf.reshape(y_pred, (-1,)), tf.float64)
        mu = tf.maximum(mu, 1e-10)

        nll = -(
            tf.math.lgamma(y + inv_a)
            - tf.math.lgamma(inv_a)
            - tf.math.lgamma(y + 1.0)
            + inv_a * tf.math.log(1.0 / (1.0 + a * mu))
            + y * tf.math.log(a * mu / (1.0 + a * mu))
        )
        return tf.reduce_mean(nll)

    loss_fn.__name__ = "negbin_nll"
    return loss_fn


# ===================================================================
# 4. Bell log-PMF (for ZI-Bell)
# ===================================================================

def _bell_log_pmf(y, mu):
    """Compute log Bell PMF for y in {0,1,2,3,4}.

    log P(Y=y|mu) = (1 - exp(W(mu))) + y*log(W(mu)) + log(B_y) - log(y!)
    """
    w = tfp.math.lambertw(tf.maximum(mu, 1e-10))
    y_int = tf.cast(tf.minimum(y, 4.0), tf.int32)

    log_bell_y = tf.gather(_LOG_BELL, y_int)
    log_fact_y = tf.gather(_LOG_FACTORIALS, y_int)

    return (1.0 - tf.exp(w)) + y * tf.math.log(tf.maximum(w, 1e-30)) + log_bell_y - log_fact_y


# ===================================================================
# 5. ZIP NLL Loss (for ZI models with 2-output architecture)
# ===================================================================

def zip_nll(y_true, y_pred):
    """Zero-Inflated Poisson NLL for 2-column output [mu, pi].

    y_pred[:, 0] = mu (count mean, already exp-transformed)
    y_pred[:, 1] = pi (zero-inflation probability, already sigmoid-transformed)
    """
    y = tf.cast(tf.reshape(y_true, (-1,)), tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    mu = tf.maximum(y_pred[:, 0], 1e-10)
    pi = tf.clip_by_value(y_pred[:, 1], 1e-7, 1.0 - 1e-7)

    is_zero = tf.equal(y, 0.0)

    # P(Y=0) = pi + (1-pi)*exp(-mu)
    log_p_zero = tf.math.log(pi + (1.0 - pi) * tf.exp(-mu))

    # P(Y=y|y>0) = (1-pi) * Poisson(y|mu)
    log_p_pos = (
        tf.math.log(1.0 - pi)
        - mu + y * tf.math.log(mu) - tf.math.lgamma(y + 1.0)
    )

    nll = -tf.where(is_zero, log_p_zero, log_p_pos)
    return tf.reduce_mean(nll)


# ===================================================================
# 6. ZINB NLL Loss
# ===================================================================

def zinb_nll(alpha):
    """Return ZI-NegBin NLL loss parameterized by dispersion alpha.

    y_pred[:, 0] = mu, y_pred[:, 1] = pi
    """
    a = tf.constant(alpha, dtype=tf.float64)
    inv_a = 1.0 / a

    def loss_fn(y_true, y_pred):
        y = tf.cast(tf.reshape(y_true, (-1,)), tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)
        mu = tf.maximum(y_pred[:, 0], 1e-10)
        pi = tf.clip_by_value(y_pred[:, 1], 1e-7, 1.0 - 1e-7)

        is_zero = tf.equal(y, 0.0)

        # NB P(0|mu,alpha) = (1/(1+a*mu))^(1/a)
        log_nb_zero = inv_a * tf.math.log(1.0 / (1.0 + a * mu))
        nb_zero = tf.exp(log_nb_zero)

        # P(Y=0) = pi + (1-pi)*NB(0)
        log_p_zero = tf.math.log(pi + (1.0 - pi) * nb_zero)

        # P(Y=y|y>0) = (1-pi) * NB(y|mu,alpha)
        log_nb_y = (
            tf.math.lgamma(y + inv_a)
            - tf.math.lgamma(inv_a)
            - tf.math.lgamma(y + 1.0)
            + inv_a * tf.math.log(1.0 / (1.0 + a * mu))
            + y * tf.math.log(a * mu / (1.0 + a * mu))
        )
        log_p_pos = tf.math.log(1.0 - pi) + log_nb_y

        nll = -tf.where(is_zero, log_p_zero, log_p_pos)
        return tf.reduce_mean(nll)

    loss_fn.__name__ = "zinb_nll"
    return loss_fn


# ===================================================================
# 7. ZI-Bell NLL Loss
# ===================================================================

def zibell_nll(y_true, y_pred):
    """Zero-Inflated Bell NLL for 2-column output [mu, pi].

    P(Y=0) = pi + (1-pi)*Bell_PMF(0|mu)
    P(Y=y) = (1-pi)*Bell_PMF(y|mu)  for y > 0
    """
    y = tf.cast(tf.reshape(y_true, (-1,)), tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    mu = tf.maximum(y_pred[:, 0], 1e-10)
    pi = tf.clip_by_value(y_pred[:, 1], 1e-7, 1.0 - 1e-7)

    is_zero = tf.equal(y, 0.0)

    # Bell PMF at 0: exp(1 - exp(W(mu)))
    log_bell_0 = _bell_log_pmf(tf.zeros_like(mu), mu)
    bell_0 = tf.exp(log_bell_0)

    # P(Y=0) = pi + (1-pi)*Bell(0|mu)
    log_p_zero = tf.math.log(pi + (1.0 - pi) * bell_0)

    # P(Y=y|y>0) = (1-pi)*Bell(y|mu)
    log_bell_y = _bell_log_pmf(y, mu)
    log_p_pos  = tf.math.log(1.0 - pi) + log_bell_y

    nll = -tf.where(is_zero, log_p_zero, log_p_pos)
    return tf.reduce_mean(nll)
