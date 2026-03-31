"""Zero-Inflated NN and CANN model builders (ZIP, ZINB, ZI-Bell).

Key difference from standard models: the output layer has 2 neurons:
  - Neuron 1 -> exp activation -> mu (count rate)
  - Neuron 2 -> sigmoid activation -> pi (zero-inflation probability)

The expected value E[Y] = (1 - pi) * mu.

For CANN variants: LogVol = log(ZI-GLM prediction).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from config import HIDDEN_LAYERS, EMBEDDING_DIM, ACTIVATION, LEARNING_RATE
from losses import zip_nll, zinb_nll, zibell_nll


def _build_zi_model(loss_fn, q0, n_brands, n_regions, d=EMBEDDING_DIM,
                    hidden_layers=HIDDEN_LAYERS, lr=LEARNING_RATE,
                    lambda_hom=0.0, pi_init=0.0):
    """Core builder for zero-inflated NN/CANN models.

    Architecture:
      Input: Design(q0) + VehBrand(1) + Region(1) + LogVol(1)
      Embeddings -> Concatenate
      -> Dense(hidden) x N (tanh)
      -> Dense(2, linear)  # [raw_mu, raw_pi]
         raw_mu + LogVol -> exp -> mu_i
         raw_pi -> sigmoid -> pi_i
      -> Concatenate([mu, pi]) as 2-column output

    Parameters
    ----------
    loss_fn : Keras loss function expecting y_pred with shape (n, 2)
    q0 : int
    n_brands, n_regions : int
    d : int — embedding dim
    hidden_layers : list of int
    lr : float
    lambda_hom : float — bias init for mu neuron
    pi_init : float — bias init for pi neuron (logit scale)

    Returns
    -------
    keras.Model with output shape (n, 2) where [:, 0]=mu, [:, 1]=pi
    """
    # Inputs
    Design   = layers.Input(shape=(q0,), name="Design")
    VehBrand = layers.Input(shape=(1,), name="VehBrand")
    Region   = layers.Input(shape=(1,), name="Region")
    LogVol   = layers.Input(shape=(1,), name="LogVol")

    # Embeddings
    brand_emb  = layers.Embedding(input_dim=n_brands, output_dim=d,
                                  input_length=1, name="BrandEmb")(VehBrand)
    brand_flat = layers.Flatten(name="BrandFlat")(brand_emb)

    region_emb  = layers.Embedding(input_dim=n_regions, output_dim=d,
                                   input_length=1, name="RegionEmb")(Region)
    region_flat = layers.Flatten(name="RegionFlat")(region_emb)

    # Concatenate
    x = layers.Concatenate()([Design, brand_flat, region_flat])

    # Hidden layers
    for i, units in enumerate(hidden_layers):
        x = layers.Dense(units, activation=ACTIVATION, name=f"hidden_{i}")(x)

    # Two-neuron output (linear)
    last_units = hidden_layers[-1]

    # Mu branch
    raw_mu = layers.Dense(
        1, activation="linear", use_bias=True,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(lambda_hom),
        name="raw_mu"
    )(x)

    # Pi branch
    raw_pi = layers.Dense(
        1, activation="linear", use_bias=True,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(pi_init),
        name="raw_pi"
    )(x)

    # Mu: add LogVol then exponentiate
    mu_plus_vol = layers.Add(name="mu_add_logvol")([raw_mu, LogVol])
    mu_out = layers.Lambda(
        lambda t: tf.exp(tf.cast(t, tf.float64)), name="mu_exp",
        dtype="float64"
    )(mu_plus_vol)

    # Pi: sigmoid
    pi_out = layers.Lambda(
        lambda t: tf.sigmoid(tf.cast(t, tf.float64)), name="pi_sigmoid",
        dtype="float64"
    )(raw_pi)

    # Concatenate mu and pi as 2-column output
    output = layers.Concatenate(name="mu_pi_output", dtype="float64")(
        [mu_out, pi_out]
    )

    model = Model(inputs=[Design, VehBrand, Region, LogVol], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Nadam(lr), loss=loss_fn)
    return model


# ===================================================================
# ZIP NN / CANN
# ===================================================================

def build_zip_nn(q0, n_brands, n_regions, lambda_hom, pi_init=0.0, **kwargs):
    """ZIP NN: LogVol = log(Exposure)."""
    return _build_zi_model(zip_nll, q0, n_brands, n_regions,
                           lambda_hom=lambda_hom, pi_init=pi_init, **kwargs)


def build_zip_cann(q0, n_brands, n_regions, lambda_hom, pi_init=0.0, **kwargs):
    """ZIP CANN: LogVol = log(zipGLM / (1-pi_zip)), i.e. log of count component mu."""
    return _build_zi_model(zip_nll, q0, n_brands, n_regions,
                           lambda_hom=lambda_hom, pi_init=pi_init, **kwargs)


# ===================================================================
# ZINB NN / CANN
# ===================================================================

def build_zinb_nn(q0, n_brands, n_regions, lambda_hom, alpha,
                  pi_init=0.0, **kwargs):
    """ZINB NN: LogVol = log(Exposure)."""
    return _build_zi_model(zinb_nll(alpha), q0, n_brands, n_regions,
                           lambda_hom=lambda_hom, pi_init=pi_init, **kwargs)


def build_zinb_cann(q0, n_brands, n_regions, lambda_hom, alpha,
                    pi_init=0.0, **kwargs):
    """ZINB CANN: LogVol = log(mu_zinb) from the count component."""
    return _build_zi_model(zinb_nll(alpha), q0, n_brands, n_regions,
                           lambda_hom=lambda_hom, pi_init=pi_init, **kwargs)


# ===================================================================
# ZI-Bell NN / CANN
# ===================================================================

def build_zibell_nn(q0, n_brands, n_regions, lambda_hom, pi_init=0.0, **kwargs):
    """ZI-Bell NN: LogVol = log(Exposure)."""
    return _build_zi_model(zibell_nll, q0, n_brands, n_regions,
                           lambda_hom=lambda_hom, pi_init=pi_init, **kwargs)


def build_zibell_cann(q0, n_brands, n_regions, lambda_hom, pi_init=0.0, **kwargs):
    """ZI-Bell CANN: LogVol = log(mu_zibell) from the count component."""
    return _build_zi_model(zibell_nll, q0, n_brands, n_regions,
                           lambda_hom=lambda_hom, pi_init=pi_init, **kwargs)
