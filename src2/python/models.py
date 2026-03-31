"""Standard NN and CANN model builders for Poisson, Bell, and NegBin.

All models follow the same architecture:
  Input: Design(q0) + VehBrand(1) + Region(1) + LogVol(1)
  Embeddings: VehBrand -> d dims, Region -> d dims
  Concatenate -> Dense layers (tanh) -> Dense(1, linear, bias=lambda_hom)
  -> Add(LogVol) -> Dense(1, exp, non-trainable) -> output (mu)

For NN:   LogVol = log(Exposure)
For CANN: LogVol = log(GLM_prediction), acts as skip connection
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from config import HIDDEN_LAYERS, EMBEDDING_DIM, ACTIVATION, LEARNING_RATE
from losses import bell_deviance, poisson_deviance, negbin_nll


def _build_standard_model(loss_fn, q0, n_brands, n_regions, d=EMBEDDING_DIM,
                          hidden_layers=HIDDEN_LAYERS, lr=LEARNING_RATE,
                          lambda_hom=0.0):
    """Core builder for standard (non-ZI) NN/CANN models.

    Parameters
    ----------
    loss_fn : Keras loss function
    q0 : int — number of continuous input features
    n_brands : int — number of VehBrand categories
    n_regions : int — number of Region categories
    d : int — embedding dimension
    hidden_layers : list of int — hidden layer sizes
    lr : float — learning rate
    lambda_hom : float — bias initializer for the last hidden->output connection

    Returns
    -------
    keras.Model
    """
    # Inputs
    Design  = layers.Input(shape=(q0,), name="Design")
    VehBrand = layers.Input(shape=(1,), name="VehBrand")
    Region  = layers.Input(shape=(1,), name="Region")
    LogVol  = layers.Input(shape=(1,), name="LogVol")

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

    # Output: single linear neuron with bias initialized at lambda_hom
    last_units = hidden_layers[-1]
    network_out = layers.Dense(
        1, activation="linear", use_bias=True,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(lambda_hom),
        name="network_out"
    )(x)

    # Add skip connection (LogVol)
    add_out = layers.Add(name="add_logvol")([network_out, LogVol])

    # Exponentiate (non-trainable)
    response = layers.Dense(
        1, activation=tf.keras.activations.exponential,
        dtype="float64", trainable=False,
        kernel_initializer="ones", bias_initializer="zeros",
        name="response"
    )(add_out)

    model = Model(inputs=[Design, VehBrand, Region, LogVol], outputs=response)
    model.compile(optimizer=tf.keras.optimizers.Nadam(lr), loss=loss_fn)
    return model


# ===================================================================
# Poisson NN / CANN
# ===================================================================

def build_poisson_nn(q0, n_brands, n_regions, lambda_hom, **kwargs):
    """Poisson NN: LogVol = log(Exposure)."""
    return _build_standard_model(poisson_deviance, q0, n_brands, n_regions,
                                 lambda_hom=lambda_hom, **kwargs)


def build_poisson_cann(q0, n_brands, n_regions, lambda_hom, **kwargs):
    """Poisson CANN: LogVol = log(poissonGLM). Same architecture, different volume."""
    return _build_standard_model(poisson_deviance, q0, n_brands, n_regions,
                                 lambda_hom=lambda_hom, **kwargs)


# ===================================================================
# Bell NN / CANN
# ===================================================================

def build_bell_nn(q0, n_brands, n_regions, lambda_hom, **kwargs):
    """Bell NN: LogVol = log(Exposure)."""
    return _build_standard_model(bell_deviance, q0, n_brands, n_regions,
                                 lambda_hom=lambda_hom, **kwargs)


def build_bell_cann(q0, n_brands, n_regions, lambda_hom, **kwargs):
    """Bell CANN: LogVol = log(bellGLM)."""
    return _build_standard_model(bell_deviance, q0, n_brands, n_regions,
                                 lambda_hom=lambda_hom, **kwargs)


# ===================================================================
# NegBin NN / CANN
# ===================================================================

def build_negbin_nn(q0, n_brands, n_regions, lambda_hom, alpha, **kwargs):
    """NegBin NN: LogVol = log(Exposure)."""
    return _build_standard_model(negbin_nll(alpha), q0, n_brands, n_regions,
                                 lambda_hom=lambda_hom, **kwargs)


def build_negbin_cann(q0, n_brands, n_regions, lambda_hom, alpha, **kwargs):
    """NegBin CANN: LogVol = log(negbinGLM)."""
    return _build_standard_model(negbin_nll(alpha), q0, n_brands, n_regions,
                                 lambda_hom=lambda_hom, **kwargs)
