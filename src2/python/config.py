"""Central configuration for Bell-CANN model comparison."""

import os

# Reproducibility
SEED = 100

# Paths (relative to python/ directory)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "R")

# Training
BATCH_SIZE = 10000
EPOCHS = 1000
PATIENCE = 10
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Architecture
HIDDEN_LAYERS = [150, 120, 100, 75, 50]
EMBEDDING_DIM = 2
ACTIVATION = "tanh"

# Cross-validation
N_FOLDS = 5

# NegBin dispersion initial guess (updated after R GLM fit)
NEGBIN_ALPHA_INIT = 1.0
