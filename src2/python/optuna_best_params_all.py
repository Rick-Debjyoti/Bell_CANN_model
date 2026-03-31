"""Best hyperparameters found by Optuna for all 12 models.

Sources:
  - Poisson_NN/CANN, Bell_NN, NegBin_NN/CANN, ZIP_NN, ZINB_NN, ZIBell_NN:
    from original run_optuna_all.py (25 trials each, 2026-03-09)
  - Bell_CANN, ZIP_CANN, ZINB_CANN, ZIBell_CANN:
    from run_optuna_improved.py (25-40 trials, 2026-03-20, L4 GPU)
"""

BEST_PARAMS = {
    # --- Standard NN models (from original tuning) ---
    "Poisson_NN": {'batch_size': 10000, 'n_layers': 5, 'first_neurons': 192, 'shrink_factor': 0.766344165080148, 'embedding_dim': 4, 'learning_rate': 0.0010522052526038752, 'activation': 'tanh', 'dropout_rate': 0.0, 'use_l2': False},
    "Bell_NN": {'batch_size': 10000, 'n_layers': 5, 'first_neurons': 240, 'shrink_factor': 0.8039327617975071, 'embedding_dim': 2, 'learning_rate': 0.0009675335505252468, 'activation': 'selu', 'dropout_rate': 0.0, 'use_l2': False},
    "NegBin_NN": {'batch_size': 5000, 'n_layers': 5, 'first_neurons': 144, 'shrink_factor': 0.6066577538174276, 'embedding_dim': 2, 'learning_rate': 0.0010751886978654494, 'activation': 'tanh', 'dropout_rate': 0.0, 'use_l2': False},

    # --- Standard CANN models ---
    "Poisson_CANN": {'batch_size': 10000, 'n_layers': 6, 'first_neurons': 192, 'shrink_factor': 0.7955868326467614, 'embedding_dim': 4, 'learning_rate': 0.0004115448866215641, 'activation': 'tanh', 'dropout_rate': 0.0, 'use_l2': False},
    "Bell_CANN": {'batch_size': 10000, 'n_layers': 4, 'first_neurons': 288, 'shrink_factor': 0.6814375494259668, 'embedding_dim': 3, 'learning_rate': 0.001035017385127823, 'activation': 'selu', 'dropout_rate': 0.029504112318204222, 'use_l2': False},
    "NegBin_CANN": {'batch_size': 5000, 'n_layers': 5, 'first_neurons': 112, 'shrink_factor': 0.6917233857113714, 'embedding_dim': 2, 'learning_rate': 0.0009388802776974618, 'activation': 'selu', 'dropout_rate': 0.0, 'use_l2': False},

    # --- ZI-NN models (from original tuning) ---
    "ZIP_NN": {'batch_size': 10000, 'n_layers': 5, 'first_neurons': 208, 'shrink_factor': 0.7472325847149591, 'embedding_dim': 2, 'learning_rate': 0.0003234376387964643, 'activation': 'relu', 'dropout_rate': 0.0, 'use_l2': False},
    "ZINB_NN": {'batch_size': 10000, 'n_layers': 3, 'first_neurons': 208, 'shrink_factor': 0.7077409808743814, 'embedding_dim': 2, 'learning_rate': 0.0007055532988587876, 'activation': 'selu', 'dropout_rate': 0.0, 'use_l2': False},
    "ZIBell_NN": {'batch_size': 10000, 'n_layers': 5, 'first_neurons': 240, 'shrink_factor': 0.8945491500576307, 'embedding_dim': 2, 'learning_rate': 0.0012031155504323792, 'activation': 'tanh', 'dropout_rate': 0.0, 'use_l2': False},

    # --- ZI-CANN models (from improved tuning, 2026-03-20) ---
    "ZIP_CANN": {'batch_size': 10000, 'n_layers': 7, 'first_neurons': 320, 'shrink_factor': 0.7288855263809064, 'embedding_dim': 3, 'learning_rate': 0.0003843021147500349, 'activation': 'selu', 'dropout_rate': 0.02653805190086723, 'use_l2': False},
    "ZINB_CANN": {'batch_size': 10000, 'n_layers': 7, 'first_neurons': 320, 'shrink_factor': 0.7288855263809064, 'embedding_dim': 3, 'learning_rate': 0.0003843021147500349, 'activation': 'selu', 'dropout_rate': 0.02653805190086723, 'use_l2': False},
    "ZIBell_CANN": {'batch_size': 10000, 'n_layers': 7, 'first_neurons': 304, 'shrink_factor': 0.9495671618786318, 'embedding_dim': 2, 'learning_rate': 0.0006174484318004588, 'activation': 'selu', 'dropout_rate': 0.05225842661067967, 'use_l2': False},
}
 