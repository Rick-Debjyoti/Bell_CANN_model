"""Best hyperparameters found by Optuna."""

# Bell CANN (tuned): test deviance = 0.269245
BELL_CANN_TUNED = {'batch_size': 10000, 'n_layers': 6, 'first_neurons': 240, 'shrink_factor': 0.5096894946970388, 'embedding_dim': 2, 'learning_rate': 0.0005328635435120163, 'activation': 'selu'}

# ZI-Bell CANN (tuned): test deviance = 0.269390
ZIBELL_CANN_TUNED = {'batch_size': 10000, 'n_layers': 3, 'first_neurons': 192, 'shrink_factor': 0.7133032376882934, 'embedding_dim': 2, 'learning_rate': 0.0009226359286511553, 'activation': 'selu'}

