# Zero-Inflated Bell CANN Model

Combined Actuarial Neural Network (CANN) enhancement of the Bell regression model for over-dispersed, zero-inflated claim frequency data.

## Overview

This project develops neural network extensions of zero-inflated Bell (ZI-Bell) regression for modelling insurance claim frequencies. We embed six distributional families -- Poisson, Bell, Negative Binomial, and their zero-inflated counterparts -- into three neural network architectures: stand-alone NNs, Combined Actuarial Neural Networks (CANN), and a novel LocalGLMnet formulation for interpretability.

All 18 models are compared on the `freMTPL2freq` motor insurance dataset (678,013 policies, 94.97% zeros) using Bayesian hyperparameter optimisation (Optuna), 5-fold cross-validation, and formal model-selection tests (LR, Vuong, Clarke).

## Key Features

- 18-model comparison: 6 GLM + 6 NN + 6 CANN across Poisson, Bell, NegBin families
- Zero-inflated Bell CANN with separate mu and pi output heads
- LocalGLMnet for dual interpretability (count drivers vs zero-inflation drivers)
- Optuna Bayesian hyperparameter tuning (TPE sampler)
- Embedding layers for categorical variables (VehBrand, Region)
- Custom Bell deviance and ZI-Bell NLL loss functions via Lambert W
- Bias regularisation for portfolio balance

## Setup

### Option A: Conda (recommended for local GPU)

```bash
# Create conda environment
conda env create -f environment.yml
conda activate bell_cann

# Or create from scratch:
conda create -p .conda_env python=3.10 cudatoolkit=11.2 cudnn=8.1 -c conda-forge -y
conda activate .conda_env
pip install -r requirements.txt
```

> **Note:** TensorFlow 2.10 is the last version with native Windows GPU support. Later versions require WSL2.

### Option B: Google Colab

See the Colab setup cell below or `src2/python/colab_setup.py`.

### Environment Details

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10 | |
| TensorFlow | 2.10.0 | Last Windows GPU-native version |
| TF Probability | 0.18.0 | Must match TF 2.10 |
| CUDA Toolkit | 11.2 | Via conda-forge |
| cuDNN | 8.1 | Via conda-forge |
| Optuna | 4.7.0 | Bayesian HPO |
| NumPy | 1.26.4 | |
| Pandas | 2.3.3 | |

## Running the Pipeline

```bash
# Activate environment
conda activate bell_cann
export PYTHONNOUSERSITE=1
cd src2/python

# Step 1: Optuna hyperparameter tuning (all 12 NN/CANN models)
python run_optuna_improved.py

# Step 2: Cross-validation with tuned architectures
python run_cv_tuned.py

# Step 3: LocalGLMnet interpretability analysis
python localglmnet.py

# Step 4: Generate plots and tables for the paper
python generate_plots.py
python generate_tables.py

# Monitor progress from another terminal:
python monitor_progress.py --watch 10
```

## Project Structure

```
├── src2/
│   ├── python/
│   │   ├── config.py              # Seeds, hyperparameters
│   │   ├── data_loader.py         # Data loading from R exports
│   │   ├── losses.py              # Bell deviance, ZI-Bell NLL, etc.
│   │   ├── metrics.py             # Deviance, AIC, BIC, log-likelihood
│   │   ├── models.py              # NN/CANN builders (Poisson, Bell, NegBin)
│   │   ├── zi_models.py           # ZI-NN/CANN builders (ZIP, ZINB, ZI-Bell)
│   │   ├── localglmnet.py         # LocalGLMnet interpretable models
│   │   ├── train.py               # Training, bias regularisation
│   │   ├── run_optuna_improved.py # Bayesian HPO for all 12 models
│   │   ├── run_cv_tuned.py        # 5-fold CV with tuned params
│   │   ├── run_comparison_tuned.py# Full 18-model comparison
│   │   ├── statistical_tests.py   # LR, Vuong, Clarke tests
│   │   ├── generate_plots.py      # Paper figures
│   │   ├── generate_tables.py     # LaTeX tables
│   │   └── monitor_progress.py    # Live progress dashboard
│   └── R/                         # R data exports and GLM fits
├── Reports/                       # Paper drafts and presentations
├── environment.yml                # Conda environment spec
├── requirements.txt               # Pip requirements (pinned)
└── README.md
```

## Data

The `freMTPL2freq` dataset is from the R `CASdatasets` package. GLM fits are exported from R to `src2/R/` as CSVs.

## License

This project is licensed under the [MIT License](LICENSE).
