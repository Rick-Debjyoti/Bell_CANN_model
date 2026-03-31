"""Data loading and preparation for Bell-CANN model comparison."""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from config import DATA_PATH, SEED, N_FOLDS


# NN continuous features (min-max scaled)
NN_FEATURES = [
    "AreaX", "VehPowerX", "VehAgeX", "DrivAgeX",
    "BonusMalusX", "VehGasX", "DensityX",
]

# GLM prediction columns
GLM_PRED_COLS = [
    "poissonGLM", "bellGLM", "negbinGLM",
    "zipGLM", "zinbGLM", "zibellGLM",
]

# ZI component columns
ZI_PI_COLS  = ["pi_zip", "pi_zinb", "pi_zibell"]
ZI_MU_COLS  = ["mu_zip", "mu_zinb", "mu_zibell"]


def load_data():
    """Load R-exported learn/test CSVs and prepare model inputs.

    Returns
    -------
    dict with keys:
        learn, test : pd.DataFrame (raw)
        Xlearn, Xtest : np.ndarray (continuous NN features)
        BrandLearn, BrandTest : np.ndarray (VehBrand embedding indices)
        RegionLearn, RegionTest : np.ndarray (Region embedding indices)
        ylearn, ytest : np.ndarray (ClaimNb targets)
        Elearn, Etest : np.ndarray (Exposure)
        glm_preds_learn, glm_preds_test : dict of {name: np.ndarray}
        n_brands : int (number of unique VehBrand levels)
        n_regions : int (number of unique Region levels)
        q0 : int (number of continuous NN features)
        negbin_alpha : float (NB dispersion from R)
        zinb_alpha : float (ZINB dispersion from R)
    """
    learn = pd.read_csv(os.path.join(DATA_PATH, "bell_learn.csv"))
    test  = pd.read_csv(os.path.join(DATA_PATH, "bell_test.csv"))
    meta  = pd.read_csv(os.path.join(DATA_PATH, "glm_metadata.csv"))

    negbin_alpha = float(meta["negbin_alpha"].iloc[0])
    zinb_alpha   = float(meta["zinb_alpha"].iloc[0])

    all_data = pd.concat([learn, test], axis=0)
    n_brands  = int(all_data["VehBrandX"].max()) + 1
    n_regions = int(all_data["RegionX"].max()) + 1

    def _extract(df):
        X      = df[NN_FEATURES].values.astype(np.float32)
        brand  = df["VehBrandX"].values.astype(np.int32)
        region = df["RegionX"].values.astype(np.int32)
        y      = df["ClaimNb"].values.astype(np.float64)
        E      = df["Exposure"].values.astype(np.float64)
        glm_preds = {col: df[col].values.astype(np.float64) for col in GLM_PRED_COLS}
        # ZI components
        for col in ZI_PI_COLS + ZI_MU_COLS:
            if col in df.columns:
                glm_preds[col] = df[col].values.astype(np.float64)
        return X, brand, region, y, E, glm_preds

    Xlearn, BrandLearn, RegionLearn, ylearn, Elearn, glm_learn = _extract(learn)
    Xtest,  BrandTest,  RegionTest,  ytest,  Etest,  glm_test  = _extract(test)

    return {
        "learn": learn, "test": test,
        "Xlearn": Xlearn, "Xtest": Xtest,
        "BrandLearn": BrandLearn, "BrandTest": BrandTest,
        "RegionLearn": RegionLearn, "RegionTest": RegionTest,
        "ylearn": ylearn, "ytest": ytest,
        "Elearn": Elearn, "Etest": Etest,
        "glm_preds_learn": glm_learn,
        "glm_preds_test": glm_test,
        "n_brands": n_brands,
        "n_regions": n_regions,
        "q0": len(NN_FEATURES),
        "negbin_alpha": negbin_alpha,
        "zinb_alpha": zinb_alpha,
    }


def get_model_inputs(X, brand, region, log_vol):
    """Package arrays into the list format expected by Keras models.

    Parameters
    ----------
    X : ndarray of shape (n, q0) — continuous features
    brand : ndarray of shape (n,) — VehBrand indices
    region : ndarray of shape (n,) — Region indices
    log_vol : ndarray of shape (n,) — log(Exposure) or log(GLM pred)

    Returns
    -------
    list : [X, brand, region, log_vol]
    """
    return [X, brand, region, log_vol.reshape(-1, 1)]


def get_kfold_splits(y, n_folds=N_FOLDS, seed=SEED):
    """Stratified K-Fold splits on the learning set.

    Stratifies by ClaimNb bins: 0 vs >= 1.

    Parameters
    ----------
    y : ndarray — target (ClaimNb)
    n_folds : int
    seed : int

    Returns
    -------
    list of (train_idx, val_idx) tuples
    """
    strat_labels = (y >= 1).astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), strat_labels))
