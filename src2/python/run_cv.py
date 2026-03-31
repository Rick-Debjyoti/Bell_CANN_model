"""Run 5-fold cross-validation for all 12 NN/CANN models.

GLMs are deterministic and don't need CV.

Usage:
    cd src2/python
    python run_cv.py
"""

import numpy as np
import pandas as pd

from config import SEED, N_FOLDS
from data_loader import load_data
from train import set_seeds
from cross_validation import run_kfold_cv
from models import (
    build_poisson_nn, build_poisson_cann,
    build_bell_nn, build_bell_cann,
    build_negbin_nn, build_negbin_cann,
)
from zi_models import (
    build_zip_nn, build_zip_cann,
    build_zinb_nn, build_zinb_cann,
    build_zibell_nn, build_zibell_cann,
)


def main():
    set_seeds(SEED)
    print("=" * 70)
    print(f"BELL-CANN MODEL COMPARISON — {N_FOLDS}-Fold Cross-Validation")
    print("=" * 70)

    data = load_data()
    q0 = data["q0"]
    n_br = data["n_brands"]
    n_re = data["n_regions"]
    nb_alpha = data["negbin_alpha"]
    zinb_alpha = data["zinb_alpha"]
    ylearn = data["ylearn"]
    Elearn = data["Elearn"]

    lambda_hom = np.sum(ylearn) / np.sum(Elearn)
    p_zero = np.mean(ylearn == 0)
    pi_logit_init = np.log(p_zero / (1 - p_zero))

    glm_learn = data["glm_preds_learn"]

    results = []

    # ------------------------------------------------------------------
    # Standard NN models
    # ------------------------------------------------------------------
    nn_models = [
        ("Poisson NN", build_poisson_nn, "poisson", None, None),
        ("Bell NN", build_bell_nn, "bell", None, None),
        ("NegBin NN", build_negbin_nn, "negbin", nb_alpha, None),
    ]

    for name, builder, dist, alpha, lv_key in nn_models:
        print(f"\n--- CV for {name} ---")
        bkw = dict(q0=q0, n_brands=n_br, n_regions=n_re, lambda_hom=lambda_hom)
        if alpha is not None:
            bkw["alpha"] = alpha

        cv = run_kfold_cv(
            builder, bkw, data, dist=dist, alpha=alpha,
            is_zi=False, log_vol_key=lv_key, verbose=1,
        )
        results.append({
            "Model": name,
            "Mean CV Dev": cv["mean_deviance"],
            "Std CV Dev": cv["std_deviance"],
            "Test Dev": cv["test_deviance"],
        })
        print(f"  => {cv['mean_deviance']:.6f} +/- {cv['std_deviance']:.6f}"
              f"  (test: {cv['test_deviance']:.6f})")

    # ------------------------------------------------------------------
    # Standard CANN models
    # ------------------------------------------------------------------
    cann_models = [
        ("Poisson CANN", build_poisson_cann, "poisson", None, "poissonGLM"),
        ("Bell CANN", build_bell_cann, "bell", None, "bellGLM"),
        ("NegBin CANN", build_negbin_cann, "negbin", nb_alpha, "negbinGLM"),
    ]

    for name, builder, dist, alpha, lv_key in cann_models:
        print(f"\n--- CV for {name} ---")
        lambda_cann = np.sum(ylearn) / np.sum(glm_learn[lv_key])
        bkw = dict(q0=q0, n_brands=n_br, n_regions=n_re,
                   lambda_hom=lambda_cann)
        if alpha is not None:
            bkw["alpha"] = alpha

        cv = run_kfold_cv(
            builder, bkw, data, dist=dist, alpha=alpha,
            is_zi=False, log_vol_key=lv_key, verbose=1,
        )
        results.append({
            "Model": name,
            "Mean CV Dev": cv["mean_deviance"],
            "Std CV Dev": cv["std_deviance"],
            "Test Dev": cv["test_deviance"],
        })
        print(f"  => {cv['mean_deviance']:.6f} +/- {cv['std_deviance']:.6f}"
              f"  (test: {cv['test_deviance']:.6f})")

    # ------------------------------------------------------------------
    # ZI-NN models
    # ------------------------------------------------------------------
    zi_nn_models = [
        ("ZIP-NN", build_zip_nn, "poisson", None, None),
        ("ZINB-NN", build_zinb_nn, "negbin", zinb_alpha, None),
        ("ZI-Bell-NN", build_zibell_nn, "bell", None, None),
    ]

    for name, builder, dist, alpha, lv_key in zi_nn_models:
        print(f"\n--- CV for {name} ---")
        bkw = dict(q0=q0, n_brands=n_br, n_regions=n_re,
                   lambda_hom=lambda_hom, pi_init=pi_logit_init)
        if alpha is not None:
            bkw["alpha"] = alpha

        cv = run_kfold_cv(
            builder, bkw, data, dist=dist, alpha=alpha,
            is_zi=True, log_vol_key=lv_key, verbose=1,
        )
        results.append({
            "Model": name,
            "Mean CV Dev": cv["mean_deviance"],
            "Std CV Dev": cv["std_deviance"],
            "Test Dev": cv["test_deviance"],
        })
        print(f"  => {cv['mean_deviance']:.6f} +/- {cv['std_deviance']:.6f}"
              f"  (test: {cv['test_deviance']:.6f})")

    # ------------------------------------------------------------------
    # ZI-CANN models
    # ------------------------------------------------------------------
    zi_cann_models = [
        ("ZIP-CANN", build_zip_cann, "poisson", None, "mu_zip"),
        ("ZINB-CANN", build_zinb_cann, "negbin", zinb_alpha, "mu_zinb"),
        ("ZI-Bell-CANN", build_zibell_cann, "bell", None, "mu_zibell"),
    ]

    for name, builder, dist, alpha, mu_key in zi_cann_models:
        print(f"\n--- CV for {name} ---")
        lambda_cann = np.sum(ylearn) / np.sum(glm_learn[mu_key])
        bkw = dict(q0=q0, n_brands=n_br, n_regions=n_re,
                   lambda_hom=lambda_cann, pi_init=pi_logit_init)
        if alpha is not None:
            bkw["alpha"] = alpha

        cv = run_kfold_cv(
            builder, bkw, data, dist=dist, alpha=alpha,
            is_zi=True, log_vol_key=mu_key, verbose=1,
        )
        results.append({
            "Model": name,
            "Mean CV Dev": cv["mean_deviance"],
            "Std CV Dev": cv["std_deviance"],
            "Test Dev": cv["test_deviance"],
        })
        print(f"  => {cv['mean_deviance']:.6f} +/- {cv['std_deviance']:.6f}"
              f"  (test: {cv['test_deviance']:.6f})")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)

    df = pd.DataFrame(results)
    print(df.to_string(index=False, float_format="%.6f"))

    df.to_csv("cv_results.csv", index=False)
    print("\nResults saved to cv_results.csv")


if __name__ == "__main__":
    main()
