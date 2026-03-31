"""Run all 18 models (6 GLM + 6 NN + 6 CANN) and output comparison table.

Saves trained model weights to saved_models/ for reuse by generate_plots.py.

Usage:
    cd src2/python
    python run_comparison.py
"""

import os
import numpy as np
import pandas as pd

from config import SEED
from data_loader import load_data, get_model_inputs
from train import (
    set_seeds, train_model, bias_regularize,
    predict_standard, predict_zi,
    evaluate_glm, evaluate_nn,
)

# Directory for saving trained model weights
SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def _save_model(model, name):
    """Save model weights to saved_models/<name>.h5."""
    safe_name = name.replace(" ", "_").replace("-", "_")
    path = os.path.join(SAVED_MODELS_DIR, f"{safe_name}.h5")
    model.save_weights(path)
    print(f"    Saved weights: {path}")
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
    print("BELL-CANN MODEL COMPARISON — 18 Models")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data = load_data()
    q0 = data["q0"]
    n_br = data["n_brands"]
    n_re = data["n_regions"]
    nb_alpha = data["negbin_alpha"]
    zinb_alpha = data["zinb_alpha"]

    ylearn = data["ylearn"]
    ytest  = data["ytest"]
    Elearn = data["Elearn"]
    Etest  = data["Etest"]
    glm_learn = data["glm_preds_learn"]
    glm_test  = data["glm_preds_test"]

    # Homogeneous rate
    lambda_hom = np.sum(ylearn) / np.sum(Elearn)
    print(f"\nHomogeneous rate (lambda_hom): {lambda_hom:.6f}")
    print(f"NegBin alpha: {nb_alpha:.6f}")
    print(f"ZINB alpha:   {zinb_alpha:.6f}")

    p_glm = 9  # number of GLM covariates

    results = []

    # ==================================================================
    # PART 1: GLM models (predictions from R)
    # ==================================================================
    print("\n--- Evaluating GLM models ---")

    # 1. Poisson GLM
    results.append(evaluate_glm(
        "Poisson GLM", ytest, glm_test["poissonGLM"], p_glm, dist="poisson"))

    # 2. Bell GLM
    results.append(evaluate_glm(
        "Bell GLM", ytest, glm_test["bellGLM"], p_glm, dist="bell"))

    # 3. NegBin GLM
    results.append(evaluate_glm(
        "NegBin GLM", ytest, glm_test["negbinGLM"], p_glm + 1,
        dist="negbin", alpha=nb_alpha))

    # 4. ZIP GLM
    results.append(evaluate_glm(
        "ZIP GLM", ytest, glm_test["zipGLM"], 2 * p_glm + 2,
        dist="zip", mu_pred=glm_test["mu_zip"], pi_pred=glm_test["pi_zip"]))

    # 5. ZINB GLM
    results.append(evaluate_glm(
        "ZINB GLM", ytest, glm_test["zinbGLM"], 2 * p_glm + 3,
        dist="zinb", alpha=zinb_alpha,
        mu_pred=glm_test["mu_zinb"], pi_pred=glm_test["pi_zinb"]))

    # 6. ZI-Bell GLM
    results.append(evaluate_glm(
        "ZI-Bell GLM", ytest, glm_test["zibellGLM"], 2 * p_glm + 1,
        dist="zibell",
        mu_pred=glm_test["mu_zibell"], pi_pred=glm_test["pi_zibell"]))

    for r in results:
        print(f"  {r['model']:20s}  dev={r['deviance']:.6f}  "
              f"AIC={r['aic']:.1f}  avg={r['portfolio_avg']:.6f}")

    # ==================================================================
    # PART 2: Standard NN models
    # ==================================================================
    print("\n--- Training NN models ---")

    nn_configs = [
        ("Poisson NN", build_poisson_nn, "poisson", None,
         np.log(Elearn), np.log(Etest)),
        ("Bell NN", build_bell_nn, "bell", None,
         np.log(Elearn), np.log(Etest)),
        ("NegBin NN", build_negbin_nn, "negbin", nb_alpha,
         np.log(Elearn), np.log(Etest)),
    ]

    for name, builder, dist, alpha, lv_learn, lv_test in nn_configs:
        print(f"\n  Training {name}...")
        set_seeds(SEED)

        bkw = dict(q0=q0, n_brands=n_br, n_regions=n_re, lambda_hom=lambda_hom)
        if alpha is not None:
            bkw["alpha"] = alpha

        model = builder(**bkw)
        inputs_learn = get_model_inputs(
            data["Xlearn"], data["BrandLearn"], data["RegionLearn"], lv_learn)
        inputs_test = get_model_inputs(
            data["Xtest"], data["BrandTest"], data["RegionTest"], lv_test)

        model, hist = train_model(model, inputs_learn, ylearn)
        _save_model(model, name)
        result = evaluate_nn(name, model, inputs_test, ytest,
                             dist=dist, alpha=alpha)
        results.append(result)
        print(f"  {name:20s}  dev={result['deviance']:.6f}  "
              f"avg={result['portfolio_avg']:.6f}")

    # ==================================================================
    # PART 3: CANN models (skip connection from GLM)
    # ==================================================================
    print("\n--- Training CANN models ---")

    cann_configs = [
        ("Poisson CANN", build_poisson_cann, "poisson", None, "poissonGLM"),
        ("Bell CANN", build_bell_cann, "bell", None, "bellGLM"),
        ("NegBin CANN", build_negbin_cann, "negbin", nb_alpha, "negbinGLM"),
    ]

    for name, builder, dist, alpha, glm_key in cann_configs:
        print(f"\n  Training {name}...")
        set_seeds(SEED)

        lv_learn = np.log(np.maximum(glm_learn[glm_key], 1e-10))
        lv_test  = np.log(np.maximum(glm_test[glm_key], 1e-10))

        lambda_cann = np.sum(ylearn) / np.sum(glm_learn[glm_key])

        bkw = dict(q0=q0, n_brands=n_br, n_regions=n_re,
                   lambda_hom=lambda_cann)
        if alpha is not None:
            bkw["alpha"] = alpha

        model = builder(**bkw)
        inputs_learn = get_model_inputs(
            data["Xlearn"], data["BrandLearn"], data["RegionLearn"], lv_learn)
        inputs_test = get_model_inputs(
            data["Xtest"], data["BrandTest"], data["RegionTest"], lv_test)

        model, hist = train_model(model, inputs_learn, ylearn)

        # Bias regularization for CANN
        model = bias_regularize(model, inputs_learn, ylearn)
        _save_model(model, name)

        result = evaluate_nn(name, model, inputs_test, ytest,
                             dist=dist, alpha=alpha)
        results.append(result)
        print(f"  {name:20s}  dev={result['deviance']:.6f}  "
              f"avg={result['portfolio_avg']:.6f}")

    # ==================================================================
    # PART 4: Zero-Inflated NN models
    # ==================================================================
    print("\n--- Training ZI-NN models ---")

    # Compute pi_init in logit scale from observed zero proportion
    p_zero = np.mean(ylearn == 0)
    pi_logit_init = np.log(p_zero / (1 - p_zero))

    zi_nn_configs = [
        ("ZIP-NN", build_zip_nn, "poisson", None,
         np.log(Elearn), np.log(Etest)),
        ("ZINB-NN", build_zinb_nn, "negbin", zinb_alpha,
         np.log(Elearn), np.log(Etest)),
        ("ZI-Bell-NN", build_zibell_nn, "bell", None,
         np.log(Elearn), np.log(Etest)),
    ]

    for name, builder, dist, alpha, lv_learn, lv_test in zi_nn_configs:
        print(f"\n  Training {name}...")
        set_seeds(SEED)

        bkw = dict(q0=q0, n_brands=n_br, n_regions=n_re,
                   lambda_hom=lambda_hom, pi_init=pi_logit_init)
        if alpha is not None:
            bkw["alpha"] = alpha

        model = builder(**bkw)
        inputs_learn = get_model_inputs(
            data["Xlearn"], data["BrandLearn"], data["RegionLearn"], lv_learn)
        inputs_test = get_model_inputs(
            data["Xtest"], data["BrandTest"], data["RegionTest"], lv_test)

        model, hist = train_model(model, inputs_learn, ylearn)
        _save_model(model, name)
        result = evaluate_nn(name, model, inputs_test, ytest,
                             dist=dist, alpha=alpha, is_zi=True)
        results.append(result)
        print(f"  {name:20s}  dev={result['deviance']:.6f}  "
              f"avg={result['portfolio_avg']:.6f}")

    # ==================================================================
    # PART 5: Zero-Inflated CANN models
    # ==================================================================
    print("\n--- Training ZI-CANN models ---")

    zi_cann_configs = [
        ("ZIP-CANN", build_zip_cann, "poisson", None, "mu_zip"),
        ("ZINB-CANN", build_zinb_cann, "negbin", zinb_alpha, "mu_zinb"),
        ("ZI-Bell-CANN", build_zibell_cann, "bell", None, "mu_zibell"),
    ]

    for name, builder, dist, alpha, mu_key in zi_cann_configs:
        print(f"\n  Training {name}...")
        set_seeds(SEED)

        # For ZI-CANN, LogVol = log(mu from GLM count component)
        lv_learn = np.log(np.maximum(glm_learn[mu_key], 1e-10))
        lv_test  = np.log(np.maximum(glm_test[mu_key], 1e-10))

        lambda_cann = np.sum(ylearn) / np.sum(glm_learn[mu_key])

        bkw = dict(q0=q0, n_brands=n_br, n_regions=n_re,
                   lambda_hom=lambda_cann, pi_init=pi_logit_init)
        if alpha is not None:
            bkw["alpha"] = alpha

        model = builder(**bkw)
        inputs_learn = get_model_inputs(
            data["Xlearn"], data["BrandLearn"], data["RegionLearn"], lv_learn)
        inputs_test = get_model_inputs(
            data["Xtest"], data["BrandTest"], data["RegionTest"], lv_test)

        model, hist = train_model(model, inputs_learn, ylearn)
        _save_model(model, name)
        result = evaluate_nn(name, model, inputs_test, ytest,
                             dist=dist, alpha=alpha, is_zi=True)
        results.append(result)
        print(f"  {name:20s}  dev={result['deviance']:.6f}  "
              f"avg={result['portfolio_avg']:.6f}")

    # ==================================================================
    # Output comparison table
    # ==================================================================
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    df = pd.DataFrame(results)
    df = df[["model", "deviance", "loglik", "aic", "bic", "portfolio_avg"]]
    df.columns = ["Model", "Test Deviance", "Log-Lik", "AIC", "BIC", "Portfolio Avg"]

    print(df.to_string(index=False, float_format="%.4f"))

    df.to_csv("comparison_results.csv", index=False)
    print("\nResults saved to comparison_results.csv")


if __name__ == "__main__":
    main()
