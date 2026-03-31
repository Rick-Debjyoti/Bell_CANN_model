"""Generate LaTeX-ready tables for the Bell-CANN paper.

Creates:
1. ClaimNb frequency distribution table
2. Main comparison table (all 18 models)
3. Cross-validation results table
4. Optuna hyperparameter summary table
5. Statistical tests summary table

Usage:
    cd src2/python
    python generate_tables.py
"""

import os
import numpy as np
import pandas as pd

from config import SEED
from data_loader import load_data
from train import set_seeds

TABLE_DIR = os.path.join(os.path.dirname(__file__), "..", "tables")
os.makedirs(TABLE_DIR, exist_ok=True)


def save_latex(df, filename, caption, label, **kwargs):
    """Save DataFrame as LaTeX table."""
    path = os.path.join(TABLE_DIR, filename)
    latex = df.to_latex(
        index=kwargs.get("index", False),
        float_format=kwargs.get("float_format", "%.4f"),
        caption=caption,
        label=label,
        escape=kwargs.get("escape", True),
        column_format=kwargs.get("column_format", None),
    )
    with open(path, "w") as f:
        f.write(latex)
    print(f"  Saved: {path}")


def table_claim_distribution(data):
    """Table 1: ClaimNb frequency distribution."""
    print("\n[T1] ClaimNb distribution...")
    y = np.concatenate([data["ylearn"], data["ytest"]])
    n = len(y)

    rows = []
    for v in range(5):
        if v < 4:
            count = int(np.sum(y == v))
            label = str(v)
        else:
            count = int(np.sum(y >= 4))
            label = "4+"
        rows.append({
            "ClaimNb": label,
            "Frequency": count,
            "Percentage": f"{count/n*100:.2f}\\%",
        })
    rows.append({
        "ClaimNb": "Total",
        "Frequency": n,
        "Percentage": "100.00\\%",
    })

    df = pd.DataFrame(rows)
    save_latex(df, "claim_distribution.tex",
               caption="Distribution of claim counts in the freMTPL2freq dataset.",
               label="tab:claim_dist",
               float_format="%.0f",
               escape=False)
    return df


def table_model_comparison():
    """Table 2: Main comparison of all 18 models."""
    print("\n[T2] Model comparison...")

    # Prefer tuned results
    tuned = os.path.join(os.path.dirname(__file__), "comparison_results_tuned.csv")
    default = os.path.join(os.path.dirname(__file__), "comparison_results.csv")
    path = tuned if os.path.exists(tuned) else default

    if not os.path.exists(path):
        print("  Skipped: no comparison results found")
        return None

    df = pd.read_csv(path)
    source = "tuned" if os.path.exists(tuned) else "default"
    print(f"  Using: {os.path.basename(path)} ({source})")

    # Format for paper
    if "Params" in df.columns:
        display = df[["Model", "Test Deviance", "Log-Lik", "AIC", "BIC",
                       "Portfolio Avg", "Params"]].copy()
    else:
        display = df[["Model", "Test Deviance", "Log-Lik", "AIC", "BIC",
                       "Portfolio Avg"]].copy()

    # Highlight best per category
    save_latex(display, "model_comparison.tex",
               caption="Comparison of all models on the test set. "
                       "Deviance, AIC, and BIC: lower is better.",
               label="tab:comparison",
               column_format="l" + "r" * (len(display.columns) - 1))
    return display


def table_cv_results():
    """Table 3: Cross-validation results."""
    print("\n[T3] CV results...")

    tuned = os.path.join(os.path.dirname(__file__), "cv_results_tuned.csv")
    default = os.path.join(os.path.dirname(__file__), "cv_results.csv")
    path = tuned if os.path.exists(tuned) else default

    if not os.path.exists(path):
        print("  Skipped: no CV results found")
        return None

    df = pd.read_csv(path)
    # Add formatted mean +/- std column
    df["CV Deviance"] = df.apply(
        lambda r: f"{r['Mean CV Dev']:.4f} $\\pm$ {r['Std CV Dev']:.4f}",
        axis=1
    )
    display = df[["Model", "CV Deviance", "Test Dev"]].copy()
    display.columns = ["Model", "CV Deviance (mean $\\pm$ std)", "Test Deviance"]

    save_latex(display, "cv_results.tex",
               caption="5-fold cross-validation results with tuned architectures.",
               label="tab:cv",
               escape=False)
    return display


def table_optuna_summary():
    """Table 4: Optuna tuned hyperparameters."""
    print("\n[T4] Optuna summary...")

    path = os.path.join(os.path.dirname(__file__), "optuna_results", "optuna_summary.csv")
    if not os.path.exists(path):
        print("  Skipped: optuna_summary.csv not found")
        return None

    df = pd.read_csv(path)

    display_cols = ["model", "test_deviance", "n_layers", "first_neurons",
                    "shrink_factor", "embedding_dim", "learning_rate",
                    "activation", "batch_size"]
    available = [c for c in display_cols if c in df.columns]
    display = df[available].copy()
    display.columns = [c.replace("_", " ").title() for c in available]

    save_latex(display, "optuna_summary.tex",
               caption="Best hyperparameters found by Optuna (25 trials per model).",
               label="tab:optuna",
               float_format="%.6f",
               column_format="l" + "r" * (len(display.columns) - 1))
    return display


def table_statistical_tests():
    """Table 5: Statistical tests — split into separate sub-tables."""
    print("\n[T5] Statistical tests...")

    path = os.path.join(os.path.dirname(__file__), "statistical_tests.csv")
    if not os.path.exists(path):
        print("  Skipped: statistical_tests.csv not found")
        return None

    df = pd.read_csv(path)

    # Build a clean combined table with one "Statistic" and one "p-value" column
    rows = []
    for _, r in df.iterrows():
        test_name = r["Test"]
        p = r["p_value"]
        if "LR:" in test_name:
            rows.append({
                "Test": test_name.replace("LR: ", ""),
                "Type": "LR",
                "Statistic": f"{r['LR_statistic']:.2f}",
                "p-value": f"{p:.2e}" if p > 0 else "$< 10^{-300}$",
                "Result": f"Signif. at 1\\%" if r.get("significant_1pct") else
                          (f"Signif. at 5\\%" if r.get("significant_5pct") else "Not signif."),
            })
        elif "Vuong:" in test_name:
            rows.append({
                "Test": test_name.replace("Vuong: ", ""),
                "Type": "Vuong",
                "Statistic": f"{r['vuong_statistic']:.2f}",
                "p-value": f"{p:.2e}" if p > 1e-300 else "$< 10^{-300}$",
                "Result": str(r.get("preferred", "")),
            })
        elif "Clarke:" in test_name:
            n1 = int(r.get("n_model1_better", 0))
            n2 = int(r.get("n_model2_better", 0))
            rows.append({
                "Test": test_name.replace("Clarke: ", ""),
                "Type": "Clarke",
                "Statistic": f"{n1} vs {n2}",
                "p-value": f"{p:.2e}" if p > 1e-300 else "$< 10^{-300}$",
                "Result": str(r.get("preferred", "")),
            })

    display = pd.DataFrame(rows)
    save_latex(display, "statistical_tests.tex",
               caption="Statistical tests for model comparison. "
                       "LR: Likelihood Ratio (nested); Vuong: non-nested; "
                       "Clarke: sign-based. Model 1 listed first in each test.",
               label="tab:stat_tests",
               escape=False,
               column_format="llllll")
    return display


def main():
    set_seeds(SEED)
    data = load_data()

    print("=" * 70)
    print("GENERATING LaTeX TABLES")
    print(f"Output directory: {TABLE_DIR}")
    print("=" * 70)

    table_claim_distribution(data)
    table_model_comparison()
    table_cv_results()
    table_optuna_summary()
    table_statistical_tests()

    print(f"\n{'=' * 70}")
    print(f"All tables saved to {TABLE_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
