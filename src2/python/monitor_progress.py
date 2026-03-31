"""Monitor Optuna and CV progress from another terminal.

Usage:
    cd src2/python
    python monitor_progress.py              # one-shot check
    python monitor_progress.py --watch      # auto-refresh every 10s
    python monitor_progress.py --watch 5    # refresh every 5s
"""

import os
import sys
import json
import time
import glob
import argparse
from datetime import datetime


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "optuna_results")
PROGRESS_FILE = os.path.join(RESULTS_DIR, "optuna_progress.json")
CV_PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "cv_progress.json")


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def format_bar(current, total, width=30):
    """Create a progress bar string."""
    if total == 0:
        return "[" + " " * width + "]"
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    pct = 100 * current / total
    return f"[{bar}] {pct:5.1f}%"


def show_optuna_progress():
    """Display Optuna tuning progress."""
    print("=" * 72)
    print("  OPTUNA HYPERPARAMETER OPTIMIZATION PROGRESS")
    print("=" * 72)

    if not os.path.exists(PROGRESS_FILE):
        # Fall back to checking CSV files
        csvs = sorted(glob.glob(os.path.join(RESULTS_DIR, "optuna_*_trials.csv")))
        if not csvs:
            print("  No Optuna results found yet.")
            return

        print(f"\n  {'Model':<20} {'Trials':>8} {'Best Deviance':>15}")
        print("  " + "-" * 50)
        for csv_path in csvs:
            name = os.path.basename(csv_path).replace("optuna_", "").replace("_trials.csv", "")
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                n_trials = len(df[df["state"] == "COMPLETE"])
                best = df[df["state"] == "COMPLETE"]["value"].min()
                print(f"  {name:<20} {n_trials:>8} {best:>15.6f}")
            except Exception:
                print(f"  {name:<20} (error reading)")
        return

    with open(PROGRESS_FILE, "r") as f:
        progress = json.load(f)

    last_update = progress.pop("_last_update", "unknown")
    print(f"  Last update: {last_update}\n")

    all_models = [
        "Poisson_NN", "Bell_NN", "NegBin_NN",
        "Poisson_CANN", "Bell_CANN", "NegBin_CANN",
        "ZIP_NN", "ZINB_NN", "ZIBell_NN",
        "ZIP_CANN", "ZINB_CANN", "ZIBell_CANN",
    ]

    completed = 0
    total = 0

    print(f"  {'Model':<16} {'Progress':<40} {'Best Dev':>10} {'Status':<12}")
    print("  " + "-" * 80)

    for model in all_models:
        if model in progress:
            info = progress[model]
            trial = info["trial"]
            n_trials = info["n_trials"]
            best = info["best_deviance"]
            status = info["status"]
            bar = format_bar(trial, n_trials)
            best_str = f"{best:.6f}" if best is not None else "---"
            total += 1
            if status == "completed":
                completed += 1
            # Color-code status
            status_display = {
                "completed": "DONE",
                "running": "RUNNING",
                "starting": "STARTING",
            }.get(status, status.upper())
            print(f"  {model:<16} {bar}  {trial:>3}/{n_trials:<3} {best_str:>10} {status_display:<12}")
        else:
            print(f"  {model:<16} {'[waiting]':<40} {'---':>10} {'PENDING':<12}")

    print(f"\n  Overall: {completed}/{len(all_models)} models completed")


def show_cv_progress():
    """Display cross-validation progress."""
    print("\n" + "=" * 72)
    print("  CROSS-VALIDATION PROGRESS")
    print("=" * 72)

    if not os.path.exists(CV_PROGRESS_FILE):
        # Check for results file
        cv_results = os.path.join(os.path.dirname(__file__), "cv_results_tuned.csv")
        if os.path.exists(cv_results):
            print("  CV completed! Results:")
            try:
                import pandas as pd
                df = pd.read_csv(cv_results)
                print(df.to_string(index=False, float_format="%.6f"))
            except Exception:
                print("  (error reading results)")
        else:
            print("  CV not started yet.")
        return

    with open(CV_PROGRESS_FILE, "r") as f:
        progress = json.load(f)

    last_update = progress.pop("_last_update", "unknown")
    print(f"  Last update: {last_update}\n")

    all_models = [
        "Poisson_NN", "Bell_NN", "NegBin_NN",
        "Poisson_CANN", "Bell_CANN", "NegBin_CANN",
        "ZIP_NN", "ZINB_NN", "ZIBell_NN",
        "ZIP_CANN", "ZINB_CANN", "ZIBell_CANN",
    ]

    print(f"  {'Model':<16} {'Fold':>6} {'Mean CV Dev':>12} {'Status':<12}")
    print("  " + "-" * 50)

    for model in all_models:
        key = model.replace("_", " ").replace("ZIBell", "ZI-Bell").replace("ZIP", "ZIP-").replace("ZINB", "ZINB-")
        # Also try the exact key
        info = progress.get(model) or progress.get(key)
        if info:
            fold = info.get("fold", "?")
            mean_dev = info.get("mean_dev")
            status = info.get("status", "?")
            mean_str = f"{mean_dev:.6f}" if mean_dev else "---"
            print(f"  {model:<16} {fold:>6} {mean_str:>12} {status:<12}")
        else:
            print(f"  {model:<16} {'---':>6} {'---':>12} {'PENDING':<12}")


def show_existing_results():
    """Show a quick summary of all existing result files."""
    print("\n" + "=" * 72)
    print("  EXISTING RESULT FILES")
    print("=" * 72)

    base = os.path.dirname(__file__)
    files_to_check = [
        ("optuna_results/optuna_summary.csv", "Optuna summary"),
        ("optuna_best_params_all.py", "Best params (all models)"),
        ("optuna_best_params.py", "Best params (Bell/ZI-Bell only)"),
        ("cv_results_tuned.csv", "CV results (tuned)"),
    ]

    for relpath, desc in files_to_check:
        path = os.path.join(base, relpath)
        exists = os.path.exists(path)
        status = "EXISTS" if exists else "MISSING"
        icon = "+" if exists else "-"
        print(f"  [{icon}] {desc:<35} {status}")

    # Check individual trial files
    csvs = glob.glob(os.path.join(RESULTS_DIR, "optuna_*_trials.csv"))
    print(f"\n  Optuna trial CSVs: {len(csvs)}/12")

    # Check saved models
    saved_dir = os.path.join(base, "saved_models_tuned")
    if os.path.exists(saved_dir):
        h5s = glob.glob(os.path.join(saved_dir, "*.h5"))
        print(f"  Saved model weights: {len(h5s)}")

    # Check plots
    plot_dir = os.path.join(base, "..", "plots")
    if os.path.exists(plot_dir):
        pngs = glob.glob(os.path.join(plot_dir, "*.png"))
        print(f"  Generated plots: {len(pngs)}")


def main():
    parser = argparse.ArgumentParser(description="Monitor Optuna/CV progress")
    parser.add_argument("--watch", nargs="?", const=10, type=int,
                        help="Auto-refresh interval in seconds (default: 10)")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                clear_screen()
                print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  (Refreshing every {args.watch}s, Ctrl+C to stop)\n")
                show_optuna_progress()
                show_cv_progress()
                show_existing_results()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n  Stopped.")
    else:
        show_optuna_progress()
        show_cv_progress()
        show_existing_results()


if __name__ == "__main__":
    main()
