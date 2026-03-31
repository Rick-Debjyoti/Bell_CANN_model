#!/bin/bash
# Run the full Bell-CANN pipeline with GPU support.
#
# Usage:
#   cd src2/python
#   bash run_all.sh
#
# Prerequisites:
#   conda activate "D:/Projects/FINAL REPOS/Bell_CANN_model/src2/.conda_env"
#   export PYTHONNOUSERSITE=1

set -e

echo "============================================"
echo "Bell-CANN Full Pipeline"
echo "============================================"

# Ensure conda env is active
if [[ -z "$CONDA_PREFIX" ]]; then
    echo "ERROR: Activate conda env first:"
    echo '  eval "$(conda shell.bash hook)"'
    echo '  conda activate "D:/Projects/FINAL REPOS/Bell_CANN_model/src2/.conda_env"'
    echo '  export PYTHONNOUSERSITE=1'
    exit 1
fi

export PYTHONNOUSERSITE=1

echo ""
echo "[Step 1/4] Running all 18 models..."
python run_comparison.py

echo ""
echo "[Step 2/4] Optuna hyperparameter tuning (all 12 models)..."
python run_optuna_all.py

echo ""
echo "[Step 3/4] Generating plots..."
python generate_plots.py

echo ""
echo "[Step 4/4] Cross-validation (5-fold)..."
python run_cv.py

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "  Results: comparison_results.csv"
echo "  Optuna:  optuna_results/"
echo "  Plots:   ../plots/"
echo "  CV:      cv_results.csv"
echo "============================================"
