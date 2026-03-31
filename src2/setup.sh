#!/bin/bash
# =============================================================
# Bell-CANN src2 — Environment Setup
# Run from: src2/
# =============================================================

set -e

echo "=== Creating Python virtual environment ==="
python -m venv .venv

echo "=== Activating venv ==="
source .venv/Scripts/activate  # Windows (Git Bash / MSYS2)
# source .venv/bin/activate    # Linux / macOS

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo ""
echo "=== Done! ==="
echo "To activate later:  source src2/.venv/Scripts/activate"
echo "Next steps:"
echo "  1. Place freMTPL2freq.csv in src2/R/"
echo "  2. cd src2/R && Rscript 01_preprocess_and_glm.R"
echo "  3. cd src2/python && python run_comparison.py"
echo "  4. cd src2/python && python run_cv.py"
