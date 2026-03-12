#!/bin/bash
# BanglaHealth-AI — Quick Setup Script
# Usage: bash setup.sh

set -e

echo "=========================================="
echo "  BanglaHealth-AI — Project Setup"
echo "=========================================="

# 1. Create virtual environment
echo ""
echo "[1/4] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "  Virtual environment created."
else
    echo "  Virtual environment already exists."
fi

# 2. Activate and install dependencies
echo ""
echo "[2/4] Installing dependencies..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Dependencies installed."

# 3. Verify datasets exist
echo ""
echo "[3/4] Checking datasets..."
if [ -f "data/raw/diabetes.csv" ] && [ -f "data/raw/heart.csv" ]; then
    echo "  Datasets found."
else
    echo "  ERROR: Datasets not found in data/raw/"
    echo "  Please ensure diabetes.csv and heart.csv are in data/raw/"
    exit 1
fi

# 4. Create necessary directories
echo ""
echo "[4/4] Ensuring directories exist..."
mkdir -p data/processed models outputs
echo "  Directories ready."

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate venv:     source venv/Scripts/activate  (Windows)"
echo "                        source venv/bin/activate      (Linux/Mac)"
echo ""
echo "  2. Run notebooks:     jupyter notebook notebooks/"
echo "     Run them in order: 01_eda -> 02_preprocessing -> 03_model_training"
echo "                        -> 04_evaluation -> 05_explainability -> 06_fairness"
echo ""
echo "  3. Launch dashboard:  streamlit run app/app.py"
echo "  4. Launch API:        uvicorn api.main:app --reload"
echo "  5. Run tests:         pytest tests/ -v"
echo ""
