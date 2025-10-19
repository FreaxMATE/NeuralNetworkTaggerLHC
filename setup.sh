#!/usr/bin/env bash
set -euo pipefail

# Setup script for Linux/macOS. Creates a venv and installs requirements.
# Usage: bash setup.sh

PY=${PY:-python3}
VENV_DIR=${VENV_DIR:-.venv}

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python not found (PY=$PY). Please install Python 3.10+ and retry." >&2
  exit 1
fi

echo "Creating virtual environment in $VENV_DIR ..."
"$PY" -m venv "$VENV_DIR"

# Detect shell for correct activation hint
SHELL_NAME=$(basename "${SHELL:-bash}")
case "$SHELL_NAME" in
  fish)
    ACTIVATE_CMD="source $VENV_DIR/bin/activate.fish";;
  zsh|bash|sh|ksh)
    ACTIVATE_CMD="source $VENV_DIR/bin/activate";;
  *)
    ACTIVATE_CMD="source $VENV_DIR/bin/activate";;
 esac

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate" || {
  echo "Failed to activate venv automatically. After this script, run: $ACTIVATE_CMD" >&2
}

pip install --upgrade pip

if [ -f requirements.txt ]; then
  echo "Installing Python dependencies from requirements.txt ..."
  pip install -r requirements.txt
  # Optional: force CPU-only PyTorch wheel on Linux to avoid large CUDA deps
  if [ "${TORCH_CPU:-0}" = "1" ]; then
    echo "Installing CPU-only PyTorch wheel (TORCH_CPU=1) ..."
    pip install --no-deps "torch==2.3.1+cpu" -f https://download.pytorch.org/whl/torch_stable.html || true
  fi
else
  echo "requirements.txt not found; installing minimal deps ..."
  pip install numpy matplotlib scikit-learn torch --extra-index-url https://download.pytorch.org/whl/cpu || pip install numpy matplotlib scikit-learn torch
fi

echo
echo "Done. Activate your environment with:"
echo "  $ACTIVATE_CMD"
echo "Then run:"
echo "  python data-exercise-template-v7.py"