#!/usr/bin/env sh
set -eu

# setup_nllb_env.sh
# Cross-platform (macOS / Linux / WSL) bootstrap for the NLLB translation scripts.
#
# What it does:
# - Creates/uses a local venv (.venv)
# - Installs Python deps: transformers, sentencepiece, accelerate
# - Installs PyTorch:
#     - On macOS: uses pip default (MPS on Apple Silicon when available)
#     - On Linux/WSL: installs CUDA build if NVIDIA GPU is detected, otherwise CPU build
#
# Usage:
#   chmod +x setup_nllb_env.sh
#   ./setup_nllb_env.sh
#
# Optional env vars:
#   PYTHON=python3.11   # choose python binary (default: python3)
#   VENV_DIR=.venv      # venv folder (default: .venv)
#   TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu128  # choose CUDA index (default cu128)

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

command -v "$PYTHON" >/dev/null 2>&1 || die "Python not found: $PYTHON (set PYTHON=python3.11 for example)"

OS="$(uname -s)"
ARCH="$(uname -m)"

say "OS=$OS ARCH=$ARCH PYTHON=$PYTHON VENV_DIR=$VENV_DIR"

# Create venv if missing
if [ ! -d "$VENV_DIR" ]; then
  say "Creating venv: $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi

# Activate venv
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

say "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

# Detect NVIDIA GPU (Linux/WSL)
has_nvidia="0"
if [ "$OS" = "Linux" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
      has_nvidia="1"
    fi
  fi
fi

install_torch() {
  if [ "$OS" = "Darwin" ]; then
    # macOS: pip default. On Apple Silicon, torch can use MPS backend (not CUDA).
    say "Installing PyTorch for macOS (pip default)..."
    python -m pip install --upgrade torch torchvision torchaudio
  else
    if [ "$has_nvidia" = "1" ]; then
      say "NVIDIA detected. Installing PyTorch CUDA from: $TORCH_CUDA_INDEX"
      python -m pip install --upgrade torch torchvision torchaudio --index-url "$TORCH_CUDA_INDEX"
    else
      say "No NVIDIA detected. Installing PyTorch CPU (pip default)..."
      python -m pip install --upgrade torch torchvision torchaudio
    fi
  fi
}

say "Installing PyTorch..."
install_torch

say "Installing NLLB dependencies..."
python -m pip install --upgrade transformers sentencepiece accelerate

# Optional: helpful libs (safe to have)
python -m pip install --upgrade tqdm

say "Sanity checks:"
python - << 'PY'
import sys
import torch
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_version:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
# macOS MPS
mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print("mps_available:", mps)
PY

say ""
say "Done."
say "Activate the venv with: . $VENV_DIR/bin/activate"
say "Run your scripts with: python <script>.py ..."
