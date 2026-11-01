#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${RSNA_DATA_ROOT:-/home/shaurya/rsna-data}"
training_env="${RSNA_TRAIN_ENV:-/home/shaurya/venvs/rsna-train}"
python_bin="${RSNA_PYTHON_BIN:-python3.12}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "ERROR: ${python_bin} is unavailable; install Python 3.12 first." >&2
  exit 2
fi

"${python_bin}" -m venv "${training_env}"
"${training_env}/bin/python" -m pip install --upgrade pip
"${training_env}/bin/python" -m pip install \
  torch==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
"${training_env}/bin/python" -m pip install \
  -r "${project_root}/requirements-training.txt"
"${training_env}/bin/python" -m ipykernel install \
  --user \
  --name rsna-train \
  --display-name "RSNA (Training PyTorch CUDA)"

RSNA_DATA_ROOT="${data_root}" \
RSNA_PROJECT_ROOT="${project_root}" \
  "${training_env}/bin/python" \
  "${project_root}/scripts/training_preflight.py" \
  --device cuda \
  --prepare-directories
