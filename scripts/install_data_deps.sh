#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -d "venv" ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

echo "[install_data_deps] Installing data download dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements_data.txt
echo "[install_data_deps] Done."
