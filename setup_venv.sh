#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found in PATH."
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
else
  echo "Virtual environment already exists at: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "Activated virtual environment: $VIRTUAL_ENV"

python -m pip install --upgrade pip

if [[ -f "$REQUIREMENTS_FILE" ]]; then
  echo "Installing dependencies from: $REQUIREMENTS_FILE"
  python -m pip install -r "$REQUIREMENTS_FILE"
else
  echo "Warning: requirements.txt not found at $REQUIREMENTS_FILE"
fi

echo "Done."
echo "If you ran this script directly, activate later with: source .venv/bin/activate"