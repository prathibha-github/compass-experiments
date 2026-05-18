#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

python3 -m venv "$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/.venv/bin/activate"
pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements.txt" -q

# Let the venv find the helm_eval_lex_quirks package
PY_VER=$(.venv/bin/python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
echo "$SCRIPT_DIR" > "$SCRIPT_DIR/.venv/lib/$PY_VER/site-packages/helm_eval_project.pth"

echo ""
echo "Setup complete.  To run:"
echo "  $SCRIPT_DIR/.venv/bin/python -m helm_eval_lex_quirks.main --api-key YOUR_KEY --dry-run"
