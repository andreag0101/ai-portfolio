#!/usr/bin/env bash
# Starts the backend (FastAPI, :8000) and frontend (Vite, :5173) together
# for local development. Ctrl+C stops both.
set -euo pipefail

# macOS reports itself to Python as version "10.16" unless this is set,
# which makes pip reject modern prebuilt wheels (numpy, opencv, scipy, ...)
# and fall back to a from-source build instead -- see README troubleshooting.
export SYSTEM_VERSION_COMPAT=0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

echo "==> Backend setup"
if [ ! -d "$BACKEND_DIR/.venv" ]; then
  echo "Creating backend/.venv"
  python3 -m venv "$BACKEND_DIR/.venv"
fi
# shellcheck disable=SC1091
source "$BACKEND_DIR/.venv/bin/activate"
pip install -q --upgrade pip
pip install -q -r "$BACKEND_DIR/requirements.txt"
deactivate

echo "==> Frontend setup"
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "Running npm install"
  (cd "$FRONTEND_DIR" && npm install)
fi

PIDS=()
cleanup() {
  echo
  echo "==> Stopping..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Starting backend on http://localhost:8000"
(
  source "$BACKEND_DIR/.venv/bin/activate"
  cd "$BACKEND_DIR"
  exec uvicorn app.main:app --reload --port 8000
) &
PIDS+=($!)

echo "==> Starting frontend on http://localhost:5173"
(
  cd "$FRONTEND_DIR"
  exec npm run dev
) &
PIDS+=($!)

wait
