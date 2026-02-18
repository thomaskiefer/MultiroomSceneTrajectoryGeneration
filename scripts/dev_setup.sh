#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_CHECKS=0

for arg in "$@"; do
  case "$arg" in
    --check)
      RUN_CHECKS=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: scripts/dev_setup.sh [--check]" >&2
      exit 2
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install from https://docs.astral.sh/uv/" >&2
  exit 1
fi

cd "$ROOT_DIR"
uv sync --extra full --group dev

echo "Tooling available:"
uv run ruff --version
uv run mypy --version

if [[ "$RUN_CHECKS" -eq 1 ]]; then
  echo "Running lint/type/test checks..."
  uv run ruff check src tests
  uv run mypy src
  PYTHONPATH=src uv run python -m unittest discover -s tests -p 'test_*.py' -v
fi

echo "Developer setup complete."
