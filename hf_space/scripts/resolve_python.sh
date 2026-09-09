#!/usr/bin/env bash

resolve_env_python() {
  local root=${1:-${ROOT:-}}
  local candidate

  if [[ -n "${ENV_PYTHON:-}" ]]; then
    printf '%s\n' "$ENV_PYTHON"
    return 0
  fi

  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s\n' "$PYTHON"
    return 0
  fi

  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    printf '%s\n' "$CONDA_PREFIX/bin/python"
    return 0
  fi

  if [[ -n "$root" ]]; then
    for candidate in \
      "$root/envs/ucfnet-pip/bin/python" \
      "$root/envs/ucfnet/bin/python" \
      "$root/envs/ucfnet-5090-cu128/bin/python"; do
      if [[ -x "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
    done
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  command -v python
}

require_env_python() {
  local python_bin=$1
  if [[ "$python_bin" == */* ]]; then
    if [[ -x "$python_bin" ]]; then
      return 0
    fi
  elif command -v "$python_bin" >/dev/null 2>&1; then
    return 0
  fi

  cat >&2 <<EOF
Missing env python: $python_bin

Activate the environment before running the project entry point:
  conda activate <your-environment>
  python train.py

Or pass it explicitly to the underlying script:
  ENV_PYTHON=\$(command -v python) bash scripts/train_ucfnet.sh

If ENV_PYTHON points at a missing repo-local env, clear it first:
  unset ENV_PYTHON
  python train.py
EOF
  return 1
}
