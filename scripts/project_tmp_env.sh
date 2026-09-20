#!/usr/bin/env bash
# Source this file before project commands: source scripts/project_tmp_env.sh
_apexgen_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export TMPDIR="${_apexgen_project_root}/artifacts/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="${_apexgen_project_root}/artifacts/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export RUFF_CACHE_DIR="$XDG_CACHE_HOME/ruff"
mkdir -p -- "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" \
    "$PYTHONPYCACHEPREFIX" "$NUMBA_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
    "$TRITON_CACHE_DIR" "$RUFF_CACHE_DIR"
unset _apexgen_project_root
