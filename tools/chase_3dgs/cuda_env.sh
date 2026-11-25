#!/usr/bin/env bash
# Source this file in every shell that invokes a native CUDA build. Login shells
# may reset PATH; exports from a separately executed check cannot repair a parent.
unreality_cuda_env() {
  local compiler root version existing
  local candidates=()
  existing=$(command -v nvcc 2>/dev/null || true)
  [ -z "$existing" ] || candidates+=("$existing")
  for root in "${CUDA_HOME:-}" "${CUDA_PATH:-}" /usr/local/cuda /usr/local/cuda-12.* /opt/cuda /opt/cuda-12.*; do
    [ -z "$root" ] || candidates+=("$root/bin/nvcc")
  done
  for compiler in "${candidates[@]}"; do
    [ -x "$compiler" ] || continue
    version=$("$compiler" --version 2>/dev/null) || continue
    printf '%s\n' "$version" | grep -Eq 'release 12\.' || continue
    root=$(python3 -c 'import pathlib,sys;print(pathlib.Path(sys.argv[1]).resolve().parent.parent)' "$compiler") || continue
    [ -f "$root/include/cuda_runtime.h" ] || continue
    export CUDA_HOME="$root" CUDA_PATH="$root" CUDACXX="$root/bin/nvcc"
    export PATH="$root/bin:$PATH"
    export LD_LIBRARY_PATH="$root/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    printf 'CUDA12_TOOLKIT_DISCOVERED\n'
    return 0
  done
  echo 'No usable CUDA 12 toolkit found: nvcc and cuda_runtime.h are required; CUDA-enabled PyTorch alone is insufficient' >&2
  return 2
}
unreality_cuda_env
