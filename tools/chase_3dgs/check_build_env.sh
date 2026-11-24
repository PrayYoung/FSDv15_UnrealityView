#!/usr/bin/env bash
set -euo pipefail
command -v nvcc >/dev/null || { echo 'Dynamic 3DGS requires a CUDA 12 development image with nvcc' >&2; exit 2; }
nvcc --version | rg -q 'release 12\.' 2>/dev/null || nvcc --version | grep -q 'release 12\.' || { echo 'CUDA 12 toolkit required' >&2; exit 2; }
