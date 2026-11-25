#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cuda_env.sh"
probe_directory=$(mktemp -d)
trap 'rm -rf "$probe_directory"' EXIT
cat > "$probe_directory/probe.cu" <<'CU'
#include <cuda_runtime.h>
__global__ void probe(float* x) { if (threadIdx.x == 0) x[0] = 1.0f; }
CU
"$CUDACXX" -c "$probe_directory/probe.cu" -o "$probe_directory/probe.o"
test -s "$probe_directory/probe.o"
echo 'CUDA_COMPILER_OBJECT_OK'
