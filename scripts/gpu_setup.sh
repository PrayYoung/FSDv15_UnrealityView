#!/usr/bin/env bash
set -euo pipefail
if ! command -v uv >/dev/null 2>&1; then
  curl --fail --location --silent --show-error https://astral.sh/uv/install.sh -o /tmp/mini-reality-uv-install.sh
  sh /tmp/mini-reality-uv-install.sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked --extra dev --extra realdata
