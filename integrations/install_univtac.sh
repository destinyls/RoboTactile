#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/install_pinned_repo.sh" \
  univtac \
  https://github.com/univtac/UniVTAC \
  05bcd3edb92237107efa40105292a24f1a9fd761 \
  Apache-2.0 \
  "$@"
