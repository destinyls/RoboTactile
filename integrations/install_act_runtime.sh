#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/install_pinned_repo.sh" \
  act_runtime \
  https://github.com/WorldArena2/WorldArena-2.0 \
  e295378c702b2e87617ebddcad193be3608e00c3 \
  Apache-2.0 \
  "$@"
