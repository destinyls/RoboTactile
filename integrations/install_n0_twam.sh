#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/install_pinned_repo.sh" \
  n0_twam \
  https://github.com/destinyls/N0-TWAM.git \
  9036c130409f8cf5494b12489fea339f7213b9d6 \
  CC-BY-NC-SA-4.0 \
  "$@"
