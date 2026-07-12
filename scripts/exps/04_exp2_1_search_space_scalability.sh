#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cat >&2 <<'EOF'
[unsupported] exp2.1 requires running each baseline under multiple operator-space
sizes. The current unified entry point, scripts/evaluate_baseline.py, does not
expose an operator allowlist/subset argument, and baselines/common/op_registry.py
loads the full operator set directly.

Add a shared operator-space CLI/config hook first, then wire its levels here.
EOF

exit 1
