#!/usr/bin/env bash
# Standalone: copy this file and audit_rec_data_readonly.py to the same directory.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:-/mnt/sdb/dengjiale/dppbench/data}"
OUTPUT_DIR="${2:-${PWD}/rec_data_audit_reports/$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" "$SCRIPT_DIR/audit_rec_data_readonly.py" \
  --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR" "${@:3}"
