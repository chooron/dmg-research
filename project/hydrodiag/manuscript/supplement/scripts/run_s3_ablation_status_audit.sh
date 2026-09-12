#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec python3 "$PROJECT_DIR/manuscript/supplement/scripts/audit_s3_remote_ablation_status.py" "$@"
