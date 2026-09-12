#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec /usr/bin/python3 "$ROOT/manuscript/supplement/scripts/audit_s3_active_paths.py"
