#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec python3 manuscript/supplement/scripts/audit_s4_foundation.py "$@"
