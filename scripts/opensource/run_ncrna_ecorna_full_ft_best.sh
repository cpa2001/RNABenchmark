#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export FREEZE_BACKBONE=0
export ECORNA_POOLING_STRATEGY="${ECORNA_POOLING_STRATEGY:-cls_tanh}"

bash scripts/opensource/run_ncrna.sh ecorna "$@"
