#!/usr/bin/env bash
# CPU regression checks for scratch AMP and playback inputs; no controller required.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/local_runtime.sh"
for test in test_amp_data.py test_amp_task.py test_amp_prior.py test_amp_replay.py test_amp_eval.py test_amp_gamepad.py test_amp_gamepad_playback.py; do
  "$SMIMIC_PY" "code/scripts/dev/$test"
done
