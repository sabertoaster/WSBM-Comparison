#!/bin/sh
# Small executable validation; extra arguments override the default task/protocol.
# Example: ./run_test.sh --tasks Swimmer-v4 --output-dir /tmp/wsbm-validation
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$TASK_ROOT/.venv/bin/python" "$TASK_ROOT/scripts/train_rl.py" --protocol legacy --env-id Swimmer-v4 --smoke "$@"
