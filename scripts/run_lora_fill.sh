#!/usr/bin/env bash
# Wait for the main queue to finish, then fill the missing lora-llm runs.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "run_all_remaining.sh" >/dev/null; do sleep 120; done
echo "=== main queue done, filling lora-llm/multilearner ==="
bash scripts/run_multilearner.sh
echo "=== LORA FILL DONE ==="
