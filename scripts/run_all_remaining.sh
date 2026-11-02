#!/usr/bin/env bash
# Sequential GPU queue (resumable): A13 shortcut -> multilearner -> multigen replay
# -> english replay -> competence+band. Launch with nohup.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."

echo "=== [1/5] shortcut A13 ==="
uv run python scripts/shortcut_factorial.py --seeds 42,43,44,45,46,47,48,49,50,51

echo "=== [2/5] multilearner ==="
bash scripts/run_neurotier.sh multilearner

echo "=== [3/5] multigen replay ==="
PHASES=replay SEEDS="42 43 44 45 46" bash scripts/run_multigenerator.sh

echo "=== [4/5] english replay ==="
bash scripts/run_english_replay.sh

echo "=== [5/5] competence + band n=10 ==="
bash scripts/run_neurotier.sh competence band

echo "=== ALL REMAINING DONE ==="
