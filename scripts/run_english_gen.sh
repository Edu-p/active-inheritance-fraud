#!/usr/bin/env bash
# T3.1 English: generate zero-shot candidates for LIAR + FakeNewsNet (deepseek,
# cheap, not OpenRouter credit). Replay is separate (GPU) — see run_neurotier.sh
# or controlled_curatorship --replay.
#
# Resumable: skips a (scenario, seed) if a complete deepseek candidate set exists.
# Usage:
#   SEEDS="42 43 44 45 46" bash scripts/run_english_gen.sh
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."

SCENARIOS="${SCENARIOS:-liar-full liar-10pct liar-5pct fakenewsnet-full fakenewsnet-10pct fakenewsnet-5pct}"
SEEDS="${SEEDS:-42 43 44 45 46}"

gen() {
  local sc=$1 seed=$2
  if python3 -c "
import json, glob, sys
hits = [p for p in glob.glob('outputs/revision-candidates/${sc}-s${seed}-*/candidates.json')
        if json.load(open(p)).get('complete') and json.load(open(p)).get('model') == 'deepseek-chat']
sys.exit(0 if hits else 1)"; then
    echo "SKIP gen $sc s$seed"; return 0
  fi
  uv run python scripts/controlled_curatorship.py --generate \
    --scenario "$sc" --seed "$seed" --model deepseek-chat \
    --cycles 5 --batch-size 64 --max-run-tokens 100000
}

for sc in $SCENARIOS; do
  for seed in $SEEDS; do
    gen "$sc" "$seed"
  done
done
echo "=== ENGLISH GEN DONE ==="
