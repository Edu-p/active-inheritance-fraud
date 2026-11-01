#!/usr/bin/env bash
# T2.2 multi-learner: replay score-k vs random-k with other downstream learners.
# Reuses the EXISTING warm-start replay (controlled_curatorship.py --replay) so the
# protocol is identical to the main study; only --learner-model changes.
# Zero API. Resumable (skips existing results.json). Overnight-safe.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."

# bump SEEDS to "42 43 44 45 46 47 48 49 50 51" for the paper's n=10
SCENARIOS="covid19br-10pct factckbr-10pct fakewhatsapp-10pct"
SEEDS="42 43 44 45 46"
# kind:model pairs — BERT-family + a Jev-style LLM (non-BERT decision models)
LEARNERS=(
  "bert:bert-base-multilingual-cased"
  "bert:neuralmind/bert-small-portuguese-cased"
  "bert:xlm-roberta-base"
  "lora-llm:Qwen/Qwen2.5-0.5B-Instruct"
)
ARMS="score-k random-k"

find_cand() {
  # canonical candidates for scenario+seed (prompt-aware: v4 whatsapp / v2 news)
  local sc=$1 seed=$2 prompt=$3
  python3 -c "
import json, glob, sys
hits = [p for p in glob.glob('outputs/revision-candidates/${sc}-s${seed}-*/candidates.json')
        if json.load(open(p)).get('complete') and json.load(open(p)).get('zero_shot')
        and json.load(open(p)).get('prompt') == '${prompt}']
print(hits[-1] if hits else '')
"
}

prompt_for() {
  case "$1" in
    fakewhatsapp-*|covid19br-*) echo v4 ;;
    *) echo v2 ;;
  esac
}

for sc in $SCENARIOS; do
  prompt=$(prompt_for "$sc")
  for pair in "${LEARNERS[@]}"; do
    kind="${pair%%:*}"; model="${pair#*:}"
    san=$(echo "$model" | tr '/.' '--')
    grp="multilearner-${sc}-${kind}-${san}"
    for seed in $SEEDS; do
      cand=$(find_cand "$sc" "$seed" "$prompt")
      [ -z "$cand" ] && { echo "MISSING candidates $sc s$seed"; continue; }
      for arm in $ARMS; do
        if compgen -G "outputs/${grp}/${arm}-s${seed}-*/results.json" >/dev/null; then
          echo "SKIP $sc s$seed $arm $kind/$model"; continue
        fi
        uv run python scripts/controlled_curatorship.py --replay \
          --candidates "$cand" --arm "$arm" --top-k 16 \
          --run-group "$grp" --learner-kind "$kind" --learner-model "$model"
      done
    done
  done
done
echo "=== MULTILEARNER DONE ==="
