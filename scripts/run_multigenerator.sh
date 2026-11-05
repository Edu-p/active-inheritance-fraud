#!/usr/bin/env bash
# T2.1 multi-generator: generate zero-shot candidates with OTHER LLMs, then replay
# score-k vs random-k. Kills the "monocultura de LLM" critique.
#
# API-GATED: spends OpenRouter credits. Do NOT run without the user's confirmation.
# Uses OPEN_ROUTER_API_KEY from .env and https://openrouter.ai/api/v1.
# Default models = one RECENT model per provider family (cheap end, under $15 total).
#
# Usage:
#   SCENARIOS="covid19br-10pct factckbr-10pct" SEEDS="42 43 44" \
#     bash scripts/run_multigenerator.sh
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."

# 2026 models (one per family) that generate CLEAN PT-BR misinformation in smoke
# tests. Note: meta-llama has NO 2026 model in OpenRouter, so llama-4-maverick
# (Apr 2025) is the newest Llama. gemini-3.x-flash (2026) truncates; gpt-6-luna
# (2026) prepends 'SIMULAÇÃO FICTÍCIA' disclaimers — both excluded.
MODELS="${MODELS:-openai/gpt-6.1-terra meta-llama/llama-4-maverick mistralai/mistral-large-4-0 google/gemma-4-31b-it anthropic/claude-haiku-5.5}"
BASE_URL="${BASE_URL:-https://openrouter.ai/api/v1}"
KEY_ENV="${KEY_ENV:-OPEN_ROUTER_API_KEY}"
SCENARIOS="${SCENARIOS:-covid19br-10pct factckbr-10pct}"
SEEDS="${SEEDS:-42 43 44}"
# PHASES="gen" generates candidates only (API, no GPU); "gen replay" also replays (GPU).
PHASES="${PHASES:-gen replay}"

if [ -z "${!KEY_ENV:-}" ] && ! grep -qE "^${KEY_ENV}=" .env 2>/dev/null; then
  echo "MISSING \$$KEY_ENV — set the provider key first."; exit 1
fi

prompt_for() {
  case "$1" in
    fakewhatsapp-*|covid19br-*) echo v4 ;;
    *) echo v2 ;;
  esac
}

gen() {
  local sc=$1 seed=$2 model=$3 prompt=$4
  # skip if a complete candidate set for this (scenario, seed, model) already exists
  if python3 -c "
import json, glob, sys
hits = [p for p in glob.glob('outputs/revision-candidates/${sc}-s${seed}-*/candidates.json')
        if json.load(open(p)).get('complete') and json.load(open(p)).get('model') == '${model}']
sys.exit(0 if hits else 1)"; then
    echo "SKIP gen $sc s$seed $model"; return 0
  fi
  uv run python scripts/controlled_curatorship.py --generate \
    --scenario "$sc" --seed "$seed" --model "$model" --base-url "$BASE_URL" \
    --prompt "$prompt" --cycles 5 --batch-size 64 --max-run-tokens 100000
}

find_cand() {
  local sc=$1 seed=$2 model=$3
  python3 -c "
import json, glob
hits = [p for p in glob.glob('outputs/revision-candidates/${sc}-s${seed}-*/candidates.json')
        if json.load(open(p)).get('complete') and json.load(open(p)).get('zero_shot')
        and json.load(open(p)).get('model') == '${model}']
print(hits[-1] if hits else '')
"
}

for model in $MODELS; do
  san=$(echo "$model" | tr '/.' '--')
  for sc in $SCENARIOS; do
    prompt=$(prompt_for "$sc")
    grp="multigenerator-${sc}-${san}"
    for seed in $SEEDS; do
      case "$PHASES" in *gen*) gen "$sc" "$seed" "$model" "$prompt" ;; *) ;; esac
      cand=$(find_cand "$sc" "$seed" "$model")
      [ -z "$cand" ] && { echo "FAILED gen $sc s$seed $model"; continue; }
      case "$PHASES" in *replay*) ;; *) continue ;; esac
      for arm in score-k random-k; do
        if compgen -G "outputs/${grp}/${arm}-s${seed}-*/results.json" >/dev/null; then
          echo "SKIP $sc s$seed $arm $model"; continue
        fi
        uv run python scripts/controlled_curatorship.py --replay \
          --candidates "$cand" --arm "$arm" --top-k 16 \
          --run-group "$grp" --learner-model neuralmind/bert-base-portuguese-cased
      done
    done
  done
done
echo "=== MULTIGENERATOR DONE ==="
