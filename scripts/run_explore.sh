#!/usr/bin/env bash
# Overnight (zero API): varredura de competência + seleção por faixa intermediária.
# Reusa candidates.json já gerados; só treina BERT na GPU. Resumível (pula done).
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."

SCENARIOS="covid19br-full,factckbr-full,fakewhatsapp-full"

echo "=== [1/2] competence sweep (score-k vs random-k, seed 42, 5 frações) ==="
uv run python scripts/explore_intermediate.py \
  --scenarios "$SCENARIOS" --seeds 42 \
  --fractions 0.05,0.10,0.20,0.50,1.00 \
  --arms score-k,random-k --group explore-competence

echo "=== [2/2] intermediate band (band-k, seeds 42-44, fração 1.00) ==="
uv run python scripts/explore_intermediate.py \
  --scenarios "$SCENARIOS" --seeds 42,43,44 \
  --fractions 1.00 --arms band-k --group explore-band

echo "=== EXPLORE DONE ==="
