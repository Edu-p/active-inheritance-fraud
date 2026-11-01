#!/usr/bin/env bash
# Orchestrator: runs every ablation/generalization tier in dependency order.
# Resumable (every sub-script skips existing results.json). Zero-API except
# multigenerator/relabel, which are separate and API-gated.
#
# Phases (pass one or more names as args; default = all zero-API phases):
#   styleblind   N2  : SemanticProbe scorer (content-only) vs random-k
#   weights      T1.2: (w_e,w_u,w_d) grid vs random-k, frozen scorer
#   multilearner T2.2: mBERT / BERTimbau-small / XLM-R / lora-llm downstream learners
#   competence   T1.1: n=10 competence sweep (5 fractions, 3 domains)
#   band         T1.1: n=10 band-k (3 domains)
#   real         A1  : real-only baseline, n=10, all 11 scenarios
#   shortcut     A13 : score-det / score-frozen / realism-k, n=10
#
# Usage:
#   bash scripts/run_neurotier.sh                       # all zero-API
#   bash scripts/run_neurotier.sh styleblind weights    # just these
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."

SEEDS10="42,43,44,45,46,47,48,49,50,51"
PHASES="${*:-styleblind weights multilearner competence band}"

for ph in $PHASES; do
  case "$ph" in
    styleblind)
      echo "=== [styleblind] N2 ==="
      uv run python scripts/ablation_replay.py --mode styleblind \
        --scenarios covid19br-10pct,factckbr-10pct,fakewhatsapp-10pct --seeds "$SEEDS10"
      ;;
    weights)
      echo "=== [weights] T1.2 ==="
      uv run python scripts/ablation_replay.py --mode weights \
        --scenarios covid19br-10pct --seeds "$SEEDS10"
      ;;
    multilearner)
      echo "=== [multilearner] T2.2 ==="
      bash scripts/run_multilearner.sh
      ;;
    competence)
      echo "=== [competence n=10] T1.1 ==="
      uv run python scripts/explore_intermediate.py \
        --scenarios covid19br-full,factckbr-full,fakewhatsapp-full --seeds "$SEEDS10" \
        --fractions 0.05,0.10,0.20,0.50,1.00 --arms score-k,random-k --group explore-competence
      ;;
    band)
      echo "=== [band n=10] T1.1 ==="
      uv run python scripts/explore_intermediate.py \
        --scenarios covid19br-full,factckbr-full,fakewhatsapp-full --seeds "$SEEDS10" \
        --fractions 1.00 --arms band-k --group explore-band
      ;;
    real)
      echo "=== [real n=10] A1 ==="
      uv run python scripts/real_baseline.py --seeds "$SEEDS10"
      ;;
    shortcut)
      echo "=== [shortcut factorial] A13 ==="
      uv run python scripts/shortcut_factorial.py --seeds "$SEEDS10"
      ;;
    *)
      echo "Unknown phase: $ph (styleblind|weights|multilearner|competence|band|real|shortcut)"
      ;;
  esac
done
echo "=== NEUROTIER DONE ==="
