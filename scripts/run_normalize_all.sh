#!/usr/bin/env bash
# Normalização completa (autorizada): Fake.br no mesmo protocolo + 10 seeds.
# Resumível: pula geração/replay cujos artifacts já existem. Fases em ordem.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."

gen() {
  local sc=$1 seed=$2
  if python3 -c "
import json,glob,sys
hits=[p for p in glob.glob('outputs/revision-candidates/${sc}-s${seed}-*/candidates.json')]
sys.exit(0 if hits and json.load(open(hits[-1])).get('complete') else 1)"; then
    echo "SKIP gen $sc s$seed"; return 0
  fi
  uv run python scripts/controlled_curatorship.py --generate --scenario "$sc" --seed "$seed" \
    --prompt v2 --cycles 5 --batch-size 64 --max-run-tokens 100000
}

replay() {
  local sc=$1 seed=$2 arm=$3 group=$4
  if compgen -G "outputs/${group}/${arm}-s${seed}-*/results.json" >/dev/null; then
    echo "SKIP replay $sc s$seed $arm"; return 0
  fi
  local cand
  cand=$(python3 -c "
import json,glob
hits=[p for p in glob.glob('outputs/revision-candidates/${sc}-s${seed}-*/candidates.json') if json.load(open(p)).get('complete')]
print(hits[-1] if hits else '')")
  [ -z "$cand" ] && { echo "MISSING candidates $sc s$seed"; return 1; }
  uv run python scripts/controlled_curatorship.py --replay --candidates "$cand" --arm "$arm" \
    --top-k 16 --run-group "$group"
}

echo "=== Phase 1: Fake.br candidates (API) ==="
for sc in fakebr-10pct fakebr-5pct; do for seed in 42 43 44; do gen "$sc" "$seed"; done; done

echo "=== Phase 1: Fake.br replay 4 arms ==="
for sc in fakebr-10pct fakebr-5pct; do
  grp="revision-controlled-${sc}-v2"
  for seed in 42 43 44; do for arm in all random-k score-k easy-k; do replay "$sc" "$seed" "$arm" "$grp"; done; done
done

echo "=== Phase 1: Fake.br competência + band-k ==="
uv run python scripts/explore_intermediate.py --scenarios fakebr-10pct,fakebr-5pct \
  --seeds 42 --fractions 0.05,0.10,0.20,0.50,1.00 --arms score-k,random-k --group explore-competence
uv run python scripts/explore_intermediate.py --scenarios fakebr-10pct,fakebr-5pct \
  --seeds 42,43,44 --fractions 1.00 --arms band-k --group explore-band

echo "=== Phase 2: 10-seed candidates, seeds 45-51 (API) ==="
for sc in fakewhatsapp-full fakewhatsapp-10pct fakewhatsapp-5pct \
          covid19br-full covid19br-10pct covid19br-5pct \
          factckbr-full factckbr-10pct factckbr-5pct \
          fakebr-10pct fakebr-5pct; do
  for seed in 45 46 47 48 49 50 51; do gen "$sc" "$seed"; done
done

echo "=== Phase 3: 10-seed replay (dias; resumível) ==="
for sc in fakewhatsapp-full fakewhatsapp-10pct fakewhatsapp-5pct \
          covid19br-full covid19br-10pct covid19br-5pct \
          factckbr-full factckbr-10pct factckbr-5pct \
          fakebr-10pct fakebr-5pct; do
  case "$sc" in factckbr*|fakebr*) p=v2;; *) p=v4;; esac
  grp="revision-controlled-${sc}-${p}"
  for seed in 42 43 44 45 46 47 48 49 50 51; do
    for arm in all random-k score-k easy-k; do replay "$sc" "$seed" "$arm" "$grp"; done
  done
done

echo "=== NORMALIZE ALL DONE ==="
