#!/usr/bin/env bash
# Chain: Phase C (2 new datasets) -> plots/analysis -> new LaTeX + PDF.
# Resumable and safe: controlled_curatorship.py never overwrites outputs.
set -u
cd "$(dirname "$0")/.." || exit 1

TS=$(date +%Y%m%dT%H%M%SZ)
LOG="outputs/revision-logs/new-datasets-full-$TS.log"
exec >"$LOG" 2>&1

echo "START full new-datasets pipeline $(date -u +%Y-%m-%dT%H:%M:%SZ)"
uv run python scripts/run_new_datasets_night.py
RC=$?
echo "Phase C exit=$RC"

if [ "$RC" -eq 0 ]; then
  uv run python scripts/finalize_new_datasets.py
  echo "finalize exit=$?"
fi

uv run python scripts/revision_status.py
echo "FINISHED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
