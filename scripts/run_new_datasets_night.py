"""Overnight orchestrator for the 2 new datasets (COVID19.BR + FACTCK.BR).

Phase C only (controlled study): one zero-shot generation per scenario/seed,
then 4 offline replays (all / random-k / score-k / easy-k) on the same
candidates. Phase B is skipped on purpose: new datasets have no historical
synthetic pools to replay. Resumable: skips any generation whose complete
candidates.json already exists and any arm whose source+arm already produced
a results.json. Logs to outputs/revision-logs/.

Usage: nohup uv run python scripts/run_new_datasets_night.py > /dev/null 2>&1 &
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

UV = ["uv", "run", "python"]

# scenario -> prompt (news/genre already fixed inside controlled_curatorship)
SCENARIOS = {
    "covid19br-full": "v4",
    "covid19br-10pct": "v4",
    "covid19br-5pct": "v4",
    "factckbr-full": "v2",
    "factckbr-10pct": "v2",
    "factckbr-5pct": "v2",
}
SEEDS = (42, 43, 44)
ARMS = ("all", "random-k", "score-k", "easy-k")

BASE = Path("outputs")
LOG_DIR = BASE / "revision-logs"
CAND_DIR = BASE / "revision-candidates"


def run(cmd):
    print("CMD " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode


def group_for(scenario, prompt):
    return f"revision-controlled-{scenario}-{prompt}"


def find_candidates(scenario, seed):
    best = None
    for p in CAND_DIR.glob(f"{scenario}-s{seed}-*/candidates.json"):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if d.get("scenario") == scenario and d.get("seed") == seed and d.get("complete"):
            best = p if best is None or p.stat().st_mtime > best.stat().st_mtime else best
    return best


def arm_done(source, arm, group):
    target = Path(source).resolve()
    for p in (BASE / group).glob("*/results.json"):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if d.get("arm") == arm and Path(d.get("source", "")).resolve() == target:
            return True
    return False


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"START {datetime.now(timezone.utc).isoformat()}", flush=True)
    for scenario, prompt in SCENARIOS.items():
        group = group_for(scenario, prompt)
        for seed in SEEDS:
            cand = find_candidates(scenario, seed)
            if cand is None:
                print(f"=== generate {scenario} s{seed} ({prompt}) ===", flush=True)
                rc = run(UV + ["scripts/controlled_curatorship.py", "--generate",
                               "--scenario", scenario, "--seed", str(seed), "--prompt", prompt,
                               "--cycles", "5", "--batch-size", "64", "--max-run-tokens", "100000"])
                if rc != 0:
                    print(f"exit={rc} generate scenario={scenario} seed={seed} — continuing", flush=True)
                    continue
                cand = find_candidates(scenario, seed)
            if cand is None:
                print(f"WARN no complete candidates for {scenario} s{seed}; skipping arms", flush=True)
                continue
            for arm in ARMS:
                if arm_done(cand, arm, group):
                    print(f"SKIP {scenario} s{seed} {arm}", flush=True)
                    continue
                rc = run(UV + ["scripts/controlled_curatorship.py", "--replay",
                               "--candidates", str(cand), "--arm", arm, "--top-k", "16",
                               "--run-group", group])
                print(f"exit={rc} replay scenario={scenario} seed={seed} arm={arm}", flush=True)
    print(f"FINISHED {datetime.now(timezone.utc).isoformat()}", flush=True)


if __name__ == "__main__":
    main()
