"""Overnight orchestrator: Phase A (checks) -> B (replay, 0 API) -> C (controlled, API-gated).

Runs under `uv run python scripts/run_revision_night.py`. Resumable: skips a scenario
whose output group already has the expected number of results, and skips a controlled
arm whose source+arm already produced a results.json. Logs to outputs/revision-logs.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

UV = ["uv", "run", "python"]

WHATSAPP = ("fakewhatsapp-full", "fakewhatsapp-10pct", "fakewhatsapp-5pct")
SCENARIOS = WHATSAPP + ("fakebr-10pct", "fakebr-5pct")
SEEDS = (42, 43, 44)
ARMS = ("all", "random-k", "score-k", "easy-k")
EXPECTED_B = 9  # real/static/ai x 3 seeds (fakebr keeps only static-158)

BASE = Path("outputs")
LOG_DIR = BASE / "revision-logs"
CAND_DIR = BASE / "revision-candidates"


def run(cmd, **kw):
    print("CMD " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], **kw).returncode


def count_group(group):
    return len(list((BASE / group).glob("*/results.json")))


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


def smoke_prompt():
    smoke = run(UV + ["scripts/controlled_curatorship.py", "--generate",
                      "--scenario", "fakewhatsapp-full", "--seed", "999", "--prompt", "v4",
                      "--cycles", "1", "--batch-size", "30", "--max-run-tokens", "20000"])
    if smoke != 0:
        return None, "smoke generation failed"
    cand = find_candidates("fakewhatsapp-full", 999)
    if cand is None:
        return None, "smoke produced no complete candidate set"
    d = json.loads(cand.read_text())
    batch = d.get("batches", [[]])[0] if d.get("batches") else []
    texts = [t for t in batch if t and t.strip()]
    if len(texts) < 25:
        return None, f"smoke too few texts ({len(texts)})"
    urgent = sum(1 for t in texts if t.lower().split()[0].rstrip("!?.,:;") == "urgente")
    if urgent / max(1, len(texts)) > 0.5:
        return None, f"smoke URGENTE rate {urgent}/{len(texts)}"
    return cand, None


def phase_a():
    if run(UV + ["-m", "tests.test_core"]) != 0:
        sys.exit("ABORT Phase A: tests failed")
    if run(UV + ["scripts/make_fakebr_val.py"]) != 0:
        sys.exit("ABORT Phase A: make_fakebr_val failed")
    if run(UV + ["-c", "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"]) != 0:
        sys.exit("ABORT Phase A: CUDA unavailable")


def phase_b():
    for s in SCENARIOS:
        group = f"{s}-weighted-replay-21oct-v2"
        if count_group(group) >= EXPECTED_B:
            print(f"SKIP B {s} (already {count_group(group)})", flush=True)
            continue
        print(f"=== Phase B {s} -> {group} ===", flush=True)
        rc = run(UV + ["scripts/replay_weighted.py", "--scenario", s,
                       "--class-weight", "--threshold-tune", "--run-group", group])
        print(f"exit={rc} phase=B scenario={s}", flush=True)


def phase_c():
    cand, err = smoke_prompt()
    if err:
        print(f"WARN Phase C smoke: {err}. Using prompt v3 (tested) instead.", flush=True)
        prompt = "v3"
    else:
        prompt = "v4"
        print(f"Phase C smoke OK ({cand}); using prompt {prompt}.", flush=True)
    for s in WHATSAPP:
        group = f"revision-controlled-{s}-{prompt}"
        for seed in SEEDS:
            cand = find_candidates(s, seed)
            if cand is None:
                print(f"=== Phase C generate {s} s{seed} ({prompt}) ===", flush=True)
                rc = run(UV + ["scripts/controlled_curatorship.py", "--generate",
                               "--scenario", s, "--seed", str(seed), "--prompt", prompt,
                               "--cycles", "5", "--batch-size", "64", "--max-run-tokens", "100000"])
                if rc != 0:
                    print(f"exit={rc} phase=C-generate scenario={s} seed={seed} — continuing", flush=True)
                    continue
                cand = find_candidates(s, seed)
            if cand is None:
                print(f"WARN no candidates for {s} s{seed}; skipping arms", flush=True)
                continue
            for arm in ARMS:
                if arm_done(cand, arm, group):
                    print(f"SKIP C {s} s{seed} {arm}", flush=True)
                    continue
                rc = run(UV + ["scripts/controlled_curatorship.py", "--replay",
                               "--candidates", str(cand), "--arm", arm, "--top-k", "16",
                               "--run-group", group])
                print(f"exit={rc} phase=C-replay scenario={s} seed={seed} arm={arm}", flush=True)


def main():
    print(f"START {datetime.now(timezone.utc).isoformat()}", flush=True)
    phase_a()
    phase_b()
    phase_c()
    run(UV + ["scripts/revision_status.py"])
    print(f"FINISHED {datetime.now(timezone.utc).isoformat()}", flush=True)


if __name__ == "__main__":
    main()
