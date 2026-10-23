#!/usr/bin/env python3
"""Sunday-only staging trainer for standalone BetSightly.

Does NOT update any active model file. Candidates remain separate until a
proper forward-in-time evaluation and operator approval.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.weekly_fixture_collector import _atomic_json


def main():
    now = datetime.now(timezone.utc)
    name = "run_" + now.strftime("%Y%m%dT%H%M%SZ")
    stage = ROOT / "models/staging" / name
    report_path = ROOT / "cache/training" / (name + ".json")
    stage.mkdir(parents=True, exist_ok=False)
    environ = os.environ.copy()
    environ["BETSIGHTLY_MODELS_OUTPUT_DIR"] = str(stage)
    cmd = [sys.executable, str(ROOT / "scripts/retrain_models.py")]
    try:
        finished = subprocess.run(
            cmd, cwd=ROOT, env=environ,
            capture_output=True, text=True, timeout=7200,
        )
        if finished.returncode:
            raise RuntimeError("model training failed: " + (finished.stderr or finished.stdout)[-600:])
        meta = json.loads((stage / "meta.json").read_text(encoding="utf-8"))
        weights = json.loads((stage / "model_weights.json").read_text(encoding="utf-8"))
        if not meta.get("models") or not weights:
            raise RuntimeError("staged model metadata or measured metrics missing")
        outcome = {
            "status": "awaiting_temporal_validation",
            "trained_at_utc": now.isoformat(),
            "candidate_dir": str(stage),
            "model_count": len(meta["models"]),
            "sample_count": meta.get("n_samples"),
            "weights": weights,
            "production_promoted": False,
        }
    except Exception as exc:
        outcome = {
            "status": "error", "error": str(exc)[:700],
            "candidate_dir": str(stage), "production_promoted": False,
        }
    _atomic_json(report_path, outcome)
    print(json.dumps(outcome, indent=2))
    return 0 if outcome["status"] == "awaiting_temporal_validation" else 1


if __name__ == "__main__":
    raise SystemExit(main())
