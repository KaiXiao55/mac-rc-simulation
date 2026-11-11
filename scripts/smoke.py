#!/usr/bin/env python3
"""Run one 2-second existing SimYOLO scenario. Requires a macOS GUI session."""
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent.parent
build_dir = Path(os.environ.get("MACRC_BUILD_DIR", str(root / ".build")))
binary = build_dir / "Build/Products/Debug/macRCtrackerSimulator.app/Contents/MacOS/macRCtrackerSimulator"
output = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root / "runs" / ("smoke-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
if not binary.is_file():
    raise SystemExit("Build first: bash scripts/build.sh")
if (output / "batch_summary.csv").exists():
    raise SystemExit("Use a new output directory; existing batches resume prior results.")
output.mkdir(parents=True, exist_ok=True)
config = {"purpose": "functional smoke only", "seed": 6001, "trackShape": 0,
          "sceneType": 0, "cameraPreset": 0, "carCount": 5, "cameraHeight": 2.5,
          "enableRamps": False, "recallRate": 0.8236, "fpRate": 0.7803,
          "simDuration": 2.0, "forceSimulatedYOLO": True}
(output / "smoke_config.json").write_text(json.dumps(config, indent=2) + "\n")
command = [str(binary), "--batch-sim", "--public-smoke", "--batch-sim-auto-exit", "--output", str(output)]
with (output / "run.log").open("w") as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=60)
summary = output / "batch_summary.csv"
if result.returncode != 0 or not summary.is_file():
    raise SystemExit("Smoke failed; inspect run.log (GUI session and macOS 26 required).")
with summary.open() as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 1 or int(rows[0]["totalFrames"]) <= 0 or rows[0]["usesSimulatedYolo"].lower() not in ("true", "1"):
    raise SystemExit("Unexpected smoke output; inspect batch_summary.csv")
(output / "summary.md").write_text(
    "# SimYOLO smoke check\n\n"
    "One fixed-seed synthetic scenario, 2 simulated seconds, no neural weights.\n\n"
    f"- Completed runs: {len(rows)}\n- Processed frames: {rows[0]['totalFrames']}\n"
    "- Mode: SimYOLO (synthetic detections from scene geometry and noise)\n"
    "- Config: smoke_config.json\n- Metrics: batch_summary.csv\n\n"
    "This verifies startup, rendering, tracking and metric export only. It is not a model-quality or speed benchmark.\n")
print(f"Smoke passed: {output / 'summary.md'}")
