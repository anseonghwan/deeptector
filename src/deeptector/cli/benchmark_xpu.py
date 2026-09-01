"""Run the bounded Intel Arc capacity benchmark for the M1 baseline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from deeptector.benchmarking.xpu_capacity import run_capacity_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/ffpp_m1.yaml")
    parser.add_argument("--manifest", default="data/manifests/ffpp_c23_smoke.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or f"runs/m1_xpu_capacity_{timestamp}/capacity_report.json"
    report = run_capacity_benchmark(
        config_path=args.config,
        manifest_path=args.manifest,
        output_path=output,
        device_request=args.device,
    )
    print(f"Capacity report: {Path(output)}")
    print(json.dumps(report["recommendation"], indent=2))


if __name__ == "__main__":
    main()
