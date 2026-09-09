"""Run a bounded real-data XPU preflight for one FF++ LOMO fold."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from deeptector.benchmarking.lomo_preflight import run_lomo_preflight
from deeptector.cli.common import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (
        f"runs/{config['experiment_name']}/preflight/{timestamp}/preflight_report.json"
    )
    report = run_lomo_preflight(args.config, output, device_request=args.device)
    print(f"LOMO preflight report: {Path(output)}")
    print(json.dumps({"success": report["success"], "device": report["device"]}, indent=2))


if __name__ == "__main__":
    main()
