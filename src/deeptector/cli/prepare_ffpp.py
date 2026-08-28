"""Build and audit an FF++ manifest from official split JSON files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from deeptector.data.ffpp import (
    OfficialFFPPSplits,
    build_ffpp_manifest,
    resolve_ffpp_root,
    select_smoke_records,
    write_manifest,
    write_split_report,
)


def main() -> None:
    """Validate official FF++ splits before writing any manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffpp-root", default=None)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--compression", default="c23")
    parser.add_argument("--output", default="data/manifests/ffpp_c23.csv")
    parser.add_argument("--report", default="data/manifests/ffpp_c23_split_report.json")
    parser.add_argument("--smoke-output", default="data/manifests/ffpp_c23_smoke.csv")
    args = parser.parse_args()
    root = resolve_ffpp_root(args.ffpp_root)
    splits = OfficialFFPPSplits.from_directory(args.splits_dir)
    records, report = build_ffpp_manifest(root, splits, compression=args.compression)
    write_manifest(records, args.output)
    write_manifest(select_smoke_records(records), args.smoke_output)
    write_split_report(report, args.report)
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
