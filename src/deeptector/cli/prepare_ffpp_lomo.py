"""Generate deterministic FF++ c23 leave-one-manipulation-out protocols."""

from __future__ import annotations

import argparse
import json

from deeptector.data.ffpp_lomo import generate_ffpp_lomo_protocol


def main() -> None:
    """Validate an FF++ c23 manifest and write all four LOMO folds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", default="data/manifests/ffpp_c23.csv")
    parser.add_argument("--output-dir", default="data/manifests/ffpp_lomo_c23")
    args = parser.parse_args()
    protocol = generate_ffpp_lomo_protocol(args.source_manifest, args.output_dir)
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
