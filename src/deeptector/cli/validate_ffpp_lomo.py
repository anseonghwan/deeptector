"""Validate a configured FF++ c23 LOMO fold before training or evaluation."""

from __future__ import annotations

import argparse
import json

from deeptector.cli.common import load_config, validate_configured_data_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    binding = validate_configured_data_protocol(config["data"])
    if binding is None:
        raise ValueError("Configuration has no LOMO protocol binding")
    print(json.dumps(binding, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
