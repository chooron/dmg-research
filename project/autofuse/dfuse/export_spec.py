"""Regenerate the materialized FUSE-78 specification snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .spec import PARAMETER_NAMES, STATE_NAMES, enumerate_structures, validate_catalog


def export(path: str | Path = Path(__file__).with_name("specs") / "catalog_78.json") -> Path:
    destination = Path(path)
    payload = {
        "schema_version": "1.0",
        "generated_from": "dfuse.spec (source inputs structures_78.json and parameter_catalog.json)",
        "validation": validate_catalog(),
        "state_union": list(STATE_NAMES),
        "parameter_union": list(PARAMETER_NAMES),
        "structures": [spec.to_dict() for spec in enumerate_structures()],
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("specs") / "catalog_78.json")
    args = parser.parse_args()
    print(export(args.output))


if __name__ == "__main__":
    main()
