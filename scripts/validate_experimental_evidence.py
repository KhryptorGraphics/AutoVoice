#!/usr/bin/env python3
"""Validate evidence gates for experimental quality upgrades."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auto_voice.experimental_registry import (
    DEFAULT_REGISTRY_PATH,
    evaluate_evidence_gates,
    load_experimental_registry,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Path to the experimental evidence registry JSON.",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path.cwd(),
        help="Project root used to resolve component and artifact paths.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/experimental_evidence/validation.json"),
        help="Where to write the validation report.",
    )
    args = parser.parse_args()

    registry = load_experimental_registry(args.registry)
    report = evaluate_evidence_gates(registry, root_dir=args.root_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    failures = []
    for feature_id, feature in report["features"].items():
        if not feature["gate_passed"]:
            failures.append(
                f"{feature_id}: missing components={feature['missing_components']} "
                f"missing evidence={feature['missing_evidence_categories']}"
            )

    if failures:
        raise SystemExit(
            "Experimental evidence validation failed:\n- " + "\n- ".join(failures)
        )

    print(
        f"Validated experimental evidence gates for {len(report['features'])} features "
        f"using {args.registry}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
