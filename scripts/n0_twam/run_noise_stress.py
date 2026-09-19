"""Freeze, prepare, execute, and report the N0-TWAM noise-drop diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.n0_fault_campaign.io import file_sha256
from robotactile_benchmark.n0_fault_campaign.stress_calibration import (
    calibrate_clean_artifacts,
)
from robotactile_benchmark.n0_fault_campaign.stress_group import (
    prepare_stress_group,
    run_stress_group,
    write_once,
)
from robotactile_benchmark.n0_fault_campaign.stress_protocol import (
    StressVariant,
    freeze_protocol,
)
from robotactile_benchmark.n0_fault_campaign.stress_selection import (
    calibration_selection_evidence,
    report_frozen_groups,
)
from robotactile_benchmark.optical.stress_templates import calibrate_contact_scars


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    freeze = subs.add_parser("freeze")
    freeze.add_argument("--spec", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    prepare = subs.add_parser("prepare")
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--clean-request", type=Path, required=True)
    prepare.add_argument("--rest-reference", type=Path)
    prepare.add_argument("--spatial-calibration", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    run = subs.add_parser("run")
    run.add_argument("--group", type=Path, required=True)
    run.add_argument("--integration-config", type=Path, required=True)
    run.add_argument("--n0-source-root", type=Path, required=True)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, required=True)
    report = subs.add_parser("report")
    report.add_argument("--protocol", type=Path, required=True)
    report.add_argument("--groups", type=Path, nargs="+")
    report.add_argument("--output", type=Path)
    select = subs.add_parser("select", help="derive minimum calibration doses or No-Go")
    select.add_argument("--protocol", type=Path, required=True)
    select.add_argument("--output", type=Path)
    calibrate = subs.add_parser("calibrate-contact")
    calibrate.add_argument(
        "--response-maps",
        type=Path,
        required=True,
        help="NPZ with left/right nonnegative Clean-calibration response maps",
    )
    calibrate.add_argument("--output", type=Path, required=True)
    from_clean = subs.add_parser("calibrate-from-clean")
    from_clean.add_argument("--clean-artifacts", type=Path, nargs="+", required=True)
    from_clean.add_argument("--development-seeds", type=int, nargs="+", required=True)
    from_clean.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        spec = read_json(args.spec)
        evidence = None
        excluded = set(spec["excluded_seeds"])
        if spec["stage"] == "confirmation":
            # Never accept a caller-authored summary or a bare selection hash.
            evidence = calibration_selection_evidence(
                read_json(Path(spec["calibration_protocol_path"]).resolve(strict=True))
            )
            parent = evidence["protocol"]
            excluded.update(parent["seeds"])
            excluded.update(parent["excluded_seeds"])
        value = freeze_protocol(
            stage=spec["stage"],
            seeds=tuple(spec["seeds"]),
            variants=tuple(
                StressVariant(v["family"], v["level"]) for v in spec["variants"]
            ),
            excluded_seeds=tuple(sorted(excluded)),
            binding=spec["binding"],
            selection_evidence_sha256=canonical_hash(evidence)
            if evidence
            else spec.get("selection_evidence_sha256"),
            spatial_calibration_sha256=spec.get("spatial_calibration_sha256"),
            group_paths=spec.get("group_paths"),
            calibration_evidence=evidence,
        )
        write_once(args.output, value)
    elif args.command == "select":
        value = calibration_selection_evidence(read_json(args.protocol))
        if args.output:
            write_once(args.output, value)
    elif args.command == "prepare":
        value = prepare_stress_group(
            args.output.absolute(),
            protocol=read_json(args.protocol),
            clean_request=args.clean_request.resolve(strict=True),
            rest_reference=args.rest_reference,
            spatial_calibration=read_json(args.spatial_calibration)
            if args.spatial_calibration
            else None,
        )
    elif args.command == "run":
        value = run_stress_group(
            args.group.resolve(strict=True),
            integration_config=args.integration_config.resolve(strict=True),
            n0_source_root=args.n0_source_root.resolve(strict=True),
            host=args.host,
            port=args.port,
        )
    elif args.command == "calibrate-contact":
        with np.load(args.response_maps, allow_pickle=False) as maps:
            value = calibrate_contact_scars(
                {slot: maps[slot] for slot in ("left", "right")},
                source_sha256=file_sha256(args.response_maps),
            )
        write_once(args.output, value)
        value = {
            "template": str(args.output),
            "spatial_calibration_sha256": canonical_hash(value),
        }
    elif args.command == "calibrate-from-clean":
        template, receipt = calibrate_clean_artifacts(
            tuple(p.resolve(strict=True) for p in args.clean_artifacts),
            allowed_seeds=tuple(args.development_seeds),
        )
        args.output.mkdir(parents=True, exist_ok=False)
        write_once(args.output / "spatial_calibration.json", template)
        write_once(args.output / "calibration_receipt.json", receipt)
        value = {
            "spatial_calibration_sha256": canonical_hash(template),
            "receipt": receipt,
        }
    else:
        value = report_frozen_groups(read_json(args.protocol), args.groups)
        if args.output:
            write_once(args.output, value)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
