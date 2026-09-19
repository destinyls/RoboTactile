"""Assess a frozen production gallery using signal-level and analytical evidence.

Curves are measured from production outputs, never sensor data synthesized by a
generative model. Analytical stimuli are explicitly separated from recordings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from robotactile_benchmark.optical.fields import spatial_weight
from robotactile_benchmark.optical.probes import run_analytical_probes
from robotactile_benchmark.runtime import apply_fault
from robotactile_benchmark.visualization.optical14 import (
    build_records,
    file_hash,
    font,
    load_source,
    make_manifest,
    production_hashes,
    source_array_hash,
    write_json,
)

COLORS = ("#2563a8", "#d57126", "#329361", "#a14585")


def plot(
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[float], list[float]]],
    path: Path,
    note: str,
) -> None:
    """Plot measured scalars; the axes, scales and evidence class stay explicit."""
    canvas = Image.new("RGB", (1080, 560), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 15), title, fill="#182438", font=font(23))
    draw.text((28, 49), note, fill="#555555", font=font(15))
    left, top, right, bottom = 110, 120, 1030, 445
    x_max = max(max(x) for _, x, _ in series)
    y_max = max(max(y) for _, _, y in series) * 1.08 or 1.0
    if any(not np.isfinite(values).all() for _, x, y in series for values in (x, y)):
        raise ValueError("non-finite plot values")
    for index in range(6):
        x = left + (right - left) * index / 5
        y = bottom - (bottom - top) * index / 5
        draw.line((left, y, right, y), fill="#e3e6eb")
        draw.text(
            (left - 75, y - 10), f"{y_max * index / 5:.3f}", font=font(15), fill="black"
        )
        draw.text(
            (x - 12, bottom + 10),
            f"{x_max * index / 5:.2f}",
            font=font(15),
            fill="black",
        )
    draw.line((left, top, left, bottom, right, bottom), fill="#182438", width=2)
    draw.text((left, 85), y_label, font=font(16), fill="black")
    draw.text((left + 240, bottom + 42), x_label, font=font(17), fill="black")
    for index, (label, xs, ys) in enumerate(series):
        if len(xs) != len(ys) or not xs:
            raise ValueError("empty or inconsistent curve")
        points = [
            (
                left + x / max(x_max, 1e-12) * (right - left),
                bottom - y / y_max * (bottom - top),
            )
            for x, y in zip(xs, ys)
        ]
        color = COLORS[index % len(COLORS)]
        if index % 2:
            # Sparse marks leave a coincident reference curve visible underneath.
            for x, y in points[:: max(1, len(points) // 24)]:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
        else:
            draw.line(points, fill=color, width=3)
        draw.line((30 + 260 * index, 525, 60 + 260 * index, 525), fill=color, width=3)
        draw.text((66 + 260 * index, 514), label, fill=color, font=font(15))
    canvas.save(path)


def detail_panels(
    source_root: Path, gallery_root: Path, output_root: Path
) -> dict[str, Any]:
    """Magnify the fixed registered ROI; never modify photometry or select by delta."""
    receipt_path = source_root / "source_receipt.json"
    if file_hash(receipt_path) != file_hash(gallery_root / "source_receipt.json"):
        raise ValueError("detail panel source differs from gallery")
    sources = json.loads(receipt_path.read_text())["sources"]
    evidence = {}
    for prefix in ("F2", "F4"):
        original = json.loads((gallery_root / f"{prefix}_provenance.json").read_text())
        spec = next(item for item in sources if item["task"] == original["source_task"])
        arrays = load_source(source_root, spec)
        records, rest = build_records(arrays, spec, file_hash(receipt_path))
        index, slot = original["display_index"], original["display_slot"]
        clean = records[index].observation.sensor(slot).payload
        if clean is None:
            raise ValueError("detail panel requires recorded clean RGB")
        manifest = make_manifest(original["operator_id"], 5, arrays, spec, rest)
        support = (
            spatial_weight(clean.shape, manifest.parameters, dead_patch=prefix == "F4")
            > 0
        )
        yy, xx = np.nonzero(support)
        # Both severities use S5 support bounds, determined from parameters alone.
        x0, y0, x1, y1 = (
            int(xx.min()),
            int(yy.min()),
            int(xx.max()) + 1,
            int(yy.max()) + 1,
        )
        crops = [clean[y0:y1, x0:x1]]
        traces = []
        for level in (3, 5):
            configured = make_manifest(
                original["operator_id"], level, arrays, spec, rest
            )
            result = apply_fault(records, configured, rest_references=rest)
            if (
                not result.validation.passed
                or result.trace_sha256
                != original["severity_sweep"][level - 1]["trace_sha256"]
            ):
                raise ValueError(
                    "detail panel is not an exact gallery production replay"
                )
            payload = result.records[index].observation.sensor(slot).payload
            if payload is None:
                raise ValueError("detail panel requires delivered RGB")
            crops.append(payload[y0:y1, x0:x1])
            traces.append(result.trace_sha256)
            del result
        size = (x1 - x0) * 3, (y1 - y0) * 3
        width, height = 350 + 3 * (size[0] + 16), size[1] + 170
        panel = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(panel)
        draw.text(
            (12, 12),
            f"{prefix} | fixed ROI detail | {spec['task']} raw{spec['raw_episode_id']} obs {index}",
            fill="#182438",
            font=font(21),
        )
        draw.text(
            (12, 43),
            "Recorded simulator input; production faults. 3x nearest-neighbor zoom; NO color/contrast enhancement.",
            fill="#555555",
            font=font(16),
        )
        context = Image.fromarray(clean)
        ImageDraw.Draw(context).rectangle(
            (x0, y0, x1 - 1, y1 - 1), outline="#ffff00", width=1
        )
        panel.paste(context, (12, 115))
        draw.text(
            (12, 86),
            "Clean context / ROI annotation only",
            fill="#182438",
            font=font(16),
        )
        for column, (label, crop) in enumerate(
            zip(("Clean ROI", "S3: same ROI", "S5: same ROI"), crops)
        ):
            x = 350 + column * (size[0] + 16)
            draw.text((x, 86), label, fill="#182438", font=font(17))
            panel.paste(
                Image.fromarray(crop).resize(size, Image.Resampling.NEAREST), (x, 115)
            )
        draw.text(
            (12, height - 35),
            "ROI fixed by registered S5 support, not by maximum image difference. Full unannotated frames remain in the gallery.",
            fill="#555555",
            font=font(15),
        )
        panel.save(output_root / f"{prefix}_detail.png")
        evidence[prefix] = {
            "bounds_xyxy": [x0, y0, x1, y1],
            "zoom": 3,
            "interpolation": "nearest",
            "photometric_enhancement": False,
            "source_selection": "unchanged gallery witness; ROI from registered parameters",
            "crop_array_sha256": [source_array_hash(crop) for crop in crops],
            "replayed_gallery_trace_sha256": traces,
        }
    return evidence


def assess(
    gallery_root: Path, output_root: Path, source_root: Path | None = None
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    gallery_path = gallery_root / "gallery_receipt.json"
    gallery = json.loads(gallery_path.read_text())
    before = production_hashes()
    if (
        gallery.get("operator_count") != 14
        or not gallery.get("all_delivery_validations_passed")
        or gallery.get("production_source_sha256") != before
    ):
        raise ValueError("requires a complete gallery bound to current production code")
    for name, expected in gallery["output_sha256"].items():
        path = (gallery_root / name).resolve(strict=True)
        if (
            not path.is_relative_to(gallery_root.resolve())
            or file_hash(path) != expected
        ):
            raise ValueError("gallery output integrity mismatch")
    output_root.mkdir(parents=True, exist_ok=False)
    profiles = {}
    region_series = []
    for prefix in ("F2", "F4", "F6", "F7"):
        evidence = json.loads((gallery_root / f"{prefix}_provenance.json").read_text())
        levels = evidence["severity_sweep"]
        if [level["severity_level"] for level in levels] != list(range(1, 6)):
            raise ValueError("requires exactly S1-S5, in order")
        values = [float(level["display_mean_absolute_delta"]) for level in levels]
        metrics = [level["validation"]["metrics"] for level in levels]
        controls = all(m["marker_changed_elements"] == 0 for m in metrics)
        if prefix in {"F2", "F4"}:
            controls = controls and all(
                m["outside_support_changed_pixels"] == 0 for m in metrics
            )
            gains = [float(m["core_response_projection_gain"]) for m in metrics]
            targets = (
                [
                    float(level["manifest"]["parameters"]["retained_gain"])
                    for level in levels
                ]
                if prefix == "F2"
                else [0.0] * 5
            )
            # 0.02 is a declared RGB-quantization verification tolerance, not material fit.
            controls = controls and all(
                m["core_response_pixel_samples"] > 0 for m in metrics
            )
            controls = controls and all(
                abs(g - target) <= 0.02 for g, target in zip(gains, targets)
            )
            region_series.append(
                (f"{prefix} measured core", [float(i) for i in range(1, 6)], gains)
            )
            region_series.append(
                (f"{prefix} registered gain", [float(i) for i in range(1, 6)], targets)
            )
        else:
            gains, targets = [], []
        profiles[prefix] = {
            "source_task": evidence["source_task"],
            "source_episode": evidence["source_episode"],
            "display_index": evidence["display_index"],
            "severity_metrics": metrics,
            "core_gain_scope": "all eligible active-window contact pixels; not just display frame",
            "mean_absolute_delta_u8": values,
            "core_gain": gains,
            "target_gain": targets,
            "increasing_display_delta": all(a < b for a, b in zip(values, values[1:])),
            "controls_passed": controls,
        }
    plot(
        "Regional optical response: actual core gain, not whole-image MAE",
        "Severity level",
        "Projection gain relative to clean optical response",
        region_series,
        output_root / "regional_response.png",
        "Recorded simulator episodes. Excludes current markers; 2/255 response floor. Not a force metric.",
    )
    probes = run_analytical_probes(level=5)
    write_json(output_root / "analytical_probes.json", probes)
    runs = probes["F6"]["runs"]
    plot(
        "F6 release tail: measured production output on controlled inputs",
        "Seconds after complete analytical release",
        "Residual optical response magnitude",
        [
            (f"{run['fps']} Hz measured", run["times_s"], run["response"])
            for run in runs
        ],
        output_root / "F6_release_tail.png",
        "ANALYTICAL STIMULUS, NOT A RECORDED RELEASE. Checks time behavior, not sensor material constants.",
    )
    curves = probes["F7"]["curves"]
    plot(
        "F7 transfer: nonlinear response compression, not camera overexposure",
        "Input RGB residual norm",
        "Delivered RGB residual norm",
        [
            (
                f"baseline {base}",
                [point[0] for point in curve],
                [point[1] for point in curve],
            )
            for base, curve in zip(probes["F7"]["baseline_levels"], curves)
        ],
        output_root / "F7_response_transfer.png",
        "ANALYTICAL STIMULUS. Equal residuals at different baselines distinguish this proxy from absolute clipping.",
    )
    details = (
        detail_panels(source_root, gallery_root, output_root)
        if source_root is not None
        else {}
    )
    if before != production_hashes():
        raise RuntimeError("production changed during scientific assessment")
    result = {
        "schema_id": "robotactile.optical14_scientific_assessment",
        "schema_version": "1.0",
        "gallery_receipt_sha256": file_hash(gallery_path),
        "production_source_sha256": before,
        "assessment_script_sha256": file_hash(Path(__file__)),
        "recorded_profiles": profiles,
        "detail_panels": details,
        "observation_level_acceptance_passed": (
            all(
                p["controls_passed"] and p["increasing_display_delta"]
                for p in profiles.values()
            )
            and all(probe["passed"] for probe in probes.values())
        ),
        "paper_scope": {
            "fourteen_operator_delivery": "verified",
            "optical_marker_scope": "photometric response isolated; clean marker dynamics deliberately retained",
            "F6_recorded_witness": "local unloading only; full recorded release tail unavailable",
            "F7_meaning": "high-response contrast compression; not pressure-calibrated or camera clipping",
            "physical_fault_calibrated": False,
            "new_closed_loop_results": False,
            "peer_review_acceptance_guaranteed": False,
        },
        "analytical_probes_artifact": "analytical_probes.json",
        "output_sha256": {
            p.name: file_hash(p) for p in sorted(output_root.iterdir()) if p.is_file()
        },
    }
    write_json(output_root / "assessment.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Optional original gallery inputs for native ROI zooms",
    )
    args = parser.parse_args()
    result = assess(
        args.gallery_root.resolve(), args.output_root.resolve(), args.source_root
    )
    if not result["observation_level_acceptance_passed"]:
        raise SystemExit("observation-level checks failed; inspect assessment.json")


if __name__ == "__main__":
    main()
