"""Compare all fourteen standard-S5 and opt-in stress production operators.

Run as ``python -m scripts.render_optical14_stress`` from the repository root.
The fixed source witnesses are selected before corruption. No policy runs,
generated sensor images or display-only fault transformations are used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    OPTICAL_MARKER_EXTREME_REGISTRY_ID,
    OPTICAL_MARKER_REGISTRY_ID,
    OPTICAL_MARKER_STRESS_REGISTRY_ID,
    operator_requires_rest_reference,
)
from robotactile_benchmark.contracts import Array, EvaluationRecord
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.rest_references import RestReferenceBundle
from robotactile_benchmark.runtime import apply_fault
from robotactile_benchmark.visualization.optical14 import (
    build_records,
    file_hash,
    load_source,
    production_hashes,
    write_json,
)
from scripts.render_optical14_paper_panels import BLUE, MUTED, ORANGE, Painter

NAMES = {
    "A1": "连续缺失",
    "A2": "间歇丢帧",
    "F1": "全局光度漂移",
    "F2": "局部光学灵敏度衰减",
    "F3": "持久表面外观伪影",
    "F4": "局部光学失响应",
    "F5": "接触图像一致形变",
    "F6": "历史光学残影",
    "F7": "高响应光学压缩",
    "T1": "共同源延迟",
    "T2": "末帧冻结",
    "T3": "左右源时间差",
    "C1": "左右来源交换",
    "C2": "图像注册错位",
}
OPTICAL_PREFIXES = frozenset({"F1", "F2", "F3", "F4", "F5", "F6", "F7", "C2"})


def dose_label(prefix: str, parameters: dict[str, Any], short_side: int) -> str:
    """Describe the actual materialized instance, not a hand-entered dose."""
    p = parameters
    if prefix == "F1":
        return f"R通道目标增益 {p['target_gain_rgb'][0]:g}"
    if prefix == "F2":
        return (
            f"核心保留{100 * p['retained_gain']:g}% / 半径{p['core_radius_fraction']:g}"
        )
    if prefix == "F3":
        return f"条纹半宽{p['scar_half_width_px']:g} px"
    if prefix == "F4":
        return f"失响应半径{p['radius_fraction']:g}"
    if prefix == "F5":
        return f"最大位移{short_side * p['displacement_fraction']:g} px"
    if prefix == "F6":
        return f"历史混合{p['history_mix']:g} / τ={p['recovery_tau_s']:g} s"
    if prefix == "F7":
        return f"响应knee={p['response_knee']:g}"
    if prefix == "C2":
        return f"向右{p['translation_xy_px'][0]:g} px"
    raise ValueError(f"not an optical panel: {prefix}")


def freeze_indices(start: int, old_hold: int, new_hold: int, stop: int) -> list[int]:
    if not 0 < old_hold < new_hold or start < 1 or start + new_hold >= stop:
        raise ValueError("freeze panel requires warm-up, longer hold and resumption")
    return [start - 1, start, start + old_hold, start + new_hold - 1, start + new_hold]


def configured(
    operator: str,
    registry: str,
    arrays: dict[str, Array],
    spec: dict[str, Any],
    rest: RestReferenceBundle,
    *,
    long_freeze: bool = False,
) -> FaultManifest:
    length = len(arrays["source_time_s"])
    start, stop = (
        (25, length - 1) if spec["task"] == "insert_HDMI" else (90, length - 1)
    )
    if long_freeze and operator.startswith("T2_"):
        start, stop = 80, length
    slots = (
        ("left", "right") if operator.startswith(("T1_", "T3_", "C1_")) else ("right",)
    )
    if operator.startswith("F6_"):
        witness = spec["f6_local_unloading_witness"]
        start, stop = witness["start_index"], witness["stop_index"]
        slots = (witness["slot"],)
    parameters: dict[str, Any] = {
        "sample_period_s": float(np.diff(arrays["source_time_s"])[0])
    }
    if operator_requires_rest_reference(operator, severity_registry=registry):
        parameters["rest_reference_sha256"] = rest.sha256
    if operator.startswith("C2_"):
        parameters["realization"] = "registered_pixels"
    return FaultManifest(
        operator,
        5,
        20260906,
        start,
        stop,
        slots,
        Observability.BLIND,
        parameters,
        severity_registry=registry,
    )


def rgb(record: EvaluationRecord, slot: str) -> Array:
    payload = record.observation.sensor(slot).payload
    if payload is None:
        raise ValueError("cannot turn absent payload into a tactile image")
    return payload


def age_ms(record: EvaluationRecord, slot: str) -> float:
    source = record.provenance_for(slot).source_time_s
    delivered = record.observation.sensor(slot).delivery_time_s
    if source is None or delivered is None:
        raise ValueError("delay visualization requires actual source/delivery clocks")
    return 1000 * (delivered - source)


def check_source_layout(spec: dict[str, Any], arrays: dict[str, Array]) -> None:
    expected = {"pull_out_key": (55, 178, 121), "insert_HDMI": (90, 118, 49)}
    actual = (
        int(spec["raw_episode_id"]),
        len(arrays["source_time_s"]),
        int(spec["contact_peak"]),
    )
    if actual != expected.get(spec["task"]) or not np.allclose(
        np.diff(arrays["source_time_s"]), 1 / 60, atol=1e-9, rtol=0
    ):
        raise ValueError(
            "fixed stress captions require the certified raw55/raw90 60Hz source layout"
        )


def stronger_checks(cases: dict[str, Any]) -> dict[str, bool]:
    """Compare native effects within each operator, never rank different faults."""
    checks = {}
    for prefix, case in cases.items():
        standard, stress = case["standard"], case["stress"]
        if prefix == "C2":
            # Periodic lattices can realign under a larger translation.
            # Delivery is signature-validated separately; MAE is not displacement.
            value = np.linalg.norm(
                stress["manifest"]["parameters"]["translation_xy_px"]
            ) > np.linalg.norm(standard["manifest"]["parameters"]["translation_xy_px"])
        elif prefix in OPTICAL_PREFIXES:
            value = (
                stress["display_mean_absolute_delta_u8"]
                > standard["display_mean_absolute_delta_u8"]
            )
        elif prefix in {"A1", "A2", "C1"}:
            value = stress["affected_count"] > standard["affected_count"]
        elif prefix == "T2":
            value = (
                stress["manifest"]["parameters"]["hold_duration_frames"]
                > standard["manifest"]["parameters"]["hold_duration_frames"]
            )
        else:
            value = max(stress["display_age_ms"]) > max(standard["display_age_ms"])
        checks[prefix] = bool(value)
    return checks


def accepted_checks(cases: dict[str, Any], *, extreme: bool) -> dict[str, str]:
    """Never relabel an at-cap tie as an increase, or excuse another regression."""
    increased = stronger_checks(cases)
    status = {}
    for prefix, value in increased.items():
        if value:
            status[prefix] = "increased"
        elif (
            extreme
            and prefix in {"A1", "C1"}
            and all(
                cases[prefix][key]["affected_count"] == 87
                for key in ("standard", "stress")
            )
        ):
            status[prefix] = "unchanged_at_100_percent_cap"
        else:
            raise ValueError(f"no demonstrated increase for {prefix}")
    return status


def strip(
    prefixes: tuple[str, ...],
    samples: dict[str, Any],
    title: str,
    font: Path,
    labels: tuple[str, str] = ("标准S5", "Stress"),
) -> Image.Image:
    height = 155 + 370 * len(prefixes)
    p = Painter(1140, height, font)
    p.title(
        title,
        f"三列：同一个Clean｜{labels[0]}｜{labels[1]} · 图像原色，无对比度增强，无模型生成",
    )
    xs = (40, 410, 780)
    for row, prefix in enumerate(prefixes):
        y = 123 + row * 370
        example = samples[prefix]
        p.text((32, y), f"{prefix}  {NAMES[prefix]}", 25)
        p.text((550, y + 2), example["source_label"], 18, MUTED)
        row_labels = ("Clean", *example["dose_labels"])
        for col, (label, image) in enumerate(zip(row_labels, example["images"])):
            x = xs[col]
            p.text(
                (x, y + 43),
                label,
                19,
                BLUE if col == 0 else ORANGE if col == 2 else MUTED,
            )
            p.frame(image, (x, y + 78))
        note = (
            "仅局部卸载，不是完整脱离；历史来自同一episode。"
            if prefix == "F6"
            else "半径单位：图像短边比例；当前marker及其保护邻域不变。"
            if prefix in {"F2", "F4"}
            else "当前图像与marker共同无折叠变形，不引入第二套点阵。"
            if prefix == "F5"
            else "压缩的是相对参考场的光学响应，不代表压力或曝光标定。"
            if prefix == "F7"
            else "完整原图；正式production delivery，无显示专用替换算法。"
        )
        p.text((40, y + 332), note, 18, MUTED)
    return p.image


def timelines(
    cases: dict[str, Any], font: Path, labels: tuple[str, str] = ("标准S5", "Stress")
) -> Image.Image:
    p = Painter(1400, 1090, font)
    p.title(
        "缺失与路由｜保留失效类别，已达100%上限者不虚称继续增强",
        "蓝色：正常输入/正常来源；橙色：来源交换；灰色：无payload · 红线界定故障窗口",
    )
    xs, width = 240, 1090
    for row, prefix in enumerate(("A1", "A2", "C1")):
        y = 135 + row * 282
        p.text((28, y), f"{prefix} {NAMES[prefix]}", 29)
        a = cases[prefix]
        for k, (label, key) in enumerate(
            (("Clean", "clean"), (labels[0], "standard"), (labels[1], "stress"))
        ):
            bar_y = y + 53 + k * 54
            states = a[key]["source_left"] if prefix == "C1" else a[key]["presence"]
            p.text(
                (40, bar_y + 6),
                label,
                23,
                BLUE if k == 0 else ORANGE if k == 2 else MUTED,
            )
            for i, state in enumerate(states):
                color = (
                    (BLUE if state else ORANGE)
                    if prefix == "C1"
                    else (BLUE if state else "#d7dce3")
                )
                p.draw.rectangle(
                    (
                        xs + width * i / len(states),
                        bar_y,
                        xs + width * (i + 1) / len(states),
                        bar_y + 32,
                    ),
                    fill=color,
                )
            for border in (90, 177):
                x = xs + width * border / len(states)
                p.draw.line((x, bar_y - 4, x, bar_y + 37), fill="#bd3546", width=2)
        fractions = [a[key]["affected_count"] for key in ("standard", "stress")]
        p.text(
            (240, y + 223),
            f"同一故障窗口87帧：{labels[0]}作用 {fractions[0]} 帧 → {labels[1]}作用 {fractions[1]} 帧",
            22,
        )
    p.text(
        (30, 1000),
        "A1/C1达到窗口内100%作用；A2保留间歇恢复帧，避免退化成A1。来源交换的单帧幅度没有“更大”的定义。",
        21,
        MUTED,
    )
    p.text(
        (30, 1041),
        "横轴为同一178帧记录的源顺序（60 Hz）；灰格表示结构性缺失，不是给模型输入灰图或黑图。",
        21,
        MUTED,
    )
    return p.image


def freeze_panel(
    sample: dict[str, Any], font: Path, labels: tuple[str, str] = ("标准S5", "Stress")
) -> Image.Image:
    p = Painter(1770, 1100, font)
    old_hold, new_hold = sample["hold_frames"]
    before, _, old_resume, _, new_resume = sample["indices"]
    p.title(
        f"T2｜将冻结从{old_hold / 60:g} s延长到{new_hold / 60:g} s，并保留恢复过程",
        f"同一记录：在第{old_resume}帧，{labels[0]}已恢复，{labels[1]}仍停在第{before}帧；第{new_resume}帧恢复当前输入",
    )
    for row, (label, key) in enumerate(
        (("Clean", "clean"), (labels[0], "standard"), (labels[1], "stress"))
    ):
        y = 140 + 300 * row
        p.text(
            (22, y + 124),
            label,
            20,
            BLUE if row == 0 else ORANGE if row == 2 else MUTED,
        )
        for col, index in enumerate(sample["indices"]):
            x = 112 + 328 * col
            source = sample[key]["sources"][col]
            p.text((x, y), f"到达{index} / 源{source}", 22)
            p.frame(sample[key]["images"][col], (x, y + 42))
    p.text(
        (30, 1060),
        f"{old_hold}帧冻结 → {new_hold}帧冻结；全部源帧和payload均来自正式注入结果。图片只展示关键时刻，不改变原始60 Hz时钟。",
        23,
        MUTED,
    )
    return p.image


def delay_panel(
    samples: dict[str, Any], font: Path, labels: tuple[str, str] = ("标准S5", "Stress")
) -> Image.Image:
    p = Painter(1400, 1480, font)
    old_age, new_age = (
        max(samples["T1"][key]["age_ms"]) for key in ("standard", "stress")
    )
    p.title(
        f"T1/T3｜源时间差从{old_age:.0f} ms增至{new_age:.0f} ms",
        "固定到达观测121；每列上下为左/右触觉图像。T1两侧一起延迟；T3只延迟右侧。",
    )
    for row, prefix in enumerate(("T1", "T3")):
        y = 130 + row * 654
        p.text((30, y), f"{prefix} {NAMES[prefix]}", 29)
        for col, (label, key) in enumerate(
            (("Clean", "clean"), (labels[0], "standard"), (labels[1], "stress"))
        ):
            x = 92 + 440 * col
            sample = samples[prefix][key]
            p.text(
                (x, y + 42),
                label,
                23,
                BLUE if col == 0 else ORANGE if col == 2 else MUTED,
            )
            for side in range(2):
                top = y + 89 + side * 278
                p.text(
                    (x, top - 5),
                    f"{'左' if side == 0 else '右'} 源{sample['sources'][side]} / {sample['age_ms'][side]:.0f} ms",
                    20,
                )
                p.frame(sample["images"][side], (x, top + 27))
    return p.image


def render(
    source_root: Path, output_root: Path, font: Path, *, extreme: bool = False
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError("stress gallery output must be new")
    before = production_hashes()
    registries = (
        (OPTICAL_MARKER_STRESS_REGISTRY_ID, OPTICAL_MARKER_EXTREME_REGISTRY_ID)
        if extreme
        else (OPTICAL_MARKER_REGISTRY_ID, OPTICAL_MARKER_STRESS_REGISTRY_ID)
    )
    labels = ("Stress", "Extreme") if extreme else ("标准S5", "Stress")
    receipt_path = source_root / "source_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    cases: dict[str, Any] = {}
    samples: dict[str, Any] = {}
    sequences: dict[str, Any] = {}
    for source_spec in receipt["sources"]:
        spec = dict(source_spec)
        unloading = receipt["f6_local_unloading_witness"]
        if unloading["task"] == spec["task"]:
            spec["f6_local_unloading_witness"] = unloading
        arrays = load_source(source_root, spec)
        check_source_layout(spec, arrays)
        clean, rest = build_records(arrays, spec, file_hash(receipt_path))
        for operator in sorted(CORE_OPERATOR_IDS):
            prefix = operator.split("_")[0]
            task = "insert_HDMI" if prefix == "F2" else "pull_out_key"
            if spec["task"] != task:
                continue
            index = (
                int(spec["f6_local_unloading_witness"]["indices"][-1])
                if prefix == "F6"
                else int(spec["contact_peak"])
            )
            if prefix == "F1":
                index = len(clean) - 2
            slot = (
                spec["f6_local_unloading_witness"]["slot"]
                if prefix == "F6"
                else "right"
            )
            samples[prefix] = {
                "source_label": f"{task} raw{spec['raw_episode_id']} / obs{index}",
                "images": [rgb(clean[index], slot)],
                "dose_labels": [],
            }
            cases[prefix] = {
                "display_index": index,
                "task": task,
                "raw_episode": spec["raw_episode_id"],
            }
            manifests = [
                configured(operator, registry, arrays, spec, rest, long_freeze=extreme)
                for registry in registries
            ]
            if prefix == "T2":
                holds = [int(m.parameters["hold_duration_frames"]) for m in manifests]
                sequences[prefix] = {
                    "indices": freeze_indices(
                        manifests[0].start_index,
                        holds[0],
                        holds[1],
                        manifests[0].stop_index,
                    ),
                    "hold_frames": holds,
                }
            if prefix in {"T1", "T3"}:
                sequences[prefix] = {}
            records_by_mode = [("clean", clean)]
            for key, manifest in zip(("standard", "stress"), manifests):
                result = apply_fault(clean, manifest, rest)
                if not result.validation.passed:
                    raise RuntimeError(f"{prefix} {key}: {result.validation.failures}")
                cases[prefix][key] = {
                    "manifest": manifest.to_dict(),
                    "trace_sha256": result.trace_sha256,
                    "validation": asdict(result.validation),
                }
                if prefix in OPTICAL_PREFIXES:
                    delivered = rgb(result.records[index], slot)
                    samples[prefix]["images"].append(delivered)
                    samples[prefix]["dose_labels"].append(
                        dose_label(
                            prefix, dict(manifest.parameters), min(delivered.shape[:2])
                        )
                    )
                    cases[prefix][key]["display_mean_absolute_delta_u8"] = float(
                        np.mean(
                            np.abs(delivered.astype(float) - rgb(clean[index], slot))
                        )
                    )
                records_by_mode.append((key, result.records))
            for key, records in records_by_mode:
                detail = cases[prefix].setdefault(key, {})
                if prefix in {"A1", "A2"}:
                    present = [
                        r.observation.sensor("right").payload_present for r in records
                    ]
                    detail.update(
                        presence=present,
                        affected_count=sum(not value for value in present[90:177]),
                    )
                elif prefix == "C1":
                    correct = [
                        r.provenance_for("left").physical_source_id
                        == clean[i].provenance_for("left").physical_source_id
                        for i, r in enumerate(records)
                    ]
                    detail.update(
                        source_left=correct,
                        affected_count=sum(not value for value in correct[90:177]),
                    )
                elif prefix == "T2":
                    selected = sequences[prefix]["indices"]
                    sequences[prefix][key] = {
                        "images": [rgb(records[i], "right") for i in selected],
                        "sources": [
                            records[i].provenance_for("right").source_index
                            for i in selected
                        ],
                    }
                    detail["display_source_indices"] = sequences[prefix][key]["sources"]
                elif prefix in {"T1", "T3"}:
                    age = [age_ms(records[index], s) for s in ("left", "right")]
                    sequences[prefix][key] = {
                        "images": [rgb(records[index], s) for s in ("left", "right")],
                        "sources": [
                            records[index].provenance_for(s).source_index
                            for s in ("left", "right")
                        ],
                        "age_ms": age,
                    }
                    detail.update(
                        display_source_indices=sequences[prefix][key]["sources"],
                        display_age_ms=age,
                    )
            del records_by_mode, result
    if len(cases) != 14:
        raise ValueError("incomplete source/operator coverage")
    increased = stronger_checks(cases)
    acceptance = accepted_checks(cases, extreme=extreme)
    figures = {
        "01_F1-F4.png": strip(
            ("F1", "F2", "F3", "F4"),
            samples,
            f"高强度光学故障｜{labels[0]} → {labels[1]}",
            font,
            labels,
        ),
        "02_F5-F7-C2.png": strip(
            ("F5", "F6", "F7", "C2"),
            samples,
            f"形变、残影、饱和与错位｜{labels[0]} → {labels[1]}",
            font,
            labels,
        ),
        "03_A1-A2-C1.png": timelines(cases, font, labels),
        "04_T2_freeze.png": freeze_panel(sequences["T2"], font, labels),
        "05_T1-T3_delays.png": delay_panel(sequences, font, labels),
    }
    if before != production_hashes():
        raise RuntimeError("production changed during generation")
    output_root.mkdir(parents=True, exist_ok=False)
    for name, picture in figures.items():
        picture.save(output_root / name, dpi=(240, 240))
    report = {
        "schema_id": "robotactile.optical14_stress_gallery",
        "version": "1.1",
        "operator_count": len(cases),
        "production_instances_passed": 28,
        "per_operator_measured_effect_increased": increased,
        "per_operator_acceptance": acceptance,
        "effect_check_units": {
            "C2": "signature_validated_translation_l2_pixels_not_image_MAE",
            "A1_A2_C1": "affected_frame_count",
            "T2": "actual_held_frame_count",
            "T1_T3": "actual_max_source_age_ms",
            "F1_through_F7": "fixed_witness_mean_absolute_delta_u8",
        },
        "standard_registry": registries[0],
        "stress_registry": registries[1],
        "source_receipt_sha256": file_hash(receipt_path),
        "production_source_sha256": before,
        "script_sha256": file_hash(Path(__file__)),
        "painter_sha256": file_hash(
            Path(__file__).with_name("render_optical14_paper_panels.py")
        ),
        "font_sha256": file_hash(font),
        "cases": cases,
        "new_closed_loop_episodes": 0,
        "generated_image_content": False,
        "scope": "separately registered engineering stress; not physical damage calibration or a standard-S5 result",
        "output_sha256": {name: file_hash(output_root / name) for name in figures},
    }
    write_json(output_root / "stress_receipt.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "output-root", "font"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--extreme",
        action="store_true",
        help="Compare prior Stress with Extreme; do not change standard-S5 defaults",
    )
    args = parser.parse_args()
    render(
        args.source_root.resolve(),
        args.output_root.resolve(),
        args.font.resolve(),
        extreme=args.extreme,
    )


if __name__ == "__main__":
    main()
