"""Chinese, source-bound mechanism figures from production tactile deliveries.

No generative images, photometric enhancement or display-specific corruptions.
This selects interpretable mechanisms, not a claim that all 14 are paper-ready.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robotactile_benchmark.contracts import Array, EvaluationRecord
from robotactile_benchmark.optical.fields import spatial_weight
from robotactile_benchmark.runtime import apply_fault
from robotactile_benchmark.visualization.optical14 import (
    build_records,
    file_hash,
    load_source,
    make_manifest,
    production_hashes,
    write_json,
)

INK = "#182d44"
MUTED = "#506277"
BLUE = "#236aa4"
ORANGE = "#b85b12"
GREEN = "#247958"


class Painter:
    def __init__(self, width: int, height: int, font_path: Path) -> None:
        self.image = Image.new("RGB", (width, height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.font_path = font_path

    def text(
        self, xy: tuple[int, int], value: str, size: int = 24, color: str = INK
    ) -> None:
        font = ImageFont.truetype(str(self.font_path), size)
        box = self.draw.multiline_textbbox(xy, value, font=font, spacing=8)
        if box[2] > self.image.width - 12 or box[3] > self.image.height - 6:
            raise ValueError(f"caption exceeds canvas: {value}")
        self.draw.multiline_text(xy, value, font=font, fill=color, spacing=8)

    def frame(self, rgb: Array, xy: tuple[int, int], scale: int = 1) -> None:
        if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8:
            raise ValueError("requires native 320x240 RGB")
        self.image.paste(
            Image.fromarray(rgb).resize(
                (320 * scale, 240 * scale), Image.Resampling.NEAREST
            ),
            xy,
        )

    def title(self, heading: str, subheading: str) -> None:
        self.text((28, 18), heading, 33)
        self.text((28, 66), subheading, 21, MUTED)
        self.draw.line((28, 106, self.image.width - 28, 106), fill="#d9e1eb", width=2)


def payload(record: EvaluationRecord, slot: str = "right") -> Array:
    value = record.observation.sensor(slot).payload
    if value is None:
        raise ValueError("mechanism panel requires present recorded payload")
    return value


def freeze_witness(
    clean: tuple[EvaluationRecord, ...],
    delivered: tuple[EvaluationRecord, ...],
    indices: list[int],
) -> dict[str, Any]:
    maps = [delivered[i].provenance_for("right").source_index for i in indices]
    first, held_a, held_b, resumed = indices
    checks = {
        "before_is_current": maps[0] == first,
        "held_source_unchanged": maps[1] == maps[2] == first,
        "held_payload_exact": np.array_equal(
            payload(delivered[held_a]), payload(delivered[held_b])
        )
        and np.array_equal(payload(delivered[held_a]), payload(clean[first])),
        "clean_actually_changes": not np.array_equal(
            payload(clean[held_a]), payload(clean[held_b])
        ),
        "resumes_current_source": maps[3] == resumed
        and np.array_equal(payload(delivered[resumed]), payload(clean[resumed])),
    }
    if not all(checks.values()):
        raise ValueError(f"freeze has no intelligible witness: {checks}")
    return {"delivery_indices": indices, "source_indices": maps, "checks": checks}


def freeze_panel(
    clean: tuple[EvaluationRecord, ...],
    delivered: tuple[EvaluationRecord, ...],
    manifest: Any,
    font_path: Path,
) -> tuple[Image.Image, dict[str, Any]]:
    start = manifest.start_index
    duration = manifest.parameters["hold_duration_frames"]
    indices = [start - 1, start, start + duration - 1, start + duration]
    witness = freeze_witness(clean, delivered, indices)
    page = Painter(1440, 845, font_path)
    page.title(
        "T2｜末帧冻结：当前帧在变化，输出仍停在同一旧帧",
        "同一 UniVTAC 记录序列 · 上排 Clean，下排正式算子输出 · 帧内容与源时间均经过核验",
    )
    page.text((28, 240), "Clean", 24, BLUE)
    page.text((28, 274), "当前帧", 21, BLUE)
    page.text((28, 538), "Faulted", 23, ORANGE)
    page.text((28, 572), "收到的帧", 21, ORANGE)
    stages = ["冻结前", "冻结开始", "冻结结束前", "恢复当前输入"]
    for column, index in enumerate(indices):
        x = 118 + column * 326
        phase_color = ORANGE if column in (1, 2) else GREEN
        page.text((x, 124), stages[column], 23, phase_color)
        now = clean[index].provenance_for("right").source_time_s
        page.text((x, 162), f"观测 {index} / t={now:.3f} s", 19)
        page.frame(payload(clean[index]), (x, 198))
        page.frame(payload(delivered[index]), (x, 491))
        source = delivered[index].provenance_for("right")
        page.text(
            (x, 451),
            f"来自帧 {source.source_index} / {source.source_time_s:.3f} s",
            20,
            phase_color,
        )
    page.text(
        (28, 764),
        f"冻结期间：连续 {duration} 个观测复用旧 payload 和源时间；delivery 时间照常推进。随后恢复。",
        23,
    )
    page.text(
        (28, 804),
        "60 Hz 观测；图中只选冻结前、首帧、末帧、恢复帧。全部为记录数据的 production delivery，无生成图像。",
        19,
        MUTED,
    )
    return page.image, witness


def spatial_panel(
    clean: tuple[EvaluationRecord, ...],
    f4: Any,
    c2: Any,
    f4_manifest: Any,
    index: int,
    font_path: Path,
) -> Image.Image:
    page = Painter(1400, 1420, font_path)
    page.title(
        "空间故障｜局部光学失响应，与全局坐标错位不是一回事",
        "同一个 Clean：pull_out_key / raw55 · 观测121 · t=2.017 s · 原色、2×最近邻显示",
    )
    page.text((28, 125), "A   F4 局部光学失响应 · S3", 29)
    page.text((40, 175), "Clean：局部圆形接触特征存在", 24, BLUE)
    page.text((720, 175), "Faulted：局部圆斑的光学特征消失", 24, ORANGE)
    page.frame(payload(clean[index]), (40, 213), 2)
    page.frame(payload(f4.records[index]), (720, 213), 2)
    support = (
        spatial_weight(
            payload(clean[index]).shape, f4_manifest.parameters, dead_patch=True
        )
        > 0
    )
    yy, xx = np.nonzero(support)
    for origin in (40, 720):
        page.draw.rectangle(
            (
                origin + 2 * int(xx.min()),
                213 + 2 * int(yy.min()),
                origin + 2 * int(xx.max()),
                213 + 2 * int(yy.max()),
            ),
            outline="#ffdc62",
            width=2,
        )
    page.text(
        (40, 707),
        "黄框只标注相同 ROI；mask 外不变。保留当前 marker 及2 px保护邻域，因而可能留有点周色边。",
        22,
        MUTED,
    )
    page.text((28, 763), "B   C2 图像注册错位 · S5 = 向右30 px", 29)
    page.text((40, 812), "Clean：原注册位置", 24, BLUE)
    page.text((720, 812), "Faulted：接触纹理与 marker 一起平移", 24, ORANGE)
    page.frame(payload(clean[index]), (40, 850), 2)
    page.frame(payload(c2.records[index]), (720, 850), 2)
    page.text(
        (40, 1343),
        "F4：仅隔离光学响应，不代表整个触觉失效。C2：像素坐标注册失配，不代表物理重装传感器。",
        22,
        MUTED,
    )
    page.text(
        (40, 1380),
        "两行 severity 不同，只用于解释机制，不构成故障强弱排名；所有注释均在显示副本上。",
        20,
        MUTED,
    )
    return page.image


def delays_panel(
    clean: tuple[EvaluationRecord, ...],
    common: Any,
    skew: Any,
    index: int,
    font_path: Path,
) -> tuple[Image.Image, dict[str, Any]]:
    page = Painter(1110, 835, font_path)
    page.title(
        "T1 与 T3｜同为200 ms延迟，左右同步关系不同",
        "固定同一到达时刻：观测121 / t=2.017 s；每幅图标注实际源帧与图像年龄",
    )
    entries = [
        ("Clean", clean[index], BLUE),
        ("T1 共同延迟 · S3", common.records[index], ORANGE),
        ("T3 单侧延迟 · S3", skew.records[index], ORANGE),
    ]
    maps = {}
    for col, (title, record, color) in enumerate(entries):
        x = 42 + col * 358
        page.text((x, 126), title, 25, color)
        ages = []
        source_indices = []
        for row, slot in enumerate(("left", "right")):
            p = record.provenance_for(slot)
            delivery_time = record.observation.sensor(slot).delivery_time_s
            if p.source_time_s is None or delivery_time is None:
                raise ValueError("delay panel needs actual source/delivery clocks")
            age = delivery_time - p.source_time_s
            ages.append(age)
            source_indices.append(p.source_index)
            y = 210 + row * 288
            page.text(
                (x, y - 36),
                f"{'左' if slot == 'left' else '右'}：源帧 {p.source_index} / 年龄 {age * 1000:.0f} ms",
                20,
                color,
            )
            page.frame(payload(record, slot), (x, y))
        maps[title] = {"source_indices": source_indices, "ages_s": ages}
        page.text(
            (x, 753),
            f"左右源时间差：{abs(ages[1] - ages[0]) * 1000:.0f} ms",
            22,
            GREEN if abs(ages[1] - ages[0]) < 1e-9 else ORANGE,
        )
    page.text(
        (28, 801),
        "T1：两侧一起落后，仍彼此同步。T3：左侧当前、右侧落后，破坏跨传感器同步。",
        21,
        MUTED,
    )
    return page.image, maps


def render(
    source_root: Path, gallery_root: Path, output_root: Path, font_path: Path
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError("paper panel output must be new")
    receipt_path = source_root / "source_receipt.json"
    if file_hash(receipt_path) != file_hash(gallery_root / "source_receipt.json"):
        raise ValueError("paper panel sources must match the frozen gallery")
    gallery = json.loads((gallery_root / "gallery_receipt.json").read_text())
    before = production_hashes()
    if gallery["production_source_sha256"] != before:
        raise ValueError("gallery does not match current production")
    for name, expected in gallery["output_sha256"].items():
        path = (gallery_root / name).resolve(strict=True)
        if (
            not path.is_relative_to(gallery_root.resolve())
            or file_hash(path) != expected
        ):
            raise ValueError("gallery output integrity mismatch")
    source = json.loads(receipt_path.read_text())
    spec = next(s for s in source["sources"] if s["task"] == "pull_out_key")
    arrays = load_source(source_root, spec)
    records, rest = build_records(arrays, spec, file_hash(receipt_path))
    operators = {
        "F4": ("F4_local_nonresponsive_patch", 3),
        "C2": ("C2_frame_misregistration", 5),
        "T1": ("T1_fixed_source_delay", 3),
        "T2": ("T2_held_last_freeze", 3),
        "T3": ("T3_inter_sensor_skew", 3),
    }
    results, manifests, checks = {}, {}, {}
    for prefix, (operator, severity) in operators.items():
        manifest = make_manifest(operator, severity, arrays, spec, rest)
        result = apply_fault(records, manifest, rest_references=rest)
        if not result.validation.passed:
            raise ValueError(
                f"production delivery failed: {result.validation.failures}"
            )
        previous = json.loads((gallery_root / f"{prefix}_provenance.json").read_text())
        if severity == 3 and result.trace_sha256 != previous["trace_sha256"]:
            raise ValueError("paper figure is not the same production delivery")
        results[prefix], manifests[prefix] = result, manifest
        checks[prefix] = {
            "manifest": manifest.to_dict(),
            "trace_sha256": result.trace_sha256,
            "validation": asdict(result.validation),
        }
    peak = int(spec["contact_peak"])
    # These source-bound explanatory captions intentionally name a specific witness.
    if peak != 121 or spec["raw_episode_id"] != 55:
        raise ValueError("this paper layout is bound to raw55 observation121")
    spatial = spatial_panel(
        records, results["F4"], results["C2"], manifests["F4"], peak, font_path
    )
    freeze, freeze_receipt = freeze_panel(
        records, results["T2"].records, manifests["T2"], font_path
    )
    delays, delay_receipt = delays_panel(
        records, results["T1"], results["T3"], peak, font_path
    )
    if before != production_hashes():
        raise RuntimeError("production changed during generation")
    output_root.mkdir(parents=True, exist_ok=False)
    for name, figure in (
        ("01_spatial.png", spatial),
        ("02_freeze.png", freeze),
        ("03_delays.png", delays),
    ):
        figure.save(output_root / name, dpi=(240, 240))
    receipt = {
        "schema_id": "robotactile.optical14_paper_panels",
        "version": "1.0",
        "script_sha256": file_hash(Path(__file__)),
        "font_sha256": file_hash(font_path),
        "source_receipt_sha256": file_hash(receipt_path),
        "production_source_sha256": before,
        "displayed_operators": checks,
        "freeze_witness": freeze_receipt,
        "delay_witness": delay_receipt,
        "generated_image_content": False,
        "scope": "recorded simulator inputs with production observation-level faults; no material calibration or closed-loop result",
        "not_accepted_as_standalone_main_figure": [
            "F2: weak naked-eye contrast",
            "F6: no recorded complete release/recovery",
        ],
        "photometric_enhancement": False,
        "output_sha256": {p.name: file_hash(p) for p in output_root.iterdir()},
    }
    write_json(output_root / "panel_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "gallery-root", "output-root", "font"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    render(
        args.source_root.resolve(),
        args.gallery_root.resolve(),
        args.output_root.resolve(),
        args.font.resolve(),
    )


if __name__ == "__main__":
    main()
