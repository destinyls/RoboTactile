"""Render fixed decision-stress artifacts and index their source-bound videos."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
from pathlib import Path
from typing import Any

from robotactile_benchmark.visualization import export_live_artifact_visualization

CONDITIONS = (
    "clean",
    "F1_global_response_drift",
    "F2_spatial_sensitivity_loss",
    "F3_persistent_surface_artifact",
    "F4_local_nonresponsive_patch",
    "F5_contact_shape_distortion",
    "F6_history_residual_imprint",
    "F7_high_load_saturation",
    "T1_fixed_source_delay",
    "T2_held_last_freeze",
    "T3_inter_sensor_skew",
    "C1_sensor_identity_misrouting",
    "C2_frame_misregistration",
    "availability_clean",
    "A1_stream_absence",
    "A2_frame_erasure",
)
TASKS = ("grasp_classify", "lift_can")
MODELS = ("n0", "act")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def _verify_video(directory: Path, artifact: Path) -> dict[str, Any]:
    receipt = _read(directory / "visualization_receipt.json")
    members = receipt.get("members")
    if not isinstance(members, dict):
        raise ValueError(f"missing visualization members: {directory}")
    for name in ("preview.mp4", "preview.png"):
        expected = members.get(name)
        if not isinstance(expected, str) or _sha256(directory / name) != expected:
            raise ValueError(f"visualization member hash mismatch: {directory / name}")
    if receipt.get("source_live_artifact_root_sha256") != _sha256(
        artifact / "root_receipt.json"
    ):
        raise ValueError(f"visualization source hash mismatch: {directory}")
    if receipt.get("fps") != 20 or receipt.get("stride") != 1:
        raise ValueError(f"unexpected video cadence: {directory}")
    return receipt


def render(results_path: Path, output: Path, seed: int) -> None:
    results = _read(results_path)
    model = results["model"]
    if model not in MODELS:
        raise ValueError(f"unexpected model: {model}")
    rows = {
        (row["task"], row["condition"]): row
        for row in results["rows"]
        if row["seed"] == seed
    }
    if len(rows) != len(TASKS) * len(CONDITIONS):
        raise ValueError("selected seed has incomplete or duplicate conditions")
    rendered: list[dict[str, Any]] = []
    for task in TASKS:
        for condition in CONDITIONS:
            row = rows[task, condition]
            artifact = Path(row["artifact"]).resolve(strict=True)
            target = output / model / task / f"seed-{seed}" / condition
            if target.exists() or target.is_symlink():
                receipt = _verify_video(target, artifact)
            else:
                export_live_artifact_visualization(
                    artifact, target, fps=20, stride=1, video=True
                )
                receipt = _verify_video(target, artifact)
            entry = {
                "model": model,
                "task": task,
                "seed": seed,
                "condition": condition,
                "protocol": row["protocol"],
                "validation_passed": row.get("validation_passed"),
                "score_eligible": row.get("score_eligible"),
                "score_success": row.get("score_success"),
                "terminal_status": row.get("terminal_status"),
                "failure_code": row.get("failure_code"),
                "validation_failure_codes": row.get("validation_failure_codes"),
                "inference_count": row.get("inference_count"),
                "post_onset_inference_count": row.get("post_onset_inference_count"),
                "onset_index": row.get("onset_index"),
                "artifact": str(artifact),
                "source_root_sha256": receipt["source_live_artifact_root_sha256"],
                "video": str(target.relative_to(output) / "preview.mp4"),
                "poster": str(target.relative_to(output) / "preview.png"),
                "receipt": str(
                    target.relative_to(output) / "visualization_receipt.json"
                ),
                "frame_count": receipt["rendered_frame_count"],
                "fps": receipt["fps"],
                "video_sha256": receipt["members"]["preview.mp4"],
            }
            rendered.append(entry)
            print(
                f"{model}/{task}/{condition}: {entry['frame_count']} frames", flush=True
            )
    _write_new(
        output / model / f"seed-{seed}-manifest.json",
        {
            "schema": "decision_stress_visualization_manifest_v1",
            "model": model,
            "seed": seed,
            "source_results_sha256": _sha256(results_path),
            "entries": rendered,
        },
    )


def _outcome(entry: dict[str, Any]) -> str:
    if entry["validation_passed"] is not True or entry["score_eligible"] is not True:
        codes = entry.get("validation_failure_codes") or []
        return "无效：" + (", ".join(codes) if codes else "验证未通过")
    if entry["score_success"] is True:
        return "成功"
    if entry["failure_code"] == "execute_failed":
        return "失败：动作越界拒绝执行"
    return "失败：" + str(entry["terminal_status"])


def _card(entry: dict[str, Any]) -> str:
    esc = html.escape
    model = "N0-TWAM" if entry["model"] == "n0" else "ACT"
    video = esc(entry["video"], quote=True)
    poster = esc(entry["poster"], quote=True)
    receipt = esc(entry["receipt"], quote=True)
    outcome = esc(_outcome(entry))
    onset = entry.get("onset_index")
    calls = entry.get("post_onset_inference_count")
    timing = (
        f"注入起点：{onset}；注入后推理：{calls} 次"
        if onset is not None
        else "Clean 对照"
    )
    return (
        f"<article><h3>{model} · {esc(entry['task'])}</h3>"
        f'<p class="status">{outcome}</p>'
        f'<video controls preload="none" poster="{poster}" src="{video}" '
        'onloadedmetadata="this.playbackRate=0.25"></video>'
        f"<p>{timing}；预览 {entry['frame_count']} 帧 / 20 FPS</p>"
        f'<p><a href="{video}">下载 MP4</a> · <a href="{receipt}">来源收据</a></p>'
        "</article>"
    )


def _statistics_table(report: Path) -> str:
    document = _read(report)
    stats = {
        (entry["model"], entry["task"], entry["condition"]): entry
        for entry in document["statistics"]
    }
    if len(stats) != len(MODELS) * len(TASKS) * len(CONDITIONS):
        raise ValueError("incomplete condition statistics")
    rows = []
    for condition in CONDITIONS:
        cells = []
        for model in MODELS:
            for task in TASKS:
                stat = stats[model, task, condition]
                label = (
                    f"{stat['success_count']}/{stat['valid_eligible']}"
                    if stat["valid_eligible"]
                    else "N/A"
                )
                if stat["invalid_or_ineligible"]:
                    label += f"（无效 {stat['invalid_or_ineligible']}）"
                cells.append(f"<td>{label}</td>")
        rows.append(
            f'<tr><th><a href="#{condition}">{condition}</a></th>'
            + "".join(cells)
            + "</tr>"
        )
    return (
        "<table><thead><tr><th>条件</th><th>N0 / grasp</th>"
        "<th>N0 / lift</th><th>ACT / grasp</th><th>ACT / lift</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def index(root: Path, seed: int, report: Path) -> None:
    entries: dict[tuple[str, str, str], dict[str, Any]] = {}
    for model in MODELS:
        manifest = _read(root / model / f"seed-{seed}-manifest.json")
        if manifest["model"] != model or manifest["seed"] != seed:
            raise ValueError(f"wrong visualization manifest: {model}")
        if manifest["source_results_sha256"] != _sha256(
            root.parent / model / "results.json"
        ):
            raise ValueError(f"result provenance mismatch: {model}")
        for entry in manifest["entries"]:
            key = (entry["model"], entry["task"], entry["condition"])
            if key in entries:
                raise ValueError(f"duplicate video entry: {key}")
            directory = root / Path(entry["video"]).parent
            receipt = _read(root / entry["receipt"])
            if _sha256(directory / "preview.mp4") != entry["video_sha256"]:
                raise ValueError(f"downloaded MP4 hash mismatch: {key}")
            if receipt["members"]["preview.mp4"] != entry["video_sha256"]:
                raise ValueError(f"downloaded receipt mismatch: {key}")
            entries[key] = entry
    if len(entries) != len(MODELS) * len(TASKS) * len(CONDITIONS):
        raise ValueError("incomplete video matrix")
    sections = []
    for condition in CONDITIONS:
        cards = "".join(
            _card(entries[model, task, condition]) for model in MODELS for task in TASKS
        )
        sections.append(
            f'<section id="{condition}"><h2>{html.escape(condition)}</h2>'
            f'<div class="grid">{cards}</div></section>'
        )
    links = " · ".join(
        f'<a href="#{condition}">{condition.split("_")[0]}</a>'
        for condition in CONDITIONS
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoboTactile：N0-TWAM / ACT 故障闭环视频</title>
<style>body{{font:16px system-ui,sans-serif;background:#10141b;color:#edf1f6;margin:0 auto;max-width:1520px;padding:24px;line-height:1.5}}a{{color:#8bc5ff}}nav{{line-height:2.2}}section{{margin:36px 0;padding-top:12px;border-top:1px solid #3a4350}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:14px}}article{{background:#1c2430;border:1px solid #384352;border-radius:10px;padding:13px}}h3{{margin:0}}.status{{font-weight:bold}}video{{width:100%;background:#080b0f}}p{{margin:7px 0}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{border-bottom:1px solid #384352;padding:7px;text-align:left}}tr:hover{{background:#1c2430}}</style></head>
<body><h1>RoboTactile：N0-TWAM / ACT 故障闭环视频</h1>
<p>任务：grasp_classify、lift_can；展示 seed {seed}。每种条件均显示两模型 × 两任务的真实闭环预览，实验统计基于 seed 5、3、7。</p>
<p>每段最多取 preview_v1 保存的 64 个真实关键帧；MP4 编码为 20 FPS，页面默认以 0.25 倍速播放以便观察。播放时长不是完整 episode 时长。画面包含 RGB、左右 Clean tactile、Delivered tactile 和仅用于显示的差异图。A1/A2 的缺失交付显示为 PAYLOAD ABSENT；模型适配层的黑图填充发生在交付之后。无效条件仍可观察画面，但不参与成功率。</p>
<p>下表为三个 seed 的成功数/有效数。点击条件名称跳转到对应四段视频；N/A 表示没有有效样本。<a href="../final/README.md">查看完整统计与逐 seed 原因</a>。</p>
{_statistics_table(report)}<nav>{links}</nav>{"".join(sections)}</body></html>"""
    with (root / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(page)


def verify(root: Path, seed: int) -> None:
    counts: list[int] = []
    for model in MODELS:
        manifest = _read(root / model / f"seed-{seed}-manifest.json")
        if len(manifest["entries"]) != len(TASKS) * len(CONDITIONS):
            raise ValueError(f"incomplete video manifest: {model}")
        for entry in manifest["entries"]:
            video = root / entry["video"]
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_frames,r_frame_rate,duration",
                    "-of",
                    "json",
                    str(video),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            if (
                int(stream["nb_frames"]) != entry["frame_count"]
                or stream["r_frame_rate"] != "20/1"
            ):
                raise ValueError(f"MP4 frame count or FPS mismatch: {video}")
            counts.append(entry["frame_count"])
    print(
        json.dumps(
            {
                "video_count": len(counts),
                "frame_count_total": sum(counts),
                "min_frames": min(counts),
                "max_frames": max(counts),
                "fps": 20,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    render_command = commands.add_parser("render")
    render_command.add_argument("--results", type=Path, required=True)
    render_command.add_argument("--output", type=Path, required=True)
    render_command.add_argument("--seed", type=int, default=5)
    index_command = commands.add_parser("index")
    index_command.add_argument("--root", type=Path, required=True)
    index_command.add_argument("--seed", type=int, default=5)
    index_command.add_argument("--report", type=Path, required=True)
    verify_command = commands.add_parser("verify")
    verify_command.add_argument("--root", type=Path, required=True)
    verify_command.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    if args.command == "render":
        render(args.results, args.output, args.seed)
    elif args.command == "index":
        index(args.root, args.seed, args.report)
    else:
        verify(args.root, args.seed)


if __name__ == "__main__":
    main()
