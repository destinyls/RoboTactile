"""Merge two completed decision-stress reports without rescoring any episode."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.decision_stress.remaining import (
    AVAILABILITY,
    PREVIOUS,
    read,
    sha,
    write,
)
from scripts.decision_stress.report import aggregate, eligible

MODELS = ("n0", "act")
TASKS = ("grasp_classify", "lift_can")
SEEDS = (5, 3, 7)
NATIVE_ORDER = (
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
)
SUPPLEMENT_ORDER = ("availability_clean", *AVAILABILITY)


def category(row: dict[str, Any]) -> str:
    if not eligible(row):
        return "invalid_or_ineligible"
    if row.get("score_success") is True:
        return "success"
    if row.get("failure_code") == "execute_failed":
        return "execution_rejected"
    return str(row["terminal_status"])


def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if eligible(r)]
    success = sum(r["score_success"] is True for r in valid)
    return {
        "completed": len(rows),
        "valid_eligible": len(valid),
        "success": success,
        "failed": len(valid) - success,
        "invalid_or_ineligible": len(rows) - len(valid),
        "outcome_counts": dict(Counter(category(r) for r in rows)),
    }


def validate_reports(documents: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    expected = {
        (task, seed, "native", op)
        for task in TASKS
        for seed in SEEDS
        for op in NATIVE_ORDER
    } | {
        (task, seed, "zero_fill_v1", op)
        for task in TASKS
        for seed in SEEDS
        for op in SUPPLEMENT_ORDER
    }
    for model in MODELS:
        doc = documents[model]
        if (
            doc["model"] != model
            or doc["new_completed"] != 66
            or doc["new_planned"] != 66
        ):
            raise ValueError("both models must have completed all 66 new conditions")
        current = doc["rows"]
        keys = [(r["task"], r["seed"], r["protocol"], r["condition"]) for r in current]
        if len(keys) != 96 or len(set(keys)) != 96 or set(keys) != expected:
            raise ValueError(
                "missing, duplicate, or unexpected condition/seed/protocol"
            )
        if sum(r["reused"] is False for r in current) != 66:
            raise ValueError("old/new accounting mismatch")
        for row in current:
            if row["model"] != model or row["status"] != "terminal":
                raise ValueError(
                    "final report cannot contain a pending or wrong-model row"
                )
            if eligible(row) and type(row["score_success"]) is not bool:
                raise ValueError(
                    "eligible terminal must have an explicit boolean score"
                )
            if row["reused"] != (row["condition"] in PREVIOUS):
                raise ValueError("historical condition reuse mismatch")
        rows.extend(current)
    return rows


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: csv_value(r.get(k)) for k in fields} for r in rows)


def cell(stat: dict[str, Any]) -> str:
    total = stat["valid_eligible"]
    if not total:
        return f"N/A（无效{stat['invalid_or_ineligible']}）"
    suffix = (
        f"；无效{stat['invalid_or_ineligible']}"
        if stat["invalid_or_ineligible"]
        else ""
    )
    return (
        f"{stat['success_count']}/{total}（{100 * stat['success_rate']:.1f}%{suffix}）"
    )


def seed_cell(row: dict[str, Any]) -> str:
    label = category(row)
    if label == "invalid_or_ineligible":
        codes = row.get("validation_failure_codes") or [
            row.get("failure_code") or "unvalidated"
        ]
        label = "无效:" + ",".join(
            c for c in codes if c != "OPERATOR_SIGNATURE_MISMATCH"
        )
    elif label == "success":
        label = "成功"
    return f"{label}；推理{row['inference_count']}/受扰{row.get('post_onset_inference_count', '—')}"


def markdown(document: dict[str, Any]) -> str:
    rows, stats = document["rows"], document["statistics"]
    lookup = {(s["model"], s["task"], s["protocol"], s["condition"]): s for s in stats}
    lines = [
        "# N0-TWAM / ACT：两任务、三seed、14故障完整对比",
        "",
        "全部132个新增episode已结束，合并此前60个结果，共192条真实Isaac闭环记录。没有重跑有效episode，也没有修改结果或评分阈值。",
        "",
        "## 1. 完成与有效性",
        "",
        "成功仅统计 validation_passed=true 且 score_eligible=true 的记录。无效注入不充当模型失败；分母不同必须一起阅读。",
        "",
        "| 模型 | 范围 | 已结束 | 有效成功 | 有效失败 | 无效/不适用 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for stage, label in (("prior", "旧30"), ("new", "新增66"), ("all", "合计96")):
            value = document["counts"][model][stage]
            lines.append(
                f"| {model} | {label} | {value['completed']} | {value['success']} | {value['failed']} | {value['invalid_or_ineligible']} |"
            )
    lines += [
        "",
        "## 2. 冻结协议与边界",
        "",
        "- 官方N0-TWAM在A800执行；官方ACT Vision+tactile权重在A5000 Pro执行。任务为grasp_classify、lift_can；seed固定为5、3、7。",
        "- optical_decision_stress_v1、severity5；双模型逐条件fault SHA已核对一致。前1–8个源观测内按seed决定onset，持续至最多301观测。",
        "- N0保持cold12/warm24、原生60Hz EE控制；ACT保持chunk50 temporal aggregation、execute1 QPOS8。不把模型原生控制或硬件差异解释为纯架构因果。",
        "- native Clean/F1/F3/T2/T3为旧结果；8个新增native故障跨进程复用旧Clean。新旧初态哈希不一致时，SR差仅为描述性比较，不能声称严格同初态的噪声因果效应。",
        "- A1/A2属于独立zero_fill_v1协议：传感器交付证据仍为缺失，模型侧填原生尺寸uint8黑图。对照是本组新availability_clean，不是旧native Clean；这不是模型原生缺失输入能力。",
        "- 只有3个探索seed，且有无效记录与条件选择过程。不得作论文级显著性、真实故障频率或论文成功率复现声明。",
    ]
    for protocol, title, order in (
        ("native", "3. 原生协议：Clean与12个F/T/C故障", NATIVE_ORDER),
        ("zero_fill_v1", "4. 缺失输入补充协议", SUPPLEMENT_ORDER),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            "表中为成功数/有效数（SR）。N/A不等于0%成功率。",
            "",
            "| 条件 | N0 / grasp_classify | ACT / grasp_classify | N0 / lift_can | ACT / lift_can |",
            "|---|---:|---:|---:|---:|",
        ]
        for op in order:
            values = [cell(lookup[m, t, protocol, op]) for t in TASKS for m in MODELS]
            lines.append("| " + op + " | " + " | ".join(values) + " |")
    lines += [
        "",
        "## 5. A1同初态配对",
        "",
        "只在Clean和A1都有效且初态哈希相同的配对内计算；第三个seed若Clean本身失败，不充当噪声新造成的失败。",
        "",
        "| 模型/任务 | 有效配对数 | 配对Clean SR | 配对A1 SR | SR下降百分点 | Clean成功→A1失败 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for task in TASKS:
            paired = [
                r
                for r in rows
                if r["model"] == model
                and r["task"] == task
                and r["condition"] == "A1_stream_absence"
                and r.get("valid_pair") is True
            ]
            total = len(paired)
            clean = sum(r["paired_clean_success"] is True for r in paired)
            noisy = sum(r["score_success"] is True for r in paired)
            flips = sum(
                r["paired_clean_success"] is True and r["score_success"] is False
                for r in paired
            )
            drop = f"{100 * (clean - noisy) / total:.1f}" if total else "N/A"
            lines.append(
                f"| {model}/{task} | {total} | {clean}/{total} | {noisy}/{total} | {drop} | {flips}/{clean} |"
            )
    lines += [
        "",
        "## 6. 实际结论与不可推断的内容",
        "",
        "1. ACT/grasp_classify：旧F1/F3及新A1显示失败，而其他可验证的新故障仍成功。支持故障类型的影响具有选择性，不支持所有强噪声都会降低SR。",
        "2. N0/grasp_classify：A1下三个seed仍成功。N0/lift_can对缺失输入表现不同；必须结合本组Clean和终止原因，不能以旧Clean统一解释。",
        "3. ACT/lift_can：Clean本身0/3。C1/C2在seed7反而成功；保留这些有利扰动结果，不选择性省略，也不据此宣称噪声普遍有益。新旧native初态并非严格相同。",
        "4. N0若出现execute_failed且message为actions violate frozen action bounds，是预测动作被当前冻结执行边界拒绝。协议内计不成功，但应与普通timeout/early_stop区分，不称为Isaac基础设施崩溃。",
        "5. F6的NO_RELEASE_SAMPLE表示未观察到满足阈值的局部卸载见证，无法验证历史残余压痕；不能把此类execution_status=success当作有效F6成功。",
        "6. A2的A2_RESUME_MISSING来自旧规则要求最后计划丢帧后恢复一次，与本轮到episode终止的窗口不适配。本轮A2不能提供有效SR。已有episode_censored_v1可用于未来独立协议，但未用于本轮，未追溯改分。",
        "7. marker mask has no local shading support; sensor profile is inapplicable表示当前图像无法支持该光学场估计；相关F2/F3/F4/F6/F7行排除，不能当作模型失败。",
        "",
        "## 7. 失败/不适用明细",
        "",
        "| 模型 | 任务 | seed | 条件 | 类别 | 具体原因 |",
        "|---|---|---:|---|---|---|",
    ]
    for row in rows:
        if category(row) == "success":
            continue
        reason = "; ".join(row.get("validation_failure_codes") or [])
        runner = row.get("runner_failure") or {}
        if runner.get("message"):
            reason += ("; " if reason else "") + str(runner["message"])
        if not reason:
            reason = str(row["terminal_status"]) + "（未提供更细任务失败原因）"
        lines.append(
            f"| {row['model']} | {row['task']} | {row['seed']} | {row['condition']} | {category(row)} | {reason.replace('|', '/')} |"
        )
    lines += [
        "",
        "## 8. 逐seed结果与真实推理次数",
        "",
        "受扰次数为source_step_index ≥ onset的推理调用数，不是动作步数。不同模型动作表示与chunk长度不同，不能直接按调用次数比较计算效率。",
    ]
    for model in MODELS:
        for task in TASKS:
            lines += [
                "",
                f"### {model} / {task}",
                "",
                "| 条件 | seed5 | seed3 | seed7 |",
                "|---|---|---|---|",
            ]
            for op in (*NATIVE_ORDER, *SUPPLEMENT_ORDER):
                values = [
                    seed_cell(
                        next(
                            r
                            for r in rows
                            if r["model"] == model
                            and r["task"] == task
                            and r["seed"] == seed
                            and r["condition"] == op
                        )
                    )
                    for seed in SEEDS
                ]
                lines.append("| " + op + " | " + " | ".join(values) + " |")
    lines += [
        "",
        "## 9. 耗时与工件",
        "",
        "总墙钟包含每组模型/Isaac冷启动、仿真、推理与保存，不是GPU kernel基准；preview_v1最多64关键帧，不是完整视频时长。",
    ]
    for model in MODELS:
        meta = document["sources"][model]
        lines.append(
            f"- {model}：新增campaign墙钟 {meta['campaign_wall_s'] / 60:.1f} 分钟；原始汇总 SHA256 `{meta['results_sha256']}`。"
        )
    lines += [
        "",
        "- [192条逐episode CSV](episode_metrics.csv)",
        "- [64组条件统计CSV](condition_metrics.csv)",
        "- [完整JSON及来源](combined_results.json)",
        "- [分析代码/输出SHA收据](analysis_receipt.json)",
        "",
        "所有原始工件路径、terminal文件SHA、验证字段和错误信息均保留于CSV/JSON；不包含虚构/生成模型数据。",
    ]
    return "\n".join(lines) + "\n"


def combine(root: Path, output: Path, n0_sha: str, act_sha: str) -> None:
    if output.exists():
        raise FileExistsError(output)
    documents, sources, plans = {}, {}, {}
    for model, expected_sha in zip(MODELS, (n0_sha, act_sha)):
        result_path = root / model / "results.json"
        if sha(result_path) != expected_sha:
            raise ValueError("downloaded result SHA differs from remote final artifact")
        documents[model] = read(result_path)
        finish = read(root / model / "finished.json")
        start = read(root / model / "supervisor.json")
        plans[model] = read(root / model / "plan.json")
        sources[model] = {
            "results_path": str(result_path),
            "results_sha256": expected_sha,
            "plan_sha256": sha(root / model / "plan.json"),
            "finished_sha256": sha(root / model / "finished.json"),
            "supervisor": start,
            "completed_unix": finish["completed_unix"],
            "campaign_wall_s": finish["completed_unix"] - start["started_unix"],
        }
    rows = validate_reports(documents)
    for left, right in zip(plans["n0"]["groups"], plans["act"]["groups"]):
        if (left["task"], left["seed"]) != (right["task"], right["seed"]):
            raise ValueError("cross-model group order mismatch")
        if len(left["cells"]) != 11 or len(right["cells"]) != 11:
            raise ValueError("unexpected prepared group size")
        for a, b in zip(left["cells"], right["cells"]):
            if (a["condition"], a["fault_sha256"]) != (
                b["condition"],
                b["fault_sha256"],
            ):
                raise ValueError("cross-model frozen fault mismatch")
    stats = aggregate(rows)
    for stat in stats:
        base = next(
            s
            for s in stats
            if s["model"] == stat["model"]
            and s["task"] == stat["task"]
            and s["protocol"] == stat["protocol"]
            and s["condition"]
            == ("clean" if stat["protocol"] == "native" else "availability_clean")
        )
        stat["descriptive_sr_drop_pp"] = (
            100 * (base["success_rate"] - stat["success_rate"])
            if base["success_rate"] is not None and stat["success_rate"] is not None
            else None
        )
    summaries = {}
    for model in MODELS:
        selected = [r for r in rows if r["model"] == model]
        summaries[model] = {
            "all": counts(selected),
            "prior": counts([r for r in selected if r["reused"]]),
            "new": counts([r for r in selected if not r["reused"]]),
        }
    document = {
        "schema": "decision_stress_two_model_final_v1",
        "evidence": "CLOSED-LOOP diagnostic",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "counts": summaries,
        "rows": rows,
        "statistics": stats,
        "boundary": "3 exploratory seeds; native/cross-process contrasts not strict causal pairs; missing inputs use separately matched zero-fill protocol; no rescoring",
    }
    output.mkdir(parents=True, exist_ok=False)
    write(output / "combined_results.json", document)
    first_fields = [
        "model",
        "task",
        "seed",
        "protocol",
        "condition",
        "reused",
        "validation_passed",
        "score_eligible",
        "score_success",
        "terminal_status",
        "failure_code",
        "failure_stage",
        "validation_failure_codes",
        "runner_failure",
        "inference_count",
        "post_onset_inference_count",
        "observation_count",
        "onset_index",
        "initial_state_match",
        "valid_pair",
        "paired_clean_success",
        "episode_wall_s",
        "artifact",
        "terminal_file_sha256",
    ]
    extra_fields = sorted(set().union(*(r.keys() for r in rows)) - set(first_fields))
    write_csv(output / "episode_metrics.csv", rows, [*first_fields, *extra_fields])
    write_csv(output / "condition_metrics.csv", stats, list(stats[0]))
    with (output / "README.md").open("x", encoding="utf-8") as stream:
        stream.write(markdown(document))
    write(
        output / "analysis_receipt.json",
        {
            "source_results": sources,
            "analyzer_path": str(Path(__file__).resolve()),
            "analyzer_sha256": sha(Path(__file__)),
            "row_count": len(rows),
            "condition_count": len(stats),
            "output_sha256": {
                p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()
            },
            "raw_results_modified": False,
            "experimental_protocol_modified": False,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n0-sha", required=True)
    p.add_argument("--act-sha", required=True)
    a = p.parse_args()
    combine(a.root.resolve(strict=True), a.output.absolute(), a.n0_sha, a.act_sha)


if __name__ == "__main__":
    main()
