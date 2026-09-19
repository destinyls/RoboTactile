"""Fixed-terminal statistics for old and complementary decision-stress runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.decision_stress.remaining import read, sha, write


def eligible(result: dict[str, Any]) -> bool:
    return (
        result.get("validation_passed") is True and result.get("score_eligible") is True
    )


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"], row["task"], row["protocol"], row["condition"]].append(
            row
        )
    output = []
    for (model, task, protocol, condition), cells in sorted(grouped.items()):
        terminal = [r for r in cells if r["status"] == "terminal"]
        valid = [r for r in terminal if eligible(r)]
        paired = [r for r in valid if r.get("valid_pair") is True]
        vulnerable = [r for r in paired if r["paired_clean_success"] is True]
        output.append(
            {
                "model": model,
                "task": task,
                "protocol": protocol,
                "condition": condition,
                "planned": len(cells),
                "completed": len(terminal),
                "valid_eligible": len(valid),
                "pending": len(cells) - len(terminal),
                "invalid_or_ineligible": len(terminal) - len(valid),
                "success_count": sum(r["score_success"] is True for r in valid),
                "success_rate": sum(r["score_success"] is True for r in valid)
                / len(valid)
                if valid
                else None,
                "valid_pair_count": len(paired),
                "paired_clean_success_to_failure": sum(
                    r["score_success"] is False for r in vulnerable
                ),
                "paired_clean_success_count": len(vulnerable),
                "paired_sr_drop": (
                    sum(
                        float(r["paired_clean_success"]) - float(r["score_success"])
                        for r in paired
                    )
                    / len(paired)
                )
                if paired
                else None,
                "initial_state_mismatch_count": sum(
                    r.get("initial_state_match") is False for r in terminal
                ),
                "seeds": [
                    {
                        k: r.get(k)
                        for k in (
                            "seed",
                            "status",
                            "score_success",
                            "validation_passed",
                            "terminal_status",
                            "failure_code",
                            "inference_count",
                            "post_onset_inference_count",
                        )
                    }
                    for r in cells
                ],
            }
        )
    return output


def report(output: Path) -> dict[str, Any]:
    plan = read(output / "plan.json")
    rows = []
    for group in plan["groups"]:
        native_clean = (
            Path(group["prior_cells"][0]["artifact"]) / "terminal_result.json"
        )
        availability_clean = (
            Path(group["path"]) / "artifacts/availability_clean/terminal_result.json"
        )
        for old, cells in ((True, group["prior_cells"]), (False, group["cells"])):
            for cell in cells:
                artifact = Path(cell["artifact"])
                terminal = artifact / "terminal_result.json"
                protocol = "native" if old else cell["protocol"]
                row: dict[str, Any] = {
                    "model": plan["model"],
                    "task": group["task"],
                    "seed": group["seed"],
                    "condition": cell["condition"],
                    "protocol": protocol,
                    "reused": old,
                    "artifact": str(artifact),
                    "status": "pending",
                }
                if terminal.exists():
                    if old and sha(terminal) != cell["terminal_file_sha256"]:
                        raise ValueError("historical terminal result changed")
                    result = read(terminal)
                    row.update(
                        result, status="terminal", terminal_file_sha256=sha(terminal)
                    )
                    actions = read(artifact / "action_trace.json")["entries"]
                    row["inference_count"] = len(actions)
                    fault = read(artifact / "fault_manifest.json")
                    if fault:
                        row["onset_index"] = fault["start_index"]
                        row["post_onset_inference_count"] = sum(
                            e["source_step_index"] >= fault["start_index"]
                            for e in actions
                        )
                        clean_path = (
                            native_clean if protocol == "native" else availability_clean
                        )
                        if clean_path.exists():
                            clean = read(clean_path)
                            match = (
                                result["initial_state_sha256"] is not None
                                and result["initial_state_sha256"]
                                == clean["initial_state_sha256"]
                            )
                            row.update(
                                initial_state_match=match,
                                paired_clean_success=clean["score_success"],
                                valid_pair=match
                                and eligible(clean)
                                and eligible(result),
                            )
                    trace = read(artifact / "transition_trace.json")
                    row["runner_failure"] = trace.get("initial_diagnostics", {}).get(
                        "runner_failure"
                    )
                    receipt = (
                        Path(group["path"]) / "receipts" / f"{cell['condition']}.json"
                    )
                    if not old and receipt.exists():
                        timing = read(receipt)
                        row["episode_wall_s"] = (
                            timing["completed_unix"] - timing["started_unix"]
                        )
                rows.append(row)
    return {
        "schema": "remaining_decision_stress_statistics_v1",
        "model": plan["model"],
        "new_completed": sum(
            r["status"] == "terminal" and not r["reused"] for r in rows
        ),
        "new_planned": plan["planned_episode_count"],
        "rows": rows,
        "statistics": aggregate(rows),
        "boundary": plan["boundary"],
        "prior": plan["prior"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    document = report(a.campaign)
    if a.output is None:
        print(json.dumps(document, indent=2, allow_nan=False))
    else:
        write(a.output, document)


if __name__ == "__main__":
    main()
