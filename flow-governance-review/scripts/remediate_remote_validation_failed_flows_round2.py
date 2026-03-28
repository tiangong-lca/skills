#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
FLOW_GOVERNANCE_ROOT = SCRIPT_DIR.parent
LCA_SKILLS_ROOT = FLOW_GOVERNANCE_ROOT.parent
PROCESS_AUTOMATED_BUILDER_ROOT = LCA_SKILLS_ROOT / "process-automated-builder"
FLOW_PROCESSING_ARTIFACT_ROOT = FLOW_GOVERNANCE_ROOT / "assets" / "artifacts" / "flow-processing"
FLOW_PROCESSING_REMEDIATION_DIR = FLOW_PROCESSING_ARTIFACT_ROOT / "remediation"
FLOW_PROCESSING_MCP_SYNC_DIR = FLOW_PROCESSING_REMEDIATION_DIR / "mcp-sync"
FLOW_PROCESSING_ROUND2_DIR = FLOW_PROCESSING_REMEDIATION_DIR / "round2"
if str(PROCESS_AUTOMATED_BUILDER_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESS_AUTOMATED_BUILDER_ROOT))

try:
    from tidas_sdk import create_flow
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        "tidas_sdk is required. Run this script with process-automated-builder/.venv/bin/python."
    ) from exc


DEFAULT_INPUT_FILE = (
    FLOW_PROCESSING_MCP_SYNC_DIR / "flows_tidas_sdk_plus_classification_remote_validation_failed.jsonl"
)
DEFAULT_ALL_FILE = (
    FLOW_PROCESSING_ROUND2_DIR / "flows_tidas_sdk_plus_classification_remote_validation_failed_round2_all.jsonl"
)
DEFAULT_READY_FILE = (
    FLOW_PROCESSING_ROUND2_DIR
    / "flows_tidas_sdk_plus_classification_remote_validation_failed_round2_ready_for_mcp.jsonl"
)
DEFAULT_RESIDUAL_FILE = (
    FLOW_PROCESSING_ROUND2_DIR
    / "flows_tidas_sdk_plus_classification_remote_validation_failed_round2_residual_manual_queue.jsonl"
)
DEFAULT_AUDIT_FILE = (
    FLOW_PROCESSING_ROUND2_DIR
    / "flows_tidas_sdk_plus_classification_remote_validation_failed_round2_audit.jsonl"
)
DEFAULT_REPORT_FILE = (
    FLOW_PROCESSING_ROUND2_DIR
    / "flows_tidas_sdk_plus_classification_remote_validation_failed_round2_report.json"
)
DEFAULT_PROMPT_FILE = (
    FLOW_PROCESSING_ROUND2_DIR
    / "flows_tidas_sdk_plus_classification_remote_validation_failed_round2_residual_manual_queue_prompt.md"
)

TARGETED_PATHS = [
    "flowDataSet.flowInformation.dataSetInformation.name.baseName",
    "flowDataSet.flowInformation.dataSetInformation.name.treatmentStandardsRoutes",
    "flowDataSet.flowInformation.dataSetInformation.name.mixAndLocationTypes",
    "flowDataSet.flowInformation.dataSetInformation.name.flowProperties",
    "flowDataSet.flowInformation.dataSetInformation.common:synonyms",
    "flowDataSet.flowInformation.dataSetInformation.common:generalComment",
    "flowDataSet.flowInformation.technology.technologicalApplicability",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Round-2 deterministic remediation for rows rejected by remote MCP validation. "
            "Canonicalize each raw flow row through tidas_sdk, materialize the strict schema "
            "defaults back into json_ordered, and re-validate with validation_mode='both'."
        )
    )
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_INPUT_FILE),
        help="Input JSONL containing rows that failed remote MCP validation.",
    )
    parser.add_argument(
        "--out-all-file",
        default=str(DEFAULT_ALL_FILE),
        help="Output JSONL containing all round-2 remediated rows.",
    )
    parser.add_argument(
        "--out-ready-file",
        default=str(DEFAULT_READY_FILE),
        help="Output JSONL containing rows that pass strict local SDK validation after remediation.",
    )
    parser.add_argument(
        "--out-residual-file",
        default=str(DEFAULT_RESIDUAL_FILE),
        help="Output JSONL containing rows that still fail strict local SDK validation.",
    )
    parser.add_argument(
        "--out-audit-file",
        default=str(DEFAULT_AUDIT_FILE),
        help="Output JSONL audit trail path.",
    )
    parser.add_argument(
        "--out-report-file",
        default=str(DEFAULT_REPORT_FILE),
        help="Output JSON summary report path.",
    )
    parser.add_argument(
        "--out-prompt-file",
        default=str(DEFAULT_PROMPT_FILE),
        help="Output Markdown prompt path for any residual manual queue.",
    )
    parser.add_argument(
        "--validation-mode",
        choices=("pydantic", "jsonschema", "both"),
        default="both",
        help="Local SDK validation mode. Use 'both' for strict validation.",
    )
    return parser.parse_args()


def load_jsonl_rows(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def dump_json(path: Path | str, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: Path | str, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def parse_validation_error(exc: Exception) -> list[dict[str, str]]:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        results: list[dict[str, str]] = []
        for item in errors():
            loc = item.get("loc") or ()
            path = ".".join(str(part) for part in loc) or "<exception>"
            results.append(
                {
                    "validator": "tidas_sdk",
                    "path": path,
                    "message": str(item.get("msg") or str(exc)),
                    "code": str(item.get("type") or "validation_error"),
                }
            )
        if results:
            return results
    return [
        {
            "validator": "tidas_sdk",
            "path": "<exception>",
            "message": str(exc),
            "code": "exception",
        }
    ]


def path_exists(payload: Any, dotted_path: str) -> bool:
    current = payload
    for segment in dotted_path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return False
        current = current[segment]
    return True


def _walk_diff(before: Any, after: Any, prefix: str, out: set[str]) -> None:
    if type(before) is not type(after):
        out.add(prefix or "<root>")
        return
    if isinstance(before, dict):
        before_keys = set(before.keys())
        after_keys = set(after.keys())
        for key in sorted(before_keys | after_keys):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                out.add(child_prefix)
                continue
            _walk_diff(before[key], after[key], child_prefix, out)
        return
    if isinstance(before, list):
        if len(before) != len(after):
            out.add(prefix or "<root>")
            return
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            child_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            _walk_diff(left, right, child_prefix, out)
        return
    if before != after:
        out.add(prefix or "<root>")


def changed_paths(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    out: set[str] = set()
    _walk_diff(before, after, "", out)
    return sorted(out)


def reason_summary(reasons: list[dict[str, Any]]) -> str:
    if not reasons:
        return "none"
    ordered = sorted({(str(item.get("path") or ""), str(item.get("code") or "")) for item in reasons})
    return "; ".join(f"{path or '<root>'}:{code}" for path, code in ordered)


def build_prompt(
    *,
    manual_rows: list[dict[str, Any]],
    manual_file: Path,
    prompt_file: Path,
) -> str:
    if not manual_rows:
        return f"""本轮 round-2 remote-validation remediation 已完成，residual manual queue 为 0。

输入文件：
`{manual_file}`

结果：
1. 本轮不需要再调用 `flow-governance-review` 做 process 证据补齐。
2. 也不需要让 OpenClaw 基于 basename 做自由补写。
3. 如后续再次遇到远端失败，再只对新的 residual 子集触发 `flow-governance-review openclaw-entry`。
"""

    queue_ids = [str(row.get("id") or "").strip() for row in manual_rows]
    id_lines = "\n".join(f"- `{value}`" for value in queue_ids) if queue_ids else "- None"
    return f"""你现在要处理 round-2 strict remediation 之后仍未通过本地 `tidas_sdk.create_flow(validate=True, validation_mode=\"both\")` 的 flow。

输入文件：
`{manual_file}`

执行边界：
1. 优先使用 `flow-governance-review` 的 `openclaw-entry` + `--process-pool-file` 提取并复用 process 证据。
2. 只有在 process 证据不足时，才允许最小语义补齐。
3. 任何改动命名、classification、treatment/routes、mix/location 的项，后续发布前都必须再走 dedup。
4. 不要基于 basename 自由发挥通用内容；如果没有证据，保守留空。

当前 residual manual queue 共 `{len(manual_rows)}` 条：
{id_lines}

推荐命令骨架：
```bash
bash /home/huimin/projects/lca-skills/flow-governance-review/scripts/run-flow-governance-review.sh openclaw-entry \\
  --subject-flows-file {manual_file} \\
  --candidate-flows-file /home/huimin/projects/lca-skills/flows.json \\
  --process-pool-file /home/huimin/projects/lca-skills/flow-governance-review/assets/artifacts/flow-processing/datasets/process_pool.jsonl \\
  --out-dir /home/huimin/.openclaw/workspace/review/flow-governance/round2-residual
```
"""


def remediate_row(row: dict[str, Any], *, validation_mode: str) -> dict[str, Any]:
    working = deepcopy(row)
    payload = working.get("json_ordered")
    if not isinstance(payload, dict) or not isinstance(payload.get("flowDataSet"), dict):
        final_reasons = [
            {
                "validator": "tidas_sdk",
                "path": "json_ordered.flowDataSet",
                "message": "Missing flowDataSet payload",
                "code": "missing_flow_dataset",
            }
        ]
        working["reason"] = final_reasons
        return {
            "row": working,
            "valid": False,
            "strict_validated": False,
            "changed_paths": [],
            "materialized_paths": [],
            "final_reasons": final_reasons,
        }

    try:
        entity = create_flow(payload, validate=True, validation_mode=validation_mode)
        canonical_payload = entity.model.model_dump(by_alias=True, exclude_none=True, mode="json")
        create_flow(canonical_payload, validate=True, validation_mode=validation_mode)
        delta_paths = changed_paths(payload, canonical_payload)
        materialized = [
            path
            for path in TARGETED_PATHS
            if not path_exists(payload, path) and path_exists(canonical_payload, path)
        ]
        working["json_ordered"] = canonical_payload
        working["reason"] = []
        return {
            "row": working,
            "valid": True,
            "strict_validated": True,
            "changed_paths": delta_paths,
            "materialized_paths": materialized,
            "final_reasons": [],
        }
    except Exception as exc:  # noqa: BLE001
        final_reasons = parse_validation_error(exc)
        working["reason"] = final_reasons
        return {
            "row": working,
            "valid": False,
            "strict_validated": False,
            "changed_paths": [],
            "materialized_paths": [],
            "final_reasons": final_reasons,
        }


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\.main")

    input_file = Path(args.input_file).expanduser().resolve()
    out_all_file = Path(args.out_all_file).expanduser().resolve()
    out_ready_file = Path(args.out_ready_file).expanduser().resolve()
    out_residual_file = Path(args.out_residual_file).expanduser().resolve()
    out_audit_file = Path(args.out_audit_file).expanduser().resolve()
    out_report_file = Path(args.out_report_file).expanduser().resolve()
    out_prompt_file = Path(args.out_prompt_file).expanduser().resolve()
    for path in (
        out_all_file,
        out_ready_file,
        out_residual_file,
        out_audit_file,
        out_report_file,
        out_prompt_file,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl_rows(input_file)

    remediated_rows: list[dict[str, Any]] = []
    ready_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    changed_path_counts: Counter[str] = Counter()
    materialized_path_counts: Counter[str] = Counter()
    original_reason_counts: Counter[str] = Counter()
    residual_reason_counts: Counter[str] = Counter()

    for row in rows:
        reasons = row.get("reason") or []
        if isinstance(reasons, dict):
            reasons = [reasons]
        original_reason_counts[reason_summary(reasons)] += 1

        result = remediate_row(row, validation_mode=args.validation_mode)
        remediated_row = result["row"]
        remediated_rows.append(remediated_row)

        if result["valid"]:
            ready_rows.append(remediated_row)
            for path in result["changed_paths"]:
                changed_path_counts[path] += 1
            for path in result["materialized_paths"]:
                materialized_path_counts[path] += 1
        else:
            residual_rows.append(remediated_row)
            residual_reason_counts[reason_summary(result["final_reasons"])] += 1

        audit_rows.append(
            {
                "id": row.get("id"),
                "state_code": row.get("state_code"),
                "strict_validated": result["strict_validated"],
                "valid_after_round2": result["valid"],
                "original_reason": reasons,
                "final_reason": result["final_reasons"],
                "changed_paths": result["changed_paths"],
                "materialized_paths": result["materialized_paths"],
            }
        )

    dump_jsonl(out_all_file, remediated_rows)
    dump_jsonl(out_ready_file, ready_rows)
    dump_jsonl(out_residual_file, residual_rows)
    dump_jsonl(out_audit_file, audit_rows)

    prompt_text = build_prompt(
        manual_rows=residual_rows,
        manual_file=out_residual_file,
        prompt_file=out_prompt_file,
    )
    out_prompt_file.write_text(prompt_text, encoding="utf-8")

    report = {
        "input_file": str(input_file),
        "validation_mode": args.validation_mode,
        "counts": {
            "input_rows": len(rows),
            "round2_rows": len(remediated_rows),
            "strict_ready_rows": len(ready_rows),
            "residual_manual_rows": len(residual_rows),
            "changed_rows": sum(1 for item in audit_rows if item["changed_paths"]),
            "unchanged_rows": sum(1 for item in audit_rows if not item["changed_paths"]),
        },
        "original_reason_counts": dict(original_reason_counts.most_common()),
        "materialized_path_counts": dict(materialized_path_counts.most_common()),
        "top_changed_paths": dict(changed_path_counts.most_common(25)),
        "residual_reason_counts": dict(residual_reason_counts.most_common()),
        "files": {
            "all_round2": str(out_all_file),
            "ready_for_mcp": str(out_ready_file),
            "residual_manual_queue": str(out_residual_file),
            "audit": str(out_audit_file),
            "prompt": str(out_prompt_file),
        },
    }
    dump_json(out_report_file, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
