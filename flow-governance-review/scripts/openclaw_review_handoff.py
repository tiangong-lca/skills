#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from flow_governance_common import (
    apply_entity_text_patch,
    build_flow_indexes,
    deep_get,
    deep_diff_paths,
    dump_json,
    dump_jsonl,
    ensure_dir,
    entity_text_fields,
    exchange_records,
    extract_entity_identity,
    extract_process_identity,
    flow_dataset_from_row,
    lang_entries,
    lang_text,
    load_json_or_jsonl,
    load_rows_from_file,
    patched_flow_reference,
    path_contains_allowed_text_change,
    process_dataset_from_row,
    safe_entity_key,
    safe_process_key,
    unified_json_diff,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FLOW_METHODOLOGY = REPO_ROOT / "flow-governance-review" / "references" / "tidas_flows.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export OpenClaw review packs and apply validated decisions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_ref = subparsers.add_parser(
        "export-ref-review-pack",
        help="Build an OpenClaw review pack for residual process-flow reference decisions.",
    )
    export_ref.add_argument("--processes-file", required=True)
    export_ref.add_argument("--review-queue-file", required=True)
    export_ref.add_argument("--scope-flow-files", nargs="+", required=True)
    export_ref.add_argument("--out-dir", required=True)

    apply_ref = subparsers.add_parser(
        "apply-ref-decisions",
        help="Apply OpenClaw reference decisions back into process rows.",
    )
    apply_ref.add_argument("--processes-file", required=True)
    apply_ref.add_argument("--review-pack", required=True)
    apply_ref.add_argument("--decisions-file", required=True)
    apply_ref.add_argument("--scope-flow-files", nargs="+", required=True)
    apply_ref.add_argument("--out-dir", required=True)

    export_text = subparsers.add_parser(
        "export-text-review-pack",
        help="Build an OpenClaw review pack for flow/process text optimization.",
    )
    export_text.add_argument("--entity-type", choices=("flow", "process"), required=True)
    export_text.add_argument("--rows-file", required=True)
    export_text.add_argument("--out-dir", required=True)
    export_text.add_argument("--max-items", type=int, default=0)
    export_text.add_argument(
        "--processes-file",
        help="Optional process rows JSON/JSONL file used to attach linked process context when entity-type=flow.",
    )
    export_text.add_argument(
        "--max-process-contexts",
        type=int,
        default=3,
        help="Maximum linked process contexts to include per flow review item.",
    )
    export_text.add_argument(
        "--methodology-file",
        help="Optional methodology YAML/JSON file for flow naming/text constraints. Defaults to tidas_flows.yaml when present.",
    )
    export_text.add_argument(
        "--methodology-id",
        help="Identifier written into review-pack-context.json for the loaded methodology constraints.",
    )

    apply_text = subparsers.add_parser(
        "apply-text-decisions",
        help="Apply OpenClaw text decisions back into flow/process rows.",
    )
    apply_text.add_argument("--entity-type", choices=("flow", "process"), required=True)
    apply_text.add_argument("--rows-file", required=True)
    apply_text.add_argument("--review-pack", required=True)
    apply_text.add_argument("--decisions-file", required=True)
    apply_text.add_argument("--out-dir", required=True)

    validate_text = subparsers.add_parser(
        "validate-text-decisions",
        help="Validate text-only changes that came back from OpenClaw.",
    )
    validate_text.add_argument("--entity-type", choices=("flow", "process"), required=True)
    validate_text.add_argument("--original-rows-file", required=True)
    validate_text.add_argument("--patched-rows-file", required=True)
    validate_text.add_argument("--out-dir", required=True)
    validate_text.add_argument("--tidas-mode", choices=("auto", "required", "skip"), default="auto")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "export-ref-review-pack":
        export_ref_review_pack(args)
        return
    if args.command == "apply-ref-decisions":
        apply_ref_decisions(args)
        return
    if args.command == "export-text-review-pack":
        export_text_review_pack(args)
        return
    if args.command == "apply-text-decisions":
        apply_text_decisions(args)
        return
    if args.command == "validate-text-decisions":
        validate_text_decisions(args)
        return
    raise RuntimeError(f"Unsupported command: {args.command}")


def export_ref_review_pack(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.out_dir)
    processes = load_rows_from_file(args.processes_file)
    process_map = {f"{pid}@{version}": row for row in processes for pid, version, _name in [extract_process_identity(row)]}
    flow_rows: list[dict[str, Any]] = []
    for path in args.scope_flow_files:
        flow_rows.extend(load_rows_from_file(path))
    flow_index = build_flow_indexes(flow_rows)
    review_queue = load_json_or_jsonl(args.review_queue_file)

    review_pack: list[dict[str, Any]] = []
    carryover_queue: list[dict[str, Any]] = []

    for item in review_queue:
        candidate_refs = [candidate for candidate in item.get("candidate_refs", []) if isinstance(candidate, dict)]
        process_key = f"{item.get('process_id', '')}@{item.get('process_version', '')}"
        process_row = process_map.get(process_key)
        if process_row is None:
            carryover_queue.append({**item, "carryover_reason": "missing_process_row"})
            continue
        exchange = _find_exchange(process_row, str(item.get("exchange_internal_id") or "").strip())
        if exchange is None:
            carryover_queue.append({**item, "carryover_reason": "missing_exchange"})
            continue
        if not candidate_refs:
            carryover_queue.append({**item, "carryover_reason": "no_candidate_refs"})
            continue

        enriched_candidates: list[dict[str, Any]] = []
        for candidate in candidate_refs:
            candidate_key = f"{candidate.get('id')}@{candidate.get('version')}"
            record = flow_index["by_uuid_version"].get(candidate_key)
            enriched_candidates.append(
                {
                    "id": str(candidate.get("id") or ""),
                    "version": str(candidate.get("version") or ""),
                    "name": str(candidate.get("name") or getattr(record, "name", "")),
                    "flow_type": getattr(record, "flow_type", ""),
                    "short_description": getattr(record, "short_description", None),
                }
            )
        review_pack.append(
            {
                "review_id": _ref_review_id(item),
                "review_kind": "process_flow_reference",
                "allowed_actions": ["keep_as_is", "choose_candidate_ref", "manual_review"],
                "process_id": str(item.get("process_id") or ""),
                "process_version": str(item.get("process_version") or ""),
                "process_name": str(item.get("process_name") or ""),
                "exchange_internal_id": str(item.get("exchange_internal_id") or ""),
                "exchange_direction": item.get("exchange_direction"),
                "exchange_text": str(item.get("exchange_text") or ""),
                "current_reference": deepcopy(exchange.get("referenceToFlowDataSet") or {}),
                "current_issue_type": item.get("current_issue_type"),
                "candidate_refs": enriched_candidates,
                "instructions": (
                    "Choose exactly one candidate only when it is clearly the correct flow. "
                    "Otherwise return keep_as_is or manual_review."
                ),
            }
        )

    dump_json(out_dir / "review-pack.json", review_pack)
    dump_jsonl(out_dir / "review-pack.jsonl", review_pack)
    dump_jsonl(out_dir / "carryover-manual-review.jsonl", carryover_queue)
    dump_json(
        out_dir / "review-pack-summary.json",
        {
            "review_item_count": len(review_pack),
            "carryover_count": len(carryover_queue),
        },
    )
    (out_dir / "openclaw-instructions.md").write_text(_ref_review_instructions(), encoding="utf-8")


def apply_ref_decisions(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.out_dir)
    processes = load_rows_from_file(args.processes_file)
    original_map = {f"{pid}@{version}": row for row in processes for pid, version, _name in [extract_process_identity(row)]}
    working_map = {key: deepcopy(row) for key, row in original_map.items()}
    review_pack = load_rows_from_file(args.review_pack)
    pack_map = {str(item.get("review_id") or ""): item for item in review_pack}
    decisions = load_json_or_jsonl(args.decisions_file)

    flow_rows: list[dict[str, Any]] = []
    for path in args.scope_flow_files:
        flow_rows.extend(load_rows_from_file(path))
    flow_index = build_flow_indexes(flow_rows)

    patch_root = ensure_dir(out_dir / "process-patches")
    evidence_by_process: dict[str, list[dict[str, Any]]] = {}
    invalid_decisions: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for decision in decisions:
        review_id = str(decision.get("review_id") or "").strip()
        pack_item = pack_map.get(review_id)
        if pack_item is None:
            invalid_decisions.append({"review_id": review_id, "reason": "review_id_not_in_pack", "decision": decision})
            continue
        action = str(decision.get("action") or "").strip()
        if action == "keep_as_is":
            continue
        if action == "manual_review":
            unresolved.append(_merge_decision_context(pack_item, decision))
            continue
        if action != "choose_candidate_ref":
            invalid_decisions.append({"review_id": review_id, "reason": "unsupported_action", "decision": decision})
            continue

        chosen = _normalize_candidate_choice(decision)
        if chosen is None:
            invalid_decisions.append({"review_id": review_id, "reason": "missing_chosen_candidate", "decision": decision})
            continue
        candidate_map = {
            f"{item.get('id')}@{item.get('version')}": item for item in pack_item.get("candidate_refs", []) if isinstance(item, dict)
        }
        chosen_key = f"{chosen['id']}@{chosen['version']}"
        if chosen_key not in candidate_map:
            invalid_decisions.append({"review_id": review_id, "reason": "chosen_candidate_not_allowed", "decision": decision})
            continue

        process_key = f"{pack_item.get('process_id')}@{pack_item.get('process_version')}"
        process_row = working_map.get(process_key)
        if process_row is None:
            invalid_decisions.append({"review_id": review_id, "reason": "missing_process_row", "decision": decision})
            continue
        exchange = _find_exchange(process_row, str(pack_item.get("exchange_internal_id") or ""))
        if exchange is None:
            invalid_decisions.append({"review_id": review_id, "reason": "missing_exchange", "decision": decision})
            continue

        record = flow_index["by_uuid_version"].get(chosen_key)
        if record is None:
            invalid_decisions.append({"review_id": review_id, "reason": "chosen_candidate_missing_in_scope_catalog", "decision": decision})
            continue

        current_ref = exchange.get("referenceToFlowDataSet") or {}
        exchange["referenceToFlowDataSet"] = patched_flow_reference(current_ref, record)
        evidence_by_process.setdefault(process_key, []).append(
            {
                "review_id": review_id,
                "action": action,
                "chosen_candidate": chosen,
                "reason": str(decision.get("reason") or ""),
                "confidence": decision.get("confidence"),
            }
        )

    changed_rows: list[dict[str, Any]] = []
    for process_key, evidence_rows in evidence_by_process.items():
        before = original_map[process_key]
        after = working_map[process_key]
        if json.dumps(before, ensure_ascii=False, sort_keys=True) == json.dumps(after, ensure_ascii=False, sort_keys=True):
            continue
        process_id, version, _name = extract_process_identity(after)
        process_dir = ensure_dir(patch_root / safe_process_key(process_id, version))
        dump_json(process_dir / "before.json", before)
        dump_json(process_dir / "after.json", after)
        dump_json(process_dir / "evidence.json", evidence_rows)
        (process_dir / "diff.patch").write_text(unified_json_diff(before, after), encoding="utf-8")
        changed_rows.append(after)

    unanswered = [item for item in review_pack if str(item.get("review_id") or "") not in {str(d.get("review_id") or "") for d in decisions}]
    unresolved.extend(unanswered)

    dump_json(out_dir / "patched-processes.json", changed_rows)
    dump_jsonl(out_dir / "invalid-decisions.jsonl", invalid_decisions)
    dump_jsonl(out_dir / "manual-review-unresolved.jsonl", unresolved)
    dump_json(
        out_dir / "decision-summary.json",
        {
            "review_items": len(review_pack),
            "decisions_received": len(decisions),
            "patched_processes": len(changed_rows),
            "invalid_decisions": len(invalid_decisions),
            "unresolved_items": len(unresolved),
        },
    )


def export_text_review_pack(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.out_dir)
    rows = load_rows_from_file(args.rows_file)
    if args.max_items and args.max_items > 0:
        rows = rows[: args.max_items]
    review_pack: list[dict[str, Any]] = []
    context_payload = {
        "entity_type": args.entity_type,
        "source_rows_file": str(Path(args.rows_file).resolve()),
    }
    methodology_context = _load_text_review_methodology(args.entity_type, args.methodology_file, args.methodology_id)
    if methodology_context is not None:
        context_payload["methodology"] = methodology_context

    linked_process_context_by_flow_key: dict[str, dict[str, Any]] = {}
    if args.entity_type == "flow" and args.processes_file:
        linked_process_context_by_flow_key = _build_flow_process_context(
            flow_rows=rows,
            process_rows=load_rows_from_file(args.processes_file),
            max_items=max(args.max_process_contexts, 0),
        )
        context_payload["linked_process_context_source"] = {
            "processes_file": str(Path(args.processes_file).resolve()),
            "max_process_contexts_per_flow": max(args.max_process_contexts, 0),
        }

    for row in rows:
        entity_id, version, name = extract_entity_identity(args.entity_type, row)
        fields = entity_text_fields(
            row,
            args.entity_type,
            include_placeholders=(args.entity_type == "flow"),
        )
        if not fields:
            continue
        review_pack.append(
            {
                "review_id": _text_review_id(args.entity_type, entity_id, version),
                "review_kind": "text_fields",
                "entity_type": args.entity_type,
                "entity_id": entity_id,
                "entity_version": version,
                "entity_name": name,
                "allowed_actions": ["keep_as_is", "patch_text_fields", "manual_review"],
                "allowed_field_ids": sorted(fields),
                "fields": {field_id: item["value"] for field_id, item in fields.items()},
                "instructions": (
                    "Only improve wording, naming consistency, or short descriptions. "
                    "Do not invent new technical facts or change structure."
                ),
            }
        )
        if methodology_context is not None:
            review_pack[-1]["review_context_refs"] = ["flow_text_methodology"]
            guidance_refs = _field_guidance_refs(sorted(fields))
            if guidance_refs:
                review_pack[-1]["field_guidance_refs"] = guidance_refs
        if args.entity_type == "flow":
            flow_key = f"{entity_id}@{version}"
            process_context = linked_process_context_by_flow_key.get(flow_key)
            if process_context:
                review_pack[-1]["linked_process_context_summary"] = process_context["summary"]
                review_pack[-1]["linked_process_contexts"] = process_context["items"]

    dump_json(out_dir / "review-pack.json", review_pack)
    dump_jsonl(out_dir / "review-pack.jsonl", review_pack)
    dump_json(
        out_dir / "review-pack-summary.json",
        {
            "review_item_count": len(review_pack),
            "has_methodology_context": methodology_context is not None,
            "has_linked_process_context": bool(linked_process_context_by_flow_key),
        },
    )
    dump_json(out_dir / "review-pack-context.json", context_payload)
    (out_dir / "openclaw-instructions.md").write_text(
        _text_review_instructions(
            args.entity_type,
            has_methodology=methodology_context is not None,
            has_process_context=bool(linked_process_context_by_flow_key),
        ),
        encoding="utf-8",
    )


def apply_text_decisions(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.out_dir)
    rows = load_rows_from_file(args.rows_file)
    original_map = {
        f"{entity_id}@{version}": row for row in rows for entity_id, version, _name in [extract_entity_identity(args.entity_type, row)]
    }
    working_map = {key: deepcopy(row) for key, row in original_map.items()}
    review_pack = load_rows_from_file(args.review_pack)
    pack_map = {str(item.get("review_id") or ""): item for item in review_pack}
    decisions = load_json_or_jsonl(args.decisions_file)
    patch_root = ensure_dir(out_dir / f"{args.entity_type}-patches")
    evidence_by_entity: dict[str, list[dict[str, Any]]] = {}
    invalid_decisions: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for decision in decisions:
        review_id = str(decision.get("review_id") or "").strip()
        pack_item = pack_map.get(review_id)
        if pack_item is None:
            invalid_decisions.append({"review_id": review_id, "reason": "review_id_not_in_pack", "decision": decision})
            continue
        action = str(decision.get("action") or "").strip()
        if action == "keep_as_is":
            continue
        if action == "manual_review":
            unresolved.append(_merge_decision_context(pack_item, decision))
            continue
        if action != "patch_text_fields":
            invalid_decisions.append({"review_id": review_id, "reason": "unsupported_action", "decision": decision})
            continue

        patches = decision.get("patches")
        if not isinstance(patches, list) or not patches:
            invalid_decisions.append({"review_id": review_id, "reason": "missing_patches", "decision": decision})
            continue

        entity_key = f"{pack_item.get('entity_id')}@{pack_item.get('entity_version')}"
        row = working_map.get(entity_key)
        if row is None:
            invalid_decisions.append({"review_id": review_id, "reason": "missing_entity_row", "decision": decision})
            continue

        allowed_field_ids = set(pack_item.get("allowed_field_ids") or [])
        current_fields = pack_item.get("fields") or {}
        evidence_rows: list[dict[str, Any]] = []
        for patch in patches:
            if not isinstance(patch, dict):
                invalid_decisions.append({"review_id": review_id, "reason": "patch_not_object", "decision": decision})
                evidence_rows = []
                break
            field_id = str(patch.get("field_id") or "").strip()
            new_value = str(patch.get("value") or "").strip()
            if field_id not in allowed_field_ids:
                invalid_decisions.append(
                    {"review_id": review_id, "reason": "field_not_allowed", "field_id": field_id, "decision": decision}
                )
                evidence_rows = []
                break
            if not new_value:
                invalid_decisions.append({"review_id": review_id, "reason": "empty_value_not_allowed", "decision": decision})
                evidence_rows = []
                break
            if new_value == str(current_fields.get(field_id) or ""):
                continue
            apply_entity_text_patch(row, args.entity_type, field_id, new_value)
            evidence_rows.append(
                {
                    "review_id": review_id,
                    "field_id": field_id,
                    "before": str(current_fields.get(field_id) or ""),
                    "after": new_value,
                    "reason": str(patch.get("reason") or decision.get("reason") or ""),
                    "confidence": patch.get("confidence", decision.get("confidence")),
                }
            )
        if evidence_rows:
            evidence_by_entity.setdefault(entity_key, []).extend(evidence_rows)

    changed_rows: list[dict[str, Any]] = []
    for entity_key, evidence_rows in evidence_by_entity.items():
        before = original_map[entity_key]
        after = working_map[entity_key]
        if json.dumps(before, ensure_ascii=False, sort_keys=True) == json.dumps(after, ensure_ascii=False, sort_keys=True):
            continue
        entity_id, version, _name = extract_entity_identity(args.entity_type, after)
        entity_dir = ensure_dir(patch_root / safe_entity_key(entity_id, version))
        dump_json(entity_dir / "before.json", before)
        dump_json(entity_dir / "after.json", after)
        dump_json(entity_dir / "evidence.json", evidence_rows)
        (entity_dir / "diff.patch").write_text(unified_json_diff(before, after), encoding="utf-8")
        changed_rows.append(after)

    unanswered = [item for item in review_pack if str(item.get("review_id") or "") not in {str(d.get("review_id") or "") for d in decisions}]
    unresolved.extend(unanswered)

    dump_json(out_dir / f"patched-{args.entity_type}s.json", changed_rows)
    dump_jsonl(out_dir / "invalid-decisions.jsonl", invalid_decisions)
    dump_jsonl(out_dir / "manual-review-unresolved.jsonl", unresolved)
    dump_json(
        out_dir / "decision-summary.json",
        {
            "review_items": len(review_pack),
            "decisions_received": len(decisions),
            "patched_entities": len(changed_rows),
            "invalid_decisions": len(invalid_decisions),
            "unresolved_items": len(unresolved),
        },
    )


def validate_text_decisions(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.out_dir)
    original_rows = load_rows_from_file(args.original_rows_file)
    patched_rows = load_rows_from_file(args.patched_rows_file)
    original_map = {
        f"{entity_id}@{version}": row for row in original_rows for entity_id, version, _name in [extract_entity_identity(args.entity_type, row)]
    }
    validator = _load_tidas_validator(args.entity_type, args.tidas_mode)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    summary = {"patched_entity_count": len(patched_rows), "passed": 0, "failed": 0, "tidas_validation": validator is not None}

    for patched_row in patched_rows:
        entity_id, version, name = extract_entity_identity(args.entity_type, patched_row)
        key = f"{entity_id}@{version}"
        original_row = original_map.get(key)
        result = _validate_text_patch(args.entity_type, original_row, patched_row, validator)
        result["entity_id"] = entity_id
        result["entity_version"] = version
        result["entity_name"] = name
        results.append(result)
        if result["ok"]:
            summary["passed"] += 1
        else:
            summary["failed"] += 1
            failures.append(result)

    dump_json(out_dir / "validation-report.json", {"summary": summary, "results": results})
    dump_jsonl(out_dir / "validation-failures.jsonl", failures)


def _validate_text_patch(entity_type: str, original_row: dict[str, Any] | None, patched_row: dict[str, Any], validator) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if original_row is None:
        issues.append({"type": "missing_original_row", "severity": "error"})
        return {"ok": False, "issues": issues}

    diff_paths = deep_diff_paths(original_row, patched_row)
    illegal_paths = [path for path in diff_paths if not path_contains_allowed_text_change(path, entity_type)]
    if illegal_paths:
        issues.append(
            {
                "type": "non_text_changes_detected",
                "severity": "error",
                "paths": [".".join(str(part) for part in path) for path in illegal_paths],
            }
        )

    patched_fields = entity_text_fields(patched_row, entity_type)
    if not any(field_id.startswith("base_name:") and item.get("value") for field_id, item in patched_fields.items()):
        issues.append({"type": "base_name_missing_after_patch", "severity": "error"})
    empty_fields = [field_id for field_id, item in patched_fields.items() if not str(item.get("value") or "").strip()]
    if empty_fields:
        issues.append({"type": "empty_text_field_after_patch", "severity": "error", "field_ids": empty_fields})

    if validator is not None:
        payload = process_dataset_from_row(patched_row) if entity_type == "process" else flow_dataset_from_row(patched_row)
        entity = validator(payload)
        if hasattr(entity, "validateEnhanced"):
            validation = entity.validateEnhanced()
        elif hasattr(entity, "validate"):
            validation = entity.validate()
        else:
            validation = None
        success = bool(getattr(validation, "success", False))
        if not success:
            issues.append(
                {
                    "type": "tidas_validation_failed",
                    "severity": "error",
                    "details": str(validation) if validation is not None else "Validator returned no result.",
                }
            )

    return {"ok": not issues, "issues": issues}


def _load_tidas_validator(entity_type: str, mode: str):
    if mode == "skip":
        return None
    try:
        module = importlib.import_module("tidas_sdk")
    except Exception:
        if mode == "required":
            raise
        return None
    factory_name = "create_process" if entity_type == "process" else "create_flow"
    factory = getattr(module, factory_name, None)
    if factory is None:
        if mode == "required":
            raise RuntimeError(f"tidas_sdk.{factory_name} not available.")
        return None
    return factory


def _find_exchange(process_row: dict[str, Any], exchange_internal_id: str) -> dict[str, Any] | None:
    for exchange in exchange_records(process_row):
        if str(exchange.get("@dataSetInternalID") or "").strip() == exchange_internal_id:
            return exchange
    return None


def _ref_review_id(item: dict[str, Any]) -> str:
    return (
        "process-flow-ref:"
        f"{item.get('process_id', '')}@{item.get('process_version', '')}:"
        f"{item.get('exchange_internal_id', '')}"
    )


def _text_review_id(entity_type: str, entity_id: str, version: str) -> str:
    return f"{entity_type}-text:{entity_id}@{version}"


def _normalize_candidate_choice(decision: dict[str, Any]) -> dict[str, str] | None:
    chosen = decision.get("chosen_candidate")
    if isinstance(chosen, dict):
        chosen_id = str(chosen.get("id") or "").strip()
        chosen_version = str(chosen.get("version") or "").strip()
        if chosen_id and chosen_version:
            return {"id": chosen_id, "version": chosen_version}
    chosen_id = str(decision.get("target_flow_id") or "").strip()
    chosen_version = str(decision.get("target_flow_version") or "").strip()
    if chosen_id and chosen_version:
        return {"id": chosen_id, "version": chosen_version}
    return None


def _merge_decision_context(pack_item: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    return {
        **deepcopy(pack_item),
        "decision": deepcopy(decision),
    }


def _load_text_review_methodology(
    entity_type: str,
    methodology_file: str | None,
    methodology_id: str | None,
) -> dict[str, Any] | None:
    if entity_type != "flow":
        return None
    source = Path(methodology_file).expanduser().resolve() if methodology_file else None
    if source is None:
        source = DEFAULT_FLOW_METHODOLOGY if DEFAULT_FLOW_METHODOLOGY.exists() else None
    if source is None or not source.exists():
        return None
    payload = _load_structured_payload(source)
    guidance: dict[str, Any] = {}
    field_paths = {
        "base_name": ["flowDataSet", "flowInformation", "dataSetInformation", "name", "baseName", "<rules>"],
        "treatment_standards_routes": [
            "flowDataSet",
            "flowInformation",
            "dataSetInformation",
            "name",
            "treatmentStandardsRoutes",
            "<rules>",
        ],
        "mix_and_location_types": [
            "flowDataSet",
            "flowInformation",
            "dataSetInformation",
            "name",
            "mixAndLocationTypes",
            "<rules>",
        ],
        "flow_properties_text": ["flowDataSet", "flowInformation", "dataSetInformation", "name", "flowProperties", "<rules>"],
        "synonyms": ["flowDataSet", "flowInformation", "dataSetInformation", "common:synonyms", "<rules>"],
    }
    for field_kind, path in field_paths.items():
        rules = deep_get(payload, path, [])
        condensed_rules = [_condense_methodology_rule(rule) for rule in rules if isinstance(rule, dict)]
        condensed_rules = [rule for rule in condensed_rules if rule]
        if condensed_rules:
            guidance[field_kind] = {
                "methodology_path": ".".join(path[:-1]),
                "rules": condensed_rules,
            }
    if not guidance:
        return None
    return {
        "context_id": "flow_text_methodology",
        "methodology_id": methodology_id or source.name,
        "source_file": str(source),
        "guidance": guidance,
    }


def _load_structured_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError(f"PyYAML is required to load methodology file: {path}") from exc
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object payload in methodology file: {path}")
    return payload


def _condense_methodology_rule(rule: dict[str, Any]) -> dict[str, Any]:
    condensed: dict[str, Any] = {}
    language = str(rule.get("language") or "all").strip()
    condensed["language"] = language or "all"
    for key in (
        "requirement",
        "purpose",
        "naming_convention",
        "usage",
        "content_requirements",
        "best_practices",
        "examples",
    ):
        value = rule.get(key)
        if isinstance(value, str) and value.strip():
            condensed[key] = value.strip()
        elif isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
            if items:
                condensed[key] = items[:8]
    return condensed


def _field_guidance_refs(field_ids: list[str]) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for field_id in field_ids:
        kind = field_id.split(":", 1)[0]
        if kind == "short_description":
            continue
        if kind not in seen:
            refs.append(kind)
            seen.add(kind)
    return refs


def _build_flow_process_context(
    *,
    flow_rows: list[dict[str, Any]],
    process_rows: list[dict[str, Any]],
    max_items: int,
) -> dict[str, dict[str, Any]]:
    if max_items <= 0:
        return {}
    flow_index = build_flow_indexes(flow_rows)
    by_flow_key: dict[str, list[dict[str, Any]]] = {f"{record.id}@{record.version}": [] for record in flow_index["records"]}
    for process_row in process_rows:
        process_id, process_version, process_name = extract_process_identity(process_row)
        dataset = process_dataset_from_row(process_row)
        ref_exchange_id = str(
            deep_get(dataset, ["processInformation", "quantitativeReference", "referenceToReferenceFlow"]) or ""
        ).strip()
        functional_unit = _first_text(
            deep_get(dataset, ["processInformation", "quantitativeReference", "functionalUnitOrOther"])
        )
        process_short_description = _first_text(
            deep_get(dataset, ["processInformation", "dataSetInformation", "common:shortDescription"])
        )
        for exchange in exchange_records(process_row):
            ref = exchange.get("referenceToFlowDataSet") or {}
            flow_id = str(ref.get("@refObjectId") or "").strip()
            flow_version = str(ref.get("@version") or "").strip()
            if not flow_id:
                continue
            matched_keys = _match_flow_keys(flow_index, flow_id, flow_version)
            if not matched_keys:
                continue
            item = {
                "process_id": process_id,
                "process_version": process_version,
                "process_name": process_name,
                "process_short_description": process_short_description,
                "functional_unit_or_other": functional_unit,
                "exchange_internal_id": str(exchange.get("@dataSetInternalID") or "").strip(),
                "exchange_direction": str(exchange.get("exchangeDirection") or "").strip(),
                "exchange_name": str(exchange.get("exchangeName") or exchange.get("name") or "").strip(),
                "reference_flow_text": lang_text(ref.get("common:shortDescription")),
                "link_role": _link_role(exchange, ref_exchange_id),
                "match_mode": "exact_version" if flow_version else "uuid_only",
            }
            for flow_key in matched_keys:
                by_flow_key.setdefault(flow_key, []).append(deepcopy(item))

    context: dict[str, dict[str, Any]] = {}
    for flow_key, items in by_flow_key.items():
        if not items:
            continue
        ranked = sorted(items, key=_process_context_sort_key)
        selected = ranked[:max_items]
        producer_count = sum(1 for item in items if item["link_role"].startswith("producer"))
        consumer_count = sum(1 for item in items if item["link_role"].startswith("consumer"))
        context[flow_key] = {
            "summary": {
                "linked_process_count": len({(item["process_id"], item["process_version"]) for item in items}),
                "linked_exchange_count": len(items),
                "producer_like_count": producer_count,
                "consumer_like_count": consumer_count,
                "selected_count": len(selected),
            },
            "items": selected,
        }
    return context


def _match_flow_keys(flow_index: dict[str, Any], flow_id: str, flow_version: str) -> list[str]:
    exact_key = f"{flow_id}@{flow_version}" if flow_version else ""
    if exact_key and exact_key in flow_index["by_uuid_version"]:
        return [exact_key]
    candidates = flow_index["by_uuid"].get(flow_id) or []
    return [f"{record.id}@{record.version}" for record in candidates]


def _link_role(exchange: dict[str, Any], ref_exchange_id: str) -> str:
    exchange_id = str(exchange.get("@dataSetInternalID") or "").strip()
    direction = str(exchange.get("exchangeDirection") or "").strip().lower()
    if exchange_id and exchange_id == ref_exchange_id:
        if direction == "output":
            return "producer_reference_flow"
        return "reference_flow_link"
    if direction == "output":
        return "producer_output_exchange"
    if direction == "input":
        return "consumer_input_exchange"
    return "other_exchange_link"


def _process_context_sort_key(item: dict[str, Any]) -> tuple[int, str, str, str]:
    role_priority = {
        "producer_reference_flow": 0,
        "producer_output_exchange": 1,
        "reference_flow_link": 2,
        "consumer_input_exchange": 3,
        "other_exchange_link": 4,
    }
    return (
        role_priority.get(str(item.get("link_role") or ""), 9),
        str(item.get("process_name") or ""),
        str(item.get("process_id") or ""),
        str(item.get("exchange_internal_id") or ""),
    )


def _first_text(value: Any) -> str:
    entries = lang_entries(value)
    if entries:
        return str(entries[0].get("text") or "").strip()
    return lang_text(value)


def _ref_review_instructions() -> str:
    return """# OpenClaw Review Instructions

Return a JSON array. Each object must contain:

- `review_id`
- `action`: one of `keep_as_is`, `choose_candidate_ref`, `manual_review`
- optional `chosen_candidate`: `{ "id": "...", "version": "..." }`
- optional `reason`
- optional `confidence`

Rules:

- Only choose from the provided `candidate_refs`.
- If evidence is weak, use `manual_review`.
- Do not invent a new flow id/version.
"""


def _text_review_instructions(entity_type: str, *, has_methodology: bool, has_process_context: bool) -> str:
    extra_rules: list[str] = []
    if has_methodology:
        extra_rules.append("- Follow `review-pack-context.json.methodology.guidance` for field-specific naming constraints.")
    if has_process_context and entity_type == "flow":
        extra_rules.append(
            "- Use `linked_process_contexts` only as contextual evidence to make flow naming more specific; do not rewrite the flow into a process description."
        )
    if entity_type == "flow":
        extra_rules.append(
            "- Keep flow naming fields consistent across `base_name`, `treatment_standards_routes`, `mix_and_location_types`, `flow_properties_text`, and `synonyms` when you patch them."
        )
    extra_rules_text = "\n".join(extra_rules)
    if extra_rules_text:
        extra_rules_text = "\n" + extra_rules_text
    return f"""# OpenClaw Review Instructions

Return a JSON array. Each object must contain:

- `review_id`
- `action`: one of `keep_as_is`, `patch_text_fields`, `manual_review`
- optional `patches`: a list of objects with `field_id`, `value`, optional `reason`, optional `confidence`
- optional `reason`
- optional `confidence`

Rules:

- Only patch `allowed_field_ids`.
- Only improve wording for {entity_type} text fields.
- Do not add new technical facts.
- Do not change identifiers, exchanges, amounts, directions, or references.
{extra_rules_text}
"""


if __name__ == "__main__":
    main()
