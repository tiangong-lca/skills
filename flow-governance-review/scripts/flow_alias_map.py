#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any

from flow_governance_common import (
    alias_lookup,
    build_flow_indexes,
    dump_json,
    dump_jsonl,
    ensure_dir,
    load_json,
    load_rows_from_file,
    normalize_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic flow alias map from pre/post governance snapshots.")
    parser.add_argument("--old-flow-files", nargs="+", required=True)
    parser.add_argument("--new-flow-files", nargs="+", required=True)
    parser.add_argument("--seed-alias-map")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    old_rows: list[dict[str, Any]] = []
    new_rows: list[dict[str, Any]] = []
    for path in args.old_flow_files:
        old_rows.extend(load_rows_from_file(path))
    for path in args.new_flow_files:
        new_rows.extend(load_rows_from_file(path))

    old_index = build_flow_indexes(old_rows, scope_group="old")
    new_index = build_flow_indexes(new_rows, scope_group="new")
    seed_alias_map = load_json(args.seed_alias_map) if args.seed_alias_map else {}

    alias_plan: list[dict[str, Any]] = []
    manual_queue: list[dict[str, Any]] = []
    version_alias_map: dict[str, dict[str, Any]] = {}
    uuid_targets: dict[str, set[str]] = {}

    for old_record in old_index["records"]:
        action = plan_alias(old_record, new_index, seed_alias_map)
        alias_plan.append(action)
        if action["decision"] == "alias_map_entry":
            target = {
                "id": action["target_flow_id"],
                "version": action["target_flow_version"],
                "reason": action["reason"],
            }
            version_alias_map[f"{old_record.id}@{old_record.version}"] = target
            uuid_targets.setdefault(old_record.id, set()).add(f"{target['id']}@{target['version']}")
        elif action["decision"] == "manual_review":
            manual_queue.append(action)

    flow_alias_map = dict(version_alias_map)
    for old_uuid, targets in uuid_targets.items():
        if len(targets) == 1:
            target_id, target_version = next(iter(targets)).split("@", 1)
            flow_alias_map[old_uuid] = {
                "id": target_id,
                "version": target_version,
                "reason": "all_versions_share_same_target",
            }

    summary = {
        "old_flow_count": len(old_index["records"]),
        "new_flow_count": len(new_index["records"]),
        "alias_entries_versioned": len(version_alias_map),
        "alias_entries_uuid_only": len(flow_alias_map) - len(version_alias_map),
        "manual_review_count": len(manual_queue),
        "decision_counts": _decision_counts(alias_plan),
    }

    dump_json(out_dir / "alias-plan.json", alias_plan)
    dump_jsonl(out_dir / "alias-plan.jsonl", alias_plan)
    dump_json(out_dir / "flow-alias-map.json", flow_alias_map)
    dump_jsonl(out_dir / "manual-review-queue.jsonl", manual_queue)
    dump_json(out_dir / "alias-summary.json", summary)


def plan_alias(old_record, new_index: dict[str, Any], seed_alias_map: dict[str, Any]) -> dict[str, Any]:
    base = {
        "old_flow_id": old_record.id,
        "old_flow_version": old_record.version,
        "old_flow_name": old_record.name,
        "old_flow_type": old_record.flow_type,
        "decision": "manual_review",
        "reason": "no_deterministic_match",
    }

    existing_target = new_index["by_uuid_version"].get(f"{old_record.id}@{old_record.version}")
    if existing_target:
        return {
            **base,
            "decision": "no_alias_needed",
            "reason": "already_present_in_target",
            "target_flow_id": existing_target.id,
            "target_flow_version": existing_target.version,
        }

    seeded_target = alias_lookup(seed_alias_map, old_record.id, old_record.version or None)
    if seeded_target:
        target_record = new_index["by_uuid_version"].get(f"{seeded_target.get('id')}@{seeded_target.get('version')}")
        if target_record:
            return {
                **base,
                "decision": "alias_map_entry",
                "reason": "seed_alias_map",
                "target_flow_id": target_record.id,
                "target_flow_version": target_record.version,
                "target_flow_name": target_record.name,
                "target_flow_type": target_record.flow_type,
            }

    same_uuid_candidates = new_index["by_uuid"].get(old_record.id, [])
    if len(same_uuid_candidates) == 1:
        candidate = same_uuid_candidates[0]
        return {
            **base,
            "decision": "alias_map_entry",
            "reason": "same_uuid_single_target_version",
            "target_flow_id": candidate.id,
            "target_flow_version": candidate.version,
            "target_flow_name": candidate.name,
            "target_flow_type": candidate.flow_type,
        }
    if len(same_uuid_candidates) > 1:
        return {
            **base,
            "decision": "manual_review",
            "reason": "same_uuid_multiple_target_versions",
            "candidate_refs": [{"id": item.id, "version": item.version, "name": item.name} for item in same_uuid_candidates],
        }

    by_name = new_index["by_name"].get(normalize_text(old_record.name), [])
    name_and_type_candidates = [item for item in by_name if item.flow_type == old_record.flow_type]
    if len(name_and_type_candidates) == 1:
        candidate = name_and_type_candidates[0]
        return {
            **base,
            "decision": "alias_map_entry",
            "reason": "unique_exact_name_and_type_match",
            "target_flow_id": candidate.id,
            "target_flow_version": candidate.version,
            "target_flow_name": candidate.name,
            "target_flow_type": candidate.flow_type,
        }
    if len(name_and_type_candidates) > 1:
        return {
            **base,
            "decision": "manual_review",
            "reason": "ambiguous_name_and_type_match",
            "candidate_refs": [{"id": item.id, "version": item.version, "name": item.name} for item in name_and_type_candidates],
        }

    if by_name:
        return {
            **base,
            "decision": "manual_review",
            "reason": "name_match_flow_type_mismatch",
            "candidate_refs": [{"id": item.id, "version": item.version, "name": item.name, "flow_type": item.flow_type} for item in by_name],
        }

    return base


def _decision_counts(actions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in actions:
        decision = str(action.get("decision") or "UNKNOWN")
        counts[decision] = counts.get(decision, 0) + 1
    return counts


if __name__ == "__main__":
    main()
