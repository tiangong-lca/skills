#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flow_governance_common import (
    deep_get,
    dump_json,
    dump_jsonl,
    ensure_dir,
    entity_text_fields,
    extract_flow_identity,
    FLOW_GOVERNANCE_ROOT,
    FLOW_PROCESSING_DATASETS_DIR,
    FLOW_PROCESSING_NAMING_DIR,
    flow_dataset_from_row,
    lang_entries,
    lang_text,
    load_rows_from_file,
)
from openclaw_review_handoff import (
    _build_flow_process_context,
    _field_guidance_refs,
    _load_text_review_methodology,
    _text_review_id,
)


DEFAULT_CHECKLIST_FILE = (
    FLOW_PROCESSING_NAMING_DIR / "remaining-after-aggressive" / "remaining-incomplete-zero-process.json"
)
DEFAULT_ROWS_FILE = (
    FLOW_PROCESSING_DATASETS_DIR / "flows_tidas_sdk_plus_classification_round2_sdk018_all_final_resolved.jsonl"
)
DEFAULT_PROCESSES_FILE = FLOW_PROCESSING_DATASETS_DIR / "process_pool.jsonl"
DEFAULT_OUT_DIR = FLOW_PROCESSING_NAMING_DIR / "zero-process-completion-pack"
DEFAULT_METHODOLOGY_FILE = FLOW_GOVERNANCE_ROOT / "references" / "tidas_flows.yaml"
DEFAULT_MAX_SAME_CLASS_EXAMPLES = 3


FIELD_KIND_BY_MISSING_NAME_FIELD = {
    "baseName": "base_name",
    "treatmentStandardsRoutes": "treatment_standards_routes",
    "mixAndLocationTypes": "mix_and_location_types",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a missing-name-field completion pack for LLM review. "
            "The pack keeps only missing flow naming fields editable, while attaching "
            "methodology guidance and selected process context."
        )
    )
    parser.add_argument("--checklist-file", default=str(DEFAULT_CHECKLIST_FILE))
    parser.add_argument("--rows-file", default=str(DEFAULT_ROWS_FILE))
    parser.add_argument("--processes-file", default=str(DEFAULT_PROCESSES_FILE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--methodology-file", default=str(DEFAULT_METHODOLOGY_FILE))
    parser.add_argument("--methodology-id", default="tidas_flows.yaml")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-process-contexts", type=int, default=5)
    parser.add_argument("--max-same-class-examples", type=int, default=DEFAULT_MAX_SAME_CLASS_EXAMPLES)
    parser.add_argument("--flow-id", action="append", dest="flow_ids", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    batches_dir = ensure_dir(out_dir / "batches")

    checklist_rows = load_rows_from_file(args.checklist_file)
    source_rows = load_rows_from_file(args.rows_file)
    process_rows = load_rows_from_file(args.processes_file)

    flow_filter = {item.strip() for item in args.flow_ids if str(item or "").strip()}
    if flow_filter:
        checklist_rows = [
            row
            for row in checklist_rows
            if str((normalize_checklist_flow(row).get("flow_id")) or "").strip() in flow_filter
        ]
    if args.max_items > 0:
        checklist_rows = checklist_rows[: args.max_items]

    row_by_key = {
        f"{flow_id}@{version}": row
        for row in source_rows
        for flow_id, version, _name in [extract_flow_identity(row)]
        if flow_id and version
    }
    scope_rows = []
    missing_scope_keys = []
    for item in checklist_rows:
        flow = normalize_checklist_flow(item)
        flow_key = str(flow.get("flow_key") or "").strip()
        row = row_by_key.get(flow_key)
        if row is None:
            missing_scope_keys.append(flow_key)
            continue
        scope_rows.append(row)

    linked_process_context_by_flow_key = _build_flow_process_context(
        flow_rows=scope_rows,
        process_rows=process_rows,
        max_items=max(args.max_process_contexts, 0),
    )
    same_class_examples_by_flow_key = build_same_class_examples_by_flow_key(
        checklist_rows=checklist_rows,
        source_rows=source_rows,
        max_examples=max(args.max_same_class_examples, 0),
    )
    methodology_context = _load_text_review_methodology(
        "flow",
        args.methodology_file,
        args.methodology_id,
    )

    review_pack: list[dict[str, Any]] = []
    skipped_items: list[dict[str, Any]] = []
    for item in checklist_rows:
        pack_item = build_review_item(
            checklist_item=item,
            row_by_key=row_by_key,
            linked_process_context_by_flow_key=linked_process_context_by_flow_key,
            same_class_examples_by_flow_key=same_class_examples_by_flow_key,
            include_methodology=methodology_context is not None,
        )
        if pack_item is None:
            skipped_items.append(
                {
                    "flow_key": str(((item.get("flow") or {}).get("flow_key")) or ""),
                    "reason": "missing_scope_row_or_editable_missing_fields",
                }
            )
            continue
        review_pack.append(pack_item)

    context_payload: dict[str, Any] = {
        "entity_type": "flow",
        "review_scope": "missing_name_field_completion",
        "source_checklist_file": str(Path(args.checklist_file).expanduser().resolve()),
        "source_rows_file": str(Path(args.rows_file).expanduser().resolve()),
        "source_processes_file": str(Path(args.processes_file).expanduser().resolve()),
        "max_process_contexts_per_flow": max(args.max_process_contexts, 0),
        "max_same_class_examples_per_flow": max(args.max_same_class_examples, 0),
        "notes": [
            "Only the currently missing naming fields are editable in this pack.",
            "Model decisions should stay within methodology guidance and supplied local evidence.",
            "Use same-class complete flow examples before searching the web.",
            "If local evidence is insufficient, targeted web research is allowed for stable generic naming qualifiers.",
            "If evidence remains ambiguous after search, return keep_as_is instead of guessing.",
        ],
        "web_search_policy": {
            "enabled": True,
            "intent": "Fill missing naming qualifiers when local process/context evidence is absent or weak.",
            "rules": [
                "Search only for stable generic technical qualifiers, common industrial route names, grades, intended-use qualifiers, and standard delivery-point phrasing.",
                "Do not infer supplier-specific, plant-specific, or region-specific claims unless the current flow evidence clearly points to them.",
                "If search results remain conflicting or variant-specific, keep_as_is.",
            ],
        },
    }
    if methodology_context is not None:
        context_payload["methodology"] = methodology_context

    scope_flows_file = out_dir / "scope-flows.jsonl"
    dump_jsonl(scope_flows_file, scope_rows)
    dump_json(out_dir / "review-pack.json", review_pack)
    dump_jsonl(out_dir / "review-pack.jsonl", review_pack)
    dump_json(out_dir / "review-pack-context.json", context_payload)
    dump_jsonl(out_dir / "skipped-items.jsonl", skipped_items)
    dump_json(
        out_dir / "review-pack-summary.json",
        {
            "review_item_count": len(review_pack),
            "scope_flow_count": len(scope_rows),
            "skipped_item_count": len(skipped_items),
            "missing_scope_row_count": len([key for key in missing_scope_keys if key]),
            "has_methodology_context": methodology_context is not None,
            "has_process_context": bool(linked_process_context_by_flow_key),
            "same_class_example_item_count": sum(1 for value in same_class_examples_by_flow_key.values() if value),
            "batch_size": max(args.batch_size, 0),
        },
    )
    (out_dir / "openclaw-instructions.md").write_text(
        openclaw_instructions(has_methodology=methodology_context is not None),
        encoding="utf-8",
    )
    (out_dir / "openclaw-prompt.md").write_text(
        openclaw_prompt(
            out_dir=out_dir,
            scope_flows_file=scope_flows_file,
            batch_size=max(args.batch_size, 0),
            review_item_count=len(review_pack),
        ),
        encoding="utf-8",
    )
    (out_dir / "openclaw-nightly-run.md").write_text(
        openclaw_nightly_run_prompt(out_dir=out_dir),
        encoding="utf-8",
    )

    batch_manifest = write_batches(
        batches_dir=batches_dir,
        review_pack=review_pack,
        context_payload=context_payload,
        scope_rows=scope_rows,
        batch_size=max(args.batch_size, 0),
    )
    dump_json(out_dir / "batch-manifest.json", batch_manifest)
    print(str(out_dir / "review-pack-summary.json"))


def build_review_item(
    *,
    checklist_item: dict[str, Any],
    row_by_key: dict[str, dict[str, Any]],
    linked_process_context_by_flow_key: dict[str, dict[str, Any]],
    same_class_examples_by_flow_key: dict[str, list[dict[str, Any]]],
    include_methodology: bool,
) -> dict[str, Any] | None:
    flow = normalize_checklist_flow(checklist_item)
    flow_id = str(flow.get("flow_id") or "").strip()
    flow_version = str(flow.get("flow_version") or "").strip()
    flow_key = str(flow.get("flow_key") or "").strip()
    flow_name = str(flow.get("flow_name") or "").strip()
    if not flow_id or not flow_version or not flow_key:
        return None
    row = row_by_key.get(flow_key)
    if row is None:
        return None
    row_context = build_row_name_context(row)
    if not flow_name:
        _flow_id, _flow_version, fallback_name = extract_flow_identity(row)
        flow_name = fallback_name

    missing_name_fields = [
        value
        for value in flow.get("missing_name_fields", [])
        if str(value or "").strip() in FIELD_KIND_BY_MISSING_NAME_FIELD
    ]
    editable_kinds = [FIELD_KIND_BY_MISSING_NAME_FIELD[value] for value in missing_name_fields]
    all_fields = entity_text_fields(row, "flow", include_placeholders=True)
    allowed_field_ids = sorted(
        field_id
        for field_id in all_fields
        if field_id.split(":", 1)[0] in set(editable_kinds)
    )
    if not allowed_field_ids:
        return None

    fields = {field_id: all_fields[field_id]["value"] for field_id in allowed_field_ids}
    process_context = linked_process_context_by_flow_key.get(flow_key) or {}
    item: dict[str, Any] = {
        "review_id": _text_review_id("flow", flow_id, flow_version),
        "review_kind": "text_fields",
        "review_scope": "missing_name_field_completion",
        "entity_type": "flow",
        "entity_id": flow_id,
        "entity_version": flow_version,
        "entity_name": flow_name,
        "allowed_actions": ["keep_as_is", "patch_text_fields"],
        "allowed_field_ids": allowed_field_ids,
        "fields": fields,
        "missing_name_fields": missing_name_fields,
        "current_name_context": {
            "name_fields": flow.get("name_fields") or row_context["name_fields"],
            "name_field_presence": flow.get("name_field_presence") or {},
            "classification_path": flow.get("classification_path") or row_context["classification_path"],
            "classification_leaf": flow.get("classification_leaf") or row_context["classification_leaf"],
            "reference_flow_property": flow.get("reference_flow_property") or row_context["reference_flow_property"],
            "synonyms": flow.get("synonyms") or row_context["synonyms"],
            "general_comment": flow.get("general_comment") or row_context["general_comment"],
        },
        "process_ref_stats": checklist_item.get("process_ref_stats") or {},
        "completion_evidence": checklist_item.get("completion_evidence") or {},
        "instructions": (
            "Fill only the currently missing name fields. "
            "Use local evidence first, including same-class complete flow examples when present. "
            "If local evidence is insufficient, targeted web research is allowed for stable generic naming qualifiers. "
            "If ambiguity remains after search, return keep_as_is."
        ),
    }
    if include_methodology:
        item["review_context_refs"] = ["flow_text_methodology"]
        guidance_refs = _field_guidance_refs(allowed_field_ids)
        if guidance_refs:
            item["field_guidance_refs"] = guidance_refs
    same_class_examples = same_class_examples_by_flow_key.get(flow_key) or []
    if same_class_examples:
        item["same_class_examples"] = same_class_examples
    if process_context:
        item["linked_process_context_summary"] = process_context.get("summary") or {}
        item["linked_process_contexts"] = process_context.get("items") or []
    return item


def normalize_checklist_flow(checklist_item: dict[str, Any]) -> dict[str, Any]:
    flow = checklist_item.get("flow") if isinstance(checklist_item.get("flow"), dict) else checklist_item
    if not isinstance(flow, dict):
        return {}
    normalized = dict(flow)
    flow_id = str(normalized.get("flow_id") or normalized.get("id") or "").strip()
    flow_version = str(normalized.get("flow_version") or normalized.get("version") or "").strip()
    if flow_id and flow_version and not str(normalized.get("flow_key") or "").strip():
        normalized["flow_key"] = f"{flow_id}@{flow_version}"
    if not str(normalized.get("type_of_dataset") or "").strip() and str(normalized.get("typeOfDataSet") or "").strip():
        normalized["type_of_dataset"] = str(normalized.get("typeOfDataSet") or "").strip()
    return normalized


def build_row_name_context(row: dict[str, Any]) -> dict[str, Any]:
    dataset = flow_dataset_from_row(row)
    info = deep_get(dataset, ["flowInformation", "dataSetInformation"], {})
    name_block = info.get("name") if isinstance(info.get("name"), dict) else {}
    reference_flow_property = (
        deep_get(dataset, ["flowProperties", "flowProperty", "referenceToFlowPropertyDataSet"], {})
        if isinstance(deep_get(dataset, ["flowProperties", "flowProperty", "referenceToFlowPropertyDataSet"], {}), dict)
        else {}
    )
    classification_path = _classification_path(info) if isinstance(info, dict) else []
    return {
        "name_fields": {
            "baseName": _lang_map(name_block.get("baseName")) if isinstance(name_block, dict) else {},
            "treatmentStandardsRoutes": _lang_map(name_block.get("treatmentStandardsRoutes")) if isinstance(name_block, dict) else {},
            "mixAndLocationTypes": _lang_map(name_block.get("mixAndLocationTypes")) if isinstance(name_block, dict) else {},
            "flowProperties": _lang_map(name_block.get("flowProperties")) if isinstance(name_block, dict) else {},
        },
        "classification_path": classification_path,
        "classification_leaf": classification_path[-1] if classification_path else "",
        "reference_flow_property": reference_flow_property,
        "synonyms": lang_entries(info.get("common:synonyms")) if isinstance(info, dict) else [],
        "general_comment": lang_entries(info.get("common:generalComment")) if isinstance(info, dict) else [],
    }


def build_same_class_examples_by_flow_key(
    *,
    checklist_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    max_examples: int,
) -> dict[str, list[dict[str, Any]]]:
    if max_examples <= 0:
        return {}
    index = build_same_class_example_index(source_rows)
    examples_by_flow_key: dict[str, list[dict[str, Any]]] = {}
    for checklist_item in checklist_rows:
        flow = normalize_checklist_flow(checklist_item)
        flow_key = str(flow.get("flow_key") or "").strip()
        flow_id = str(flow.get("flow_id") or "").strip()
        flow_type = str(flow.get("type_of_dataset") or flow.get("typeOfDataSet") or "").strip()
        classification_leaf = str(flow.get("classification_leaf") or "").strip()
        if not flow_key or not flow_type or not classification_leaf:
            continue
        candidates = index.get(_same_class_key(flow_type, classification_leaf), [])
        picked: list[dict[str, Any]] = []
        for candidate in candidates:
            if str(candidate.get("flow_id") or "").strip() == flow_id:
                continue
            picked.append({key: value for key, value in candidate.items() if key != "flow_key"})
            if len(picked) >= max_examples:
                break
        if picked:
            examples_by_flow_key[flow_key] = picked
    return examples_by_flow_key


def build_same_class_example_index(source_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for row in source_rows:
        example = build_same_class_example(row)
        if example is None:
            continue
        index.setdefault(_same_class_key(example["type_of_dataset"], example["classification_leaf"]), []).append(example)
    for candidates in index.values():
        candidates.sort(
            key=lambda item: (
                str(item.get("flow_name") or ""),
                str(item.get("flow_version") or ""),
                str(item.get("flow_id") or ""),
            )
        )
    return index


def build_same_class_example(row: dict[str, Any]) -> dict[str, Any] | None:
    flow_id, flow_version, flow_name = extract_flow_identity(row)
    if not flow_id or not flow_version:
        return None
    dataset = flow_dataset_from_row(row)
    info = deep_get(dataset, ["flowInformation", "dataSetInformation"], {})
    if not isinstance(info, dict):
        return None
    name_block = info.get("name") if isinstance(info.get("name"), dict) else {}
    if not isinstance(name_block, dict):
        return None
    base_name = _lang_map(name_block.get("baseName"))
    treatment = _lang_map(name_block.get("treatmentStandardsRoutes"))
    mix = _lang_map(name_block.get("mixAndLocationTypes"))
    if not (_has_text_map(base_name) and _has_text_map(treatment) and _has_text_map(mix)):
        return None
    classification_path = _classification_path(info)
    classification_leaf = classification_path[-1] if classification_path else ""
    flow_type = str(
        deep_get(dataset, ["modellingAndValidation", "LCIMethodAndAllocation", "typeOfDataSet"])
        or deep_get(dataset, ["modellingAndValidation", "LCIMethod", "typeOfDataSet"])
        or row.get("typeOfDataSet")
        or ""
    ).strip()
    if not flow_type or not classification_leaf:
        return None
    return {
        "flow_key": f"{flow_id}@{flow_version}",
        "flow_id": flow_id,
        "flow_version": flow_version,
        "flow_name": flow_name,
        "type_of_dataset": flow_type,
        "classification_path": classification_path,
        "classification_leaf": classification_leaf,
        "name_fields": {
            "base_name": base_name,
            "treatment_standards_routes": treatment,
            "mix_and_location_types": mix,
            "flow_properties_text": _lang_map(name_block.get("flowProperties")),
        },
    }


def _same_class_key(flow_type: str, classification_leaf: str) -> str:
    return f"{str(flow_type or '').strip().lower()}||{str(classification_leaf or '').strip().lower()}"


def _classification_path(info: dict[str, Any]) -> list[str]:
    classes = deep_get(info, ["classificationInformation", "common:classification", "common:class"], [])
    if isinstance(classes, dict):
        classes = [classes]
    if not isinstance(classes, list):
        return []
    path: list[str] = []
    for item in classes:
        if not isinstance(item, dict):
            continue
        text = str(item.get("#text") or "").strip()
        if text:
            path.append(text)
    return path


def _lang_map(value: Any) -> dict[str, str]:
    values_by_lang: dict[str, str] = {}
    for item in lang_entries(value):
        lang = str(item.get("lang") or "en").strip() or "en"
        text = str(item.get("text") or "").strip()
        if text and lang not in values_by_lang:
            values_by_lang[lang] = text
    if not values_by_lang:
        fallback = lang_text(value)
        if fallback:
            values_by_lang["en"] = fallback
    return values_by_lang


def _has_text_map(values_by_lang: dict[str, str]) -> bool:
    return any(str(value or "").strip() for value in values_by_lang.values())


def write_batches(
    *,
    batches_dir: Path,
    review_pack: list[dict[str, Any]],
    context_payload: dict[str, Any],
    scope_rows: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    if batch_size <= 0:
        batch_size = len(review_pack) or 1
    scope_row_by_key = {
        f"{flow_id}@{version}": row
        for row in scope_rows
        for flow_id, version, _name in [extract_flow_identity(row)]
        if flow_id and version
    }
    batches: list[dict[str, Any]] = []
    for index in range(0, len(review_pack), batch_size):
        batch_items = review_pack[index : index + batch_size]
        batch_no = len(batches) + 1
        batch_slug = f"batch-{batch_no:04d}"
        batch_dir = ensure_dir(batches_dir / batch_slug)
        flow_keys = [
            f"{item.get('entity_id')}@{item.get('entity_version')}"
            for item in batch_items
            if str(item.get("entity_id") or "").strip() and str(item.get("entity_version") or "").strip()
        ]
        batch_scope_rows = [scope_row_by_key[key] for key in flow_keys if key in scope_row_by_key]
        dump_json(batch_dir / "review-pack.json", batch_items)
        dump_jsonl(batch_dir / "review-pack.jsonl", batch_items)
        dump_json(batch_dir / "review-pack-context.json", context_payload)
        dump_jsonl(batch_dir / "scope-flows.jsonl", batch_scope_rows)
        (batch_dir / "openclaw-instructions.md").write_text(
            openclaw_instructions(has_methodology=bool(context_payload.get("methodology"))),
            encoding="utf-8",
        )
        (batch_dir / "openclaw-prompt.md").write_text(
            openclaw_prompt(
                out_dir=batch_dir,
                scope_flows_file=batch_dir / "scope-flows.jsonl",
                batch_size=len(batch_items),
                review_item_count=len(batch_items),
            ),
            encoding="utf-8",
        )
        batches.append(
            {
                "batch_no": batch_no,
                "batch_slug": batch_slug,
                "review_item_count": len(batch_items),
                "review_pack_file": str(batch_dir / "review-pack.json"),
                "review_pack_context_file": str(batch_dir / "review-pack-context.json"),
                "scope_flows_file": str(batch_dir / "scope-flows.jsonl"),
                "openclaw_prompt_file": str(batch_dir / "openclaw-prompt.md"),
            }
        )
    return {
        "batch_count": len(batches),
        "batch_size": batch_size,
        "batches": batches,
    }


def openclaw_instructions(*, has_methodology: bool) -> str:
    extra_rules = [
        "- Patch only `allowed_field_ids`.",
        "- Fill only fields that are currently missing or empty in the pack.",
        "- Do not rewrite already populated naming fields.",
        "- Use `same_class_examples`, `linked_process_contexts`, `completion_evidence`, and the current classification/property context as local evidence first.",
        "- If local evidence is insufficient, perform targeted web research for stable generic naming qualifiers before choosing `keep_as_is`.",
        "- Use web research only for generic technical qualifiers, common route names, grades, intended-use qualifiers, and delivery-point phrasing.",
        "- If web results remain conflicting, variant-specific, or weak, use `keep_as_is` instead of guessing.",
        "- Do not transform the flow into a process description.",
        "- Keep English and Chinese fields aligned in meaning when both are patched.",
        "- Do not add identifiers, numeric properties, standards, routes, locations, or uses that are absent from the evidence.",
        "- Never output placeholder wording such as `NA`, `N/A`, `Unspecified`, or `Unspecified treatment`.",
        "- For `Product flow`, when `mix_and_location_types:*` is missing and there is no contrary evidence, `Production mix, at plant` / `生产混合，在工厂` is allowed as a conservative default.",
        "- For `Waste flow`, do not invent a default mix/location phrase; patch it only when evidence supports it.",
        "- Do not fabricate `treatment_standards_routes:*`; if no specific technical qualifier is supportable, use `keep_as_is`.",
    ]
    if has_methodology:
        extra_rules.append("- Follow `review-pack-context.json.methodology.guidance` for field-specific naming constraints.")
    extra_rules_text = "\n".join(extra_rules)
    return f"""# OpenClaw Review Instructions

Return a JSON array. Each object must contain:

- `review_id`
- `action`: one of `keep_as_is`, `patch_text_fields`
- optional `patches`: a list of objects with `field_id`, `value`, optional `reason`, optional `confidence`
- optional `reason`
- optional `confidence`

Rules:

{extra_rules_text}
"""


def openclaw_prompt(*, out_dir: Path, scope_flows_file: Path, batch_size: int, review_item_count: int) -> str:
    return f"""请执行这批 flow 缺失命名字段补全。

工作目录：
- `{out_dir}`

输入文件：
- `review-pack.json`
- `review-pack-context.json`
- `openclaw-instructions.md`

目标：
- 仅补全当前缺失的 flow naming 字段，主要是 `treatment_standards_routes:*` 和 `mix_and_location_types:*`
- 必须先使用 pack 里已有本地 evidence：`same_class_examples`、`linked_process_contexts`、`completion_evidence`、当前 flow 的 classification / property / synonyms / general comment
- 必须遵守 `review-pack-context.json.methodology.guidance`
- 本地证据不足时，主动上网检索该 flow 术语在行业中的通用技术限定词、路线词、等级词、用途限定词或交付点表达
- 检索后仍不能确定时，返回 `keep_as_is`
- `Product flow` 缺少 `mixAndLocationTypes` 且没有相反证据时，可保守补为 `Production mix, at plant` / `生产混合，在工厂`
- `Waste flow` 不要套用上述默认值
- `treatmentStandardsRoutes` 禁止写 `Unspecified treatment`、`Unspecified`、`NA`、`N/A`

输出要求：
- 生成 `openclaw-decisions.json`
- 返回 JSON array，格式严格遵守 `openclaw-instructions.md`
- 不要输出解释性正文，不要输出 markdown

后续本地 apply 命令：
```bash
python3 /home/huimin/projects/lca-skills/flow-governance-review/scripts/openclaw_review_handoff.py apply-text \\
  --entity-type flow \\
  --rows-file {scope_flows_file} \\
  --review-pack {out_dir / "review-pack.json"} \\
  --decisions-file {out_dir / "openclaw-decisions.json"} \\
  --out-dir {out_dir / "applied"}
```

当前 batch 信息：
- `review_item_count={review_item_count}`
- `batch_size={batch_size}`
"""


def openclaw_nightly_run_prompt(*, out_dir: Path) -> str:
    return f"""请把这次缺失 flow naming 字段补全作为一个长任务连续跑完，不要等待用户逐批确认。

固定工作目录：
- `{out_dir}`

目标：
1. 按 `batch-manifest.json` 顺序处理全部 batch。
2. 对每个 batch：
   - 读取 `review-pack.json`、`review-pack-context.json`、`openclaw-instructions.md`
   - 生成 `openclaw-decisions.json`
   - 只能补当前缺失字段，不要改已有字段
   - 先用本地 evidence 和 `same_class_examples`
   - 本地 evidence 不够时，主动上网检索
   - 检索后仍不确定就返回 `keep_as_is`
3. 所有 batch 跑完后，执行本地后处理脚本：
```bash
python3 /home/huimin/projects/lca-skills/flow-governance-review/scripts/run_missing_name_completion_postprocess.py \\
  --pack-root {out_dir}
```
4. 如果 `aggregate/pipeline-status.json` 显示：
   - `pending_batch_count=0`
   - `blocked_batch_count=0`
   - `patched_flow_row_count>0`
   则继续执行真实 publish：
```bash
python3 /home/huimin/projects/lca-skills/flow-governance-review/scripts/run_missing_name_completion_postprocess.py \\
  --pack-root {out_dir} \\
  --commit
```

执行约束：
- 不要逐批停下来问用户。
- 中途如果对话上下文中断，先读取：
  - `batch-manifest.json`
  - `aggregate/pipeline-status.json`（如果存在）
  - 各 batch 目录里是否已有 `openclaw-decisions.json`
  然后从未完成的 batch 继续。
- 如果本地 evidence 不够，先检索；检索后仍不确定就 `keep_as_is`，不要硬猜。
- publish 只在所有 batch 都没有 pending/blocked 时执行。
- `publish --commit` 只会写入当前 MCP/CRUD 账号；如果后续还要提到 `state_code=100`，那是 publish 之后的独立步骤，不在这里猜测执行。
"""


if __name__ == "__main__":
    main()
