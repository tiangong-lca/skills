#!/usr/bin/env python3
"""Read-only planner and local builder for lifecycle model automated builder."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST: dict[str, Any] = {
    "run_label": "lifecyclemodel-automated-builder",
    "allow_remote_write": False,
    "discovery": {
        "sources": [
            {"kind": "account_processes", "selector": "all-accessible"},
            {
                "kind": "public_open_data",
                "table": "processes",
                "filters": {"state_code": 100},
            },
        ],
        "supporting_open_tables": ["flows", "sources", "lifecyclemodels"],
        "batch_limit": 200,
        "reference_model_queries": [],
        "reference_model_select_limit": 3,
    },
    "selection": {
        "mode": "llm_judgement",
        "max_models": 25,
        "max_processes_per_model": 12,
        "decision_factors": [
            "shared product system or classification lineage",
            "explicit exchange connectivity",
            "quantitative reference completeness",
            "geography and time coherence",
        ],
    },
    "reuse": {
        "reusable_process_dirs": [],
        "include_reference_model_resulting_processes": True,
    },
    "output": {
        "write_local_models": True,
        "emit_validation_report": True,
    },
    "local_runs": [],
    "publish": {
        "enabled": False,
        "mode": "mcp_insert_only",
        "target_runs": [],
        "select_after_insert": True,
        "max_attempts": 5,
        "retry_delay_seconds": 2.0,
    },
}

TIDAS_TOOLS_PYTHON = Path("/home/huimin/projects/tidas-tools/.venv/bin/python")
TIDAS_TOOLS_SRC = Path("/home/huimin/projects/tidas-tools/src")


@dataclass
class Edge:
    src: str
    dst: str
    flow_uuid: str
    downstream_input_amount: Decimal
    confidence: Decimal
    reasons: list[str]


@dataclass
class ProcessRecord:
    process_uuid: str
    version: str
    path: Path
    raw: dict[str, Any]
    reference_exchange_internal_id: str
    reference_flow_uuid: str
    reference_direction: str
    reference_amount: Decimal
    input_amounts: dict[str, Decimal]
    output_amounts: dict[str, Decimal]
    name_en: str
    name_zh: str
    route_en: str
    mix_en: str
    geography_code: str
    classification_path: list[str]
    token_set: set[str]
    source_kind: str
    source_label: str
    included_process_ref_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a read-only execution plan or build local lifecycle models from process run artifacts."
    )
    parser.add_argument("--manifest", required=True, help="Path to the request manifest JSON.")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where local planning artifacts should be written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved execution plan instead of writing artifacts.",
    )
    return parser.parse_args()


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = deepcopy(base)
        for key, value in override.items():
            merged[key] = deep_merge(merged.get(key), value)
        return merged
    return deepcopy(override)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("manifest root must be a JSON object")
    return deep_merge(DEFAULT_MANIFEST, raw)


def decimal_or_zero(value: Any) -> Decimal:
    if value is None:
        return Decimal("0")
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError):
        return Decimal("0")


def copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def multilang_from_text(en_text: str, zh_text: str | None = None) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if en_text:
        items.append({"@xml:lang": "en", "#text": en_text})
    if zh_text:
        items.append({"@xml:lang": "zh", "#text": zh_text})
    return items


def first_text(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0].get("#text", ""))
    if isinstance(value, dict):
        return str(value.get("#text", ""))
    return ""


def tokenize_text(value: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in value)
    return {token for token in cleaned.split() if len(token) >= 3}


def multilang_text(value: Any) -> tuple[str, str]:
    if isinstance(value, list):
        en = ""
        zh = ""
        for item in value:
            if not isinstance(item, dict):
                continue
            lang = str(item.get("@xml:lang", "")).lower()
            text = str(item.get("#text", "")).strip()
            if not text:
                continue
            if lang.startswith("en") and not en:
                en = text
            if lang.startswith("zh") and not zh:
                zh = text
        return en, zh
    text = first_text(value)
    return text, text


def lang_text_map(value: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in ensure_list(value):
        if not isinstance(item, dict):
            continue
        lang = str(item.get("@xml:lang", "")).strip().lower() or "en"
        text = str(item.get("#text", "")).strip()
        if text and lang not in mapping:
            mapping[lang] = text
    if mapping:
        return mapping
    text = first_text(value).strip()
    return {"en": text} if text else {}


def localized_text(value: Any, preferred: str = "zh") -> str:
    mapping = lang_text_map(value)
    if preferred in mapping:
        return mapping[preferred]
    if preferred.startswith("zh"):
        for candidate in ("zh-cn", "zh-hans", "zh"):
            if candidate in mapping:
                return mapping[candidate]
    if "en" in mapping:
        return mapping["en"]
    return next(iter(mapping.values()), "")


def build_name_summary(name_info: dict[str, Any]) -> list[dict[str, str]]:
    base = lang_text_map((name_info or {}).get("baseName"))
    route = lang_text_map((name_info or {}).get("treatmentStandardsRoutes"))
    mix = lang_text_map((name_info or {}).get("mixAndLocationTypes"))
    functional = lang_text_map((name_info or {}).get("functionalUnitFlowProperties"))
    lang_order: list[str] = []
    for mapping in (base, route, mix, functional):
        for lang in mapping:
            if lang not in lang_order:
                lang_order.append(lang)
    if not lang_order:
        return []

    def fallback(mapping: dict[str, str], lang: str) -> str:
        return (
            mapping.get(lang)
            or mapping.get("zh")
            or mapping.get("zh-cn")
            or mapping.get("en")
            or next(iter(mapping.values()), "")
        )

    summary: list[dict[str, str]] = []
    for lang in lang_order:
        parts = [
            fallback(base, lang),
            fallback(route, lang),
            fallback(mix, lang),
            fallback(functional, lang),
        ]
        text = "; ".join(part for part in parts if part).strip()
        if text:
            summary.append({"@xml:lang": lang, "#text": text})
    return summary


def extract_classification_path(classification_info: Any) -> list[str]:
    carrier = (classification_info or {}).get("common:classification", {})
    items = ensure_list(carrier.get("common:class"))
    labels: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("#text", "")).strip()
            if text:
                labels.append(text)
    return labels


def classification_overlap(left: list[str], right: list[str]) -> int:
    overlap = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        overlap += 1
    return overlap


def runtime_config() -> dict[str, Any]:
    return {
        "remote": {
            "transport": os.getenv("TIANGONG_LCA_REMOTE_TRANSPORT", "streamable_http"),
            "service_name": os.getenv(
                "TIANGONG_LCA_REMOTE_SERVICE_NAME", "TianGong_LCA_Remote"
            ),
            "url": os.getenv("TIANGONG_LCA_REMOTE_URL", "https://lcamcp.tiangong.earth/mcp"),
            "api_key_present": bool(os.getenv("TIANGONG_LCA_REMOTE_API_KEY")),
        },
        "openai": {
            "model": os.getenv("OPENAI_MODEL", "gpt-5"),
            "api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        },
    }


def selection_brief(manifest: dict[str, Any]) -> str:
    decision_factors = manifest["selection"]["decision_factors"]
    decision_lines = "\n".join(f"- {item}" for item in decision_factors)
    reference_queries = [
        str(item).strip()
        for item in ensure_list(manifest["discovery"].get("reference_model_queries"))
        if str(item).strip()
    ]
    reuse_dirs = [
        str(Path(item).expanduser())
        for item in ensure_list((manifest.get("reuse") or {}).get("reusable_process_dirs"))
        if str(item).strip()
    ]
    extra_lines: list[str] = []
    if reference_queries:
        extra_lines.append(
            "Reference lifecycle models should be retrieved read-only via MCP to learn model shape and resulting-process reuse."
        )
        extra_lines.extend(f"- reference model query: {item}" for item in reference_queries)
    if reuse_dirs:
        extra_lines.append("Reusable resulting processes should be considered as candidate upstream or downstream building blocks.")
        extra_lines.extend(f"- reusable process dir: {item}" for item in reuse_dirs)
    return (
        "# Selection Brief\n\n"
        "This run is read-only and should only produce local planning artifacts.\n\n"
        "Candidate processes should be selected by AI using these factors:\n"
        f"{decision_lines}\n"
        + ("\n" + "\n".join(extra_lines) + "\n" if extra_lines else "")
    )


def build_global_reference(
    ref_object_id: str,
    version: str,
    short_description: Any,
    dataset_type: str,
    uri: str,
) -> dict[str, Any]:
    return {
        "@refObjectId": ref_object_id,
        "@type": dataset_type,
        "@uri": uri,
        "@version": version,
        "common:shortDescription": copy_json(short_description),
    }


def load_process_record(
    path: Path,
    *,
    source_kind: str = "local_run_export",
    source_label: str = "",
) -> ProcessRecord:
    raw = json.loads(path.read_text(encoding="utf-8"))
    dataset = raw["processDataSet"]
    info = dataset["processInformation"]
    data_info = info["dataSetInformation"]
    name_info = data_info.get("name", {})
    publication = dataset["administrativeInformation"]["publicationAndOwnership"]
    geography = info.get("geography", {})
    technology = info.get("technology") or {}
    process_uuid = data_info["common:UUID"]
    version = publication["common:dataSetVersion"]
    ref_internal_id = info["quantitativeReference"]["referenceToReferenceFlow"]
    exchanges = ensure_list(dataset.get("exchanges", {}).get("exchange"))
    ref_exchange = next(
        (item for item in exchanges if item.get("@dataSetInternalID") == ref_internal_id),
        None,
    )
    if ref_exchange is None:
        raise ValueError(f"reference exchange {ref_internal_id} not found in {path}")

    input_amounts: dict[str, Decimal] = {}
    output_amounts: dict[str, Decimal] = {}
    for exchange in exchanges:
        flow_uuid = (
            exchange.get("referenceToFlowDataSet", {}) or {}
        ).get("@refObjectId")
        if not flow_uuid:
            continue
        amount = decimal_or_zero(exchange.get("meanAmount") or exchange.get("resultingAmount"))
        direction = str(exchange.get("exchangeDirection", ""))
        if direction == "Input":
            input_amounts[flow_uuid] = amount
        elif direction == "Output":
            output_amounts[flow_uuid] = amount

    return ProcessRecord(
        process_uuid=process_uuid,
        version=version,
        path=path,
        raw=raw,
        reference_exchange_internal_id=ref_internal_id,
        reference_flow_uuid=ref_exchange["referenceToFlowDataSet"]["@refObjectId"],
        reference_direction=ref_exchange["exchangeDirection"],
        reference_amount=decimal_or_zero(
            ref_exchange.get("meanAmount") or ref_exchange.get("resultingAmount")
        ),
        input_amounts=input_amounts,
        output_amounts=output_amounts,
        name_en=multilang_text(name_info.get("baseName"))[0],
        name_zh=multilang_text(name_info.get("baseName"))[1],
        route_en=multilang_text(name_info.get("treatmentStandardsRoutes"))[0],
        mix_en=multilang_text(name_info.get("mixAndLocationTypes"))[0],
        geography_code=str(
            (geography.get("locationOfOperationSupplyOrProduction") or {}).get("@location", "")
        ).strip(),
        classification_path=extract_classification_path(
            data_info.get("classificationInformation") or {}
        ),
        token_set=tokenize_text(
            " ".join(
                filter(
                    None,
                    [
                        multilang_text(name_info.get("baseName"))[0],
                        multilang_text(name_info.get("treatmentStandardsRoutes"))[0],
                        multilang_text(name_info.get("mixAndLocationTypes"))[0],
                        " ".join(
                            extract_classification_path(
                                data_info.get("classificationInformation") or {}
                            )
                        ),
                    ],
                )
            )
        ),
        source_kind=source_kind,
        source_label=source_label or path.parent.name,
        included_process_ref_count=len(
            [
                item
                for item in ensure_list(technology.get("referenceToIncludedProcesses"))
                if isinstance(item, dict)
            ]
        ),
    )


def count_output_connections(process_instances: list[dict[str, Any]]) -> int:
    connection_count = 0
    for instance in process_instances:
        output_items = ensure_list(
            (instance.get("connections") or {}).get("outputExchange")
        )
        for output_item in output_items:
            if not isinstance(output_item, dict):
                continue
            connection_count += len(ensure_list(output_item.get("downstreamProcess")))
    return connection_count


def lifecyclemodel_rows_from_select_result(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("data")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    if payload.get("id") and (
        payload.get("json_ordered") or payload.get("jsonOrdered")
    ):
        return [payload]
    return []


def process_rows_from_select_result(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("data")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    if payload.get("id") and (
        payload.get("json_ordered") or payload.get("jsonOrdered")
    ):
        return [payload]
    return []


def summarize_reference_model_row(row: dict[str, Any]) -> dict[str, Any]:
    json_ordered = row.get("json_ordered") or row.get("jsonOrdered") or {}
    json_tg = row.get("json_tg") or row.get("jsonTg") or {}
    dataset = json_ordered.get("lifeCycleModelDataSet", {})
    info = dataset.get("lifeCycleModelInformation", {})
    data_info = info.get("dataSetInformation", {})
    tech = info.get("technology") or {}
    process_instances = ensure_list(
        ((tech.get("processes") or {}).get("processInstance"))
    )
    reference_process_refs = [
        ref
        for ref in (
            (instance.get("referenceToProcess") or {})
            for instance in process_instances
            if isinstance(instance, dict)
        )
        if isinstance(ref, dict)
    ]
    name_info = data_info.get("name") or {}
    resulting_refs = [
        item
        for item in ensure_list(data_info.get("referenceToResultingProcess"))
        if isinstance(item, dict)
    ]
    xflow = json_tg.get("xflow") or {}
    connection_count = len(ensure_list(xflow.get("edges"))) or count_output_connections(
        process_instances
    )
    submodels = [
        item for item in ensure_list(json_tg.get("submodels")) if isinstance(item, dict)
    ]
    return {
        "id": row.get("id") or data_info.get("common:UUID"),
        "version": row.get("version")
        or (
            (
                (dataset.get("administrativeInformation") or {}).get(
                    "publicationAndOwnership"
                )
                or {}
            ).get("common:dataSetVersion")
        ),
        "name_en": multilang_text(name_info.get("baseName"))[0],
        "name_zh": multilang_text(name_info.get("baseName"))[1],
        "classification_path": extract_classification_path(
            data_info.get("classificationInformation") or {}
        ),
        "process_count": len(process_instances),
        "connection_count": connection_count,
        "reference_process_internal_id": (
            (info.get("quantitativeReference") or {}).get("referenceToReferenceProcess")
        ),
        "process_ids": [
            ref.get("@refObjectId")
            for ref in reference_process_refs
            if ref.get("@refObjectId")
        ],
        "resulting_process_ids": [
            ref.get("@refObjectId") for ref in resulting_refs if ref.get("@refObjectId")
        ],
        "submodel_count": len(submodels),
        "submodel_types": [
            str(item.get("type", "")).strip()
            for item in submodels
            if str(item.get("type", "")).strip()
        ],
        "submodel_ids": [item.get("id") for item in submodels if item.get("id")],
        "rule_verification": row.get("rule_verification"),
    }


def normalize_model_shape(summary: dict[str, Any]) -> str:
    process_count = int(summary.get("process_count") or 0)
    connection_count = int(summary.get("connection_count") or 0)
    if process_count <= 1:
        return "single-process"
    if connection_count <= 0:
        return "disconnected"
    if connection_count == process_count - 1:
        return "chain-like"
    if connection_count > process_count:
        return "branched"
    return "mixed"


def collect_reference_model_resulting_process_ids(
    selected_details: list[dict[str, Any]],
) -> list[str]:
    process_ids: list[str] = []
    seen: set[str] = set()
    for detail in selected_details:
        for row in lifecyclemodel_rows_from_select_result(detail):
            summary = summarize_reference_model_row(row)
            candidate_ids = (summary.get("resulting_process_ids") or []) + (
                summary.get("submodel_ids") or []
            )
            for process_id in candidate_ids:
                if process_id and process_id not in seen:
                    seen.add(process_id)
                    process_ids.append(process_id)
    return process_ids


def collect_reusable_process_dirs(manifest: dict[str, Any], out_dir: Path) -> list[Path]:
    reuse_cfg = manifest.get("reuse") or {}
    dirs: list[Path] = []
    for item in ensure_list(reuse_cfg.get("reusable_process_dirs")):
        candidate = Path(item).expanduser().resolve()
        if candidate.is_dir():
            dirs.append(candidate)
    if reuse_cfg.get("include_reference_model_resulting_processes", True):
        discovered_dir = out_dir / "discovery" / "reference-model-resulting-processes"
        if discovered_dir.is_dir():
            dirs.append(discovered_dir)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for item in dirs:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def load_reusable_process_records(manifest: dict[str, Any], out_dir: Path) -> dict[str, ProcessRecord]:
    process_map: dict[str, ProcessRecord] = {}
    for reusable_dir in collect_reusable_process_dirs(manifest, out_dir):
        source_kind = (
            "reference_model_resulting_process"
            if reusable_dir.name == "reference-model-resulting-processes"
            else "reusable_generated_process"
        )
        for path in sorted(reusable_dir.glob("*.json")):
            record = load_process_record(
                path,
                source_kind=source_kind,
                source_label=str(reusable_dir),
            )
            process_map.setdefault(record.process_uuid, record)
    return process_map


def score_edge_candidate(
    src: ProcessRecord,
    dst: ProcessRecord,
    flow_uuid: str,
    downstream_amount: Decimal,
) -> tuple[Decimal, list[str]]:
    score = Decimal("10.0")
    reasons = ["shared flow UUID"]
    if src.reference_flow_uuid == flow_uuid and src.reference_direction == "Output":
        score += Decimal("3.0")
        reasons.append("upstream reference flow matches shared flow")
    if dst.reference_flow_uuid == flow_uuid and dst.reference_direction == "Input":
        score += Decimal("3.0")
        reasons.append("downstream reference flow matches shared flow")
    class_overlap = classification_overlap(src.classification_path, dst.classification_path)
    if class_overlap:
        score += Decimal(str(min(class_overlap, 3)))
        reasons.append(f"classification prefix overlap={class_overlap}")
    if src.geography_code and dst.geography_code and src.geography_code == dst.geography_code:
        score += Decimal("1.0")
        reasons.append(f"same geography={src.geography_code}")
    token_overlap = len(src.token_set & dst.token_set)
    if token_overlap:
        score += Decimal(str(min(token_overlap, 4))) / Decimal("4")
        reasons.append(f"token overlap={token_overlap}")
    upstream_amount = src.output_amounts.get(flow_uuid, Decimal("0"))
    if upstream_amount > 0 and downstream_amount > 0:
        ratio = downstream_amount / upstream_amount
        if Decimal("0.5") <= ratio <= Decimal("2.0"):
            score += Decimal("1.0")
            reasons.append(f"amount ratio plausible={format_decimal(ratio)}")
    if (
        src.source_kind != "local_run_export"
        and src.reference_flow_uuid == flow_uuid
        and src.included_process_ref_count > 1
    ):
        score += Decimal("0.5")
        reasons.append(
            f"reusable aggregated upstream process ({src.included_process_ref_count} included processes)"
        )
    if (
        dst.source_kind != "local_run_export"
        and dst.reference_flow_uuid == flow_uuid
        and dst.included_process_ref_count > 1
    ):
        score += Decimal("0.5")
        reasons.append(
            f"reusable aggregated downstream process ({dst.included_process_ref_count} included processes)"
        )
    return score, reasons


def infer_edges(process_map: dict[str, ProcessRecord]) -> list[Edge]:
    by_flow: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"producers": [], "consumers": []}
    )
    for process_id, record in process_map.items():
        for flow_uuid in record.output_amounts:
            by_flow[flow_uuid]["producers"].append(process_id)
        for flow_uuid in record.input_amounts:
            by_flow[flow_uuid]["consumers"].append(process_id)

    edge_map: dict[tuple[str, str, str], Edge] = {}
    for flow_uuid, participants in by_flow.items():
        producers = set(participants["producers"])
        consumers = set(participants["consumers"])
        if not producers or not consumers:
            continue
        pass_through = producers & consumers
        candidate_pairs: list[tuple[str, str]] = []
        if pass_through:
            for producer in producers - pass_through:
                for bridge in pass_through:
                    candidate_pairs.append((producer, bridge))
            for bridge in pass_through:
                for consumer in consumers - pass_through:
                    candidate_pairs.append((bridge, consumer))
        else:
            for producer in producers:
                for consumer in consumers:
                    candidate_pairs.append((producer, consumer))

        for src, dst in candidate_pairs:
            if src == dst:
                continue
            downstream_amount = process_map[dst].input_amounts.get(flow_uuid)
            if downstream_amount is None:
                continue
            confidence, reasons = score_edge_candidate(
                process_map[src],
                process_map[dst],
                flow_uuid,
                downstream_amount,
            )
            edge_map[(src, dst, flow_uuid)] = Edge(
                src=src,
                dst=dst,
                flow_uuid=flow_uuid,
                downstream_input_amount=downstream_amount,
                confidence=confidence,
                reasons=reasons,
            )
    edges = list(edge_map.values())
    grouped_by_dst: dict[tuple[str, str], list[Edge]] = defaultdict(list)
    for edge in edges:
        grouped_by_dst[(edge.dst, edge.flow_uuid)].append(edge)

    filtered: list[Edge] = []
    for group in grouped_by_dst.values():
        ordered_group = sorted(group, key=lambda item: (-item.confidence, item.src, item.dst))
        if len(ordered_group) >= 2:
            first = ordered_group[0]
            second = ordered_group[1]
            if first.confidence - second.confidence >= Decimal("2.0"):
                filtered.append(first)
                continue
        filtered.extend(ordered_group)

    return sorted(filtered, key=lambda item: (-item.confidence, item.flow_uuid, item.src, item.dst))


def choose_reference_process(
    process_map: dict[str, ProcessRecord],
    edges: list[Edge],
    state: dict[str, Any],
    preferred_process_ids: set[str] | None = None,
) -> str:
    target_flow_uuid = state.get("flow_summary", {}).get("uuid")
    indegree = defaultdict(int)
    outdegree = defaultdict(int)
    for edge in edges:
        indegree[edge.dst] += 1
        outdegree[edge.src] += 1

    candidate_pool = preferred_process_ids or set(process_map)

    candidates = [
        process_id
        for process_id, record in process_map.items()
        if process_id in candidate_pool
        if record.reference_flow_uuid == target_flow_uuid
        and record.reference_direction == "Output"
    ]
    if not candidates and target_flow_uuid:
        candidates = [
            process_id
            for process_id, record in process_map.items()
            if process_id in candidate_pool
            if target_flow_uuid in record.output_amounts
        ]
    if not candidates:
        candidates = sorted(candidate_pool)

    def rank(process_id: str) -> tuple[int, int, int]:
        record = process_map[process_id]
        target_outputs = 1 if target_flow_uuid and target_flow_uuid in record.output_amounts else 0
        return (
            target_outputs,
            1 if outdegree[process_id] == 0 else 0,
            indegree[process_id],
        )

    return max(candidates, key=rank)


def collect_reachable(final_process_id: str, edges: list[Edge]) -> set[str]:
    reverse_adj: dict[str, list[str]] = defaultdict(list)
    reachable = {final_process_id}
    for edge in edges:
        reverse_adj[edge.dst].append(edge.src)

    queue = deque([final_process_id])
    while queue:
        current = queue.popleft()
        for upstream in reverse_adj.get(current, []):
            if upstream in reachable:
                continue
            reachable.add(upstream)
            queue.append(upstream)
    return reachable


def topological_order(process_ids: set[str], edges: list[Edge]) -> list[str]:
    indegree = {process_id: 0 for process_id in process_ids}
    adj: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.src not in process_ids or edge.dst not in process_ids:
            continue
        indegree[edge.dst] += 1
        adj[edge.src].append(edge.dst)

    queue = deque(sorted([process_id for process_id, degree in indegree.items() if degree == 0]))
    ordered: list[str] = []
    while queue:
        current = queue.popleft()
        ordered.append(current)
        for downstream in sorted(adj.get(current, [])):
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                queue.append(downstream)

    if len(ordered) != len(process_ids):
        missing = sorted(process_ids - set(ordered))
        ordered.extend(missing)
    return ordered


def compute_multiplication_factors(
    process_map: dict[str, ProcessRecord],
    reachable: set[str],
    edges: list[Edge],
    order: list[str],
    final_process_id: str,
) -> dict[str, Decimal]:
    factors: dict[str, Decimal] = {process_id: Decimal("0") for process_id in reachable}
    factors[final_process_id] = Decimal("1")
    incoming: dict[str, list[Edge]] = defaultdict(list)
    for edge in edges:
        if edge.src in reachable and edge.dst in reachable:
            incoming[edge.dst].append(edge)

    for current in reversed(order):
        current_factor = factors.get(current, Decimal("0"))
        if current_factor == 0:
            continue
        for edge in incoming.get(current, []):
            upstream_output = process_map[edge.src].output_amounts.get(
                edge.flow_uuid, Decimal("1")
            )
            if upstream_output == 0:
                upstream_output = Decimal("1")
            delta = (current_factor * edge.downstream_input_amount) / upstream_output
            factors[edge.src] = factors.get(edge.src, Decimal("0")) + delta
    return factors


def format_decimal(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def decimal_to_json_number(value: Decimal) -> int | float:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return int(normalized)
    return float(normalized)


def exchange_amount(exchange: dict[str, Any]) -> Decimal:
    return decimal_or_zero(exchange.get("meanAmount") or exchange.get("resultingAmount"))


def clone_exchange_with_amount(
    exchange: dict[str, Any],
    amount: Decimal,
    internal_id: str,
    *,
    quantitative_reference: bool = False,
) -> dict[str, Any]:
    cloned = copy_json(exchange)
    cloned["@dataSetInternalID"] = internal_id
    amount_value = decimal_to_json_number(amount)
    cloned["meanAmount"] = amount_value
    if "resultingAmount" in cloned:
        cloned["resultingAmount"] = amount_value
    cloned["quantitativeReference"] = quantitative_reference
    return cloned


def build_process_instances(
    process_map: dict[str, ProcessRecord],
    order: list[str],
    edges: list[Edge],
    factors: dict[str, Decimal],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    internal_ids = {process_id: str(index + 1) for index, process_id in enumerate(order)}
    outgoing: dict[str, list[Edge]] = defaultdict(list)
    for edge in edges:
        if edge.src in internal_ids and edge.dst in internal_ids:
            outgoing[edge.src].append(edge)

    instances: list[dict[str, Any]] = []
    for process_id in order:
        record = process_map[process_id]
        data_info = record.raw["processDataSet"]["processInformation"]["dataSetInformation"]
        short_description = copy_json(data_info["name"]["baseName"])
        output_groups: list[dict[str, Any]] = []
        by_flow: dict[str, list[Edge]] = defaultdict(list)
        for edge in outgoing.get(process_id, []):
            by_flow[edge.flow_uuid].append(edge)
        for flow_uuid, grouped_edges in by_flow.items():
            downstream_payload = [
                {
                    "@id": internal_ids[edge.dst],
                    "@flowUUID": flow_uuid,
                    "@dominant": "true",
                }
                for edge in grouped_edges
            ]
            output_groups.append(
                {
                    "@dominant": "true",
                    "@flowUUID": flow_uuid,
                    "downstreamProcess": downstream_payload[0]
                    if len(downstream_payload) == 1
                    else downstream_payload,
                }
            )

        instances.append(
            {
                "@dataSetInternalID": internal_ids[process_id],
                "@multiplicationFactor": format_decimal(factors.get(process_id, Decimal("0"))),
                "referenceToProcess": build_global_reference(
                    ref_object_id=record.process_uuid,
                    version=record.version,
                    short_description=short_description,
                    dataset_type="process data set",
                    uri=f"../processes/{record.process_uuid}_{record.version}.xml",
                ),
                "connections": {
                    "outputExchange": output_groups[0]
                    if len(output_groups) == 1
                    else output_groups
                }
                if output_groups
                else {},
            }
        )
    return instances, internal_ids


def build_xflow(
    process_map: dict[str, ProcessRecord],
    order: list[str],
    edges: list[Edge],
    internal_ids: dict[str, str],
    factors: dict[str, Decimal],
    final_process_id: str,
) -> dict[str, Any]:
    primary_color = "#5c246a"
    background_color = "#ffffff"
    muted_text_color = "rgba(0,0,0,0.45)"
    body_text_color = "#000"
    node_width = 350
    base_x = -150
    step_x = 430
    node_y = 160
    chain_second_row_y = 380
    port_start_y = 65
    port_step_y = 20
    paired_input_start_y = 58
    paired_output_start_y = 78
    paired_port_step_y = 40

    def exchanges_for(record: ProcessRecord) -> list[dict[str, Any]]:
        carrier = record.raw["processDataSet"].get("exchanges", {})
        return [item for item in ensure_list(carrier.get("exchange")) if isinstance(item, dict)]

    def find_exchange(
        record: ProcessRecord, flow_uuid: str, direction: str
    ) -> dict[str, Any] | None:
        matches: list[dict[str, Any]] = []
        for exchange in exchanges_for(record):
            if str(exchange.get("exchangeDirection", "")) != direction:
                continue
            flow_ref = exchange.get("referenceToFlowDataSet", {}) or {}
            if flow_ref.get("@refObjectId") == flow_uuid:
                matches.append(exchange)
        if not matches:
            return None
        for candidate in matches:
            if candidate.get("quantitativeReference") is True:
                return candidate
        return matches[0]

    def port_text_data(exchange: dict[str, Any]) -> tuple[Any, str]:
        flow_ref = exchange.get("referenceToFlowDataSet", {}) or {}
        text_lang = copy_json(flow_ref.get("common:shortDescription") or [])
        return text_lang, localized_text(text_lang)

    nodes = []
    graph_ids = {process_id: process_id for process_id in order}

    port_specs_by_process: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
        process_id: {} for process_id in order
    }

    def register_port(
        process_id: str,
        side: str,
        exchange: dict[str, Any] | None,
        *,
        quantitative_reference: bool = False,
    ) -> None:
        if not exchange:
            return
        flow_ref = exchange.get("referenceToFlowDataSet", {}) or {}
        flow_uuid = str(flow_ref.get("@refObjectId", "")).strip()
        if not flow_uuid:
            return
        key = (side, flow_uuid)
        existing = port_specs_by_process[process_id].get(key)
        text_lang, display_text = port_text_data(exchange)
        spec = {
            "side": side,
            "flow_uuid": flow_uuid,
            "flow_version": str(flow_ref.get("@version", "")).strip(),
            "text_lang": text_lang,
            "display_text": display_text,
            "quantitative_reference": quantitative_reference
            or bool(exchange.get("quantitativeReference")),
            "allocations": copy_json(exchange.get("allocations")) if exchange.get("allocations") else None,
        }
        if existing:
            if spec["quantitative_reference"] and not existing["quantitative_reference"]:
                port_specs_by_process[process_id][key] = spec
            return
        port_specs_by_process[process_id][key] = spec

    for process_id in order:
        record = process_map[process_id]
        reference_exchange = find_exchange(
            record, record.reference_flow_uuid, record.reference_direction
        )
        register_port(
            process_id,
            "INPUT" if record.reference_direction == "Input" else "OUTPUT",
            reference_exchange,
            quantitative_reference=process_id == final_process_id,
        )

    for edge in edges:
        if edge.src in process_map:
            register_port(edge.src, "OUTPUT", find_exchange(process_map[edge.src], edge.flow_uuid, "Output"))
        if edge.dst in process_map:
            register_port(edge.dst, "INPUT", find_exchange(process_map[edge.dst], edge.flow_uuid, "Input"))

    positions = {
        process_id: {"x": base_x + index * step_x, "y": node_y}
        for index, process_id in enumerate(order)
    }

    if len(order) > 5 and len(edges) == len(order) - 1:
        incoming_counts = {process_id: 0 for process_id in order}
        outgoing_counts = {process_id: 0 for process_id in order}
        for edge in edges:
            if edge.src in outgoing_counts:
                outgoing_counts[edge.src] += 1
            if edge.dst in incoming_counts:
                incoming_counts[edge.dst] += 1
        source_count = sum(1 for process_id in order if incoming_counts[process_id] == 0)
        sink_count = sum(1 for process_id in order if outgoing_counts[process_id] == 0)
        linear_chain = (
            source_count == 1
            and sink_count == 1
            and all(incoming_counts[process_id] <= 1 for process_id in order)
            and all(outgoing_counts[process_id] <= 1 for process_id in order)
        )
        if linear_chain:
            for tail_index, process_id in enumerate(order[5:]):
                positions[process_id] = {
                    "x": base_x + (3 + tail_index) * step_x,
                    "y": chain_second_row_y,
                }

    nodes = []
    for index, process_id in enumerate(order):
        record = process_map[process_id]
        info = record.raw["processDataSet"]["processInformation"]["dataSetInformation"]
        name_info = copy_json(info.get("name") or {})
        short_description = build_name_summary(name_info)
        port_specs = list(port_specs_by_process[process_id].values())
        input_specs = [item for item in port_specs if item["side"] == "INPUT"]
        output_specs = [item for item in port_specs if item["side"] == "OUTPUT"]
        has_inputs = bool(input_specs)
        has_outputs = bool(output_specs)
        if has_inputs and has_outputs:
            pair_count = max(len(input_specs), len(output_specs))
            height = max(110, 110 + (pair_count - 1) * paired_port_step_y)
        else:
            height = max(100, 60 + 20 * max(len(input_specs), len(output_specs), 2))

        def port_y(side: str, port_index: int) -> int:
            if has_inputs and has_outputs:
                if side == "INPUT":
                    return paired_input_start_y + port_index * paired_port_step_y
                return paired_output_start_y + port_index * paired_port_step_y
            return port_start_y + port_index * port_step_y

        port_items: list[dict[str, Any]] = []
        for port_index, spec in enumerate(input_specs):
            port_items.append(
                {
                    "id": f"INPUT:{spec['flow_uuid']}",
                    "group": "groupInput",
                    "args": {"x": 0, "y": port_y("INPUT", port_index)},
                    "attrs": {
                        "text": {
                            "text": spec["display_text"],
                            "title": spec["display_text"],
                            "cursor": "pointer",
                            "fill": primary_color if spec["quantitative_reference"] else muted_text_color,
                            "font-weight": "bold" if spec["quantitative_reference"] else "normal",
                        }
                    },
                    "data": {
                        "textLang": spec["text_lang"],
                        "flowId": spec["flow_uuid"],
                        "flowVersion": spec["flow_version"],
                        "quantitativeReference": spec["quantitative_reference"],
                        "allocations": spec["allocations"],
                    },
                    "tools": [{"id": "portTool"}],
                }
            )
        for port_index, spec in enumerate(output_specs):
            port_items.append(
                {
                    "id": f"OUTPUT:{spec['flow_uuid']}",
                    "group": "groupOutput",
                    "args": {"x": "100%", "y": port_y("OUTPUT", port_index)},
                    "attrs": {
                        "text": {
                            "text": spec["display_text"],
                            "title": spec["display_text"],
                            "cursor": "pointer",
                            "fill": primary_color if spec["quantitative_reference"] else muted_text_color,
                            "font-weight": "bold" if spec["quantitative_reference"] else "normal",
                        }
                    },
                    "data": {
                        "textLang": spec["text_lang"],
                        "flowId": spec["flow_uuid"],
                        "flowVersion": spec["flow_version"],
                        "quantitativeReference": spec["quantitative_reference"],
                        "allocations": spec["allocations"],
                    },
                    "tools": [{"id": "portTool"}],
                }
            )

        nodes.append(
            {
                "id": graph_ids[process_id],
                "shape": "rect",
                "position": positions[process_id],
                "size": {"width": node_width, "height": height},
                "attrs": {
                    "body": {
                        "stroke": primary_color,
                        "strokeWidth": 1,
                        "fill": background_color,
                        "rx": 6,
                        "ry": 6,
                    },
                    "label": {
                        "fill": body_text_color,
                        "refX": 0.5,
                        "refY": 8,
                        "textAnchor": "middle",
                        "textVerticalAnchor": "top",
                    },
                    "text": {
                        "text": localized_text(short_description or name_info.get("baseName")),
                    },
                },
                "isMyProcess": True,
                "data": {
                    "id": process_id,
                    "version": record.version,
                    "index": internal_ids[process_id],
                    "label": name_info,
                    "shortDescription": short_description,
                    "quantitativeReference": "1" if process_id == final_process_id else "0",
                    "targetAmount": "1" if process_id == final_process_id else "",
                    "multiplicationFactor": format_decimal(
                        factors.get(process_id, Decimal("0"))
                    ),
                },
                "ports": {
                    "groups": {
                        "groupInput": {
                            "position": {"name": "absolute"},
                            "label": {"position": {"name": "right"}},
                            "attrs": {
                                "circle": {
                                    "stroke": primary_color,
                                    "fill": background_color,
                                    "strokeWidth": 1,
                                    "r": 4,
                                    "magnet": True,
                                },
                                "text": {"fill": muted_text_color, "fontSize": 14},
                            },
                        },
                        "groupOutput": {
                            "position": {"name": "absolute"},
                            "label": {"position": {"name": "left"}},
                            "attrs": {
                                "circle": {
                                    "stroke": primary_color,
                                    "fill": background_color,
                                    "strokeWidth": 1,
                                    "r": 4,
                                    "magnet": True,
                                },
                                "text": {"fill": muted_text_color, "fontSize": 14},
                            },
                        },
                    },
                    "items": port_items,
                },
                "tools": {"name": None, "items": []},
                "visible": True,
                "zIndex": 1,
            }
        )

    edge_payload = []
    for edge in edges:
        if edge.src not in internal_ids or edge.dst not in internal_ids:
            continue
        edge_payload.append(
            {
                "id": str(uuid.uuid4()),
                "shape": "edge",
                "source": {
                    "cell": graph_ids[edge.src],
                    "port": f"OUTPUT:{edge.flow_uuid}",
                },
                "target": {
                    "cell": graph_ids[edge.dst],
                    "port": f"INPUT:{edge.flow_uuid}",
                },
                "labels": [],
                "attrs": {"line": {"stroke": primary_color}},
                "data": {
                    "connection": {
                        "outputExchange": {
                            "@flowUUID": edge.flow_uuid,
                            "downstreamProcess": {
                                "@id": internal_ids[edge.dst],
                                "@flowUUID": edge.flow_uuid,
                            },
                        },
                        "connectionConfidence": format_decimal(edge.confidence),
                        "connectionReasons": edge.reasons,
                        "isBalanced": True,
                        "unbalancedAmount": "0",
                        "exchangeAmount": format_decimal(edge.downstream_input_amount),
                    },
                    "node": {
                        "sourceNodeID": graph_ids[edge.src],
                        "sourceProcessId": edge.src,
                        "sourceProcessVersion": process_map[edge.src].version,
                        "targetNodeID": graph_ids[edge.dst],
                        "targetProcessId": edge.dst,
                        "targetProcessVersion": process_map[edge.dst].version,
                    },
                },
                "zIndex": 4,
            }
        )

    return {"nodes": nodes, "edges": edge_payload}


def build_primary_resulting_process(
    *,
    model_uuid: str,
    model_version: str,
    run_name: str,
    state: dict[str, Any],
    process_map: dict[str, ProcessRecord],
    order: list[str],
    edges: list[Edge],
    factors: dict[str, Decimal],
    final_process_id: str,
    process_instances: list[dict[str, Any]],
    model_name: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    final_record = process_map[final_process_id]
    final_process = copy_json(final_record.raw)
    final_dataset = final_process["processDataSet"]

    totals: dict[tuple[str, str], dict[str, Any]] = {}
    for process_id in order:
        factor = factors.get(process_id, Decimal("0"))
        if factor == 0:
            continue
        exchanges = ensure_list(process_map[process_id].raw["processDataSet"].get("exchanges", {}).get("exchange"))
        for exchange in exchanges:
            flow_ref = exchange.get("referenceToFlowDataSet", {}) or {}
            flow_uuid = flow_ref.get("@refObjectId")
            direction = str(exchange.get("exchangeDirection", ""))
            if not flow_uuid or direction not in {"Input", "Output"}:
                continue
            key = (flow_uuid, direction)
            scaled_amount = exchange_amount(exchange) * factor
            if key not in totals:
                totals[key] = {
                    "amount": Decimal("0"),
                    "exchange": copy_json(exchange),
                }
            totals[key]["amount"] += scaled_amount

    for edge in edges:
        internal_amount = edge.downstream_input_amount * factors.get(edge.dst, Decimal("0"))
        for key in ((edge.flow_uuid, "Output"), (edge.flow_uuid, "Input")):
            if key in totals:
                totals[key]["amount"] -= internal_amount
                if abs(totals[key]["amount"]) < Decimal("0.0000000001"):
                    totals[key]["amount"] = Decimal("0")

    target_flow_uuid = state.get("flow_summary", {}).get("uuid")
    exchange_items: list[dict[str, Any]] = []
    next_internal_id = 1
    ref_exchange_internal_id = ""
    for (flow_uuid, direction), payload in sorted(totals.items(), key=lambda item: (item[0][1], item[0][0])):
        amount = payload["amount"]
        if amount <= 0:
            continue
        quantitative_reference = (
            flow_uuid == target_flow_uuid and direction == "Output" and not ref_exchange_internal_id
        )
        internal_id = str(next_internal_id)
        next_internal_id += 1
        if quantitative_reference:
            ref_exchange_internal_id = internal_id
        exchange_items.append(
            clone_exchange_with_amount(
                payload["exchange"],
                amount,
                internal_id,
                quantitative_reference=quantitative_reference,
            )
        )

    if not ref_exchange_internal_id:
        raise ValueError(
            f"could not build resulting process reference exchange for run {run_name}"
        )

    info = final_dataset["processInformation"]["dataSetInformation"]
    info["common:UUID"] = model_uuid
    info["name"] = copy_json(model_name)
    info["classificationInformation"] = copy_json(
        final_record.raw["processDataSet"]["processInformation"]["dataSetInformation"]["classificationInformation"]
    )
    general_comment = copy_json(info.get("common:generalComment") or [])
    general_comment.extend(
        multilang_from_text(
            f"Local primary resulting process generated from lifecycle model {model_uuid}; exchanges are aggregated from included processes with internal linked flows cancelled.",
            f"本地为生命周期模型 {model_uuid} 生成的 primary resulting process；其 exchanges 由包含过程聚合并抵消内部连接 flow 后得到。",
        )
    )
    info["common:generalComment"] = general_comment

    final_dataset["processInformation"]["quantitativeReference"]["referenceToReferenceFlow"] = (
        ref_exchange_internal_id
    )
    technology = final_dataset["processInformation"].get("technology")
    if not isinstance(technology, dict):
        technology = {}
        final_dataset["processInformation"]["technology"] = technology
    included_refs = [copy_json(item["referenceToProcess"]) for item in process_instances]
    technology["referenceToIncludedProcesses"] = included_refs[0] if len(included_refs) == 1 else included_refs

    final_dataset["exchanges"] = {
        "exchange": exchange_items[0] if len(exchange_items) == 1 else exchange_items
    }

    publication = final_dataset["administrativeInformation"]["publicationAndOwnership"]
    publication["common:dataSetVersion"] = model_version
    publication["common:permanentDataSetURI"] = (
        f"https://local.tiangong.invalid/processes/{model_uuid}?version={model_version}"
    )

    resulting_reference = build_global_reference(
        ref_object_id=model_uuid,
        version=model_version,
        short_description=copy_json(model_name["baseName"]),
        dataset_type="process data set",
        uri=f"../processes/{model_uuid}_{model_version}.xml",
    )
    submodel_info = {
        "id": model_uuid,
        "type": "primary",
        "finalId": {
            "nodeId": final_process_id,
            "processId": final_process_id,
            "referenceToFlowDataSet": {
                "@refObjectId": target_flow_uuid,
                "@exchangeDirection": "Output",
            },
        },
    }
    summary = {
        "resulting_process_uuid": model_uuid,
        "resulting_process_version": model_version,
        "resulting_process_exchange_count": len(exchange_items),
        "reference_exchange_internal_id": ref_exchange_internal_id,
    }
    return final_process, resulting_reference, {"submodel": submodel_info, "summary": summary}


def build_lifecycle_model_dataset(
    run_name: str,
    state: dict[str, Any],
    process_map: dict[str, ProcessRecord],
    order: list[str],
    edges: list[Edge],
    factors: dict[str, Decimal],
    final_process_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    final_record = process_map[final_process_id]
    final_dataset = final_record.raw["processDataSet"]
    final_info = final_dataset["processInformation"]["dataSetInformation"]
    flow_dataset = state.get("flow_dataset", {}).get("flowDataSet", {})
    flow_info = flow_dataset.get("flowInformation", {}).get("dataSetInformation", {})
    flow_name = flow_info.get("name", {})
    model_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, run_name))
    model_version = "01.01.000"
    process_instances, internal_ids = build_process_instances(
        process_map, order, edges, factors
    )

    commissioner_and_goal = copy_json(
        final_dataset["administrativeInformation"]["common:commissionerAndGoal"]
    )
    data_entry_by = copy_json(final_dataset["administrativeInformation"]["dataEntryBy"])
    entering_ref = data_entry_by.get("common:referenceToPersonOrEntityEnteringTheData")
    if entering_ref is not None:
        data_entry_by["common:referenceToPersonOrEntityEnteringTheDataSet"] = copy_json(
            entering_ref
        )
    publication = copy_json(
        final_dataset["administrativeInformation"]["publicationAndOwnership"]
    )
    publication["common:dataSetVersion"] = model_version
    publication["common:permanentDataSetURI"] = (
        f"https://local.tiangong.invalid/lifecyclemodels/{model_uuid}?version={model_version}"
    )
    publication["common:accessRestrictions"] = publication.get("common:accessRestrictions") or []

    compliance = copy_json(
        final_dataset["modellingAndValidation"]["complianceDeclarations"]["compliance"]
    )
    compliance.setdefault("common:approvalOfOverallCompliance", "Fully compliant")
    compliance.setdefault("common:nomenclatureCompliance", "Not defined")
    compliance.setdefault("common:methodologicalCompliance", "Not defined")
    compliance.setdefault("common:reviewCompliance", "Not defined")
    compliance.setdefault("common:documentationCompliance", "Not defined")
    compliance.setdefault("common:qualityCompliance", "Not defined")

    review_ref = (
        commissioner_and_goal.get("common:referenceToCommissioner")
        or entering_ref
        or publication.get("common:referenceToOwnershipOfDataSet")
    )
    review = {
        "common:referenceToNameOfReviewerAndInstitution": copy_json(review_ref),
        "common:otherReviewDetails": multilang_from_text(
            "Local automated-builder test artifact; not independently reviewed.",
            "本地 automated-builder 测试产物，未经过独立评审。",
        ),
    }

    general_comment = copy_json(final_info.get("common:generalComment") or [])
    general_comment.extend(
        multilang_from_text(
            f"Built locally from process-automated-builder run {run_name}. {state.get('technical_description', '')}".strip(),
            f"本地基于 process-automated-builder 运行 {run_name} 生成。".strip(),
        )
    )
    general_comment.extend(
        multilang_from_text(
            "This skill emits native json_ordered only. Platform visualization fields and downstream resulting-process handling are delegated to MCP or application-side publishing workflows.",
            "该 skill 只产出原生 json_ordered。平台可视化字段和后续 resulting process 处理交由 MCP 或应用侧发布流程完成。",
        )
    )

    resulting_process = build_global_reference(
        ref_object_id=model_uuid,
        version=model_version,
        short_description=copy_json(flow_name.get("baseName") or final_info["name"]["baseName"]),
        dataset_type="process data set",
        uri=f"../processes/{model_uuid}_{model_version}.xml",
    )

    model = {
        "lifeCycleModelDataSet": {
            "@xmlns": "http://eplca.jrc.ec.europa.eu/ILCD/LifeCycleModel/2017",
            "@xmlns:acme": "http://acme.com/custom",
            "@xmlns:common": "http://lca.jrc.it/ILCD/Common",
            "@xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "@locations": "../ILCDLocations.xml",
            "@version": "1.1",
            "@xsi:schemaLocation": "http://eplca.jrc.ec.europa.eu/ILCD/LifeCycleModel/2017 ../../schemas/ILCD_LifeCycleModelDataSet.xsd",
            "lifeCycleModelInformation": {
                "dataSetInformation": {
                    "common:UUID": model_uuid,
                    "name": {
                        "baseName": copy_json(flow_name.get("baseName") or final_info["name"]["baseName"]),
                        "treatmentStandardsRoutes": copy_json(
                            flow_name.get("treatmentStandardsRoutes")
                            or final_info["name"]["treatmentStandardsRoutes"]
                        ),
                        "mixAndLocationTypes": copy_json(
                            flow_name.get("mixAndLocationTypes")
                            or final_info["name"]["mixAndLocationTypes"]
                        ),
                        "functionalUnitFlowProperties": multilang_from_text(
                            f"Reference process output scaled to 1 unit of {state.get('flow_summary', {}).get('base_name_en', first_text(flow_name.get('baseName')))}",
                            f"参考过程输出缩放到 1 单位 {state.get('flow_summary', {}).get('base_name_zh', '')}",
                        ),
                    },
                    "classificationInformation": copy_json(
                        final_info["classificationInformation"]
                    ),
                    "referenceToResultingProcess": resulting_process,
                    "common:generalComment": general_comment,
                },
                "quantitativeReference": {
                    "referenceToReferenceProcess": internal_ids[final_process_id]
                },
                "technology": {
                    "processes": {
                        "processInstance": process_instances,
                    }
                },
            },
            "modellingAndValidation": {
                "dataSourcesTreatmentEtc": {
                    "useAdviceForDataSet": multilang_from_text(
                        state.get("scope", "")
                        or "Built from locally exported process datasets and inferred process-to-process flow links.",
                        "基于本地导出的 process 数据集和推断出的过程间 flow 连接生成。",
                    )
                },
                "validation": {
                    "review": review,
                },
                "complianceDeclarations": {
                    "compliance": compliance,
                },
            },
            "administrativeInformation": {
                "common:commissionerAndGoal": commissioner_and_goal,
                "dataEntryBy": data_entry_by,
                "publicationAndOwnership": publication,
            },
        }
    }

    summary = {
        "run_name": run_name,
        "model_uuid": model_uuid,
        "model_version": model_version,
        "reference_process_uuid": final_process_id,
        "reference_process_internal_id": internal_ids[final_process_id],
        "reference_flow_uuid": state.get("flow_summary", {}).get("uuid"),
        "reference_to_resulting_process_uuid": model_uuid,
        "process_count": len(order),
        "edge_count": len([edge for edge in edges if edge.src in internal_ids and edge.dst in internal_ids]),
        "multiplication_factors": {
            process_id: format_decimal(factors.get(process_id, Decimal("0")))
            for process_id in order
        },
        "ordered_processes": [
            {
                "process_uuid": process_id,
                "name_en": process_map[process_id].name_en,
                "name_zh": process_map[process_id].name_zh,
                "route_en": process_map[process_id].route_en,
                "geography_code": process_map[process_id].geography_code,
                "classification_path": process_map[process_id].classification_path,
            }
            for process_id in order
        ],
    }
    return model, summary


def validator_script() -> str:
    return """
import importlib.resources as pkg_resources
import json
import sys
from jsonschema import Draft7Validator
from referencing import Registry
from referencing.jsonschema import DRAFT7

sys.path.insert(0, sys.argv[1])
from tidas_tools.validate import retrieve_schema, validate_processes_classification_hierarchy
import tidas_tools.tidas.schemas as schemas

model_path = sys.argv[2]
schema_path = pkg_resources.files(schemas) / "tidas_lifecyclemodels.json"
schema_uri = f"file://{schema_path}"
with schema_path.open() as handle:
    schema = json.load(handle)
registry = Registry(retrieve=retrieve_schema)
registry = registry.with_resource(schema_uri, DRAFT7.create_resource(schema))
validator = Draft7Validator(schema, registry=registry)
with open(model_path, "r", encoding="utf-8") as handle:
    item = json.load(handle)
errors = []
for schema_error in validator.iter_errors(item):
    location = "/".join(str(part) for part in schema_error.path) if schema_error.path else "<root>"
    errors.append(f"Schema Error at {location}: {schema_error.message}")
try:
    class_items = item["lifeCycleModelDataSet"]["lifeCycleModelInformation"]["dataSetInformation"]["classificationInformation"]["common:classification"]["common:class"]
    class_check = validate_processes_classification_hierarchy(class_items)
    if not class_check["valid"]:
        errors.extend(class_check["errors"])
except Exception as exc:
    errors.append(f"Classification validation error: {exc}")
print(json.dumps({"errors": errors}, ensure_ascii=False))
sys.exit(1 if errors else 0)
""".strip()


def validate_model_file(model_path: Path) -> dict[str, Any]:
    if not TIDAS_TOOLS_PYTHON.exists():
        return {
            "validator": str(TIDAS_TOOLS_PYTHON),
            "executed": False,
            "errors": [f"validator python not found: {TIDAS_TOOLS_PYTHON}"],
        }

    command = [
        str(TIDAS_TOOLS_PYTHON),
        "-c",
        validator_script(),
        str(TIDAS_TOOLS_SRC),
        str(model_path),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    payload_text = result.stdout.strip() or "{}"
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        payload = {"errors": [payload_text or result.stderr.strip()]}
    payload["validator"] = str(TIDAS_TOOLS_PYTHON)
    payload["executed"] = True
    payload["returncode"] = result.returncode
    if result.stderr.strip():
        payload["stderr"] = result.stderr.strip()
    return payload


def jsonrpc_payload(method: str, params: dict[str, Any], request_id: int) -> str:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        },
        ensure_ascii=False,
    )


def parse_mcp_response(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise ValueError("empty MCP response")
    data_lines = [
        line[len("data: ") :].strip()
        for line in text.splitlines()
        if line.startswith("data: ")
    ]
    payload_text = "\n".join(data_lines).strip() if data_lines else text
    return json.loads(payload_text)


def run_curl_mcp_request(
    *,
    api_key: str,
    url: str,
    method: str,
    params: dict[str, Any],
    request_id: int,
    max_attempts: int,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    body = jsonrpc_payload(method, params, request_id)
    command = [
        "curl",
        "-sS",
        "--max-time",
        "30",
        "-X",
        "POST",
        url,
        "-H",
        "Content-Type: application/json",
        "-H",
        "Accept: application/json, text/event-stream",
        "-H",
        f"Authorization: Bearer {api_key}",
        "--data",
        body,
    ]
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return parse_mcp_response(result.stdout)
        stderr_text = result.stderr.strip() or result.stdout.strip() or f"curl exit {result.returncode}"
        last_error = f"attempt {attempt}/{max_attempts}: {stderr_text}"
        if attempt < max_attempts:
            time.sleep(retry_delay_seconds)
    raise RuntimeError(f"MCP request failed: {last_error}")


def mcp_call_tool(
    *,
    api_key: str,
    url: str,
    tool_name: str,
    arguments: dict[str, Any],
    request_id: int,
    max_attempts: int,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    payload = run_curl_mcp_request(
        api_key=api_key,
        url=url,
        method="tools/call",
        params={"name": tool_name, "arguments": arguments},
        request_id=request_id,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    return payload


def parse_tool_text_json(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result") or {}
    content = result.get("content")
    if not isinstance(content, list):
        raise ValueError("tool response missing content list")
    text_blocks = [
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    joined = "\n".join(block for block in text_blocks if block).strip()
    if not joined:
        return {}
    return json.loads(joined)

def parse_search_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload.get("result") or {}
    content = result.get("content")
    if not isinstance(content, list):
        return []
    text_blocks = [
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    joined = "\n".join(block for block in text_blocks if block).strip()
    if not joined:
        return []
    parsed = json.loads(joined)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
        return [item for item in parsed["data"] if isinstance(item, dict)]
    return []


def discover_reference_models(
    manifest: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any] | None:
    queries = [str(item).strip() for item in ensure_list(manifest["discovery"].get("reference_model_queries")) if str(item).strip()]
    if not queries:
        return None

    api_key = os.getenv("TIANGONG_LCA_REMOTE_API_KEY", "").strip()
    if not api_key:
        return {
            "executed": False,
            "error": "TIANGONG_LCA_REMOTE_API_KEY is not set",
            "queries": queries,
        }

    url = os.getenv("TIANGONG_LCA_REMOTE_URL", "https://lcamcp.tiangong.earth/mcp")
    max_attempts = int((manifest.get("publish") or {}).get("max_attempts") or 5)
    retry_delay_seconds = float((manifest.get("publish") or {}).get("retry_delay_seconds") or 2.0)
    select_limit = int(manifest["discovery"].get("reference_model_select_limit") or 3)
    request_id = 2000

    search_results: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    selected_details: list[dict[str, Any]] = []

    for query in queries:
        payload = mcp_call_tool(
            api_key=api_key,
            url=url,
            tool_name="Search_Life_Cycle_Models_Tool",
            arguments={"query": query},
            request_id=request_id,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        request_id += 1
        rows = parse_search_payload(payload)
        search_results.append({"query": query, "rows": rows})
        for row in rows:
            row_id = str(row.get("id", "")).strip()
            if row_id and row_id not in selected_ids:
                selected_ids.append(row_id)
            if len(selected_ids) >= select_limit:
                break
        if len(selected_ids) >= select_limit:
            break

    for row_id in selected_ids:
        payload = mcp_call_tool(
            api_key=api_key,
            url=url,
            tool_name="Database_CRUD_Tool",
            arguments={
                "operation": "select",
                "table": "lifecyclemodels",
                "id": row_id,
            },
            request_id=request_id,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        request_id += 1
        selected_details.append(parse_tool_text_json(payload))

    model_patterns: list[dict[str, Any]] = []
    for detail in selected_details:
        for row in lifecyclemodel_rows_from_select_result(detail):
            pattern = summarize_reference_model_row(row)
            pattern["shape"] = normalize_model_shape(pattern)
            model_patterns.append(pattern)

    resulting_process_ids = collect_reference_model_resulting_process_ids(selected_details)
    resulting_process_payloads: list[dict[str, Any]] = []
    resulting_process_dir = out_dir / "discovery" / "reference-model-resulting-processes"
    resulting_process_dir.mkdir(parents=True, exist_ok=True)
    resulting_process_index: list[dict[str, Any]] = []
    for process_id in resulting_process_ids:
        payload = mcp_call_tool(
            api_key=api_key,
            url=url,
            tool_name="Database_CRUD_Tool",
            arguments={
                "operation": "select",
                "table": "processes",
                "id": process_id,
            },
            request_id=request_id,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        request_id += 1
        parsed = parse_tool_text_json(payload)
        resulting_process_payloads.append(parsed)
        for row in process_rows_from_select_result(parsed):
            process_json = row.get("json_ordered") or row.get("jsonOrdered")
            process_version = row.get("version")
            if not isinstance(process_json, dict):
                continue
            if not process_version:
                process_version = (
                    (
                        (
                            process_json.get("processDataSet", {})
                            .get("administrativeInformation", {})
                            .get("publicationAndOwnership", {})
                        )
                    ).get("common:dataSetVersion")
                    or "01.00.000"
                )
            process_file = resulting_process_dir / f"{process_id}_{process_version}.json"
            process_file.write_text(
                json.dumps(process_json, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            record = load_process_record(
                process_file,
                source_kind="reference_model_resulting_process",
                source_label=process_id,
            )
            resulting_process_index.append(
                {
                    "process_uuid": record.process_uuid,
                    "version": record.version,
                    "name_en": record.name_en,
                    "name_zh": record.name_zh,
                    "reference_flow_uuid": record.reference_flow_uuid,
                    "reference_direction": record.reference_direction,
                    "reference_amount": format_decimal(record.reference_amount),
                    "classification_path": record.classification_path,
                    "geography_code": record.geography_code,
                    "included_process_ref_count": record.included_process_ref_count,
                    "path": str(process_file),
                }
            )

    discovery_dir = out_dir / "discovery"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    search_file = discovery_dir / "reference-model-search-results.json"
    details_file = discovery_dir / "reference-model-details.json"
    patterns_file = discovery_dir / "reference-model-patterns.json"
    resulting_process_file = discovery_dir / "reference-model-resulting-process-details.json"
    resulting_process_index_file = discovery_dir / "reference-model-resulting-process-index.json"
    summary_file = discovery_dir / "reference-model-summary.json"

    search_file.write_text(
        json.dumps(search_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    details_file.write_text(
        json.dumps(selected_details, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    patterns_file.write_text(
        json.dumps(model_patterns, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    resulting_process_file.write_text(
        json.dumps(resulting_process_payloads, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    resulting_process_index_file.write_text(
        json.dumps(resulting_process_index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "executed": True,
        "queries": queries,
        "selected_ids": selected_ids,
        "search_file": str(search_file),
        "details_file": str(details_file),
        "patterns_file": str(patterns_file),
        "resulting_process_details_file": str(resulting_process_file),
        "resulting_process_index_file": str(resulting_process_index_file),
        "resulting_process_dir": str(resulting_process_dir),
        "selected_count": len(selected_ids),
        "resulting_process_count": len(resulting_process_index),
    }
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary["summary_file"] = str(summary_file)
    return summary


def build_local_models(manifest: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    run_dirs = [Path(item).expanduser().resolve() for item in manifest.get("local_runs", [])]
    if not run_dirs:
        return []

    models_dir = out_dir / "models"
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    run_reports: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        state_path = run_dir / "cache" / "process_from_flow_state.json"
        process_dir = run_dir / "exports" / "processes"
        if not state_path.is_file():
            raise ValueError(f"missing state file in run dir: {run_dir}")
        if not process_dir.is_dir():
            raise ValueError(f"missing exported processes in run dir: {run_dir}")

        state = json.loads(state_path.read_text(encoding="utf-8"))
        local_process_map = {
            record.process_uuid: record
            for record in (
                load_process_record(
                    path,
                    source_kind="local_run_export",
                    source_label=str(run_dir),
                )
                for path in sorted(process_dir.glob("*.json"))
            )
        }
        reusable_process_map = load_reusable_process_records(manifest, out_dir)
        process_map = dict(reusable_process_map)
        process_map.update(local_process_map)
        edges = infer_edges(process_map)
        final_process_id = choose_reference_process(
            process_map,
            edges,
            state,
            preferred_process_ids=set(local_process_map),
        )
        reachable = collect_reachable(final_process_id, edges)
        order = topological_order(reachable, edges)
        factors = compute_multiplication_factors(
            process_map, reachable, edges, order, final_process_id
        )
        filtered_edges = [
            edge for edge in edges if edge.src in reachable and edge.dst in reachable
        ]
        model, summary = build_lifecycle_model_dataset(
            run_name=run_dir.name,
            state=state,
            process_map=process_map,
            order=order,
            edges=filtered_edges,
            factors=factors,
            final_process_id=final_process_id,
        )

        run_out = models_dir / run_dir.name
        bundle_out = run_out / "tidas_bundle" / "lifecyclemodels"
        bundle_out.mkdir(parents=True, exist_ok=True)

        model_uuid = summary["model_uuid"]
        model_version = summary["model_version"]
        model_file = bundle_out / f"{model_uuid}_{model_version}.json"
        summary_file = run_out / "summary.json"
        connections_file = run_out / "connections.json"
        process_catalog_file = run_out / "process-catalog.json"
        used_reusable_processes = [
            process_map[process_id]
            for process_id in order
            if process_map[process_id].source_kind != "local_run_export"
        ]
        reusable_source_counts = defaultdict(int)
        for record in reusable_process_map.values():
            reusable_source_counts[record.source_kind] += 1
        used_source_counts = defaultdict(int)
        for record in used_reusable_processes:
            used_source_counts[record.source_kind] += 1
        summary["source_counts"] = {
            "local_run_export": len(local_process_map),
            "used_local_run_processes": len(
                [
                    process_id
                    for process_id in order
                    if process_map[process_id].source_kind == "local_run_export"
                ]
            ),
            "available_reusable_processes": len(reusable_process_map),
            "used_reusable_processes": len(used_reusable_processes),
            "available_reusable_by_kind": dict(reusable_source_counts),
            "used_reusable_by_kind": dict(used_source_counts),
        }
        summary["reused_candidate_processes"] = [
            {
                "process_uuid": record.process_uuid,
                "version": record.version,
                "name_en": record.name_en,
                "reference_flow_uuid": record.reference_flow_uuid,
                "source_label": record.source_label,
                "included_process_ref_count": record.included_process_ref_count,
            }
            for record in used_reusable_processes
        ]

        model_file.write_text(
            json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summary_file.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        connections_file.write_text(
            json.dumps(
                [
                    {
                        "src": edge.src,
                        "dst": edge.dst,
                        "flow_uuid": edge.flow_uuid,
                        "downstream_input_amount": format_decimal(
                            edge.downstream_input_amount
                        ),
                        "confidence": format_decimal(edge.confidence),
                        "reasons": edge.reasons,
                        "src_name_en": process_map[edge.src].name_en,
                        "dst_name_en": process_map[edge.dst].name_en,
                        "src_source_kind": process_map[edge.src].source_kind,
                        "dst_source_kind": process_map[edge.dst].source_kind,
                    }
                    for edge in filtered_edges
                ],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        process_catalog_file.write_text(
            json.dumps(
                [
                    {
                        "process_uuid": process_id,
                        "name_en": record.name_en,
                        "name_zh": record.name_zh,
                        "route_en": record.route_en,
                        "mix_en": record.mix_en,
                        "geography_code": record.geography_code,
                        "classification_path": record.classification_path,
                        "reference_flow_uuid": record.reference_flow_uuid,
                        "reference_direction": record.reference_direction,
                        "reference_amount": format_decimal(record.reference_amount),
                        "input_flow_count": len(record.input_amounts),
                        "output_flow_count": len(record.output_amounts),
                        "source_kind": record.source_kind,
                        "source_label": record.source_label,
                        "included_process_ref_count": record.included_process_ref_count,
                    }
                    for process_id, record in sorted(process_map.items())
                ],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        validation = validate_model_file(model_file)
        validation_file = reports_dir / f"{run_dir.name}-validation.json"
        validation_file.write_text(
            json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        run_reports.append(
            {
                "run_dir": str(run_dir),
                "model_file": str(model_file),
                "summary_file": str(summary_file),
                "connections_file": str(connections_file),
                "process_catalog_file": str(process_catalog_file),
                "validation_file": str(validation_file),
                "validation": validation,
                "summary": summary,
            }
        )

    return run_reports
def publish_local_models_via_mcp(
    manifest: dict[str, Any],
    out_dir: Path,
    build_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    publish_cfg = manifest.get("publish") or {}
    if not publish_cfg.get("enabled"):
        return []
    publish_mode = str(publish_cfg.get("mode") or "mcp_insert_only")
    if publish_mode != "mcp_insert_only":
        raise ValueError("publish.mode only supports 'mcp_insert_only'")

    api_key = os.getenv("TIANGONG_LCA_REMOTE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("publish requested but TIANGONG_LCA_REMOTE_API_KEY is not set")
    url = os.getenv("TIANGONG_LCA_REMOTE_URL", "https://lcamcp.tiangong.earth/mcp")
    max_attempts = int(publish_cfg.get("max_attempts") or 5)
    retry_delay_seconds = float(publish_cfg.get("retry_delay_seconds") or 2.0)
    target_runs = {
        Path(item).name if "/" in str(item) else str(item)
        for item in ensure_list(publish_cfg.get("target_runs"))
        if str(item).strip()
    }
    selected_reports = [
        report
        for report in build_reports
        if not target_runs or report["summary"]["run_name"] in target_runs
    ]
    if not selected_reports:
        raise ValueError("publish requested but no local build reports matched publish.target_runs")

    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    publish_reports: list[dict[str, Any]] = []
    request_id = 1000

    for report in selected_reports:
        summary = report["summary"]
        run_name = summary["run_name"]
        model_file = Path(report["model_file"])
        model_json = json.loads(model_file.read_text(encoding="utf-8"))
        model_uuid = summary["model_uuid"]
        model_version = summary["model_version"]

        publish_report: dict[str, Any] = {
            "run_name": run_name,
            "model_uuid": model_uuid,
            "model_version": model_version,
            "mode": publish_mode,
            "status": "pending",
            "steps": [],
        }

        try:
            select_before_payload = mcp_call_tool(
                api_key=api_key,
                url=url,
                tool_name="Database_CRUD_Tool",
                arguments={
                    "operation": "select",
                    "table": "lifecyclemodels",
                    "id": model_uuid,
                },
                request_id=request_id,
                max_attempts=max_attempts,
                retry_delay_seconds=retry_delay_seconds,
            )
            request_id += 1
            select_before = parse_tool_text_json(select_before_payload)
            publish_report["steps"].append(
                {
                    "name": "select_before_insert",
                    "response": select_before,
                }
            )

            existing_rows = select_before.get("data") or []
            if existing_rows:
                publish_report["status"] = "skipped_existing"
            else:
                insert_payload = mcp_call_tool(
                    api_key=api_key,
                    url=url,
                    tool_name="Database_CRUD_Tool",
                    arguments={
                        "operation": "insert",
                        "table": "lifecyclemodels",
                        "id": model_uuid,
                        "jsonOrdered": model_json,
                    },
                    request_id=request_id,
                    max_attempts=max_attempts,
                    retry_delay_seconds=retry_delay_seconds,
                )
                request_id += 1
                insert_result = parse_tool_text_json(insert_payload)
                publish_report["steps"].append(
                    {
                        "name": "insert",
                        "response": insert_result,
                    }
                )
                publish_report["status"] = "inserted"

            if publish_cfg.get("select_after_insert", True):
                select_after_payload = mcp_call_tool(
                    api_key=api_key,
                    url=url,
                    tool_name="Database_CRUD_Tool",
                    arguments={
                        "operation": "select",
                        "table": "lifecyclemodels",
                        "id": model_uuid,
                    },
                    request_id=request_id,
                    max_attempts=max_attempts,
                    retry_delay_seconds=retry_delay_seconds,
                )
                request_id += 1
                select_after = parse_tool_text_json(select_after_payload)
                publish_report["steps"].append(
                    {
                        "name": "select_after_insert",
                        "response": select_after,
                    }
                )
                publish_report["verified_present"] = bool(select_after.get("data"))
        except Exception as exc:  # noqa: BLE001
            publish_report["status"] = "failed"
            publish_report["error"] = str(exc)

        publish_file = reports_dir / f"{run_name}-publish.json"
        publish_file.write_text(
            json.dumps(publish_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        publish_report["publish_file"] = str(publish_file)
        publish_reports.append(publish_report)

    return publish_reports


def build_plan(manifest: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    if manifest.get("allow_remote_write"):
        raise ValueError(
            "allow_remote_write=true is not supported by this initial scaffold. "
            "Keep the run read-only."
        )

    outputs = {
        "root": str(out_dir),
        "run_plan": str(out_dir / "run-plan.json"),
        "resolved_manifest": str(out_dir / "resolved-manifest.json"),
        "selection_brief": str(out_dir / "selection" / "selection-brief.md"),
        "discovery_dir": str(out_dir / "discovery"),
        "reference_model_summary": str(out_dir / "discovery" / "reference-model-summary.json"),
        "models_dir": str(out_dir / "models"),
        "reports_dir": str(out_dir / "reports"),
        "publish_reports_dir": str(out_dir / "reports"),
    }

    plan = {
        "skill": "lifecyclemodel-automated-builder",
        "mode": (
            "read_only_planning"
            if not manifest.get("local_runs")
            else "local_build_with_publish_gate"
            if (manifest.get("publish") or {}).get("enabled")
            else "read_only_local_build"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest,
        "runtime": runtime_config(),
        "guardrails": [
            "never persist TIANGONG_LCA_REMOTE_API_KEY in repo files or artifacts",
            "remote CRUD is select-only unless an explicit MCP publish mode is enabled",
            "generate local json_ordered lifecyclemodel artifacts before considering any upload",
            "the skill does not emit json_tg, rule_verification, or generated process artifacts",
            "publish.mode=mcp_insert_only only performs lifecyclemodels select and insert with jsonOrdered",
            "publish mode never performs remote delete operations",
        ],
        "artifacts": outputs,
        "stages": [
            {
                "name": "discover-account-processes",
                "mode": "read_only",
                "description": "List processes accessible to the target account.",
            },
            {
                "name": "discover-public-open-data",
                "mode": "read_only",
                "description": "List public records with state_code=100, centered on processes.",
            },
            {
                "name": "normalize-candidates",
                "mode": "local",
                "description": "Reduce rows to fields needed for AI scoring and graph assembly.",
            },
            {
                "name": "discover-reference-models",
                "mode": "read_only",
                "description": "Optional MCP search/select on lifecyclemodels and related processes to learn model shape and candidate packaging patterns.",
            },
            {
                "name": "ai-process-selection",
                "mode": "local",
                "description": "Choose candidate process groups and record reasoning.",
            },
            {
                "name": "assemble-lifecyclemodels",
                "mode": "local",
                "description": "Build native json_ordered lifecyclemodel artifacts only.",
            },
            {
                "name": "validate",
                "mode": "local",
                "description": "Run local lifecyclemodel schema and classification checks.",
            },
            {
                "name": "publish-via-mcp",
                "mode": "remote_insert_only",
                "description": "Optional explicit gate: MCP select then lifecyclemodels insert using jsonOrdered only.",
            },
        ],
    }
    if manifest.get("local_runs"):
        plan["local_runs"] = [str(Path(item).expanduser().resolve()) for item in manifest["local_runs"]]
    return plan


def write_artifacts(plan: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "discovery").mkdir(parents=True, exist_ok=True)
    (out_dir / "selection").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    with (out_dir / "run-plan.json").open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with (out_dir / "resolved-manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(plan["manifest"], handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with (out_dir / "selection" / "selection-brief.md").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(selection_brief(plan["manifest"]))


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    try:
        manifest = load_manifest(manifest_path)
        plan = build_plan(manifest, out_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    try:
        write_artifacts(plan, out_dir)
        try:
            reference_model_summary = discover_reference_models(manifest, out_dir)
        except Exception as exc:  # noqa: BLE001
            reference_model_summary = {
                "executed": False,
                "queries": [
                    str(item).strip()
                    for item in ensure_list(
                        (manifest.get("discovery") or {}).get("reference_model_queries")
                    )
                    if str(item).strip()
                ],
                "error": str(exc),
            }
            summary_path = out_dir / "discovery" / "reference-model-summary.json"
            summary_path.write_text(
                json.dumps(reference_model_summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            reference_model_summary["summary_file"] = str(summary_path)
        if reference_model_summary:
            plan["reference_model_discovery"] = reference_model_summary
        build_reports = build_local_models(manifest, out_dir)
        if build_reports:
            plan["local_build_reports"] = build_reports
        publish_reports = publish_local_models_via_mcp(manifest, out_dir, build_reports)
        if publish_reports:
            plan["publish_reports"] = publish_reports
        with (out_dir / "run-plan.json").open("w", encoding="utf-8") as handle:
            json.dump(plan, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
