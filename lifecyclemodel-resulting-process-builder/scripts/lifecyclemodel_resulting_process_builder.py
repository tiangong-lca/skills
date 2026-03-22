#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
REQUEST_SCHEMA_PATH = SKILL_DIR / "assets" / "request.schema.json"


@dataclass
class ProcessRecord:
    process_uuid: str
    version: str
    raw: dict[str, Any]
    source_label: str
    source_path: str | None
    reference_exchange_internal_id: str
    reference_flow_uuid: str
    reference_direction: str
    reference_amount: Decimal
    input_amounts: dict[str, Decimal]
    output_amounts: dict[str, Decimal]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def unique_strings(values: list[str | None]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def first_non_empty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def lifecyclemodel_root(model: dict[str, Any]) -> dict[str, Any]:
    wrapped = model.get("lifeCycleModelDataSet")
    if isinstance(wrapped, dict):
        return wrapped
    return model


def copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def decimal_or_zero(value: Any) -> Decimal:
    if value is None:
        return Decimal("0")
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError):
        return Decimal("0")


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


def multilang_from_text(en_text: str, zh_text: str | None = None) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    if en_text:
        payload.append({"@xml:lang": "en", "#text": en_text})
    if zh_text:
        payload.append({"@xml:lang": "zh", "#text": zh_text})
    return payload


def process_dataset_root(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("processDataSet"), dict):
        return payload["processDataSet"]
    for key in ("json_ordered", "json"):
        nested = payload.get(key)
        if isinstance(nested, dict) and isinstance(nested.get("processDataSet"), dict):
            return nested["processDataSet"]
    raise ValueError("payload does not contain processDataSet")


def resolve_local_ref(root_schema: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"Only local schema refs are supported: {ref}")
    node: Any = root_schema
    for part in ref[2:].split("/"):
        node = node[part]
    if not isinstance(node, dict):
        raise ValueError(f"Schema ref does not resolve to an object: {ref}")
    return node


def _matches_type(value: Any, schema_type: str) -> bool:
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "null":
        return value is None
    return True


def _schema_is_valid(value: Any, schema: dict[str, Any], root_schema: dict[str, Any]) -> bool:
    return not validate_against_schema(value, schema, root_schema=root_schema)


def validate_against_schema(
    value: Any,
    schema: dict[str, Any],
    *,
    root_schema: dict[str, Any] | None = None,
    path: str = "$",
) -> list[str]:
    root = root_schema or schema
    errors: list[str] = []

    if "$ref" in schema:
        return validate_against_schema(
            value,
            resolve_local_ref(root, str(schema["$ref"])),
            root_schema=root,
            path=path,
        )

    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        if not any(_schema_is_valid(value, candidate, root) for candidate in any_of):
            errors.append(f"{path}: value does not satisfy anyOf")
        return errors

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for candidate in all_of:
            errors.extend(validate_against_schema(value, candidate, root_schema=root, path=path))

    if_schema = schema.get("if")
    then_schema = schema.get("then")
    if isinstance(if_schema, dict) and isinstance(then_schema, dict):
        if _schema_is_valid(value, if_schema, root):
            errors.extend(validate_against_schema(value, then_schema, root_schema=root, path=path))

    schema_type = schema.get("type")
    if isinstance(schema_type, str) and not _matches_type(value, schema_type):
        return [f"{path}: expected {schema_type}, got {type(value).__name__}"]

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"{path}: expected one of {enum_values}, got {value!r}")

    if "const" in schema and value != schema.get("const"):
        errors.append(f"{path}: expected const {schema.get('const')!r}, got {value!r}")

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            errors.append(f"{path}: string shorter than minLength={min_length}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{path}: number smaller than minimum={minimum}")

    if isinstance(value, list):
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for index, item in enumerate(value):
                errors.extend(
                    validate_against_schema(
                        item,
                        items_schema,
                        root_schema=root,
                        path=f"{path}[{index}]",
                    )
                )

    if isinstance(value, dict):
        required = schema.get("required") or []
        for field in required:
            if field not in value:
                errors.append(f"{path}: missing required field '{field}'")

        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, subschema in properties.items():
                if key in value and isinstance(subschema, dict):
                    errors.extend(
                        validate_against_schema(
                            value[key],
                            subschema,
                            root_schema=root,
                            path=f"{path}.{key}",
                        )
                    )

            additional = schema.get("additionalProperties", True)
            if additional is False:
                unexpected = sorted(set(value) - set(properties))
                for key in unexpected:
                    errors.append(f"{path}: unexpected property '{key}'")
            elif isinstance(additional, dict):
                for key in sorted(set(value) - set(properties)):
                    errors.extend(
                        validate_against_schema(
                            value[key],
                            additional,
                            root_schema=root,
                            path=f"{path}.{key}",
                        )
                    )

    return errors


def validate_request_schema(request: dict[str, Any]) -> None:
    schema = load_json(REQUEST_SCHEMA_PATH)
    errors = validate_against_schema(request, schema)
    source_model = request.get("source_model") or {}
    if not any(
        source_model.get(field)
        for field in ("id", "json_ordered", "json_ordered_path")
    ):
        errors.append("$.source_model: provide at least one of id/json_ordered/json_ordered_path")
    if errors:
        raise ValueError("request schema validation failed:\n- " + "\n- ".join(errors))


def resolve_path(base_dir: Path, raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if "://" in text or text.startswith("file:"):
        return text
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def resolve_name_field(name_payload: Any) -> str | None:
    if isinstance(name_payload, str):
        return name_payload.strip() or None
    if isinstance(name_payload, list):
        for item in name_payload:
            text = resolve_name_field(item)
            if text:
                return text
        return None
    if not isinstance(name_payload, dict):
        return None
    direct_text = first_non_empty(
        name_payload.get("@index"),
        name_payload.get("#text"),
        name_payload.get("text"),
    )
    if direct_text:
        return direct_text
    for key in ("baseName", "shortName", "name"):
        value = name_payload.get(key)
        if isinstance(value, dict):
            text = first_non_empty(
                value.get("@index"),
                value.get("#text"),
                value.get("text"),
            )
            if text:
                return text
        text = first_non_empty(value)
        if text:
            return text
    return None


def normalized_name_info(name_payload: Any, fallback_text: str) -> dict[str, Any]:
    if isinstance(name_payload, dict) and name_payload:
        return copy_json(name_payload)
    text = resolve_name_field(name_payload) or fallback_text
    return {
        "baseName": [
            {
                "@xml:lang": "en",
                "#text": text,
            }
        ]
    }


def model_dataset_version(model: dict[str, Any]) -> str | None:
    root = lifecyclemodel_root(model)
    administrative = root.get("administrativeInformation") or {}
    publication = (
        administrative.get("publicationAndOwnership")
        or administrative.get("common:publicationAndOwnership")
        or {}
    )
    return first_non_empty(publication.get("common:dataSetVersion"))


def process_reference_pairs(model: dict[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in extract_process_instances(model):
        key = (item["process_id"], item["process_version"])
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def parse_process_record(
    payload: dict[str, Any],
    *,
    source_label: str,
    source_path: str | None,
) -> ProcessRecord:
    dataset = process_dataset_root(payload)
    info = dataset.get("processInformation") or {}
    data_info = info.get("dataSetInformation") or {}
    publication = ((dataset.get("administrativeInformation") or {}).get("publicationAndOwnership") or {})
    process_uuid = first_non_empty(data_info.get("common:UUID"))
    version = first_non_empty(publication.get("common:dataSetVersion")) or "00.00.001"
    if not process_uuid:
        raise ValueError(f"processDataSet missing common:UUID from {source_label}")
    ref_internal_id = first_non_empty(
        ((info.get("quantitativeReference") or {}).get("referenceToReferenceFlow"))
    )
    exchanges = ensure_list((dataset.get("exchanges") or {}).get("exchange"))
    ref_exchange = None
    if ref_internal_id:
        ref_exchange = next(
            (item for item in exchanges if isinstance(item, dict) and str(item.get("@dataSetInternalID")) == ref_internal_id),
            None,
        )
    if not isinstance(ref_exchange, dict):
        raise ValueError(f"reference exchange {ref_internal_id!r} not found for process {process_uuid}")

    input_amounts: dict[str, Decimal] = {}
    output_amounts: dict[str, Decimal] = {}
    for exchange in exchanges:
        if not isinstance(exchange, dict):
            continue
        flow_uuid = (((exchange.get("referenceToFlowDataSet") or {})).get("@refObjectId"))
        if not flow_uuid:
            continue
        amount = decimal_or_zero(exchange.get("meanAmount") or exchange.get("resultingAmount"))
        direction = str(exchange.get("exchangeDirection") or "").strip()
        if direction == "Input":
            input_amounts[flow_uuid] = input_amounts.get(flow_uuid, Decimal("0")) + amount
        elif direction == "Output":
            output_amounts[flow_uuid] = output_amounts.get(flow_uuid, Decimal("0")) + amount

    return ProcessRecord(
        process_uuid=process_uuid,
        version=version,
        raw={"processDataSet": dataset},
        source_label=source_label,
        source_path=source_path,
        reference_exchange_internal_id=str(ref_internal_id),
        reference_flow_uuid=(((ref_exchange.get("referenceToFlowDataSet") or {})).get("@refObjectId") or ""),
        reference_direction=str(ref_exchange.get("exchangeDirection") or ""),
        reference_amount=decimal_or_zero(ref_exchange.get("meanAmount") or ref_exchange.get("resultingAmount")),
        input_amounts=input_amounts,
        output_amounts=output_amounts,
    )


def parse_mcp_event_stream(text: str) -> str:
    lines = [line[len("data: "):] for line in text.splitlines() if line.startswith("data: ")]
    return "\n".join(lines).strip() if lines else text.strip()


def mcp_select_process(process_id: str, version: str | None) -> dict[str, Any] | None:
    api_key = os.getenv("TIANGONG_LCA_REMOTE_API_KEY", "").strip()
    url = os.getenv("TIANGONG_LCA_REMOTE_URL", "").strip()
    if not api_key or not url:
        return None
    arguments: dict[str, Any] = {
        "operation": "select",
        "table": "processes",
        "id": process_id,
    }
    if version:
        arguments["version"] = version
    body = {
        "jsonrpc": "2.0",
        "id": 9001,
        "method": "tools/call",
        "params": {
            "name": "Database_CRUD_Tool",
            "arguments": arguments,
        },
    }
    cmd = [
        "curl",
        "-sS",
        "--max-time",
        "120",
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
        json.dumps(body, ensure_ascii=False),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    raw_text = parse_mcp_event_stream(result.stdout)
    if not raw_text:
        return None
    try:
        outer = json.loads(raw_text)
        content = ((outer.get("result") or {}).get("content") or [])
        if not content:
            return None
        payload = json.loads(content[0]["text"])
        rows = payload.get("data") or []
        if not rows:
            return None
        return rows[0]
    except Exception:
        return None


def model_identifier(model: dict[str, Any], source_model: dict[str, Any]) -> tuple[str, str, str]:
    root = lifecyclemodel_root(model)
    info = root.get("lifeCycleModelInformation") or {}
    data_info = info.get("dataSetInformation") or {}
    model_id = first_non_empty(
        source_model.get("id"),
        data_info.get("common:UUID"),
        root.get("@id"),
        root.get("id"),
        data_info.get("identifierOfSubDataSet"),
    ) or f"lm-{sha256_text(canonical_json(model))[:12]}"
    version = first_non_empty(
        source_model.get("version"),
        model_dataset_version(model),
        root.get("@version"),
        root.get("version"),
        data_info.get("@version"),
    ) or "00.00.001"
    name = first_non_empty(
        source_model.get("name"),
        resolve_name_field(data_info.get("name")),
        root.get("name"),
        model_id,
    ) or model_id
    return model_id, version, name


def extract_process_instances(model: dict[str, Any]) -> list[dict[str, Any]]:
    root = lifecyclemodel_root(model)
    info = root.get("lifeCycleModelInformation") or {}
    technology = info.get("technology") or root.get("technology") or {}
    processes = technology.get("processes") or {}
    instances = ensure_list(processes.get("processInstance"))
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(instances, start=1):
        if not isinstance(item, dict):
            continue
        ref = item.get("referenceToProcess") or {}
        normalized.append(
            {
                "instance_id": first_non_empty(
                    item.get("@dataSetInternalID"),
                    item.get("@id"),
                    item.get("id"),
                    f"pi-{index}",
                )
                or f"pi-{index}",
                "process_id": first_non_empty(ref.get("@refObjectId"), ref.get("id"), ref.get("processId")) or f"proc-{index}",
                "process_version": first_non_empty(ref.get("@version"), ref.get("version")) or "00.00.001",
                "label": first_non_empty(
                    resolve_name_field(ref.get("common:shortDescription")),
                    resolve_name_field(ref.get("shortDescription")),
                    resolve_name_field(ref.get("name")),
                    ref.get("@refObjectId"),
                )
                or f"process-{index}",
                "multiplication_factor": decimal_or_zero(item.get("@multiplicationFactor")),
                "reference_to_process": copy_json(ref),
                "raw": item,
            }
        )
    return normalized


def extract_edges(model: dict[str, Any]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for instance in extract_process_instances(model):
        raw = instance["raw"]
        connections = raw.get("connections") or {}
        outputs = ensure_list(connections.get("outputExchange"))
        for edge_index, exchange in enumerate(outputs, start=1):
            if not isinstance(exchange, dict):
                continue
            downstream_items = ensure_list(exchange.get("downstreamProcess"))
            for downstream_index, downstream in enumerate(downstream_items, start=1):
                if not isinstance(downstream, dict):
                    continue
                downstream_id = first_non_empty(
                    downstream.get("@refObjectId"),
                    downstream.get("@id"),
                    downstream.get("id"),
                    exchange.get("downstreamProcessId"),
                )
                if not downstream_id:
                    continue
                edge_id = first_non_empty(
                    exchange.get("@id"),
                    exchange.get("id"),
                    f"{instance['instance_id']}-edge-{edge_index}-{downstream_index}",
                ) or f"{instance['instance_id']}-edge-{edge_index}-{downstream_index}"
                edges.append(
                    {
                        "edge_id": edge_id,
                        "from": instance["instance_id"],
                        "to": downstream_id,
                        "exchange_id": first_non_empty(exchange.get("@id"), exchange.get("id")),
                        "flow_uuid": first_non_empty(
                            exchange.get("@flowUUID"),
                            downstream.get("@flowUUID"),
                            exchange.get("flowUUID"),
                        ),
                    }
                )
    return edges


def auto_detect_process_catalog_path(model_path: str | None) -> str | None:
    if not model_path:
        return None
    path = Path(model_path).expanduser().resolve()
    try:
        if path.parent.name != "lifecyclemodels" or path.parent.parent.name != "tidas_bundle":
            return None
    except IndexError:
        return None
    candidate = path.parent.parent.parent / "process-catalog.json"
    if candidate.is_file():
        return str(candidate)
    return None


def auto_detect_process_json_dirs(model_path: str | None) -> list[str]:
    if not model_path:
        return []
    path = Path(model_path).expanduser().resolve()
    candidates: list[Path] = []
    if path.parent.name == "lifecyclemodels":
        candidates.append(path.parent.parent / "processes")
    candidates.append(path.parent / "processes")
    stem = path.stem
    if stem.endswith("-model"):
        prefix = stem[: -len("-model")].strip()
        if prefix:
            candidates.append(path.parent / f"{prefix}-processes")
    if stem.endswith("_model"):
        prefix = stem[: -len("_model")].strip()
        if prefix:
            candidates.append(path.parent / f"{prefix}_processes")
    return unique_strings([str(candidate) for candidate in candidates if candidate.is_dir()])


def process_record_key(process_id: str, version: str) -> str:
    return f"{process_id}@{version}"


def process_source_dirs(process_sources: dict[str, Any]) -> list[Path]:
    dirs: list[Path] = []
    for raw in ensure_list(process_sources.get("process_json_dirs")):
        if raw:
            path = Path(str(raw)).expanduser().resolve()
            if path.is_dir():
                dirs.append(path)
    for raw in ensure_list(process_sources.get("run_dirs")):
        if raw:
            run_dir = Path(str(raw)).expanduser().resolve()
            process_dir = run_dir / "exports" / "processes"
            if process_dir.is_dir():
                dirs.append(process_dir)
    catalog_path = process_sources.get("process_catalog_path")
    if catalog_path:
        catalog_file = Path(str(catalog_path)).expanduser().resolve()
        if catalog_file.is_file():
            try:
                catalog_payload = json.loads(catalog_file.read_text(encoding="utf-8"))
                for item in ensure_list(catalog_payload):
                    if not isinstance(item, dict):
                        continue
                    source_label = item.get("source_label")
                    if not source_label:
                        continue
                    process_dir = Path(str(source_label)).expanduser().resolve() / "exports" / "processes"
                    if process_dir.is_dir():
                        dirs.append(process_dir)
            except Exception:
                pass
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in dirs:
        text = str(path)
        if text in seen:
            continue
        seen.add(text)
        deduped.append(path)
    return deduped


def locate_local_process_file(
    process_id: str,
    version: str,
    *,
    process_dirs: list[Path],
    process_files: list[Path],
) -> Path | None:
    target_name = f"{process_id}_{version}.json"
    for directory in process_dirs:
        candidate = directory / target_name
        if candidate.is_file():
            return candidate
    for file_path in process_files:
        if file_path.is_file() and file_path.name == target_name:
            return file_path
    return None


def resolve_process_records(
    normalized_request: dict[str, Any],
    *,
    source_model_json: dict[str, Any],
    model_path: str | None,
) -> tuple[dict[str, ProcessRecord], dict[str, Any]]:
    process_sources = normalized_request.get("process_sources") or {}
    required_pairs = process_reference_pairs(source_model_json)
    process_files = [
        Path(str(raw)).expanduser().resolve()
        for raw in ensure_list(process_sources.get("process_json_files"))
        if str(raw).strip()
    ]
    process_dirs = process_source_dirs(process_sources)
    records: dict[str, ProcessRecord] = {}
    resolution_items: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    remote_hits = 0

    for process_id, version in required_pairs:
        record_key = process_record_key(process_id, version)
        local_path = locate_local_process_file(
            process_id,
            version,
            process_dirs=process_dirs,
            process_files=process_files,
        )
        if local_path is not None:
            record = parse_process_record(
                load_json(local_path),
                source_label=str(local_path.parent),
                source_path=str(local_path),
            )
            records[record_key] = record
            resolution_items.append(
                {
                    "process_id": process_id,
                    "version": version,
                    "resolution": "local_file",
                    "source_path": str(local_path),
                }
            )
            continue

        if process_sources.get("allow_mcp_lookup", True):
            remote_row = mcp_select_process(process_id, version)
            if isinstance(remote_row, dict):
                record = parse_process_record(
                    remote_row,
                    source_label="mcp_select_process",
                    source_path=None,
                )
                records[record_key] = record
                remote_hits += 1
                resolution_items.append(
                    {
                        "process_id": process_id,
                        "version": version,
                        "resolution": "mcp_select",
                        "source_path": None,
                    }
                )
                continue

        unresolved.append({"process_id": process_id, "version": version})

    if unresolved:
        missing = ", ".join(f"{item['process_id']}@{item['version']}" for item in unresolved[:10])
        raise ValueError(
            "could not resolve referenced process datasets for lifecycle model: "
            f"{missing}. Provide process_sources.run_dirs/process_json_dirs/process_json_files "
            "or enable readable remote process rows via TIANGONG_LCA_REMOTE_*."
        )

    return records, {
        "required_process_count": len(required_pairs),
        "resolved_process_count": len(records),
        "local_process_dir_count": len(process_dirs),
        "explicit_process_file_count": len(process_files),
        "remote_resolution_count": remote_hits,
        "items": resolution_items,
    }


def normalize_request(
    request: dict[str, Any],
    *,
    base_dir: Path,
) -> dict[str, Any]:
    normalized = copy.deepcopy(request)
    source_model = normalized.setdefault("source_model", {})
    source_model["json_ordered_path"] = resolve_path(base_dir, source_model.get("json_ordered_path"))
    projection = normalized.setdefault("projection", {})
    projection.setdefault("mode", "primary-only")
    projection["attach_graph_snapshot_uri"] = resolve_path(
        base_dir,
        projection.get("attach_graph_snapshot_uri"),
    )
    process_sources = normalized.setdefault("process_sources", {})
    model_path = source_model.get("json_ordered_path")
    should_auto_detect_process_dirs = not any(
        [
            first_non_empty(process_sources.get("process_catalog_path")),
            ensure_list(process_sources.get("run_dirs")),
            ensure_list(process_sources.get("process_json_dirs")),
            ensure_list(process_sources.get("process_json_files")),
        ]
    )
    auto_catalog = auto_detect_process_catalog_path(model_path)
    resolved_catalog = resolve_path(
        base_dir,
        process_sources.get("process_catalog_path") or auto_catalog,
    )
    if resolved_catalog:
        process_sources["process_catalog_path"] = resolved_catalog
    else:
        process_sources.pop("process_catalog_path", None)
    process_sources["run_dirs"] = [
        path
        for path in (
            resolve_path(base_dir, raw)
            for raw in ensure_list(process_sources.get("run_dirs"))
        )
        if path
    ]
    process_sources["process_json_dirs"] = [
        path
        for path in (
            resolve_path(base_dir, raw)
            for raw in ensure_list(process_sources.get("process_json_dirs"))
        )
        if path
    ]
    if should_auto_detect_process_dirs:
        process_sources["process_json_dirs"] = unique_strings(
            process_sources.get("process_json_dirs", []) + auto_detect_process_json_dirs(model_path)
        )
    process_sources["process_json_files"] = [
        path
        for path in (
            resolve_path(base_dir, raw)
            for raw in ensure_list(process_sources.get("process_json_files"))
        )
        if path
    ]
    process_sources["allow_mcp_lookup"] = bool(process_sources.get("allow_mcp_lookup", True))
    publish = normalized.setdefault("publish", {})
    publish.setdefault("intent", "dry_run")
    publish.setdefault("prepare_process_payloads", True)
    publish.setdefault("prepare_relation_payloads", True)
    return normalized


def synthesize_request_from_model(
    *,
    model_file: Path,
    projection_mode: str,
) -> dict[str, Any]:
    process_sources: dict[str, Any] = {
        "allow_mcp_lookup": True,
    }
    auto_catalog = auto_detect_process_catalog_path(str(model_file.resolve()))
    if auto_catalog:
        process_sources["process_catalog_path"] = auto_catalog
    auto_process_dirs = auto_detect_process_json_dirs(str(model_file.resolve()))
    if auto_process_dirs:
        process_sources["process_json_dirs"] = auto_process_dirs
    return {
        "source_model": {
            "json_ordered_path": str(model_file.resolve()),
        },
        "projection": {
            "mode": projection_mode,
            "metadata_overrides": {},
            "attach_graph_snapshot": False,
        },
        "process_sources": process_sources,
        "publish": {
            "intent": "dry_run",
            "prepare_process_payloads": True,
            "prepare_relation_payloads": True,
        },
    }


def load_request_from_args(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if args.request:
        request_path = Path(args.request).expanduser().resolve()
        request = load_json(request_path)
        base_dir = request_path.parent
    else:
        if not args.model_file:
            raise ValueError("Either --request or --model-file is required.")
        request = synthesize_request_from_model(
            model_file=Path(args.model_file).expanduser().resolve(),
            projection_mode="all-subproducts" if args.projection_role == "all" else "primary-only",
        )
        base_dir = Path.cwd()
    validate_request_schema(request)
    return normalize_request(request, base_dir=base_dir), base_dir


def load_source_model(normalized_request: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    source_model = normalized_request.get("source_model") or {}
    embedded = source_model.get("json_ordered")
    if isinstance(embedded, dict):
        return copy.deepcopy(embedded), None
    model_path = source_model.get("json_ordered_path")
    if model_path:
        resolved = Path(str(model_path)).expanduser().resolve()
        return load_json(resolved), str(resolved)
    raise ValueError("source_model must include json_ordered or json_ordered_path for local projection.")


def reference_to_resulting_process(model: dict[str, Any]) -> tuple[str | None, str | None]:
    root = lifecyclemodel_root(model)
    info = root.get("lifeCycleModelInformation") or {}
    data_info = info.get("dataSetInformation") or {}
    ref = data_info.get("referenceToResultingProcess") or {}
    if not isinstance(ref, dict):
        return None, None
    return (
        first_non_empty(ref.get("@refObjectId"), ref.get("id")),
        first_non_empty(ref.get("@version"), ref.get("version")),
    )


def reference_process_instance_id(model: dict[str, Any]) -> str | None:
    root = lifecyclemodel_root(model)
    info = root.get("lifeCycleModelInformation") or {}
    quantitative = info.get("quantitativeReference") or {}
    ref = quantitative.get("referenceToReferenceProcess") or {}
    if isinstance(ref, dict):
        return first_non_empty(ref.get("@refObjectId"), ref.get("id"))
    return first_non_empty(ref)


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


def choose_reference_instance(
    process_instances: list[dict[str, Any]],
    requested_instance_id: str | None,
) -> dict[str, Any]:
    by_id = {item["instance_id"]: item for item in process_instances}
    if requested_instance_id and requested_instance_id in by_id:
        return by_id[requested_instance_id]
    for item in process_instances:
        if item.get("multiplication_factor", Decimal("0")) > 0:
            return item
    if process_instances:
        return process_instances[0]
    raise ValueError("lifecycle model does not contain any process instances")


def build_resulting_process_payload(
    *,
    source_model_id: str,
    source_model_version: str,
    source_model_name: str,
    source_model_name_info: dict[str, Any],
    process_id: str,
    process_version: str,
    role: str,
    projection_signature: str,
    process_instances: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    process_records: dict[str, ProcessRecord],
    reference_process_instance: str | None,
    metadata_overrides: dict[str, Any],
    attach_graph_snapshot_uri: str | None,
) -> dict[str, Any]:
    chosen_instance = choose_reference_instance(process_instances, reference_process_instance)
    chosen_key = process_record_key(
        chosen_instance["process_id"],
        chosen_instance["process_version"],
    )
    final_record = process_records[chosen_key]
    final_process = copy_json(final_record.raw)
    final_dataset = final_process["processDataSet"]
    totals: dict[tuple[str, str], dict[str, Any]] = {}

    for instance in process_instances:
        factor = instance.get("multiplication_factor", Decimal("0"))
        if factor == 0:
            continue
        record = process_records[
            process_record_key(instance["process_id"], instance["process_version"])
        ]
        exchanges = ensure_list((record.raw["processDataSet"].get("exchanges") or {}).get("exchange"))
        for exchange in exchanges:
            if not isinstance(exchange, dict):
                continue
            flow_ref = exchange.get("referenceToFlowDataSet") or {}
            flow_uuid = flow_ref.get("@refObjectId")
            direction = str(exchange.get("exchangeDirection") or "").strip()
            if not flow_uuid or direction not in {"Input", "Output"}:
                continue
            key = (flow_uuid, direction)
            scaled_amount = decimal_or_zero(exchange.get("meanAmount") or exchange.get("resultingAmount")) * factor
            if key not in totals:
                totals[key] = {
                    "amount": Decimal("0"),
                    "exchange": copy_json(exchange),
                }
            totals[key]["amount"] += scaled_amount

    instance_by_id = {item["instance_id"]: item for item in process_instances}
    for edge in edges:
        flow_uuid = edge.get("flow_uuid")
        downstream_instance = instance_by_id.get(edge.get("to"))
        if not flow_uuid or not downstream_instance:
            continue
        downstream_record = process_records.get(
            process_record_key(
                downstream_instance["process_id"],
                downstream_instance["process_version"],
            )
        )
        if downstream_record is None:
            continue
        internal_amount = downstream_record.input_amounts.get(flow_uuid, Decimal("0")) * downstream_instance.get(
            "multiplication_factor",
            Decimal("0"),
        )
        for key in ((flow_uuid, "Output"), (flow_uuid, "Input")):
            if key in totals:
                totals[key]["amount"] -= internal_amount
                if abs(totals[key]["amount"]) < Decimal("0.0000000001"):
                    totals[key]["amount"] = Decimal("0")

    target_flow_uuid = final_record.reference_flow_uuid
    target_direction = final_record.reference_direction or "Output"
    exchange_items: list[dict[str, Any]] = []
    next_internal_id = 1
    reference_exchange_internal_id = ""
    for (flow_uuid, direction), payload in sorted(totals.items(), key=lambda item: (item[0][1], item[0][0])):
        amount = payload["amount"]
        if amount <= 0:
            continue
        quantitative_reference = (
            flow_uuid == target_flow_uuid
            and direction == target_direction
            and not reference_exchange_internal_id
        )
        internal_id = str(next_internal_id)
        next_internal_id += 1
        if quantitative_reference:
            reference_exchange_internal_id = internal_id
        exchange_items.append(
            clone_exchange_with_amount(
                payload["exchange"],
                amount,
                internal_id,
                quantitative_reference=quantitative_reference,
            )
        )

    if not reference_exchange_internal_id:
        raise ValueError(
            f"could not build resulting process reference exchange for lifecycle model {source_model_id}"
        )

    data_info = ((final_dataset.get("processInformation") or {}).get("dataSetInformation") or {})
    data_info["common:UUID"] = process_id
    data_info["name"] = copy_json(source_model_name_info)
    general_comment = copy_json(data_info.get("common:generalComment") or [])
    general_comment.extend(
        multilang_from_text(
            f"Local {role} resulting process generated from lifecycle model {source_model_id}; exchanges are aggregated from included processes with internal linked flows cancelled.",
            f"本地为生命周期模型 {source_model_id} 生成的 {role} resulting process；其 exchanges 由包含过程聚合并抵消内部连接 flow 后得到。",
        )
    )
    data_info["common:generalComment"] = general_comment

    process_information = final_dataset.setdefault("processInformation", {})
    quantitative_reference = process_information.setdefault("quantitativeReference", {})
    quantitative_reference["referenceToReferenceFlow"] = reference_exchange_internal_id
    technology = process_information.setdefault("technology", {})
    included_refs = [copy_json(item["reference_to_process"]) for item in process_instances]
    if included_refs:
        technology["referenceToIncludedProcesses"] = (
            included_refs[0] if len(included_refs) == 1 else included_refs
        )

    final_dataset["exchanges"] = {
        "exchange": exchange_items[0] if len(exchange_items) == 1 else exchange_items
    }

    publication = (
        final_dataset.setdefault("administrativeInformation", {}).setdefault("publicationAndOwnership", {})
    )
    publication["common:dataSetVersion"] = process_version
    publication["common:permanentDataSetURI"] = (
        f"https://local.tiangong.invalid/processes/{process_id}?version={process_version}"
    )

    lci_method = (
        final_dataset.setdefault("modellingAndValidation", {}).setdefault("LCIMethodAndAllocation", {})
    )
    type_of_data_set = str(
        metadata_overrides.get("type_of_data_set")
        or "partly terminated system"
    )
    lci_method["typeOfDataSet"] = type_of_data_set

    metadata = {
        "generated_from_lifecyclemodel_id": source_model_id,
        "generated_from_lifecyclemodel_version": source_model_version,
        "projection_role": role,
        "projection_signature": projection_signature,
        "type_of_data_set": type_of_data_set,
    }
    metadata.update(metadata_overrides)
    if attach_graph_snapshot_uri:
        metadata["graph_snapshot_uri"] = attach_graph_snapshot_uri
    data_info["generatedFromLifecycleModel"] = {
        "id": source_model_id,
        "version": source_model_version,
        "role": role,
    }
    final_process["projectionMetadata"] = metadata
    final_process["topologySummary"] = {
        "process_instance_count": len(process_instances),
        "edge_count": len(edges),
    }
    return final_process


def build_projection_bundle(
    normalized_request: dict[str, Any],
    source_model_json: dict[str, Any],
    *,
    model_path: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_model_cfg = normalized_request.get("source_model") or {}
    projection = normalized_request.get("projection") or {}
    publish = normalized_request.get("publish") or {}
    root = lifecyclemodel_root(source_model_json)
    model_info = root.get("lifeCycleModelInformation") or {}
    model_data_info = model_info.get("dataSetInformation") or {}
    source_model_id, source_model_version, source_model_name = model_identifier(
        source_model_json,
        source_model_cfg,
    )
    source_model_name_info = normalized_name_info(
        model_data_info.get("name"),
        source_model_name,
    )
    process_instances = extract_process_instances(source_model_json)
    edges = extract_edges(source_model_json)
    process_records, process_resolution = resolve_process_records(
        normalized_request,
        source_model_json=source_model_json,
        model_path=model_path,
    )
    ref_process_id, ref_process_version = reference_to_resulting_process(source_model_json)
    ref_process_instance = reference_process_instance_id(source_model_json)
    signature_seed = {
        "source_model_id": source_model_id,
        "source_model_version": source_model_version,
        "projection_mode": projection.get("mode"),
        "process_instances": [
            {
                "instance_id": item["instance_id"],
                "process_id": item["process_id"],
                "process_version": item["process_version"],
            }
            for item in process_instances
        ],
        "edges": edges,
    }
    base_signature = f"sha256:{sha256_text(canonical_json(signature_seed))}"
    metadata_overrides = dict(projection.get("metadata_overrides") or {})
    attach_graph_snapshot_uri = projection.get("attach_graph_snapshot_uri")
    primary_process_id = first_non_empty(
        projection.get("process_id"),
        ref_process_id,
        f"{source_model_id}-resulting-process",
    ) or f"{source_model_id}-resulting-process"
    primary_process_version = first_non_empty(
        projection.get("process_version"),
        ref_process_version,
        source_model_version,
    ) or source_model_version
    projected_processes: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    notes = [
        "This skill materializes a deterministic resulting processDataSet by aggregating included process exchanges and cancelling internal linked flows.",
        "Remote writes remain gated behind an explicit publish layer.",
    ]

    primary_payload = build_resulting_process_payload(
        source_model_id=source_model_id,
        source_model_version=source_model_version,
        source_model_name=source_model_name,
        source_model_name_info=source_model_name_info,
        process_id=primary_process_id,
        process_version=primary_process_version,
        role="primary",
        projection_signature=base_signature,
        process_instances=process_instances,
        edges=edges,
        process_records=process_records,
        reference_process_instance=ref_process_instance,
        metadata_overrides=metadata_overrides,
        attach_graph_snapshot_uri=attach_graph_snapshot_uri,
    )
    projected_processes.append(
        {
            "role": "primary",
            "id": primary_process_id,
            "version": primary_process_version,
            "name": source_model_name,
            "json_ordered": primary_payload,
            "metadata": primary_payload["projectionMetadata"],
        }
    )
    relations.append(
        {
            "lifecyclemodel_id": source_model_id,
            "lifecyclemodel_version": source_model_version,
            "resulting_process_id": primary_process_id,
            "resulting_process_version": primary_process_version,
            "projection_role": "primary",
            "projection_signature": base_signature,
            "is_primary": True,
        }
    )

    if projection.get("mode") == "all-subproducts":
        json_tg = root.get("json_tg") or {}
        submodels = [item for item in ensure_list(json_tg.get("submodels")) if isinstance(item, dict)]
        if submodels:
            notes.append(
                "Subproduct projection was requested, but this lifecycle model only carries submodel metadata and no submodel-specific topology slices; only the primary aggregated resulting process was emitted."
            )
        else:
            notes.append(
                "Subproduct projection was requested, but the lifecycle model does not expose submodel topology metadata; only the primary aggregated resulting process was emitted."
            )

    source_model_summary = {
        "id": source_model_id,
        "version": source_model_version,
        "name": source_model_name,
        "json_ordered_path": model_path,
        "reference_to_resulting_process_id": ref_process_id,
        "reference_to_resulting_process_version": ref_process_version,
        "reference_process_instance_id": ref_process_instance,
        "resolved_process_summary": process_resolution,
    }
    report = {
        "generated_at": now_iso(),
        "status": (
            "projected_local_bundle"
            if publish.get("intent") == "publish"
            else "prepared_local_bundle"
        ),
        "source_model": source_model_summary,
        "projection_mode": projection.get("mode", "primary-only"),
        "node_count": len(process_instances),
        "edge_count": len(edges),
        "reference_process_instance_id": ref_process_instance,
        "process_instance_preview": [
            {
                "instance_id": item["instance_id"],
                "process_id": item["process_id"],
                "label": item["label"],
            }
            for item in process_instances[:10]
        ],
        "edge_preview": edges[:10],
        "projection_signature": base_signature,
        "attach_graph_snapshot_uri": attach_graph_snapshot_uri,
        "resolved_process_summary": process_resolution,
        "projected_process_count": len(projected_processes),
        "notes": notes,
    }
    bundle = {
        "source_model": source_model_summary,
        "projected_processes": projected_processes,
        "relations": relations,
        "report": report,
        "projection": {
            "mode": projection.get("mode", "primary-only"),
            "metadata_overrides": metadata_overrides,
            "attach_graph_snapshot_uri": attach_graph_snapshot_uri,
        },
    }
    return bundle, report, source_model_summary


def write_projection_artifacts(
    *,
    out_dir: Path,
    normalized_request: dict[str, Any],
    source_model_json: dict[str, Any],
    source_model_summary: dict[str, Any],
    bundle: dict[str, Any],
    report: dict[str, Any],
) -> None:
    dump_json(out_dir / "request.normalized.json", normalized_request)
    dump_json(out_dir / "source-model.normalized.json", source_model_json)
    dump_json(out_dir / "source-model.summary.json", source_model_summary)
    dump_json(out_dir / "projection-report.json", report)
    dump_json(out_dir / "process-projection-bundle.json", bundle)


def cmd_prepare_or_project(args: argparse.Namespace) -> int:
    normalized_request, _ = load_request_from_args(args)
    out_dir = Path(args.out_dir).expanduser().resolve()
    source_model_json, model_path = load_source_model(normalized_request)
    bundle, report, source_model_summary = build_projection_bundle(
        normalized_request,
        source_model_json,
        model_path=model_path,
    )
    write_projection_artifacts(
        out_dir=out_dir,
        normalized_request=normalized_request,
        source_model_json=source_model_json,
        source_model_summary=source_model_summary,
        bundle=bundle,
        report=report,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "command": args.command,
                "out_dir": str(out_dir),
                "projected_process_count": len(bundle["projected_processes"]),
                "relation_count": len(bundle["relations"]),
                "status": report["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    bundle_path = run_dir / "process-projection-bundle.json"
    report_path = run_dir / "projection-report.json"
    if not bundle_path.exists():
        raise ValueError(f"Missing projection bundle: {bundle_path}")
    if not report_path.exists():
        raise ValueError(f"Missing projection report: {report_path}")
    bundle = load_json(bundle_path)
    report = load_json(report_path)
    publish_bundle = {
        "generated_at": now_iso(),
        "run_dir": str(run_dir),
        "source_model": bundle.get("source_model") or {},
        "publish_processes": bool(args.publish_processes),
        "publish_relations": bool(args.publish_relations),
        "status": "prepared_local_publish_bundle",
        "projected_processes": bundle.get("projected_processes") if args.publish_processes else [],
        "relations": bundle.get("relations") if args.publish_relations else [],
        "report": report,
    }
    dump_json(run_dir / "publish-bundle.json", publish_bundle)
    dump_json(
        run_dir / "publish-intent.json",
        {
            "ok": True,
            "command": "publish",
            "run_dir": str(run_dir),
            "publish_processes": bool(args.publish_processes),
            "publish_relations": bool(args.publish_relations),
            "status": "prepared_local_publish_bundle",
        },
    )
    print(json.dumps(publish_bundle, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lifecycle model resulting process builder")
    sub = p.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--request")
    prepare.add_argument("--model-file")
    prepare.add_argument("--out-dir", required=True)
    prepare.add_argument("--projection-role", choices=["primary", "all"], default="primary")
    prepare.set_defaults(func=cmd_prepare_or_project)

    build = sub.add_parser("build")
    build.add_argument("--request")
    build.add_argument("--model-file")
    build.add_argument("--out-dir", required=True)
    build.add_argument("--projection-role", choices=["primary", "all"], default="primary")
    build.set_defaults(func=cmd_prepare_or_project)

    project = sub.add_parser("project")
    project.add_argument("--request")
    project.add_argument("--model-file")
    project.add_argument("--out-dir", required=True)
    project.add_argument("--projection-role", choices=["primary", "all"], default="primary")
    project.set_defaults(func=cmd_prepare_or_project)

    publish = sub.add_parser("publish")
    publish.add_argument("--run-dir", required=True)
    publish.add_argument("--publish-processes", action="store_true")
    publish.add_argument("--publish-relations", action="store_true")
    publish.set_defaults(func=cmd_publish)
    return p


if __name__ == "__main__":
    parsed = parser().parse_args()
    raise SystemExit(parsed.func(parsed))
