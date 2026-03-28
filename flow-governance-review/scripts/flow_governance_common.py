#!/usr/bin/env python3
from __future__ import annotations

import http.client
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
FLOW_GOVERNANCE_ROOT = SCRIPT_DIR.parent
FLOW_PROCESSING_ARTIFACT_ROOT = FLOW_GOVERNANCE_ROOT / "assets" / "artifacts" / "flow-processing"
FLOW_PROCESSING_DATASETS_DIR = FLOW_PROCESSING_ARTIFACT_ROOT / "datasets"
FLOW_PROCESSING_VALIDATION_DIR = FLOW_PROCESSING_ARTIFACT_ROOT / "validation"
FLOW_PROCESSING_NAMING_DIR = FLOW_PROCESSING_ARTIFACT_ROOT / "naming"

DEFAULT_USER_ID = ""
PUBLIC_TARGET_FLOW_TYPES = {"Product flow", "Waste flow"}
FLOW_REF_KEYS = ("@refObjectId", "@version", "@type", "@uri", "common:shortDescription")
EMERGY_TEXT_KEYWORDS = (
    "emergy",
    "能值",
    "太阳能值",
    "solar emergy",
    "solar emjoule",
    "sej",
)


def ensure_dir(path: Path | str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: Path | str, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: Path | str, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def load_json_or_jsonl(path: Path | str) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        return rows
    payload = load_json(source)
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected list payload in {path}")
    return [row for row in payload if isinstance(row, dict)]


def listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def deep_get(obj: Any, path: list[str], default: Any = None) -> Any:
    cur = obj
    for part in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
        if cur is None:
            return default
    return cur


def lang_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        text = value.get("#text")
        if isinstance(text, str):
            return text.strip()
        for nested in value.values():
            candidate = lang_text(nested)
            if candidate:
                return candidate
        return ""
    if isinstance(value, list):
        for item in value:
            candidate = lang_text(item)
            if candidate:
                return candidate
        return ""
    return ""


def first_lang_entry(value: Any, fallback: str = "") -> dict[str, Any] | None:
    if isinstance(value, dict) and "#text" in value and lang_text(value):
        return {"@xml:lang": value.get("@xml:lang") or "en", "#text": lang_text(value)}
    if isinstance(value, list):
        for item in value:
            result = first_lang_entry(item)
            if result:
                return result
    if isinstance(value, dict):
        for nested in value.values():
            result = first_lang_entry(nested)
            if result:
                return result
    if fallback:
        return {"@xml:lang": "en", "#text": fallback}
    return None


def normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", value or "")
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def dataset_payload_from_row(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("json_ordered"), dict):
        return row["json_ordered"]
    if isinstance(row.get("json"), dict):
        return row["json"]
    return row


def flow_dataset_from_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = dataset_payload_from_row(row)
    if isinstance(payload.get("flowDataSet"), dict):
        return payload["flowDataSet"]
    return payload


def process_dataset_from_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = dataset_payload_from_row(row)
    if isinstance(payload.get("processDataSet"), dict):
        return payload["processDataSet"]
    return payload


@dataclass
class FlowRecord:
    id: str
    version: str
    name: str
    flow_type: str
    short_description: dict[str, Any] | None
    scope_group: str
    row: dict[str, Any]


def extract_flow_record(row: dict[str, Any], scope_group: str = "") -> FlowRecord:
    dataset = flow_dataset_from_row(row)
    info = deep_get(dataset, ["flowInformation", "dataSetInformation"], {})
    flow_uuid = str(row.get("id") or info.get("common:UUID") or "").strip()
    version = str(
        row.get("version")
        or deep_get(dataset, ["administrativeInformation", "publicationAndOwnership", "common:dataSetVersion"], "01.00.000")
    ).strip() or "01.00.000"
    name_block = deep_get(info, ["name", "baseName"]) or deep_get(info, ["name"]) or info.get("common:shortDescription")
    name = lang_text(name_block) or flow_uuid
    flow_type = str(
        deep_get(dataset, ["modellingAndValidation", "LCIMethodAndAllocation", "typeOfDataSet"])
        or deep_get(dataset, ["modellingAndValidation", "LCIMethod", "typeOfDataSet"])
        or row.get("typeOfDataSet", "")
        or ""
    ).strip()
    short_description = first_lang_entry(info.get("common:shortDescription"), fallback=name)
    return FlowRecord(
        id=flow_uuid,
        version=version,
        name=name,
        flow_type=flow_type,
        short_description=short_description,
        scope_group=scope_group,
        row=row,
    )


def select_reference_flow_property(dataset: dict[str, Any]) -> dict[str, Any]:
    properties = [item for item in listify(deep_get(dataset, ["flowProperties", "flowProperty"], [])) if isinstance(item, dict)]
    if not properties:
        return {}
    quant_ref_id = str(
        deep_get(dataset, ["flowInformation", "quantitativeReference", "referenceToReferenceFlowProperty"]) or ""
    ).strip()
    for candidate in properties:
        if str(candidate.get("@dataSetInternalID") or "").strip() == quant_ref_id and isinstance(
            candidate.get("referenceToFlowPropertyDataSet"), dict
        ):
            return candidate["referenceToFlowPropertyDataSet"]
    for candidate in properties:
        if str(candidate.get("@dataSetInternalID") or "").strip() == "0" and isinstance(
            candidate.get("referenceToFlowPropertyDataSet"), dict
        ):
            return candidate["referenceToFlowPropertyDataSet"]
    first = properties[0].get("referenceToFlowPropertyDataSet")
    return first if isinstance(first, dict) else {}


def extract_process_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    dataset = process_dataset_from_row(row)
    info = deep_get(dataset, ["processInformation", "dataSetInformation"], {})
    name = lang_text(deep_get(info, ["name", "baseName"]) or info.get("name")) or str(row.get("id") or "")
    return (
        str(row.get("id") or "").strip(),
        str(row.get("version") or "").strip(),
        name,
    )


def extract_flow_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    record = extract_flow_record(row)
    return record.id, record.version, record.name


def extract_entity_identity(entity_type: str, row: dict[str, Any]) -> tuple[str, str, str]:
    if entity_type == "flow":
        return extract_flow_identity(row)
    if entity_type == "process":
        return extract_process_identity(row)
    raise RuntimeError(f"Unsupported entity_type: {entity_type}")


def exchange_records(process_row: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = process_dataset_from_row(process_row)
    exchanges = deep_get(dataset, ["exchanges", "exchange"], [])
    return [item for item in listify(exchanges) if isinstance(item, dict)]


def extract_process_reference_exchange(process_row: dict[str, Any]) -> dict[str, Any] | None:
    dataset = process_dataset_from_row(process_row)
    ref_internal_id = str(
        deep_get(dataset, ["processInformation", "quantitativeReference", "referenceToReferenceFlow"], "")
    ).strip()
    exchanges = exchange_records(process_row)
    if ref_internal_id:
        for exchange in exchanges:
            if str(exchange.get("@dataSetInternalID") or "").strip() == ref_internal_id:
                return exchange
    for exchange in exchanges:
        if str(exchange.get("exchangeDirection") or "").strip().lower() == "output":
            return exchange
    return exchanges[0] if exchanges else None


def extract_process_reference_flow_ref(process_row: dict[str, Any]) -> dict[str, str]:
    exchange = extract_process_reference_exchange(process_row) or {}
    ref = exchange.get("referenceToFlowDataSet") or {}
    return {
        "flow_id": str(ref.get("@refObjectId") or "").strip(),
        "flow_version": str(ref.get("@version") or "").strip(),
        "flow_text": lang_text(ref.get("common:shortDescription")),
        "exchange_internal_id": str(exchange.get("@dataSetInternalID") or "").strip(),
    }


def process_row_key(process_row: dict[str, Any]) -> str:
    process_id, version, _name = extract_process_identity(process_row)
    if not process_id or not version:
        return ""
    return f"{process_id}@{version}"


def process_row_sort_key(process_row: dict[str, Any]) -> tuple[str, tuple[int, ...], str]:
    process_id, version, name = extract_process_identity(process_row)
    return (process_id, version_key(version), name)


def process_reference_flow_ids(process_row: dict[str, Any]) -> list[str]:
    flow_ids: list[str] = []
    seen: set[str] = set()
    for exchange in exchange_records(process_row):
        ref = exchange.get("referenceToFlowDataSet") or {}
        if not isinstance(ref, dict):
            continue
        flow_id = str(ref.get("@refObjectId") or "").strip()
        if not flow_id or flow_id in seen:
            continue
        seen.add(flow_id)
        flow_ids.append(flow_id)
    return flow_ids


def process_references_flow_id(process_row: dict[str, Any], flow_id: str) -> bool:
    target = str(flow_id or "").strip()
    if not target:
        return False
    return target in set(process_reference_flow_ids(process_row))


def build_flow_indexes(flow_rows: list[dict[str, Any]], scope_group: str = "") -> dict[str, Any]:
    by_uuid: dict[str, list[FlowRecord]] = {}
    by_uuid_version: dict[str, FlowRecord] = {}
    by_name: dict[str, list[FlowRecord]] = {}
    records: list[FlowRecord] = []
    for row in flow_rows:
        record = extract_flow_record(row, scope_group=scope_group)
        records.append(record)
        by_uuid.setdefault(record.id, []).append(record)
        by_uuid_version[f"{record.id}@{record.version}"] = record
        by_name.setdefault(normalize_text(record.name), []).append(record)
    return {
        "records": records,
        "by_uuid": by_uuid,
        "by_uuid_version": by_uuid_version,
        "by_name": by_name,
    }


def merge_flow_indexes(*indexes: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index in indexes:
        for record in index.get("records", []):
            rows.append(record.row)
    return build_flow_indexes(rows)


def flow_emergy_scope_decision(
    row: dict[str, Any],
    flow_property_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = extract_flow_record(row)
    dataset = flow_dataset_from_row(row)
    name_node = deep_get(dataset, ["flowInformation", "dataSetInformation", "name"], {})
    name_candidates: list[str] = []
    name_candidates.extend(item.get("text", "") for item in lang_entries(name_node))
    signals: list[str] = []
    for text in _unique_nonempty_texts(name_candidates):
        if _text_has_emergy_keyword(text):
            signals.append(f"emergy_name:{text}")
    return {
        "entity_type": "flow",
        "flow_id": record.id,
        "version": record.version,
        "name": record.name,
        "flow_type": record.flow_type,
        "excluded": bool(signals),
        "reason": "emergy_named_flow" if signals else "",
        "signals": signals,
    }


def filter_emergy_named_flows(
    rows: list[dict[str, Any]],
    flow_property_meta: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        decision = flow_emergy_scope_decision(row, flow_property_meta=flow_property_meta)
        if decision["excluded"]:
            excluded.append(decision)
            continue
        kept.append(row)
    return kept, excluded


def process_emergy_scope_decision(
    process_row: dict[str, Any],
    flow_index: dict[str, Any] | None = None,
    flow_property_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    process_id, process_version, process_name = extract_process_identity(process_row)
    dataset = process_dataset_from_row(process_row)
    ref_info = extract_process_reference_flow_ref(process_row)
    flow_id = ref_info["flow_id"]
    flow_version = ref_info["flow_version"]
    ref_text = ref_info["flow_text"]
    signals: list[str] = []
    matched_record = None
    if flow_id and flow_version and flow_index:
        matched_record = flow_index.get("by_uuid_version", {}).get(f"{flow_id}@{flow_version}")
    if flow_id and not matched_record and flow_index:
        candidates = flow_index.get("by_uuid", {}).get(flow_id) or []
        matched_record = candidates[0] if candidates else None
    if matched_record is not None:
        flow_decision = flow_emergy_scope_decision(matched_record.row, flow_property_meta=flow_property_meta)
        if flow_decision["excluded"]:
            signals.extend(flow_decision["signals"])
    if not signals:
        functional_unit_texts = [item.get("text", "") for item in lang_entries(deep_get(dataset, ["processInformation", "quantitativeReference", "functionalUnitOrOther"]))]
        heuristic_texts = [ref_text, *functional_unit_texts]
        for text in _unique_nonempty_texts(heuristic_texts):
            if _text_has_emergy_keyword(text):
                signals.append(f"process_ref_text:{text}")
    return {
        "entity_type": "process",
        "process_id": process_id,
        "process_version": process_version,
        "process_name": process_name,
        "reference_flow_id": flow_id,
        "reference_flow_version": flow_version,
        "reference_exchange_internal_id": ref_info["exchange_internal_id"],
        "excluded": bool(signals),
        "reason": "reference_flow_name_mentions_emergy" if signals else "",
        "signals": signals,
    }


def filter_emergy_named_processes(
    rows: list[dict[str, Any]],
    flow_index: dict[str, Any] | None = None,
    flow_property_meta: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        decision = process_emergy_scope_decision(row, flow_index=flow_index, flow_property_meta=flow_property_meta)
        if decision["excluded"]:
            excluded.append(decision)
            continue
        kept.append(row)
    return kept, excluded


def flow_reference_from_record(record: FlowRecord) -> dict[str, Any]:
    reference = {
        "@type": "flow data set",
        "@refObjectId": record.id,
        "@version": record.version,
        "@uri": f"../flows/{record.id}_{record.version}.xml",
    }
    if record.short_description:
        reference["common:shortDescription"] = deepcopy(record.short_description)
    else:
        reference["common:shortDescription"] = {"@xml:lang": "en", "#text": record.name}
    return reference


def patched_flow_reference(current_ref: dict[str, Any], record: FlowRecord) -> dict[str, Any]:
    target_ref = flow_reference_from_record(record)
    patched = deepcopy(current_ref) if isinstance(current_ref, dict) else {}
    patched["@type"] = str(current_ref.get("@type") or target_ref["@type"])
    patched["@refObjectId"] = target_ref["@refObjectId"]
    patched["@version"] = target_ref["@version"]
    patched["@uri"] = target_ref["@uri"]
    patched["common:shortDescription"] = _preserve_short_description_shape(
        current_ref.get("common:shortDescription"),
        target_ref["common:shortDescription"],
    )
    return patched


def postgrest_auth_password(base_url: str, apikey: str, email: str, password: str) -> str:
    payload = json.dumps({"email": email, "password": password}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/auth/v1/token?grant_type=password",
        data=payload,
        headers={
            "apikey": apikey,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    token = str(body.get("access_token") or "").strip()
    if not token:
        raise RuntimeError("Supabase password grant did not return an access token.")
    return token


def build_rest_headers(apikey: str, access_token: str | None = None) -> dict[str, str]:
    headers = {
        "apikey": apikey,
        "Accept": "application/json",
    }
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    return headers


def postgrest_select_all(
    *,
    base_url: str,
    apikey: str,
    table: str,
    filters: dict[str, Any],
    access_token: str | None = None,
    columns: str = "*",
    page_size: int = 500,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        batch = postgrest_select_page(
            base_url=base_url,
            apikey=apikey,
            table=table,
            raw_filters={key: f"eq.{value}" for key, value in filters.items()},
            access_token=access_token,
            columns=columns,
            limit=page_size,
            offset=offset,
        )
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected PostgREST response for table '{table}': expected list.")
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def postgrest_select_page(
    *,
    base_url: str,
    apikey: str,
    table: str,
    raw_filters: dict[str, str] | None = None,
    access_token: str | None = None,
    columns: str = "*",
    limit: int = 500,
    offset: int = 0,
    order: str | None = None,
) -> list[dict[str, Any]]:
    query = {"select": columns, "limit": str(limit), "offset": str(offset)}
    for key, value in (raw_filters or {}).items():
        query[key] = value
    if order:
        query["order"] = order
    url = f"{base_url.rstrip('/')}/rest/v1/{table}?{urllib.parse.urlencode(query)}"
    last_error: Exception | None = None
    for attempt in range(4):
        request = urllib.request.Request(url, headers=build_rest_headers(apikey, access_token), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                batch = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504} or attempt >= 3:
                raise RuntimeError(f"PostgREST query failed for table '{table}': {exc.code} {body}") from exc
            last_error = RuntimeError(f"retryable PostgREST HTTP error for table '{table}': {exc.code} {body}")
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt >= 3:
                raise RuntimeError(f"PostgREST query failed for table '{table}': {exc}") from exc
        time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"PostgREST query failed for table '{table}': {last_error}")
    if not isinstance(batch, list):
        raise RuntimeError(f"Unexpected PostgREST response for table '{table}': expected list.")
    return batch


def alias_lookup(alias_map: dict[str, Any], flow_uuid: str, flow_version: str | None) -> dict[str, Any] | None:
    version_key = f"{flow_uuid}@{flow_version}" if flow_version else None
    for key in (version_key, flow_uuid):
        if key and key in alias_map and isinstance(alias_map[key], dict):
            return alias_map[key]
    return None


def unified_json_diff(before: Any, after: Any) -> str:
    import difflib

    before_text = json.dumps(before, ensure_ascii=False, indent=2).splitlines()
    after_text = json.dumps(after, ensure_ascii=False, indent=2).splitlines()
    diff = difflib.unified_diff(before_text, after_text, fromfile="before.json", tofile="after.json", lineterm="")
    return "\n".join(diff) + "\n"


def deep_diff_paths(before: Any, after: Any, prefix: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    if type(before) is not type(after):
        return [prefix]
    if isinstance(before, dict):
        paths: list[tuple[Any, ...]] = []
        keys = sorted(set(before) | set(after))
        for key in keys:
            if key not in before or key not in after:
                paths.append(prefix + (key,))
                continue
            paths.extend(deep_diff_paths(before[key], after[key], prefix + (key,)))
        return paths
    if isinstance(before, list):
        paths: list[tuple[Any, ...]] = []
        max_len = max(len(before), len(after))
        for index in range(max_len):
            if index >= len(before) or index >= len(after):
                paths.append(prefix + (index,))
                continue
            paths.extend(deep_diff_paths(before[index], after[index], prefix + (index,)))
        return paths
    if before != after:
        return [prefix]
    return []


def path_contains_reference_to_flow(path: tuple[Any, ...]) -> bool:
    return any(part == "referenceToFlowDataSet" for part in path)


def safe_process_key(process_id: str, version: str) -> str:
    version_slug = re.sub(r"[^0-9A-Za-z._-]+", "_", version or "unknown")
    return f"{process_id}__{version_slug}"


def safe_entity_key(entity_id: str, version: str) -> str:
    version_slug = re.sub(r"[^0-9A-Za-z._-]+", "_", version or "unknown")
    return f"{entity_id}__{version_slug}"


def load_rows_from_file(path: Path | str) -> list[dict[str, Any]]:
    return load_json_or_jsonl(path)


def write_rows_file(path: Path | str, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    if target.suffix.lower() == ".jsonl":
        dump_jsonl(target, rows)
        return
    dump_json(target, rows)


def version_key(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in str(version or "").split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def json_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, ensure_ascii=False, sort_keys=True) == json.dumps(right, ensure_ascii=False, sort_keys=True)


def merge_rows_by_identity(
    existing_rows: list[dict[str, Any]],
    incoming_rows: list[dict[str, Any]],
    *,
    entity_type: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    merged_rows = [deepcopy(row) for row in existing_rows if isinstance(row, dict)]
    key_to_index: dict[str, int] = {}
    for index, row in enumerate(merged_rows):
        entity_id, version, _name = extract_entity_identity(entity_type, row)
        if entity_id and version:
            key_to_index[f"{entity_id}@{version}"] = index

    inserted = 0
    updated = 0
    unchanged = 0
    skipped_invalid = 0
    for row in incoming_rows:
        if not isinstance(row, dict):
            skipped_invalid += 1
            continue
        entity_id, version, _name = extract_entity_identity(entity_type, row)
        if not entity_id or not version:
            skipped_invalid += 1
            continue
        key = f"{entity_id}@{version}"
        if key not in key_to_index:
            key_to_index[key] = len(merged_rows)
            merged_rows.append(deepcopy(row))
            inserted += 1
            continue
        existing_index = key_to_index[key]
        if json_equal(merged_rows[existing_index], row):
            unchanged += 1
            continue
        merged_rows[existing_index] = deepcopy(row)
        updated += 1
    return merged_rows, {
        "inserted": inserted,
        "updated": updated,
        "unchanged": unchanged,
        "skipped_invalid": skipped_invalid,
    }


def sync_process_pool_file(pool_file: Path | str, incoming_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pool_path = Path(pool_file)
    if pool_path.parent and not pool_path.parent.exists():
        ensure_dir(pool_path.parent)
    existing_rows = load_rows_from_file(pool_path) if pool_path.exists() else []
    merged_rows, counts = merge_rows_by_identity(existing_rows, incoming_rows, entity_type="process")
    merged_rows = sorted(merged_rows, key=process_row_sort_key)
    write_rows_file(pool_path, merged_rows)
    return {
        "pool_file": str(pool_path),
        "pool_pre_count": len(existing_rows),
        "incoming_count": len(incoming_rows),
        "pool_post_count": len(merged_rows),
        **counts,
    }


def lang_entries(value: Any) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    if isinstance(value, list):
        for item in value:
            entries.extend(lang_entries(item))
        return entries
    if isinstance(value, dict):
        if "#text" in value or "@xml:lang" in value:
            text = lang_text(value)
            if text:
                entries.append({"lang": str(value.get("@xml:lang") or "en"), "text": text})
            return entries
        for nested in value.values():
            entries.extend(lang_entries(nested))
    return entries


def entity_text_fields(
    row: dict[str, Any],
    entity_type: str,
    *,
    include_placeholders: bool = False,
) -> dict[str, dict[str, Any]]:
    info = _entity_dataset_info(row, entity_type)
    specs = _entity_text_field_specs(entity_type)
    languages = _entity_text_languages(info, specs) if include_placeholders else []
    fields: dict[str, dict[str, Any]] = {}
    for spec in specs:
        values_by_lang: dict[str, str] = {}
        for item in lang_entries(_info_value(info, spec["path"])):
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            values_by_lang[str(item.get("lang") or "en")] = text
        target_langs = list(values_by_lang)
        if include_placeholders:
            for lang in languages:
                if lang not in target_langs:
                    target_langs.append(lang)
        for lang in target_langs:
            field_id = f"{spec['kind']}:{lang}"
            fields[field_id] = {
                "field_id": field_id,
                "field_kind": spec["kind"],
                "lang": lang,
                "value": values_by_lang.get(lang, ""),
            }
    return fields


def apply_entity_text_patch(row: dict[str, Any], entity_type: str, field_id: str, new_value: str) -> None:
    kind, lang = _parse_text_field_id(field_id)
    info = _entity_dataset_info(row, entity_type, create=True)
    for spec in _entity_text_field_specs(entity_type):
        if spec["kind"] == kind:
            _upsert_info_lang_text(info, spec["path"], lang, new_value)
            return
    raise RuntimeError(f"Unsupported text field kind for {entity_type}: {kind}")


def path_contains_allowed_text_change(path: tuple[Any, ...], entity_type: str) -> bool:
    for prefix in allowed_text_change_prefixes(entity_type):
        if len(path) >= len(prefix) and tuple(path[: len(prefix)]) == prefix:
            return True
    return False


def flow_type_of_dataset(row: dict[str, Any]) -> str:
    dataset = flow_dataset_from_row(row)
    return str(
        deep_get(dataset, ["modellingAndValidation", "LCIMethodAndAllocation", "typeOfDataSet"])
        or deep_get(dataset, ["modellingAndValidation", "LCIMethod", "typeOfDataSet"])
        or row.get("typeOfDataSet", "")
        or ""
    ).strip()


def classification_branch_for_flow_type(flow_type: str) -> str:
    normalized = str(flow_type or "").strip().lower()
    if normalized in {"product flow", "waste flow"}:
        return "product"
    if normalized == "elementary flow":
        return "elementary"
    return ""


def flow_classification_entries(row: dict[str, Any], branch: str) -> list[dict[str, str]]:
    info = _entity_dataset_info(row, "flow")
    if branch == "product":
        raw_entries = deep_get(info, ["classificationInformation", "common:classification", "common:class"], [])
        id_keys = ("@classId", "@catId", "@code")
        target_key = "@classId"
    elif branch == "elementary":
        raw_entries = deep_get(
            info,
            ["classificationInformation", "common:elementaryFlowCategorization", "common:category"],
            [],
        )
        id_keys = ("@catId", "@classId", "@code")
        target_key = "@catId"
    else:
        raise RuntimeError(f"Unsupported flow classification branch: {branch}")

    entries: list[dict[str, str]] = []
    for item in listify(raw_entries):
        if not isinstance(item, dict):
            continue
        code = ""
        for key in id_keys:
            code = str(item.get(key) or "").strip()
            if code:
                break
        text = str(item.get("#text") or "").strip() or lang_text(item)
        level = str(item.get("@level") or "").strip()
        if not any((level, code, text)):
            continue
        entries.append(
            {
                "@level": level,
                target_key: code,
                "#text": text,
            }
        )
    return entries


def flow_classification_state(row: dict[str, Any]) -> dict[str, Any]:
    flow_type = flow_type_of_dataset(row)
    expected_branch = classification_branch_for_flow_type(flow_type)
    product_entries = flow_classification_entries(row, "product")
    elementary_entries = flow_classification_entries(row, "elementary")
    active_branch = expected_branch
    if not active_branch:
        if product_entries and not elementary_entries:
            active_branch = "product"
        elif elementary_entries and not product_entries:
            active_branch = "elementary"
    active_entries = product_entries if active_branch == "product" else elementary_entries if active_branch == "elementary" else []
    leaf = deepcopy(active_entries[-1]) if active_entries else {}
    return {
        "type_of_dataset": flow_type,
        "expected_branch": expected_branch,
        "active_branch": active_branch,
        "product_entries": deepcopy(product_entries),
        "elementary_entries": deepcopy(elementary_entries),
        "active_entries": deepcopy(active_entries),
        "leaf": leaf,
        "has_both_branches": bool(product_entries and elementary_entries),
    }


def apply_flow_classification_patch(
    row: dict[str, Any],
    branch: str,
    entries: list[dict[str, Any]],
    *,
    clear_other_branch: bool = True,
) -> None:
    info = _entity_dataset_info(row, "flow", create=True)
    classification_info = info.get("classificationInformation")
    if not isinstance(classification_info, dict):
        info["classificationInformation"] = {}
        classification_info = info["classificationInformation"]

    normalized_entries = [deepcopy(item) for item in entries if isinstance(item, dict)]
    if branch == "product":
        classification_info["common:classification"] = {"common:class": normalized_entries}
        if clear_other_branch:
            classification_info.pop("common:elementaryFlowCategorization", None)
        return
    if branch == "elementary":
        classification_info["common:elementaryFlowCategorization"] = {"common:category": normalized_entries}
        if clear_other_branch:
            classification_info.pop("common:classification", None)
        return
    raise RuntimeError(f"Unsupported flow classification branch: {branch}")


def path_contains_allowed_flow_classification_change(path: tuple[Any, ...]) -> bool:
    for prefix in allowed_flow_classification_change_prefixes():
        if len(path) >= len(prefix) and tuple(path[: len(prefix)]) == prefix:
            return True
    return False


def allowed_flow_classification_change_prefixes() -> list[tuple[Any, ...]]:
    dataset_roots = (("json_ordered", "flowDataSet"), ("flowDataSet",))
    subpaths = (
        (
            "flowInformation",
            "dataSetInformation",
            "classificationInformation",
            "common:classification",
            "common:class",
        ),
        (
            "flowInformation",
            "dataSetInformation",
            "classificationInformation",
            "common:elementaryFlowCategorization",
            "common:category",
        ),
    )
    prefixes: list[tuple[Any, ...]] = []
    for root in dataset_roots:
        for subpath in subpaths:
            prefixes.append(root + subpath)
    return prefixes


def allowed_text_change_prefixes(entity_type: str) -> list[tuple[Any, ...]]:
    dataset_roots: list[tuple[Any, ...]]
    if entity_type == "flow":
        dataset_roots = (("json_ordered", "flowDataSet"), ("flowDataSet",))
        subpaths = tuple(
            ("flowInformation", "dataSetInformation", *spec["path"]) for spec in _entity_text_field_specs("flow")
        )
    elif entity_type == "process":
        dataset_roots = (("json_ordered", "processDataSet"), ("processDataSet",))
        subpaths = tuple(
            ("processInformation", "dataSetInformation", *spec["path"]) for spec in _entity_text_field_specs("process")
        )
    else:
        raise RuntimeError(f"Unsupported entity_type: {entity_type}")
    prefixes: list[tuple[Any, ...]] = []
    for root in dataset_roots:
        for subpath in subpaths:
            prefixes.append(root + subpath)
    return prefixes


def _entity_dataset_info(row: dict[str, Any], entity_type: str, create: bool = False) -> dict[str, Any]:
    if entity_type == "flow":
        payload = dataset_payload_from_row(row)
        dataset = payload.get("flowDataSet")
        if not isinstance(dataset, dict):
            if not create:
                return {}
            payload["flowDataSet"] = {}
            dataset = payload["flowDataSet"]
        info_root = dataset.get("flowInformation")
        if not isinstance(info_root, dict):
            if not create:
                return {}
            dataset["flowInformation"] = {}
            info_root = dataset["flowInformation"]
        info = info_root.get("dataSetInformation")
        if not isinstance(info, dict):
            if not create:
                return {}
            info_root["dataSetInformation"] = {}
            info = info_root["dataSetInformation"]
        return info
    if entity_type == "process":
        payload = dataset_payload_from_row(row)
        dataset = payload.get("processDataSet")
        if not isinstance(dataset, dict):
            if not create:
                return {}
            payload["processDataSet"] = {}
            dataset = payload["processDataSet"]
        info_root = dataset.get("processInformation")
        if not isinstance(info_root, dict):
            if not create:
                return {}
            dataset["processInformation"] = {}
            info_root = dataset["processInformation"]
        info = info_root.get("dataSetInformation")
        if not isinstance(info, dict):
            if not create:
                return {}
            info_root["dataSetInformation"] = {}
            info = info_root["dataSetInformation"]
        return info
    raise RuntimeError(f"Unsupported entity_type: {entity_type}")


def _parse_text_field_id(field_id: str) -> tuple[str, str]:
    if ":" not in field_id:
        raise RuntimeError(f"Invalid text field id: {field_id}")
    kind, lang = field_id.split(":", 1)
    supported_kinds = {spec["kind"] for entity_type in ("flow", "process") for spec in _entity_text_field_specs(entity_type)}
    if kind not in supported_kinds or not lang:
        raise RuntimeError(f"Invalid text field id: {field_id}")
    return kind, lang


def _entity_text_field_specs(entity_type: str) -> list[dict[str, Any]]:
    if entity_type == "flow":
        return [
            {"kind": "base_name", "path": ("name", "baseName")},
            {"kind": "treatment_standards_routes", "path": ("name", "treatmentStandardsRoutes")},
            {"kind": "mix_and_location_types", "path": ("name", "mixAndLocationTypes")},
            {"kind": "flow_properties_text", "path": ("name", "flowProperties")},
            {"kind": "synonyms", "path": ("common:synonyms",)},
            {"kind": "short_description", "path": ("common:shortDescription",)},
        ]
    if entity_type == "process":
        return [
            {"kind": "base_name", "path": ("name", "baseName")},
            {"kind": "short_description", "path": ("common:shortDescription",)},
        ]
    raise RuntimeError(f"Unsupported entity_type: {entity_type}")


def _entity_text_languages(info: dict[str, Any], specs: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    for spec in specs:
        for item in lang_entries(_info_value(info, spec["path"])):
            lang = str(item.get("lang") or "en")
            if lang not in ordered:
                ordered.append(lang)
    if "en" not in ordered:
        ordered.insert(0, "en")
    return ordered


def _info_value(info: dict[str, Any], path: tuple[str, ...]) -> Any:
    return deep_get(info, list(path))


def _upsert_info_lang_text(info: dict[str, Any], path: tuple[str, ...], lang: str, text: str) -> None:
    current = info
    for key in path[:-1]:
        nested = current.get(key)
        if not isinstance(nested, dict):
            current[key] = {}
            nested = current[key]
        current = nested
    final_key = path[-1]
    current[final_key] = _upsert_lang_text(current.get(final_key), lang, text)


def _upsert_lang_text(existing: Any, lang: str, text: str) -> Any:
    if isinstance(existing, list):
        patched_items = [deepcopy(item) for item in existing]
        for item in patched_items:
            if isinstance(item, dict) and str(item.get("@xml:lang") or "en") == lang:
                item["@xml:lang"] = lang
                item["#text"] = text
                return patched_items
        patched_items.append({"@xml:lang": lang, "#text": text})
        return patched_items
    if isinstance(existing, dict):
        patched = deepcopy(existing)
        patched["@xml:lang"] = lang
        patched["#text"] = text
        return patched
    return [{"@xml:lang": lang, "#text": text}]


def _preserve_short_description_shape(existing: Any, target: dict[str, Any]) -> Any:
    if isinstance(existing, list):
        if existing and isinstance(existing[0], dict):
            patched_entry = deepcopy(existing[0])
            patched_entry["@xml:lang"] = target.get("@xml:lang") or patched_entry.get("@xml:lang") or "en"
            patched_entry["#text"] = target.get("#text", "")
            return [patched_entry]
        return [deepcopy(target)]
    if isinstance(existing, dict):
        patched_entry = deepcopy(existing)
        patched_entry["@xml:lang"] = target.get("@xml:lang") or patched_entry.get("@xml:lang") or "en"
        patched_entry["#text"] = target.get("#text", "")
        return patched_entry
    return deepcopy(target)


def _text_has_emergy_keyword(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    return any(normalize_text(keyword) in normalized for keyword in EMERGY_TEXT_KEYWORDS)


def _unique_nonempty_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result
