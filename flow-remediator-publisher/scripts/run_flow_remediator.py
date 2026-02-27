#!/usr/bin/env python3
"""Initial flow remediation + append-only publish pipeline.

This script is intentionally split into subcommands so it can be used as:
- a one-shot pipeline (`pipeline`)
- or a staged workflow (`fetch`, `review`, `propose-fix`, `validate`, `publish`)

Design goals for the initial version:
- No direct DB access; all remote access goes through MCP CRUD.
- Reuse `process-automated-builder` services for CRUD and product flow regeneration.
- Keep review/fix logic minimal but structured so it can later hand off to `lci-review --profile flow`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping


UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

DATABASE_TOOL_NAME = "Database_CRUD_Tool"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_json(data: Any) -> str:
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        text = value.get("#text")
        if isinstance(text, str):
            return text.strip()
    return str(value).strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _deep_get(obj: Mapping[str, Any] | None, path: tuple[str, ...]) -> Any:
    current: Any = obj
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _walk_strings(node: Any) -> Iterator[str]:
    if isinstance(node, str):
        text = node.strip()
        if text:
            yield text
        return
    if isinstance(node, Mapping):
        text = node.get("#text")
        if isinstance(text, str):
            stripped = text.strip()
            if stripped:
                yield stripped
        for value in node.values():
            yield from _walk_strings(value)
        return
    if isinstance(node, list):
        for item in node:
            yield from _walk_strings(item)


def _flow_root(doc: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(doc.get("flowDataSet"), Mapping):
        return copy.deepcopy(doc["flowDataSet"])  # type: ignore[index]
    return copy.deepcopy(dict(doc))


def _flow_wrapper(doc: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(doc.get("flowDataSet"), Mapping):
        return copy.deepcopy(dict(doc))
    return {"flowDataSet": copy.deepcopy(dict(doc))}


def _flow_uuid(flow_ds: Mapping[str, Any]) -> str:
    return _coerce_text(
        _deep_get(flow_ds, ("flowInformation", "dataSetInformation", "common:UUID"))
    )


def _flow_version(flow_ds: Mapping[str, Any]) -> str:
    return _coerce_text(
        _deep_get(
            flow_ds,
            ("administrativeInformation", "publicationAndOwnership", "common:dataSetVersion"),
        )
    )


def _set_flow_version(flow_ds: MutableMapping[str, Any], version: str) -> None:
    flow_ds.setdefault("administrativeInformation", {})
    admin = flow_ds["administrativeInformation"]
    if not isinstance(admin, MutableMapping):
        flow_ds["administrativeInformation"] = {}
        admin = flow_ds["administrativeInformation"]
    admin.setdefault("publicationAndOwnership", {})
    pub = admin["publicationAndOwnership"]
    if not isinstance(pub, MutableMapping):
        admin["publicationAndOwnership"] = {}
        pub = admin["publicationAndOwnership"]
    pub["common:dataSetVersion"] = version


def _flow_type(flow_ds: Mapping[str, Any]) -> str:
    return _coerce_text(
        _deep_get(flow_ds, ("modellingAndValidation", "LCIMethod", "typeOfDataSet"))
    )


def _name_node(flow_ds: Mapping[str, Any]) -> Any:
    return _deep_get(flow_ds, ("flowInformation", "dataSetInformation", "name"))


def _name_texts(flow_ds: Mapping[str, Any]) -> list[str]:
    return list(_walk_strings(_name_node(flow_ds)))


def _classification_classes(flow_ds: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = _deep_get(
        flow_ds,
        (
            "flowInformation",
            "dataSetInformation",
            "classificationInformation",
            "common:classification",
            "common:class",
        ),
    )
    out: list[dict[str, Any]] = []
    for item in _as_list(raw):
        if isinstance(item, Mapping):
            out.append(dict(item))
    return out


def _classification_leaf(flow_ds: Mapping[str, Any]) -> dict[str, str]:
    classes = _classification_classes(flow_ds)
    if not classes:
        return {"class_id": "", "text": "", "key": ""}
    leaf = classes[-1]
    class_id = _coerce_text(leaf.get("@classId"))
    text = _coerce_text(leaf.get("#text"))
    key = f"{class_id}|{text}".strip("|")
    return {"class_id": class_id, "text": text, "key": key}


def _lang_text(items: Any, lang: str) -> str:
    for item in _as_list(items):
        if isinstance(item, Mapping):
            if _coerce_text(item.get("@xml:lang")).lower() == lang.lower():
                text = _coerce_text(item.get("#text"))
                if text:
                    return text
    # Fallback to first textual entry
    for item in _as_list(items):
        text = _coerce_text(item)
        if text:
            return text
    return ""


def _name_primary(flow_ds: Mapping[str, Any], lang: str = "en") -> str:
    name = _name_node(flow_ds)
    if isinstance(name, Mapping):
        return _lang_text(name.get("baseName"), lang)
    return ""


def _normalize_name_token(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _name_fingerprint(flow_ds: Mapping[str, Any]) -> str:
    name = _name_node(flow_ds)
    if not isinstance(name, Mapping):
        return ""
    parts = []
    for key in ("baseName", "treatmentStandardsRoutes", "mixAndLocationTypes"):
        pieces = [_coerce_text(item.get("#text") if isinstance(item, Mapping) else item) for item in _as_list(name.get(key))]
        pieces = [p for p in pieces if p]
        if pieces:
            parts.append(" | ".join(pieces))
    if not parts:
        parts = _name_texts(flow_ds)
    return _normalize_name_token(" || ".join(parts))


def _flow_properties(flow_ds: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = _deep_get(flow_ds, ("flowProperties", "flowProperty"))
    out: list[dict[str, Any]] = []
    for item in _as_list(raw):
        if isinstance(item, Mapping):
            out.append(dict(item))
    return out


def _pick_reference_flow_property(flow_ds: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    props = _flow_properties(flow_ds)
    for prop in props:
        internal_id = _coerce_text(prop.get("@dataSetInternalID"))
        if internal_id == "0":
            return prop, internal_id
    if props:
        internal_id = _coerce_text(props[0].get("@dataSetInternalID"))
        return props[0], internal_id
    return None, ""


def _quant_ref_internal_id(flow_ds: Mapping[str, Any]) -> str:
    return _coerce_text(
        _deep_get(flow_ds, ("flowInformation", "quantitativeReference", "referenceToReferenceFlowProperty"))
    )


def _set_quant_ref_internal_id(flow_ds: MutableMapping[str, Any], value: str) -> None:
    flow_ds.setdefault("flowInformation", {})
    info = flow_ds["flowInformation"]
    if not isinstance(info, MutableMapping):
        flow_ds["flowInformation"] = {}
        info = flow_ds["flowInformation"]
    info.setdefault("quantitativeReference", {})
    qref = info["quantitativeReference"]
    if not isinstance(qref, MutableMapping):
        info["quantitativeReference"] = {}
        qref = info["quantitativeReference"]
    qref["referenceToReferenceFlowProperty"] = value


def _ref_uuid_from_reference_node(node: Any) -> str:
    if isinstance(node, Mapping):
        for key in ("@refObjectId", "@uri"):
            raw = _coerce_text(node.get(key))
            if not raw:
                continue
            match = UUID_RE.search(raw)
            if match:
                return match.group(0).lower()
    raw_text = _coerce_text(node)
    match = UUID_RE.search(raw_text)
    return match.group(0).lower() if match else ""


def _ref_version_from_reference_node(node: Any) -> str:
    if isinstance(node, Mapping):
        return _coerce_text(node.get("@version"))
    return ""


def _flow_property_ref(prop: Mapping[str, Any]) -> dict[str, str]:
    ref = prop.get("referenceToFlowPropertyDataSet")
    return {
        "uuid": _ref_uuid_from_reference_node(ref),
        "version": _ref_version_from_reference_node(ref),
        "internal_id": _coerce_text(prop.get("@dataSetInternalID")),
        "short_name": _lang_text(
            (ref.get("common:shortDescription") if isinstance(ref, Mapping) else None),
            "en",
        ),
    }


def _extract_flowproperty_unitgroup_ref(flowprop_root: Mapping[str, Any]) -> str:
    candidates = [
        ("flowPropertyDataSet", "unitGroup", "referenceToUnitGroup"),
        (
            "flowPropertyDataSet",
            "flowPropertiesInformation",
            "quantitativeReference",
            "referenceToReferenceUnitGroup",
        ),
    ]
    # Allow caller to pass raw root or wrapped doc.
    root = flowprop_root
    if "flowPropertyDataSet" in flowprop_root and isinstance(flowprop_root["flowPropertyDataSet"], Mapping):
        root = flowprop_root
    else:
        root = {"flowPropertyDataSet": flowprop_root}
    for path in candidates:
        node = _deep_get(root, path)
        ref_uuid = _ref_uuid_from_reference_node(node)
        if ref_uuid:
            return ref_uuid
    return ""


def _extract_reference_unit_name(unitgroup_root: Mapping[str, Any]) -> str:
    root = unitgroup_root
    if "unitGroupDataSet" in unitgroup_root and isinstance(unitgroup_root["unitGroupDataSet"], Mapping):
        root = unitgroup_root
    else:
        root = {"unitGroupDataSet": unitgroup_root}
    units = _deep_get(root, ("unitGroupDataSet", "units", "unit"))
    unit_items = [u for u in _as_list(units) if isinstance(u, Mapping)]
    # Prefer marked reference unit.
    for unit in unit_items:
        ref_flag = unit.get("referenceUnit")
        ref_text = str(ref_flag).strip().lower() if ref_flag is not None else ""
        if ref_text == "true":
            name = _coerce_text(unit.get("name"))
            if name:
                return name
    for unit in unit_items:
        name = _coerce_text(unit.get("name"))
        if name:
            return name
    return ""


def _is_elementary_flow(flow_type: str) -> bool:
    return flow_type.strip().lower() == "elementary flow"


def _bump_ilcd_version(version: str) -> str:
    text = (version or "").strip()
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", text)
    if not match:
        return "01.01.001"
    a, b, c = match.groups()
    c_num = int(c) + 1
    return f"{int(a):0{len(a)}d}.{int(b):0{len(b)}d}.{c_num:0{len(c)}d}"


def _json_pointer_set(root: MutableMapping[str, Any], pointer: str, value: Any) -> None:
    if not pointer.startswith("/"):
        raise ValueError(f"Unsupported JSON pointer: {pointer}")
    parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer.lstrip("/").split("/")]
    current: Any = root
    for key in parts[:-1]:
        if isinstance(current, MutableMapping):
            if key not in current or not isinstance(current[key], MutableMapping):
                current[key] = {}
            current = current[key]
        else:
            raise ValueError(f"Cannot set pointer {pointer}; non-object encountered at {key}")
    last = parts[-1]
    if isinstance(current, MutableMapping):
        current[last] = value
        return
    raise ValueError(f"Cannot set pointer {pointer}; parent is not an object")


def _iter_flow_files(flows_dir: Path) -> list[Path]:
    if not flows_dir.exists():
        return []
    return sorted([p for p in flows_dir.glob("*.json") if p.is_file()])


def _parse_uuid_list_item(item: Any) -> dict[str, str] | None:
    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        return {"id": text}
    if isinstance(item, Mapping):
        flow_id = _coerce_text(item.get("id") or item.get("uuid") or item.get("flow_uuid") or item.get("flow_id"))
        if not flow_id:
            return None
        version = _coerce_text(item.get("version") or item.get("base_version"))
        row = {"id": flow_id}
        if version:
            row["version"] = version
        return row
    return None


def load_uuid_list(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    items: list[dict[str, str]] = []
    if suffix == ".json":
        data = _read_json(path)
        if isinstance(data, list):
            for item in data:
                parsed = _parse_uuid_list_item(item)
                if parsed:
                    items.append(parsed)
        elif isinstance(data, Mapping):
            for key in ("uuids", "ids", "data", "rows", "items"):
                if isinstance(data.get(key), list):
                    for item in data[key]:
                        parsed = _parse_uuid_list_item(item)
                        if parsed:
                            items.append(parsed)
                    break
    elif suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            parsed = _parse_uuid_list_item(json.loads(text))
            if parsed:
                items.append(parsed)
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parsed = _parse_uuid_list_item(text.split()[0])
            if parsed:
                items.append(parsed)
    # Deduplicate by (id, version)
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in items:
        key = (row["id"], row.get("version", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def _process_builder_root() -> Path:
    return _repo_root_from_script() / "process-automated-builder"


def _ensure_process_builder_on_syspath() -> None:
    pb_root = _process_builder_root()
    if not pb_root.exists():
        raise RuntimeError(f"process-automated-builder not found: {pb_root}")
    pb_text = str(pb_root)
    if pb_text not in sys.path:
        sys.path.insert(0, pb_text)


@dataclass
class McpCrudFacade:
    """Thin facade reusing process-automated-builder MCP + CRUD services."""

    _crud: Any
    _mcp: Any
    _settings: Any

    @classmethod
    def create(cls) -> "McpCrudFacade":
        _ensure_process_builder_on_syspath()
        from tiangong_lca_spec.core.config import get_settings
        from tiangong_lca_spec.core.mcp_client import MCPToolClient
        from tiangong_lca_spec.publishing.crud import DatabaseCrudClient

        settings = get_settings()
        mcp = MCPToolClient(settings)
        crud = DatabaseCrudClient(settings, mcp_client=mcp)
        return cls(_crud=crud, _mcp=mcp, _settings=settings)

    @property
    def server_name(self) -> str:
        return str(self._settings.flow_search_service_name)

    def close(self) -> None:
        try:
            self._crud.close()
        except Exception:
            # DatabaseCrudClient closes the shared MCP client; best effort only.
            pass

    def select_flow(self, flow_uuid: str, version: str | None = None) -> dict[str, Any] | None:
        dataset = self._crud.select_flow(flow_uuid, version=version)
        if isinstance(dataset, Mapping):
            return {"flowDataSet": copy.deepcopy(dict(dataset))}
        return None

    def select_flow_record(self, flow_uuid: str) -> dict[str, Any] | None:
        result = self._crud.select_flow_record(flow_uuid)
        return copy.deepcopy(result) if isinstance(result, Mapping) else None

    def insert_flow(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._crud.insert_flow(payload)

    def _raw_crud(self, payload: Mapping[str, Any]) -> Any:
        return self._mcp.invoke_json_tool(self.server_name, DATABASE_TOOL_NAME, payload)

    def select_table_dataset(
        self,
        table: str,
        record_id: str,
        *,
        version: str | None = None,
        root_key: str | None = None,
    ) -> dict[str, Any] | None:
        payload: dict[str, Any] = {"operation": "select", "table": table, "id": record_id}
        if version:
            payload["version"] = version
        raw = self._raw_crud(payload)
        if not isinstance(raw, Mapping):
            return None
        if root_key and isinstance(raw.get(root_key), Mapping):
            return {root_key: copy.deepcopy(dict(raw[root_key]))}
        data = raw.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, Mapping):
                for key in ("json_ordered", "json"):
                    blob = first.get(key)
                    if isinstance(blob, Mapping):
                        if root_key and isinstance(blob.get(root_key), Mapping):
                            return {root_key: copy.deepcopy(dict(blob[root_key]))}
                        if not root_key:
                            return copy.deepcopy(dict(blob))
        return None

    def select_flowproperty(self, fp_uuid: str, version: str | None = None) -> dict[str, Any] | None:
        return self.select_table_dataset("flowproperties", fp_uuid, version=version, root_key="flowPropertyDataSet")

    def select_unitgroup(self, ug_uuid: str, version: str | None = None) -> dict[str, Any] | None:
        return self.select_table_dataset("unitgroups", ug_uuid, version=version, root_key="unitGroupDataSet")


def _finding(
    *,
    flow_uuid: str,
    base_version: str,
    severity: str,
    rule_id: str,
    message: str,
    evidence: Mapping[str, Any] | None = None,
    fixability: str = "manual",
    suggested_action: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "flow_uuid": flow_uuid,
        "base_version": base_version,
        "severity": severity,
        "rule_id": rule_id,
        "message": message,
        "fixability": fixability,
    }
    if evidence:
        item["evidence"] = dict(evidence)
    if suggested_action:
        item["suggested_action"] = suggested_action
    return item


def _review_one_flow(
    doc: Mapping[str, Any],
    *,
    mcp: McpCrudFacade | None = None,
    fp_cache: dict[tuple[str, str], dict[str, Any] | None] | None = None,
    ug_cache: dict[str, dict[str, Any] | None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    flow_ds = _flow_root(doc)
    flow_uuid = _flow_uuid(flow_ds) or "(missing-uuid)"
    version = _flow_version(flow_ds)
    flow_type = _flow_type(flow_ds)
    names = _name_texts(flow_ds)
    leaf = _classification_leaf(flow_ds)

    summary: dict[str, Any] = {
        "flow_uuid": flow_uuid,
        "base_version": version,
        "type_of_dataset": flow_type,
        "classification_leaf": leaf,
        "name_primary_en": _name_primary(flow_ds, "en"),
        "name_primary_zh": _name_primary(flow_ds, "zh"),
        "name_fingerprint": _name_fingerprint(flow_ds),
        "flow_property_uuid": "",
        "flow_property_version": "",
        "flow_property_internal_id": "",
        "unitgroup_uuid": "",
        "unitgroup_reference_unit_name": "",
    }

    if not flow_uuid or flow_uuid == "(missing-uuid)":
        findings.append(
            _finding(
                flow_uuid=flow_uuid,
                base_version=version,
                severity="error",
                rule_id="missing_flow_uuid",
                message="Flow missing common:UUID.",
                fixability="manual",
            )
        )

    if not version:
        findings.append(
            _finding(
                flow_uuid=flow_uuid,
                base_version=version,
                severity="warning",
                rule_id="missing_dataset_version",
                message="Flow missing administrativeInformation.publicationAndOwnership.common:dataSetVersion.",
                fixability="auto",
                suggested_action="Set a valid ILCD version before publish.",
            )
        )

    if not flow_type:
        findings.append(
            _finding(
                flow_uuid=flow_uuid,
                base_version=version,
                severity="error",
                rule_id="missing_type_of_dataset",
                message="typeOfDataSet is missing under modellingAndValidation.LCIMethod.",
                fixability="manual",
            )
        )
    elif _is_elementary_flow(flow_type):
        findings.append(
            _finding(
                flow_uuid=flow_uuid,
                base_version=version,
                severity="error",
                rule_id="unexpected_elementary_flow",
                message="Flow is Elementary flow but this pipeline is intended for non-elementary flow remediation.",
                evidence={"typeOfDataSet": flow_type},
                fixability="manual",
            )
        )

    if not names:
        findings.append(
            _finding(
                flow_uuid=flow_uuid,
                base_version=version,
                severity="warning",
                rule_id="missing_name_text",
                message="No textual name fields found under flowInformation.dataSetInformation.name.",
                fixability="manual",
            )
        )
    else:
        if any("emergy" in text.lower() for text in names):
            findings.append(
                _finding(
                    flow_uuid=flow_uuid,
                    base_version=version,
                    severity="warning",
                    rule_id="name_contains_emergy",
                    message="Name subtree contains 'Emergy'.",
                    evidence={"matched_count": sum(1 for text in names if "emergy" in text.lower())},
                    fixability="manual",
                )
            )

    if not leaf["key"]:
        findings.append(
            _finding(
                flow_uuid=flow_uuid,
                base_version=version,
                severity="warning",
                rule_id="missing_classification_leaf",
                message="Classification leaf is missing.",
                fixability="manual",
            )
        )

    prop, chosen_internal_id = _pick_reference_flow_property(flow_ds)
    quant_ref_id = _quant_ref_internal_id(flow_ds)
    if prop is None:
        findings.append(
            _finding(
                flow_uuid=flow_uuid,
                base_version=version,
                severity="error",
                rule_id="missing_flow_property",
                message="No flowProperties.flowProperty entry found.",
                fixability="manual",
            )
        )
    else:
        pref = _flow_property_ref(prop)
        summary["flow_property_uuid"] = pref["uuid"]
        summary["flow_property_version"] = pref["version"]
        summary["flow_property_internal_id"] = pref["internal_id"]

        if not pref["uuid"]:
            findings.append(
                _finding(
                    flow_uuid=flow_uuid,
                    base_version=version,
                    severity="error",
                    rule_id="invalid_flow_property_reference",
                    message="referenceToFlowPropertyDataSet does not contain a parseable UUID.",
                    evidence={"internal_id": pref["internal_id"], "short_name": pref["short_name"]},
                    fixability="manual",
                )
            )

        if not quant_ref_id:
            findings.append(
                _finding(
                    flow_uuid=flow_uuid,
                    base_version=version,
                    severity="warning",
                    rule_id="missing_quantitative_reference",
                    message="referenceToReferenceFlowProperty is missing.",
                    evidence={"expected_internal_id": chosen_internal_id or pref["internal_id"]},
                    fixability="auto",
                    suggested_action="Set quantitative reference to the chosen flowProperty @dataSetInternalID.",
                )
            )
        elif chosen_internal_id and quant_ref_id != chosen_internal_id:
            findings.append(
                _finding(
                    flow_uuid=flow_uuid,
                    base_version=version,
                    severity="warning",
                    rule_id="quantitative_reference_mismatch",
                    message="Quantitative reference internal ID does not match the chosen reference flowProperty.",
                    evidence={"quant_ref_internal_id": quant_ref_id, "expected_internal_id": chosen_internal_id},
                    fixability="auto",
                    suggested_action="Align quantitative reference with chosen flowProperty internal ID.",
                )
            )

        if mcp and pref["uuid"]:
            fp_key = (pref["uuid"], pref["version"])
            fp_doc = None
            if fp_cache is not None and fp_key in fp_cache:
                fp_doc = fp_cache[fp_key]
            else:
                fp_doc = mcp.select_flowproperty(pref["uuid"], pref["version"] or None) or mcp.select_flowproperty(pref["uuid"])
                if fp_cache is not None:
                    fp_cache[fp_key] = fp_doc
            if not fp_doc:
                findings.append(
                    _finding(
                        flow_uuid=flow_uuid,
                        base_version=version,
                        severity="warning",
                        rule_id="flowproperty_lookup_failed",
                        message="Could not fetch referenced flow property dataset via MCP CRUD.",
                        evidence={"flow_property_uuid": pref["uuid"], "flow_property_version": pref["version"]},
                        fixability="manual",
                    )
                )
            else:
                ug_uuid = _extract_flowproperty_unitgroup_ref(fp_doc)
                summary["unitgroup_uuid"] = ug_uuid
                if not ug_uuid:
                    findings.append(
                        _finding(
                            flow_uuid=flow_uuid,
                            base_version=version,
                            severity="warning",
                            rule_id="missing_unitgroup_reference",
                            message="Referenced flow property does not expose a parseable unit group UUID.",
                            evidence={"flow_property_uuid": pref["uuid"]},
                            fixability="manual",
                        )
                    )
                else:
                    ug_doc = None
                    if ug_cache is not None and ug_uuid in ug_cache:
                        ug_doc = ug_cache[ug_uuid]
                    else:
                        ug_doc = mcp.select_unitgroup(ug_uuid)
                        if ug_cache is not None:
                            ug_cache[ug_uuid] = ug_doc
                    if not ug_doc:
                        findings.append(
                            _finding(
                                flow_uuid=flow_uuid,
                                base_version=version,
                                severity="warning",
                                rule_id="unitgroup_lookup_failed",
                                message="Could not fetch unit group dataset via MCP CRUD.",
                                evidence={"unitgroup_uuid": ug_uuid},
                                fixability="manual",
                            )
                        )
                    else:
                        ref_unit = _extract_reference_unit_name(ug_doc)
                        summary["unitgroup_reference_unit_name"] = ref_unit
                        if not ref_unit:
                            findings.append(
                                _finding(
                                    flow_uuid=flow_uuid,
                                    base_version=version,
                                    severity="warning",
                                    rule_id="missing_reference_unit_name",
                                    message="Unit group does not expose a readable reference unit name.",
                                    evidence={"unitgroup_uuid": ug_uuid},
                                    fixability="manual",
                                )
                            )
                        else:
                            # Very light heuristic: keep this as warning-only to avoid false confidence.
                            if flow_type and "product" in flow_type.lower():
                                token = ref_unit.strip().lower()
                                if token in {"kg", "g", "t", "m3", "l", "piece", "pcs", "pc", "ea", "mj", "kwh", "m2", "m"}:
                                    pass
                                else:
                                    findings.append(
                                        _finding(
                                            flow_uuid=flow_uuid,
                                            base_version=version,
                                            severity="warning",
                                            rule_id="unitgroup_needs_review",
                                            message="Reference unit is uncommon for Product flow; review flowProperty/unitgroup selection.",
                                            evidence={
                                                "typeOfDataSet": flow_type,
                                                "flow_property_uuid": pref["uuid"],
                                                "unitgroup_uuid": ug_uuid,
                                                "reference_unit_name": ref_unit,
                                            },
                                            fixability="manual",
                                        )
                                    )

    return findings, summary


def _similarity_findings(
    summaries: list[dict[str, Any]],
    *,
    threshold: float = 0.92,
    max_pairs_per_group: int = 20000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        leaf_key = _coerce_text((row.get("classification_leaf") or {}).get("key") if isinstance(row.get("classification_leaf"), Mapping) else "")
        if not leaf_key:
            continue
        fp_uuid = _coerce_text(row.get("flow_property_uuid"))
        ug_uuid = _coerce_text(row.get("unitgroup_uuid"))
        name_fp = _coerce_text(row.get("name_fingerprint"))
        if not name_fp:
            continue
        grouped[(leaf_key, fp_uuid, ug_uuid)].append(row)

    for group_key, rows in grouped.items():
        if len(rows) < 2:
            continue
        pair_count = 0
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if pair_count >= max_pairs_per_group:
                    break
                left = rows[i]
                right = rows[j]
                a = _coerce_text(left.get("name_fingerprint"))
                b = _coerce_text(right.get("name_fingerprint"))
                if not a or not b:
                    continue
                ratio = SequenceMatcher(None, a, b).ratio()
                pair_count += 1
                if ratio < threshold:
                    continue
                pair = {
                    "classification_group": group_key[0],
                    "flow_property_uuid": group_key[1],
                    "unitgroup_uuid": group_key[2],
                    "left_flow_uuid": left.get("flow_uuid"),
                    "right_flow_uuid": right.get("flow_uuid"),
                    "left_version": left.get("base_version"),
                    "right_version": right.get("base_version"),
                    "similarity": round(ratio, 6),
                    "left_name_en": left.get("name_primary_en"),
                    "right_name_en": right.get("name_primary_en"),
                }
                pairs.append(pair)
                for current, other in ((left, right), (right, left)):
                    findings.append(
                        _finding(
                            flow_uuid=_coerce_text(current.get("flow_uuid")),
                            base_version=_coerce_text(current.get("base_version")),
                            severity="warning",
                            rule_id="same_category_high_similarity",
                            message="Another flow in the same classification/flowProperty/unitgroup group is highly similar.",
                            evidence={
                                "other_flow_uuid": _coerce_text(other.get("flow_uuid")),
                                "other_version": _coerce_text(other.get("base_version")),
                                "similarity": round(ratio, 6),
                                "classification_group": group_key[0],
                            },
                            fixability="review-needed",
                            suggested_action="Review duplicate/near-duplicate risk before automated changes.",
                        )
                    )
            if pair_count >= max_pairs_per_group:
                break
    return findings, pairs


def _review_directory(
    flows_dir: Path,
    out_dir: Path,
    *,
    with_mcp_context: bool = False,
    similarity_threshold: float = 0.92,
    max_pairs_per_group: int = 20000,
) -> dict[str, Any]:
    flow_files = _iter_flow_files(flows_dir)
    if not flow_files:
        raise RuntimeError(f"No flow JSON files found in {flows_dir}")

    mcp: McpCrudFacade | None = None
    fp_cache: dict[tuple[str, str], dict[str, Any] | None] = {}
    ug_cache: dict[str, dict[str, Any] | None] = {}
    if with_mcp_context:
        mcp = McpCrudFacade.create()

    try:
        all_findings: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        per_flow: list[dict[str, Any]] = []
        for path in flow_files:
            doc = _read_json(path)
            findings, summary = _review_one_flow(doc, mcp=mcp, fp_cache=fp_cache, ug_cache=ug_cache)
            summary["source_file"] = path.name
            summaries.append(summary)
            per_flow.append(
                {
                    "flow_uuid": summary.get("flow_uuid"),
                    "base_version": summary.get("base_version"),
                    "source_file": path.name,
                    "finding_count": len(findings),
                }
            )
            all_findings.extend(findings)

        sim_findings, sim_pairs = _similarity_findings(
            summaries,
            threshold=similarity_threshold,
            max_pairs_per_group=max_pairs_per_group,
        )
        all_findings.extend(sim_findings)

        out_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out_dir / "findings.jsonl", all_findings)
        _write_jsonl(out_dir / "flow_summaries.jsonl", summaries)
        _write_jsonl(out_dir / "similarity_pairs.jsonl", sim_pairs)
        _write_json(out_dir / "review_index.json", per_flow)

        severity_counts: dict[str, int] = defaultdict(int)
        rule_counts: dict[str, int] = defaultdict(int)
        fixability_counts: dict[str, int] = defaultdict(int)
        for item in all_findings:
            severity_counts[_coerce_text(item.get("severity")) or "unknown"] += 1
            rule_counts[_coerce_text(item.get("rule_id")) or "unknown"] += 1
            fixability_counts[_coerce_text(item.get("fixability")) or "unknown"] += 1
        summary = {
            "flow_count": len(flow_files),
            "finding_count": len(all_findings),
            "similarity_pair_count": len(sim_pairs),
            "with_mcp_context": with_mcp_context,
            "severity_counts": dict(sorted(severity_counts.items())),
            "fixability_counts": dict(sorted(fixability_counts.items())),
            "rule_counts": dict(sorted(rule_counts.items())),
        }
        _write_json(out_dir / "review_summary.json", summary)
        return summary
    finally:
        if mcp is not None:
            mcp.close()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        value = json.loads(text)
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def _build_flow_file_index(flows_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in _iter_flow_files(flows_dir):
        try:
            doc = _read_json(path)
        except Exception:
            continue
        uuid_value = _flow_uuid(_flow_root(doc))
        if uuid_value:
            index[uuid_value] = path
    return index


def _apply_safe_fixes(
    flow_doc: Mapping[str, Any],
    flow_findings: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    wrapper = _flow_wrapper(flow_doc)
    flow_ds = wrapper["flowDataSet"]
    applied_ops: list[dict[str, Any]] = []
    changed = False

    chosen_prop, chosen_internal_id = _pick_reference_flow_property(flow_ds)
    expected_internal_id = chosen_internal_id or _coerce_text((chosen_prop or {}).get("@dataSetInternalID"))
    version = _flow_version(flow_ds) or "01.01.000"

    # Apply deterministic fixes only once even if multiple findings repeat.
    rule_ids = {str(item.get("rule_id")) for item in flow_findings}

    if "missing_dataset_version" in rule_ids:
        current = _flow_version(flow_ds)
        if not current:
            _set_flow_version(flow_ds, version)
            applied_ops.append(
                {
                    "op": "set",
                    "path": "/flowDataSet/administrativeInformation/publicationAndOwnership/common:dataSetVersion",
                    "value": version,
                    "rule_id": "missing_dataset_version",
                }
            )
            changed = True

    if expected_internal_id and (
        "missing_quantitative_reference" in rule_ids or "quantitative_reference_mismatch" in rule_ids
    ):
        current_quant = _quant_ref_internal_id(flow_ds)
        if current_quant != expected_internal_id:
            _set_quant_ref_internal_id(flow_ds, expected_internal_id)
            applied_ops.append(
                {
                    "op": "set",
                    "path": "/flowDataSet/flowInformation/quantitativeReference/referenceToReferenceFlowProperty",
                    "value": expected_internal_id,
                    "rule_id": "quantitative_reference_alignment",
                }
            )
            changed = True

    return wrapper, applied_ops, changed


def _summarize_findings_by_flow(findings: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in findings:
        flow_uuid = _coerce_text(item.get("flow_uuid"))
        if not flow_uuid:
            continue
        grouped[flow_uuid].append(item)
    return grouped


def _propose_fixes(
    flows_dir: Path,
    findings_path: Path,
    out_dir: Path,
    *,
    copy_unchanged: bool = False,
) -> dict[str, Any]:
    findings = _load_jsonl(findings_path)
    findings_by_flow = _summarize_findings_by_flow(findings)
    flow_files = _iter_flow_files(flows_dir)
    if not flow_files:
        raise RuntimeError(f"No flow JSON files found in {flows_dir}")

    patched_dir = out_dir / "patched_flows"
    patched_dir.mkdir(parents=True, exist_ok=True)

    proposals: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    changed_count = 0
    copied_count = 0

    for flow_file in flow_files:
        doc = _read_json(flow_file)
        flow_ds = _flow_root(doc)
        flow_uuid = _flow_uuid(flow_ds)
        if not flow_uuid:
            continue
        flow_findings = findings_by_flow.get(flow_uuid, [])
        patched_wrapper, applied_ops, changed = _apply_safe_fixes(doc, flow_findings)

        # Emit candidate entries for non-auto findings so humans know what is pending.
        for item in flow_findings:
            fixability = _coerce_text(item.get("fixability")) or "manual"
            if fixability == "auto":
                continue
            proposals.append(
                {
                    "flow_uuid": flow_uuid,
                    "base_version": _coerce_text(item.get("base_version")),
                    "mode": "candidate",
                    "rule_id": _coerce_text(item.get("rule_id")),
                    "severity": _coerce_text(item.get("severity")),
                    "message": _coerce_text(item.get("message")),
                    "next_step": "manual-review-or-regenerate",
                }
            )

        if applied_ops:
            proposals.append(
                {
                    "flow_uuid": flow_uuid,
                    "base_version": _flow_version(flow_ds),
                    "mode": "applied",
                    "rule_id": "safe_fix_batch",
                    "patch_ops": applied_ops,
                }
            )

        if not changed and not copy_unchanged:
            continue

        out_name = flow_file.name
        out_path = patched_dir / out_name
        if changed:
            _write_json(out_path, patched_wrapper)
            changed_count += 1
        else:
            _write_json(out_path, doc)
            copied_count += 1

        patched_root = _flow_root(patched_wrapper if changed else doc)
        base_version = _flow_version(flow_ds)
        manifest.append(
            {
                "flow_uuid": flow_uuid,
                "base_version": base_version,
                "patched_version_before_publish": _flow_version(patched_root),
                "source_file": str(flow_file),
                "patched_file": str(out_path),
                "changed": changed,
                "before_sha256": _sha256_json(_flow_wrapper(doc)),
                "after_sha256": _sha256_json(_flow_wrapper(patched_wrapper if changed else doc)),
            }
        )

    _write_jsonl(out_dir / "fix_proposals.jsonl", proposals)
    _write_jsonl(out_dir / "patch_manifest.jsonl", manifest)
    summary = {
        "input_flow_count": len(flow_files),
        "flows_with_findings": len(findings_by_flow),
        "patched_file_count": len(manifest),
        "changed_count": changed_count,
        "copied_unchanged_count": copied_count,
        "proposal_count": len(proposals),
    }
    _write_json(out_dir / "fix_summary.json", summary)
    return summary


def _extract_record_version_from_select_record(record: Mapping[str, Any]) -> str:
    # Prefer explicit DB columns when available; fallback to embedded JSON.
    for key in ("version", "data_set_version", "dataset_version"):
        value = _coerce_text(record.get(key))
        if value:
            return value
    for key in ("json_ordered", "json"):
        payload = record.get(key)
        if isinstance(payload, Mapping):
            if isinstance(payload.get("flowDataSet"), Mapping):
                return _flow_version(payload["flowDataSet"])
    return ""


def _publish_patched_flows(
    manifest_path: Path,
    out_dir: Path,
    *,
    mode: str = "dry-run",
    require_latest_match: bool = True,
    only_changed: bool = True,
) -> dict[str, Any]:
    manifest_rows = _load_jsonl(manifest_path)
    if not manifest_rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "row_count": 0,
            "mode": mode,
            "inserted_count": 0,
            "dry_run_count": 0,
            "conflict_count": 0,
            "error_count": 0,
            "skipped": True,
            "reason": f"empty_manifest:{manifest_path}",
        }
        _write_jsonl(out_dir / "publish_results.jsonl", [])
        _write_json(out_dir / "publish_summary.json", summary)
        return summary

    mcp = McpCrudFacade.create()
    results: list[dict[str, Any]] = []
    inserted = 0
    skipped = 0
    conflicts = 0
    errors = 0
    try:
        for row in manifest_rows:
            if only_changed and not bool(row.get("changed")):
                continue
            patched_file = Path(_coerce_text(row.get("patched_file")))
            if not patched_file.exists():
                results.append(
                    {
                        "flow_uuid": _coerce_text(row.get("flow_uuid")),
                        "status": "error",
                        "reason": f"patched_file_missing:{patched_file}",
                    }
                )
                errors += 1
                continue

            wrapper = _flow_wrapper(_read_json(patched_file))
            flow_ds = wrapper["flowDataSet"]
            flow_uuid = _flow_uuid(flow_ds)
            base_version = _coerce_text(row.get("base_version"))

            latest_doc = mcp.select_flow(flow_uuid)
            latest_version = _flow_version(_flow_root(latest_doc)) if latest_doc else ""

            if require_latest_match and base_version and latest_version and base_version != latest_version:
                results.append(
                    {
                        "flow_uuid": flow_uuid,
                        "base_version": base_version,
                        "latest_version": latest_version,
                        "status": "conflict",
                        "reason": "base_version_mismatch",
                    }
                )
                conflicts += 1
                continue

            new_version = _bump_ilcd_version(latest_version or base_version or _flow_version(flow_ds))
            _set_flow_version(flow_ds, new_version)
            after_hash = _sha256_json(wrapper)

            result_row = {
                "flow_uuid": flow_uuid,
                "base_version": base_version,
                "latest_version_checked": latest_version,
                "new_version": new_version,
                "mode": mode,
                "status": "dry-run",
                "after_sha256": after_hash,
            }

            if mode == "insert":
                try:
                    insert_result = mcp.insert_flow(wrapper)
                    result_row["status"] = "inserted"
                    result_row["insert_result"] = insert_result
                    inserted += 1
                except Exception as exc:  # noqa: BLE001
                    result_row["status"] = "error"
                    result_row["reason"] = str(exc)
                    errors += 1
            else:
                skipped += 1

            results.append(result_row)

        out_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out_dir / "publish_results.jsonl", results)
        summary = {
            "row_count": len(results),
            "mode": mode,
            "inserted_count": inserted,
            "dry_run_count": skipped,
            "conflict_count": conflicts,
            "error_count": errors,
        }
        _write_json(out_dir / "publish_summary.json", summary)
        return summary
    finally:
        mcp.close()


def _fetch_flows(uuid_list_path: Path, out_dir: Path, *, limit: int | None = None) -> dict[str, Any]:
    rows = load_uuid_list(uuid_list_path)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"No UUIDs parsed from {uuid_list_path}")

    cache_flows = out_dir / "cache" / "flows"
    cache_flows.mkdir(parents=True, exist_ok=True)
    fetch_log: list[dict[str, Any]] = []

    mcp = McpCrudFacade.create()
    ok = 0
    miss = 0
    err = 0
    try:
        for item in rows:
            flow_id = item["id"]
            version = item.get("version")
            try:
                doc = mcp.select_flow(flow_id, version=version or None)
                if not doc:
                    fetch_log.append({"id": flow_id, "requested_version": version or "", "status": "not_found"})
                    miss += 1
                    continue
                flow_ds = _flow_root(doc)
                actual_uuid = _flow_uuid(flow_ds) or flow_id
                actual_version = _flow_version(flow_ds) or (version or "")
                filename = f"{actual_uuid}_{actual_version or 'unknown'}.json"
                out_file = cache_flows / filename
                _write_json(out_file, doc)
                fetch_log.append(
                    {
                        "id": flow_id,
                        "requested_version": version or "",
                        "actual_uuid": actual_uuid,
                        "actual_version": actual_version,
                        "file": str(out_file),
                        "status": "ok",
                    }
                )
                ok += 1
            except Exception as exc:  # noqa: BLE001
                fetch_log.append(
                    {
                        "id": flow_id,
                        "requested_version": version or "",
                        "status": "error",
                        "reason": str(exc),
                    }
                )
                err += 1

        _write_jsonl(out_dir / "fetch" / "fetch_log.jsonl", fetch_log)
        summary = {
            "requested_count": len(rows),
            "ok_count": ok,
            "not_found_count": miss,
            "error_count": err,
            "cache_flows_dir": str(cache_flows),
        }
        _write_json(out_dir / "fetch" / "fetch_summary.json", summary)
        return summary
    finally:
        mcp.close()


def _request_from_existing_flow(flow_doc: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    flow_ds = _flow_root(flow_doc)
    class_entries = _classification_classes(flow_ds)
    leaf = _classification_leaf(flow_ds)
    name = _name_node(flow_ds) if isinstance(_name_node(flow_ds), Mapping) else {}
    if not isinstance(name, Mapping):
        name = {}
    comments = _deep_get(flow_ds, ("flowInformation", "dataSetInformation", "common:generalComment"))
    synonyms = _deep_get(flow_ds, ("flowInformation", "dataSetInformation", "common:synonyms"))
    prop, _ = _pick_reference_flow_property(flow_ds)
    pref = _flow_property_ref(prop or {})

    def _split_synonyms(value: Any, lang: str) -> list[str]:
        text = _lang_text(value, lang)
        if not text:
            return []
        raw = text.replace("；", ";")
        pieces = [p.strip() for chunk in raw.split(";") for p in chunk.split(",")]
        return [p for p in pieces if p]

    request = {
        "class_id": leaf["class_id"] or "unknown",
        "classification": class_entries,
        "base_name_en": _lang_text(name.get("baseName"), "en"),
        "base_name_zh": _lang_text(name.get("baseName"), "zh"),
        "treatment_en": _lang_text(name.get("treatmentStandardsRoutes"), "en"),
        "treatment_zh": _lang_text(name.get("treatmentStandardsRoutes"), "zh"),
        "mix_en": _lang_text(name.get("mixAndLocationTypes"), "en"),
        "mix_zh": _lang_text(name.get("mixAndLocationTypes"), "zh"),
        "comment_en": _lang_text(comments, "en"),
        "comment_zh": _lang_text(comments, "zh"),
        "synonyms_en": _split_synonyms(synonyms, "en"),
        "synonyms_zh": _split_synonyms(synonyms, "zh"),
        "flow_type": _flow_type(flow_ds) or "Product flow",
        "flow_uuid": _flow_uuid(flow_ds),
        "version": _flow_version(flow_ds) or "01.01.000",
        "flow_property_uuid": pref["uuid"],
        "flow_property_version": pref["version"] or "03.00.003",
        "flow_property_name_en": pref["short_name"] or "Mass",
    }
    if overrides:
        for key, value in overrides.items():
            if key in request:
                request[key] = value
    return request


def _regenerate_product_flow(
    flow_file: Path,
    out_file: Path,
    *,
    overrides_file: Path | None = None,
    allow_validation_fallback: bool = False,
) -> dict[str, Any]:
    _ensure_process_builder_on_syspath()
    from tiangong_lca_spec.product_flow_creation import ProductFlowCreateRequest, ProductFlowCreationService

    flow_doc = _read_json(flow_file)
    overrides = _read_json(overrides_file) if overrides_file else None
    if overrides is not None and not isinstance(overrides, Mapping):
        raise RuntimeError(f"Overrides must be a JSON object: {overrides_file}")
    request_payload = _request_from_existing_flow(flow_doc, overrides=overrides if isinstance(overrides, Mapping) else None)
    request = ProductFlowCreateRequest(**request_payload)
    service = ProductFlowCreationService()
    result = service.build(request, allow_validation_fallback=allow_validation_fallback)
    _write_json(out_file, result.payload)
    summary = {
        "flow_uuid": result.flow_uuid,
        "version": result.version,
        "output_file": str(out_file),
        "reused_service": "process-automated-builder/tiangong_lca_spec/product_flow_creation/service.py",
    }
    return summary


def _run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    fetch_summary = _fetch_flows(Path(args.uuid_list).resolve(), run_dir, limit=args.limit)
    review_summary = _review_directory(
        run_dir / "cache" / "flows",
        run_dir / "review",
        with_mcp_context=bool(args.with_mcp_review_context),
        similarity_threshold=args.similarity_threshold,
        max_pairs_per_group=args.max_pairs_per_group,
    )
    fix_summary = _propose_fixes(
        run_dir / "cache" / "flows",
        run_dir / "review" / "findings.jsonl",
        run_dir / "fix",
        copy_unchanged=bool(args.copy_unchanged),
    )
    patched_files = _iter_flow_files(run_dir / "fix" / "patched_flows")
    if patched_files:
        validate_summary = _review_directory(
            run_dir / "fix" / "patched_flows",
            run_dir / "validate",
            with_mcp_context=bool(args.with_mcp_review_context),
            similarity_threshold=args.similarity_threshold,
            max_pairs_per_group=args.max_pairs_per_group,
        )
    else:
        validate_summary = {
            "flow_count": 0,
            "finding_count": 0,
            "similarity_pair_count": 0,
            "with_mcp_context": bool(args.with_mcp_review_context),
            "skipped": True,
            "reason": "no_patched_flows",
        }
        _write_json(run_dir / "validate" / "review_summary.json", validate_summary)

    publish_summary: dict[str, Any] | None = None
    if args.publish_mode != "none":
        publish_summary = _publish_patched_flows(
            run_dir / "fix" / "patch_manifest.jsonl",
            run_dir / "publish",
            mode=args.publish_mode,
            require_latest_match=not args.skip_base_check,
            only_changed=not args.publish_include_unchanged,
        )

    pipeline_summary = {
        "run_dir": str(run_dir),
        "fetch": fetch_summary,
        "review": review_summary,
        "fix": fix_summary,
        "validate": validate_summary,
        "publish": publish_summary,
    }
    _write_json(run_dir / "pipeline_summary.json", pipeline_summary)
    return pipeline_summary


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flow remediation + append-only publish pipeline (initial skill version)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch flow JSONs via MCP CRUD from a UUID list.")
    p_fetch.add_argument("--uuid-list", required=True, help="Path to UUID list JSON/JSONL/TXT.")
    p_fetch.add_argument("--run-dir", required=True, help="Run directory for cache and outputs.")
    p_fetch.add_argument("--limit", type=int, help="Optional limit for dry runs.")

    p_review = sub.add_parser("review", help="Run bootstrap flow review on local flow JSON cache.")
    p_review.add_argument("--flows-dir", help="Directory of flow JSON files. Defaults to <run-dir>/cache/flows.")
    p_review.add_argument("--run-dir", help="Run directory. Used when --flows-dir omitted.")
    p_review.add_argument("--out-dir", help="Output dir. Defaults to <run-dir>/review.")
    p_review.add_argument("--with-mcp-context", action="store_true", help="Fetch flowproperty/unitgroup via MCP for richer checks.")
    p_review.add_argument("--similarity-threshold", type=float, default=0.92)
    p_review.add_argument("--max-pairs-per-group", type=int, default=20000)

    p_fix = sub.add_parser("propose-fix", help="Apply deterministic safe fixes and emit fix proposals.")
    p_fix.add_argument("--run-dir", help="Run directory. Defaults inputs/outputs under it.")
    p_fix.add_argument("--flows-dir", help="Directory of flow JSON files. Defaults to <run-dir>/cache/flows.")
    p_fix.add_argument("--findings", help="Path to findings.jsonl. Defaults to <run-dir>/review/findings.jsonl.")
    p_fix.add_argument("--out-dir", help="Output dir. Defaults to <run-dir>/fix.")
    p_fix.add_argument("--copy-unchanged", action="store_true", help="Copy unchanged flows into patched_flows for later publish inspection.")

    p_validate = sub.add_parser("validate", help="Re-run bootstrap review on patched flows.")
    p_validate.add_argument("--run-dir", help="Run directory. Defaults patched flows under <run-dir>/fix/patched_flows.")
    p_validate.add_argument("--flows-dir", help="Directory of patched flow JSON files.")
    p_validate.add_argument("--out-dir", help="Output dir. Defaults to <run-dir>/validate.")
    p_validate.add_argument("--with-mcp-context", action="store_true")
    p_validate.add_argument("--similarity-threshold", type=float, default=0.92)
    p_validate.add_argument("--max-pairs-per-group", type=int, default=20000)

    p_publish = sub.add_parser("publish", help="Append-only publish patched flows by MCP CRUD insert.")
    p_publish.add_argument("--run-dir", help="Run directory. Uses <run-dir>/fix/patch_manifest.jsonl and outputs to <run-dir>/publish.")
    p_publish.add_argument("--manifest", help="Patch manifest JSONL. Defaults to <run-dir>/fix/patch_manifest.jsonl.")
    p_publish.add_argument("--out-dir", help="Output dir. Defaults to <run-dir>/publish.")
    p_publish.add_argument("--mode", choices=["dry-run", "insert"], default="dry-run")
    p_publish.add_argument("--skip-base-check", action="store_true", help="Do not require latest DB version to match base_version.")
    p_publish.add_argument("--include-unchanged", action="store_true", help="Also publish unchanged copied flows (normally skipped).")

    p_pipeline = sub.add_parser("pipeline", help="Run fetch -> review -> propose-fix -> validate -> optional publish.")
    p_pipeline.add_argument("--uuid-list", required=True)
    p_pipeline.add_argument("--run-dir", required=True)
    p_pipeline.add_argument("--limit", type=int)
    p_pipeline.add_argument("--with-mcp-review-context", action="store_true")
    p_pipeline.add_argument("--similarity-threshold", type=float, default=0.92)
    p_pipeline.add_argument("--max-pairs-per-group", type=int, default=20000)
    p_pipeline.add_argument("--copy-unchanged", action="store_true")
    p_pipeline.add_argument("--publish-mode", choices=["none", "dry-run", "insert"], default="none")
    p_pipeline.add_argument("--skip-base-check", action="store_true")
    p_pipeline.add_argument("--publish-include-unchanged", action="store_true")

    p_regen = sub.add_parser(
        "regen-product-flow",
        help="Rebuild a product flow using process-automated-builder ProductFlowCreationService (for classification/name changes).",
    )
    p_regen.add_argument("--flow-file", required=True)
    p_regen.add_argument("--out-file", required=True)
    p_regen.add_argument("--overrides-file", help="JSON object overriding ProductFlowCreateRequest fields.")
    p_regen.add_argument("--allow-validation-fallback", action="store_true")

    return parser


def _require_run_dir(value: str | None, parser: argparse.ArgumentParser, sub_name: str) -> Path:
    if not value:
        parser.error(f"{sub_name} requires --run-dir when explicit paths are not provided")
    return Path(value).resolve()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "fetch":
            summary = _fetch_flows(Path(args.uuid_list).resolve(), Path(args.run_dir).resolve(), limit=args.limit)
            _print_json(summary)
            return 0

        if args.command == "review":
            if args.flows_dir:
                flows_dir = Path(args.flows_dir).resolve()
                if args.out_dir:
                    out_dir = Path(args.out_dir).resolve()
                else:
                    run_dir = _require_run_dir(args.run_dir, parser, "review")
                    out_dir = run_dir / "review"
            else:
                run_dir = _require_run_dir(args.run_dir, parser, "review")
                flows_dir = run_dir / "cache" / "flows"
                out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "review")
            summary = _review_directory(
                flows_dir,
                out_dir,
                with_mcp_context=bool(args.with_mcp_context),
                similarity_threshold=args.similarity_threshold,
                max_pairs_per_group=args.max_pairs_per_group,
            )
            _print_json(summary)
            return 0

        if args.command == "propose-fix":
            run_dir = Path(args.run_dir).resolve() if args.run_dir else None
            if args.flows_dir:
                flows_dir = Path(args.flows_dir).resolve()
            else:
                if run_dir is None:
                    parser.error("propose-fix requires --run-dir or --flows-dir")
                flows_dir = run_dir / "cache" / "flows"
            if args.findings:
                findings_path = Path(args.findings).resolve()
            else:
                if run_dir is None:
                    parser.error("propose-fix requires --run-dir or --findings")
                findings_path = run_dir / "review" / "findings.jsonl"
            if args.out_dir:
                out_dir = Path(args.out_dir).resolve()
            else:
                if run_dir is None:
                    parser.error("propose-fix requires --run-dir or --out-dir")
                out_dir = run_dir / "fix"
            summary = _propose_fixes(flows_dir, findings_path, out_dir, copy_unchanged=bool(args.copy_unchanged))
            _print_json(summary)
            return 0

        if args.command == "validate":
            if args.flows_dir:
                flows_dir = Path(args.flows_dir).resolve()
                if args.out_dir:
                    out_dir = Path(args.out_dir).resolve()
                else:
                    run_dir = _require_run_dir(args.run_dir, parser, "validate")
                    out_dir = run_dir / "validate"
            else:
                run_dir = _require_run_dir(args.run_dir, parser, "validate")
                flows_dir = run_dir / "fix" / "patched_flows"
                out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "validate")
            if not _iter_flow_files(flows_dir):
                summary = {
                    "flow_count": 0,
                    "finding_count": 0,
                    "similarity_pair_count": 0,
                    "with_mcp_context": bool(args.with_mcp_context),
                    "skipped": True,
                    "reason": f"no_flow_files:{flows_dir}",
                }
                _write_json(out_dir / "review_summary.json", summary)
            else:
                summary = _review_directory(
                    flows_dir,
                    out_dir,
                    with_mcp_context=bool(args.with_mcp_context),
                    similarity_threshold=args.similarity_threshold,
                    max_pairs_per_group=args.max_pairs_per_group,
                )
            _print_json(summary)
            return 0

        if args.command == "publish":
            run_dir = Path(args.run_dir).resolve() if args.run_dir else None
            manifest_path = (
                Path(args.manifest).resolve()
                if args.manifest
                else (run_dir / "fix" / "patch_manifest.jsonl" if run_dir else None)
            )
            out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "publish" if run_dir else None)
            if manifest_path is None or out_dir is None:
                parser.error("publish requires --run-dir or both --manifest and --out-dir")
            summary = _publish_patched_flows(
                manifest_path,
                out_dir,
                mode=args.mode,
                require_latest_match=not args.skip_base_check,
                only_changed=not args.include_unchanged,
            )
            _print_json(summary)
            return 0

        if args.command == "pipeline":
            summary = _run_pipeline(args)
            _print_json(summary)
            return 0

        if args.command == "regen-product-flow":
            summary = _regenerate_product_flow(
                Path(args.flow_file).resolve(),
                Path(args.out_file).resolve(),
                overrides_file=Path(args.overrides_file).resolve() if args.overrides_file else None,
                allow_validation_fallback=bool(args.allow_validation_fallback),
            )
            _print_json(summary)
            return 0

    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
