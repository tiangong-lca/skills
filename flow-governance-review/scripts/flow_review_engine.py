#!/usr/bin/env python3
"""Flow governance review engine (LLM-driven semantic review + structured evidence extraction).

This script keeps the original flow review logic local to flow-governance-review:
- deterministic local extraction builds evidence summaries
- optional LLM layer performs semantic review on the summaries
- outputs both markdown reports and machine-readable findings
"""

import argparse
import copy
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from flow_governance_common import extract_flow_record, flow_dataset_from_row, load_rows_from_file


UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


# ---------- generic helpers ----------
def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        text = value.get("#text")
        if isinstance(text, str):
            return text.strip()
    return str(value).strip()


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _deep_get(obj: Any, path: List[str], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def _walk_strings(node: Any) -> Iterable[str]:
    if isinstance(node, str):
        text = node.strip()
        if text:
            yield text
        return
    if isinstance(node, dict):
        txt = node.get("#text")
        if isinstance(txt, str) and txt.strip():
            yield txt.strip()
        for value in node.values():
            for child in _walk_strings(value):
                yield child
        return
    if isinstance(node, list):
        for item in node:
            for child in _walk_strings(item):
                yield child


def _lang_text(items: Any, lang: str) -> str:
    for item in _as_list(items):
        if not isinstance(item, dict):
            continue
        if _coerce_text(item.get("@xml:lang")).lower() == lang.lower():
            text = _coerce_text(item.get("#text"))
            if text:
                return text
    for item in _as_list(items):
        text = _coerce_text(item)
        if text:
            return text
    return ""


def _find_uuid_in_node(node: Any) -> str:
    if isinstance(node, dict):
        for key in ("@refObjectId", "@uri"):
            raw = _coerce_text(node.get(key))
            if raw:
                m = UUID_RE.search(raw)
                if m:
                    return m.group(0).lower()
    raw = _coerce_text(node)
    m = UUID_RE.search(raw)
    return m.group(0).lower() if m else ""


def _flow_root(doc: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(doc.get("flowDataSet"), dict):
        return copy.deepcopy(doc["flowDataSet"])
    return copy.deepcopy(doc)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _iter_flow_files(flows_dir: Path) -> List[Path]:
    if not flows_dir.exists():
        return []
    return sorted([p for p in flows_dir.glob("*.json") if p.is_file()])


def _materialize_rows_file(rows_file: Path, out_dir: Path) -> Path:
    rows = load_rows_from_file(rows_file)
    target_dir = out_dir / "review-input" / "flows"
    target_dir.mkdir(parents=True, exist_ok=True)

    by_key: Dict[str, Dict[str, Any]] = {}
    duplicate_count = 0
    for index, row in enumerate(rows):
        record = extract_flow_record(row)
        flow_id = record.id or f"row-{index}"
        version = record.version or "01.00.000"
        key = f"{flow_id}@{version}"
        if key in by_key:
            duplicate_count += 1
        by_key[key] = row

    manifest_rows: List[Dict[str, Any]] = []
    for key, row in sorted(by_key.items()):
        record = extract_flow_record(row)
        file_name = f"{record.id or key.split('@', 1)[0]}__{record.version or '01.00.000'}.json"
        payload = {"flowDataSet": flow_dataset_from_row(row)}
        path = target_dir / file_name
        _write_json(path, payload)
        manifest_rows.append(
            {
                "flow_key": key,
                "primary_name": record.name,
                "file": str(path),
            }
        )

    _write_json(
        out_dir / "review-input" / "materialization-summary.json",
        {
            "source_rows_file": str(rows_file),
            "input_row_count": len(rows),
            "materialized_flow_count": len(by_key),
            "duplicate_input_rows_collapsed": duplicate_count,
            "flows_dir": str(target_dir),
            "items": manifest_rows,
        },
    )
    return target_dir


def _normalize_name_token(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- optional local registry enrichment via process-automated-builder ----------
class _FlowPropertyRegistryFacade:
    def __init__(self) -> None:
        self._registry = None
        self._source = "process-automated-builder:tiangong_lca_spec.tidas.flow_property_registry"

    def open(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        pb_root = repo_root / "process-automated-builder"
        if not pb_root.exists():
            raise RuntimeError(f"process-automated-builder not found: {pb_root}")
        sys_path = str(pb_root)
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        # Skill code assumes the OpenClaw/runtime environment provides the
        # dependencies required by process-automated-builder.
        from tiangong_lca_spec.tidas.flow_property_registry import get_default_registry  # type: ignore

        self._registry = get_default_registry()

    def close(self) -> None:
        self._registry = None

    def _ensure_open(self) -> None:
        if self._registry is None:
            self.open()

    def lookup_flow_property_context(self, fp_uuid: str) -> Optional[Dict[str, Any]]:
        self._ensure_open()
        try:
            descriptor = self._registry.get(fp_uuid)  # type: ignore[union-attr]
        except Exception:
            return None
        unit_group = getattr(descriptor, "unit_group", None)
        ref_unit = getattr(unit_group, "reference_unit", None)
        return {
            "flow_property_uuid": _coerce_text(getattr(descriptor, "uuid", "")),
            "flow_property_name": _coerce_text(getattr(descriptor, "name", "")),
            "flow_property_version": _coerce_text(self._registry.get_version(descriptor.uuid)),  # type: ignore[union-attr]
            "flow_property_reference_unit_description": _coerce_text(
                getattr(descriptor, "reference_unit_description", "")
            ),
            "unitgroup_uuid": _coerce_text(getattr(unit_group, "uuid", "")) or _coerce_text(
                getattr(descriptor, "reference_unit_group_uuid", "")
            ),
            "unitgroup_name": _coerce_text(getattr(unit_group, "name", "")),
            "unitgroup_reference_unit_name": _coerce_text(getattr(ref_unit, "name", "")),
            "lookup_source": self._source,
        }


# ---------- flow extraction ----------
def _flow_uuid(flow: Dict[str, Any]) -> str:
    return _coerce_text(_deep_get(flow, ["flowInformation", "dataSetInformation", "common:UUID"]))


def _flow_version(flow: Dict[str, Any]) -> str:
    return _coerce_text(
        _deep_get(flow, ["administrativeInformation", "publicationAndOwnership", "common:dataSetVersion"])
    )


def _flow_type(flow: Dict[str, Any]) -> str:
    return _coerce_text(_deep_get(flow, ["modellingAndValidation", "LCIMethod", "typeOfDataSet"]))


def _name_node(flow: Dict[str, Any]) -> Any:
    return _deep_get(flow, ["flowInformation", "dataSetInformation", "name"])


def _name_texts(flow: Dict[str, Any]) -> List[str]:
    return list(_walk_strings(_name_node(flow)))


def _name_primary(flow: Dict[str, Any], lang: str) -> str:
    name = _name_node(flow)
    if not isinstance(name, dict):
        return ""
    return _lang_text(name.get("baseName"), lang)


def _name_fingerprint(flow: Dict[str, Any]) -> str:
    name = _name_node(flow)
    if not isinstance(name, dict):
        return ""
    parts = []
    for key in ("baseName", "treatmentStandardsRoutes", "mixAndLocationTypes"):
        values = []
        for item in _as_list(name.get(key)):
            if isinstance(item, dict):
                text = _coerce_text(item.get("#text"))
            else:
                text = _coerce_text(item)
            if text:
                values.append(text)
        if values:
            parts.append(" | ".join(values))
    if not parts:
        parts = _name_texts(flow)
    return _normalize_name_token(" || ".join(parts))


def _classification_entries(flow: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = _deep_get(
        flow,
        [
            "flowInformation",
            "dataSetInformation",
            "classificationInformation",
            "common:classification",
            "common:class",
        ],
    )
    out: List[Dict[str, str]] = []
    for item in _as_list(raw):
        if isinstance(item, dict):
            out.append(
                {
                    "level": _coerce_text(item.get("@level")),
                    "class_id": _coerce_text(item.get("@classId")),
                    "text": _coerce_text(item.get("#text")),
                }
            )
    return out


def _classification_leaf(flow: Dict[str, Any]) -> Dict[str, str]:
    items = _classification_entries(flow)
    if not items:
        return {"class_id": "", "text": "", "key": ""}
    leaf = items[-1]
    key = (leaf.get("class_id", "") + "|" + leaf.get("text", "")).strip("|")
    return {"class_id": leaf.get("class_id", ""), "text": leaf.get("text", ""), "key": key}


def _flow_properties(flow: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = _deep_get(flow, ["flowProperties", "flowProperty"])
    out = []
    for item in _as_list(raw):
        if isinstance(item, dict):
            out.append(item)
    return out


def _pick_reference_flow_property(flow: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    props = _flow_properties(flow)
    for prop in props:
        internal_id = _coerce_text(prop.get("@dataSetInternalID"))
        if internal_id == "0":
            return prop, internal_id
    if props:
        return props[0], _coerce_text(props[0].get("@dataSetInternalID"))
    return None, ""


def _quant_ref_internal_id(flow: Dict[str, Any]) -> str:
    return _coerce_text(_deep_get(flow, ["flowInformation", "quantitativeReference", "referenceToReferenceFlowProperty"]))


def _flow_property_ref(prop: Dict[str, Any]) -> Dict[str, str]:
    ref = prop.get("referenceToFlowPropertyDataSet") if isinstance(prop, dict) else None
    return {
        "uuid": _find_uuid_in_node(ref),
        "version": _coerce_text(ref.get("@version") if isinstance(ref, dict) else ""),
        "internal_id": _coerce_text(prop.get("@dataSetInternalID")) if isinstance(prop, dict) else "",
        "short_name_en": _lang_text((ref.get("common:shortDescription") if isinstance(ref, dict) else None), "en"),
    }


def _elementary_class_entries(flow: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = _deep_get(
        flow,
        [
            "flowInformation",
            "dataSetInformation",
            "classificationInformation",
            "common:elementaryFlowCategorization",
            "common:category",
        ],
    )
    out: List[Dict[str, str]] = []
    for item in _as_list(raw):
        if isinstance(item, dict):
            out.append(
                {
                    "level": _coerce_text(item.get("@level")),
                    "cat_id": _coerce_text(item.get("@catId")),
                    "text": _coerce_text(item.get("#text")),
                }
            )
    return out


def _apply_methodology_checks(
    flow: Dict[str, Any],
    *,
    flow_uuid: str,
    base_version: str,
    rule_source: str,
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    allowed_types = {"elementary flow", "product flow", "waste flow"}
    flow_type = _flow_type(flow).strip().lower()
    if flow_type and flow_type not in allowed_types:
        findings.append(_rule_finding(
            flow_uuid, base_version, "error", "methodology_invalid_type_of_dataset",
            "typeOfDataSet not in allowed set: Elementary flow | Product flow | Waste flow.",
            evidence={"typeOfDataSet": _flow_type(flow)},
            rule_source=rule_source,
        ))

    base_name_items = _as_list(_deep_get(flow, ["flowInformation", "dataSetInformation", "name", "baseName"]))
    base_name_en = _lang_text(base_name_items, "en")
    if not base_name_en:
        findings.append(_rule_finding(
            flow_uuid, base_version, "warning", "methodology_missing_base_name_en",
            "English baseName is missing (methodology marks English as mandatory).",
            fixability="auto",
            rule_source=rule_source,
        ))
    if any(";" in _coerce_text(x.get("#text") if isinstance(x, dict) else x) for x in base_name_items):
        findings.append(_rule_finding(
            flow_uuid, base_version, "warning", "methodology_basename_semicolon",
            "baseName contains semicolon; methodology requires comma-separated descriptors.",
            fixability="auto",
            rule_source=rule_source,
        ))

    # quantitative reference should point to an existing flowProperty internal ID
    quant_id = _quant_ref_internal_id(flow)
    prop_ids = {_coerce_text(p.get("@dataSetInternalID")) for p in _flow_properties(flow) if isinstance(p, dict)}
    if quant_id and quant_id not in prop_ids:
        findings.append(_rule_finding(
            flow_uuid, base_version, "error", "methodology_quant_ref_missing_target",
            "referenceToReferenceFlowProperty points to a non-existing flowProperty internal ID.",
            evidence={"referenceToReferenceFlowProperty": quant_id, "available_internal_ids": sorted(x for x in prop_ids if x)},
            rule_source=rule_source,
        ))

    # classification levels should be sequential without gaps.
    for entries, label, id_key, max_level in (
        (_classification_entries(flow), "product_classification", "class_id", 4),
        (_elementary_class_entries(flow), "elementary_classification", "cat_id", 2),
    ):
        levels: List[int] = []
        for it in entries:
            try:
                levels.append(int(_coerce_text(it.get("level"))))
            except Exception:
                continue
            if not _coerce_text(it.get(id_key)):
                findings.append(_rule_finding(
                    flow_uuid, base_version, "warning", f"methodology_missing_{id_key}",
                    f"{label} entry has level but missing {id_key}.",
                    evidence={"entry": it},
                    rule_source=rule_source,
                ))
        if levels:
            uniq = sorted(set(levels))
            expected = list(range(uniq[0], min(max_level, uniq[-1]) + 1))
            if uniq[0] != 0 or uniq != expected:
                findings.append(_rule_finding(
                    flow_uuid, base_version, "warning", f"methodology_{label}_level_gap",
                    f"{label} levels should be continuous and start from 0.",
                    evidence={"levels": uniq},
                    rule_source=rule_source,
                ))

    return findings


def _rule_finding(
    flow_uuid: str,
    base_version: str,
    severity: str,
    rule_id: str,
    message: str,
    *,
    fixability: str = "manual",
    evidence: Optional[Dict[str, Any]] = None,
    action: Optional[str] = None,
    rule_source: str = "built_in",
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "flow_uuid": flow_uuid,
        "base_version": base_version,
        "severity": severity,
        "rule_id": rule_id,
        "message": message,
        "fixability": fixability,
        "source": "rule",
        "rule_source": rule_source,
    }
    if evidence:
        row["evidence"] = evidence
    if action:
        row["action"] = action
    return row


def _flow_summary_and_rule_findings(
    doc: Dict[str, Any],
    *,
    registry_ctx: Optional[_FlowPropertyRegistryFacade],
    fp_registry_cache: Dict[str, Optional[Dict[str, Any]]],
    methodology_rule_source: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    flow = _flow_root(doc)
    flow_uuid = _flow_uuid(flow) or "(missing-uuid)"
    base_version = _flow_version(flow)
    flow_type = _flow_type(flow)
    names = _name_texts(flow)
    class_entries = _classification_entries(flow)
    class_leaf = _classification_leaf(flow)

    summary: Dict[str, Any] = {
        "flow_uuid": flow_uuid,
        "base_version": base_version,
        "type_of_dataset": flow_type,
        "names": {
            "primary_en": _name_primary(flow, "en"),
            "primary_zh": _name_primary(flow, "zh"),
            "all_texts": names[:20],
        },
        "classification": {
            "leaf": class_leaf,
            "path": class_entries[:20],
        },
        "flow_property": {},
        "quantitative_reference": {
            "reference_flow_property_internal_id": _quant_ref_internal_id(flow),
        },
        "unitgroup": {
            "uuid": "",
            "name": "",
            "reference_unit_name": "",
            "lookup_status": "not_requested" if registry_ctx is None else "pending",
            "lookup_source": "" if registry_ctx is None else "process_automated_builder_registry",
        },
        "rule_signals": [],
        "similarity_candidates": [],
        "_name_fingerprint": _name_fingerprint(flow),
    }

    findings: List[Dict[str, Any]] = []

    if not flow_type:
        findings.append(
            _rule_finding(
                flow_uuid, base_version, "error", "missing_type_of_dataset",
                "typeOfDataSet is missing under modellingAndValidation.LCIMethod."
            )
        )
    elif flow_type.strip().lower() == "elementary flow":
        findings.append(
            _rule_finding(
                flow_uuid, base_version, "warning", "elementary_flow_in_flow_review",
                "Flow type is Elementary flow; check whether this batch should exclude it.",
                fixability="review-needed",
                evidence={"typeOfDataSet": flow_type},
            )
        )

    if not names:
        findings.append(
            _rule_finding(
                flow_uuid, base_version, "warning", "missing_name_text",
                "No textual entries found under flowInformation.dataSetInformation.name.",
            )
        )
    elif any("emergy" in x.lower() for x in names):
        findings.append(
            _rule_finding(
                flow_uuid, base_version, "warning", "name_contains_emergy",
                "Name subtree contains 'Emergy'.",
                fixability="review-needed",
                evidence={"matched_count": sum(1 for x in names if "emergy" in x.lower())},
            )
        )

    if not class_leaf.get("key"):
        findings.append(
            _rule_finding(
                flow_uuid, base_version, "warning", "missing_classification_leaf",
                "Classification leaf is missing.",
            )
        )

    prop, chosen_internal_id = _pick_reference_flow_property(flow)
    if prop is None:
        findings.append(
            _rule_finding(
                flow_uuid, base_version, "error", "missing_flow_property",
                "No flowProperties.flowProperty entry found."
            )
        )
    else:
        pref = _flow_property_ref(prop)
        summary["flow_property"] = {
            "selected_internal_id": chosen_internal_id or pref["internal_id"],
            "referenced_uuid": pref["uuid"],
            "referenced_version": pref["version"],
            "referenced_short_name_en": pref["short_name_en"],
            "available_count": len(_flow_properties(flow)),
        }
        if not pref["uuid"]:
            findings.append(
                _rule_finding(
                    flow_uuid, base_version, "error", "invalid_flow_property_reference",
                    "Could not parse flow property UUID from referenceToFlowPropertyDataSet.",
                    evidence={"selected_internal_id": chosen_internal_id or pref["internal_id"]},
                )
            )
        quant_id = summary["quantitative_reference"]["reference_flow_property_internal_id"]
        if not quant_id:
            findings.append(
                _rule_finding(
                    flow_uuid, base_version, "warning", "missing_quantitative_reference",
                    "referenceToReferenceFlowProperty is missing.",
                    fixability="auto",
                    evidence={"expected_internal_id": chosen_internal_id or pref["internal_id"]},
                )
            )
        elif chosen_internal_id and quant_id != chosen_internal_id:
            findings.append(
                _rule_finding(
                    flow_uuid, base_version, "warning", "quantitative_reference_mismatch",
                    "Quantitative reference internal ID differs from selected reference flowProperty internal ID.",
                    fixability="auto",
                    evidence={"quant_ref_internal_id": quant_id, "expected_internal_id": chosen_internal_id},
                    action="Align quantitative reference internal ID to the selected flowProperty.",
                )
            )

        # Optional local registry enrichment for unitgroup/reference-unit context.
        if registry_ctx is not None and pref["uuid"]:
            fp_uuid = pref["uuid"]
            fp_ctx = fp_registry_cache.get(fp_uuid) if fp_uuid in fp_registry_cache else None
            if fp_uuid not in fp_registry_cache:
                try:
                    fp_ctx = registry_ctx.lookup_flow_property_context(fp_uuid)
                except Exception:
                    fp_ctx = None
                fp_registry_cache[fp_uuid] = fp_ctx

            if not fp_ctx:
                summary["unitgroup"]["lookup_status"] = "flowproperty_lookup_failed"
                findings.append(
                    _rule_finding(
                        flow_uuid, base_version, "warning", "flowproperty_lookup_failed",
                        "Referenced flow property was not found in process-automated-builder local registry.",
                        fixability="review-needed",
                        evidence={"flow_property_uuid": pref["uuid"], "flow_property_version": pref["version"]},
                    )
                )
            else:
                ug_uuid = _coerce_text(fp_ctx.get("unitgroup_uuid"))
                summary["unitgroup"]["uuid"] = ug_uuid
                summary["unitgroup"]["name"] = _coerce_text(fp_ctx.get("unitgroup_name"))
                summary["unitgroup"]["reference_unit_name"] = _coerce_text(fp_ctx.get("unitgroup_reference_unit_name"))
                summary["flow_property"]["registry_name"] = _coerce_text(fp_ctx.get("flow_property_name"))
                if not summary["flow_property"].get("referenced_version"):
                    summary["flow_property"]["registry_best_known_version"] = _coerce_text(fp_ctx.get("flow_property_version"))
                if not ug_uuid:
                    summary["unitgroup"]["lookup_status"] = "unitgroup_reference_missing"
                    findings.append(
                        _rule_finding(
                            flow_uuid, base_version, "warning", "missing_unitgroup_reference",
                            "Referenced flow property does not expose a parseable unit group UUID.",
                            evidence={"flow_property_uuid": pref["uuid"]},
                        )
                    )
                else:
                    summary["unitgroup"]["lookup_status"] = "ok"
                    if not summary["unitgroup"]["reference_unit_name"]:
                        findings.append(
                            _rule_finding(
                                flow_uuid, base_version, "warning", "missing_reference_unit_name",
                                "Unit group resolved via local registry but no readable reference unit name found.",
                                evidence={"unitgroup_uuid": ug_uuid},
                            )
                        )

    if methodology_rule_source:
        findings.extend(
            _apply_methodology_checks(
                flow,
                flow_uuid=flow_uuid,
                base_version=base_version,
                rule_source=methodology_rule_source,
            )
        )

    # record compact rule-based signals for LLM
    summary["rule_signals"] = [
        {
            "rule_id": f.get("rule_id"),
            "severity": f.get("severity"),
            "message": f.get("message"),
            "evidence": f.get("evidence", {}),
        }
        for f in findings[:20]
    ]
    return summary, findings


def _build_similarity(
    summaries: List[Dict[str, Any]], threshold: float
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    pairs: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    candidates_by_flow: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for s in summaries:
        leaf = s.get("classification", {}).get("leaf", {}) if isinstance(s.get("classification"), dict) else {}
        leaf_key = _coerce_text(leaf.get("key"))
        if not leaf_key:
            continue
        flow_prop_uuid = _coerce_text(s.get("flow_property", {}).get("referenced_uuid")) if isinstance(s.get("flow_property"), dict) else ""
        unitgroup_uuid = _coerce_text(s.get("unitgroup", {}).get("uuid")) if isinstance(s.get("unitgroup"), dict) else ""
        name_fp = _coerce_text(s.get("_name_fingerprint"))
        if not name_fp:
            continue
        grouped[(leaf_key, flow_prop_uuid, unitgroup_uuid)].append(s)

    for group_key, rows in grouped.items():
        if len(rows) < 2:
            continue
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a = _coerce_text(rows[i].get("_name_fingerprint"))
                b = _coerce_text(rows[j].get("_name_fingerprint"))
                if not a or not b:
                    continue
                ratio = SequenceMatcher(None, a, b).ratio()
                if ratio < threshold:
                    continue
                pair = {
                    "classification_group": group_key[0],
                    "flow_property_uuid": group_key[1],
                    "unitgroup_uuid": group_key[2],
                    "left_flow_uuid": rows[i].get("flow_uuid"),
                    "right_flow_uuid": rows[j].get("flow_uuid"),
                    "left_version": rows[i].get("base_version"),
                    "right_version": rows[j].get("base_version"),
                    "similarity": round(ratio, 6),
                    "left_name_en": rows[i].get("names", {}).get("primary_en") if isinstance(rows[i].get("names"), dict) else "",
                    "right_name_en": rows[j].get("names", {}).get("primary_en") if isinstance(rows[j].get("names"), dict) else "",
                }
                pairs.append(pair)
                for left, right in ((rows[i], rows[j]), (rows[j], rows[i])):
                    candidates_by_flow[_coerce_text(left.get("flow_uuid"))].append(
                        {
                            "other_flow_uuid": _coerce_text(right.get("flow_uuid")),
                            "other_base_version": _coerce_text(right.get("base_version")),
                            "similarity": round(ratio, 6),
                            "other_name_en": _coerce_text(
                                right.get("names", {}).get("primary_en") if isinstance(right.get("names"), dict) else ""
                            ),
                            "classification_group": group_key[0],
                        }
                    )

    # Keep top candidates by similarity for each flow.
    for flow_uuid, candidates in list(candidates_by_flow.items()):
        candidates.sort(key=lambda x: (x.get("similarity", 0), x.get("other_flow_uuid", "")), reverse=True)
        candidates_by_flow[flow_uuid] = candidates[:5]

    return pairs, candidates_by_flow


# ---------- LLM layer ----------
def _call_llm_chat(api_key: str, model: str, prompt: str, base_url: str) -> Optional[str]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "你是严谨的LCA flow复审助手。只基于输入证据判断，不得臆造。输出必须是JSON对象。"},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices") or []
            if not choices:
                return None
            return choices[0].get("message", {}).get("content")
    except Exception:
        return None


def _llm_flow_batch_review(batch_summaries: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {"enabled": False, "reason": "OPENAI_API_KEY missing"}

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip() or "https://api.openai.com/v1"
    prompt = (
        "请对以下 flow 摘要进行 LCI 复审，重点关注：\n"
        "1) flow property 与 quantitative reference 是否合理/一致；\n"
        "2) unitgroup/reference unit 是否与 flow 类型和分类大体一致（如证据不足必须写明）；\n"
        "3) 同类别高相似度 flow 的重复/近重复风险；\n"
        "4) 名称、分类、typeOfDataSet 的语义一致性与明显异常。\n\n"
        "规则：\n"
        "- 只能依据输入摘要中的证据；\n"
        "- 若 unitgroup 未成功查询到，不得强行判断；\n"
        "- 对每条问题给出 flow_uuid；\n"
        "- 输出必须是 JSON 对象。\n\n"
        "输出格式："
        "{findings:[{flow_uuid, severity, fixability, evidence, action}]}。\n\n"
        f"输入摘要:\n{json.dumps(batch_summaries, ensure_ascii=False)}"
    )
    raw = _call_llm_chat(api_key=api_key, model=model, prompt=prompt, base_url=base_url)
    if not raw:
        return {"enabled": True, "ok": False, "reason": "llm call failed"}

    try:
        start = raw.find("{")
        end = raw.rfind("}")
        parsed = json.loads(raw[start : end + 1] if start >= 0 and end > start else raw)
        return {"enabled": True, "ok": True, "result": parsed}
    except Exception:
        return {"enabled": True, "ok": False, "reason": "llm non-json output", "raw": raw[:8000]}


def _normalize_llm_finding(
    item: Dict[str, Any],
    summary_by_uuid: Dict[str, Dict[str, Any]],
    *,
    fallback_flow_uuid: str = "",
) -> Optional[Dict[str, Any]]:
    flow_uuid = _coerce_text(item.get("flow_uuid")) or fallback_flow_uuid
    if not flow_uuid:
        return None
    base_version = _coerce_text(summary_by_uuid.get(flow_uuid, {}).get("base_version"))
    severity = _coerce_text(item.get("severity")).lower() or "warning"
    if severity not in {"error", "warning", "info"}:
        severity = "warning"
    fixability = _coerce_text(item.get("fixability")) or "review-needed"
    action = _coerce_text(item.get("action")) or _coerce_text(item.get("suggestion")) or _coerce_text(item.get("suggested_action"))
    row: Dict[str, Any] = {
        "flow_uuid": flow_uuid,
        "base_version": base_version,
        "severity": severity,
        "fixability": fixability,
        "source": "llm",
        "action": action,
    }
    evidence = item.get("evidence")
    if isinstance(evidence, dict):
        row["evidence"] = evidence
    elif evidence is not None:
        row["evidence"] = {"text": _coerce_text(evidence)}
    return row


def _run_llm_review(
    summaries: List[Dict[str, Any]],
    *,
    model: str,
    batch_size: int,
    max_flows: int,
) -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY", "").strip():
        return {
            "enabled": False,
            "reason": "OPENAI_API_KEY missing",
            "batch_count": 0,
            "reviewed_flow_count": 0,
            "truncated": False,
            "batch_results": [],
            "llm_findings": [],
        }

    if max_flows > 0:
        target = summaries[:max_flows]
    else:
        target = summaries
    summary_by_uuid = {_coerce_text(s.get("flow_uuid")): s for s in summaries if _coerce_text(s.get("flow_uuid"))}

    batches: List[List[Dict[str, Any]]] = []
    batch_size = max(1, batch_size)
    for i in range(0, len(target), batch_size):
        batches.append(target[i : i + batch_size])

    batch_results = []
    all_llm_findings: List[Dict[str, Any]] = []

    for idx, batch in enumerate(batches, start=1):
        # Remove internal fields before sending to LLM.
        batch_payload = []
        for s in batch:
            cleaned = copy.deepcopy(s)
            cleaned.pop("_name_fingerprint", None)
            batch_payload.append(cleaned)

        res = _llm_flow_batch_review(batch_payload, model=model)
        batch_meta = {
            "batch_index": idx,
            "batch_size": len(batch),
            "enabled": bool(res.get("enabled", True)),
            "ok": bool(res.get("ok", False)),
            "reason": res.get("reason"),
        }
        if res.get("ok"):
            result = res.get("result") or {}
            if isinstance(result, dict):
                raw_findings = result.get("findings")
                if isinstance(raw_findings, list):
                    fallback_flow_uuid = _coerce_text(batch[0].get("flow_uuid")) if len(batch) == 1 else ""
                    for item in raw_findings:
                        if not isinstance(item, dict):
                            continue
                        norm = _normalize_llm_finding(item, summary_by_uuid, fallback_flow_uuid=fallback_flow_uuid)
                        if norm:
                            all_llm_findings.append(norm)
        else:
            if res.get("raw"):
                batch_meta["raw_preview"] = _coerce_text(res.get("raw"))[:500]
        batch_results.append(batch_meta)

    # Deduplicate LLM findings by key fields.
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for f in all_llm_findings:
        key = (
            _coerce_text(f.get("flow_uuid")),
            _coerce_text(f.get("severity")),
            _coerce_text(f.get("fixability")),
            json.dumps(f.get("evidence", {}), ensure_ascii=False, sort_keys=True),
            _coerce_text(f.get("action")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)

    return {
        "enabled": True,
        "ok": any(bool(b.get("ok")) for b in batch_results),
        "batch_count": len(batches),
        "reviewed_flow_count": len(target),
        "truncated": len(target) < len(summaries),
        "batch_results": batch_results,
        "llm_findings": deduped,
    }


def _severity_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    c: Dict[str, int] = defaultdict(int)
    for r in rows:
        c[_coerce_text(r.get("severity")) or "unknown"] += 1
    return dict(sorted(c.items()))


def _rule_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    c: Dict[str, int] = defaultdict(int)
    for r in rows:
        c[_coerce_text(r.get("rule_id")) or "unknown"] += 1
    return dict(sorted(c.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Flow governance review engine (LLM-driven)")
    ap.add_argument("--rows-file", help="Flow rows JSON/JSONL file to materialize into a review snapshot.")
    ap.add_argument("--run-root", help="Run root containing cache/flows (used when --flows-dir omitted)")
    ap.add_argument("--run-id", help="Run id for reporting")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--flows-dir", help="Directory containing flow JSON files")
    ap.add_argument("--start-ts")
    ap.add_argument("--end-ts")
    ap.add_argument("--logic-version", default="flow-v1.0-llm")
    default_llm_enabled = bool(os.getenv("OPENAI_API_KEY", "").strip())
    ap.add_argument(
        "--enable-llm",
        dest="enable_llm",
        action="store_true",
        default=default_llm_enabled,
        help="Enable LLM semantic review (default: enabled when OPENAI_API_KEY is set).",
    )
    ap.add_argument(
        "--disable-llm",
        dest="enable_llm",
        action="store_false",
        help="Disable LLM semantic review even if OPENAI_API_KEY is set.",
    )
    ap.add_argument("--llm-model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--llm-max-flows", type=int, default=120)
    ap.add_argument("--llm-batch-size", type=int, default=20)
    ap.add_argument(
        "--with-reference-context",
        action="store_true",
        help="Enable local flow property/unitgroup reference context using process-automated-builder registry.",
    )
    ap.add_argument("--similarity-threshold", type=float, default=0.92)
    ap.add_argument(
        "--methodology-file",
        help="Optional path to methodology YAML/JSON. Presence enables methodology-backed rule tags in findings.",
    )
    ap.add_argument(
        "--methodology-id",
        default="tidas_flows.yaml",
        help="Identifier written to findings.rule_source when methodology checks are enabled.",
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    input_modes = [bool(args.rows_file), bool(args.flows_dir), bool(args.run_root)]
    if sum(input_modes) != 1:
        ap.error("flow review requires exactly one of --rows-file, --flows-dir, or --run-root")

    rows_file: Optional[Path] = Path(args.rows_file).resolve() if args.rows_file else None
    materialized_dir: Optional[Path] = None
    flows_dir: Optional[Path] = Path(args.flows_dir).resolve() if args.flows_dir else None
    if rows_file is not None:
        materialized_dir = _materialize_rows_file(rows_file, out)
        flows_dir = materialized_dir
    elif flows_dir is None:
        root = Path(args.run_root).resolve()
        candidates = [root / "cache" / "flows", root / "exports" / "flows"]
        flows_dir = next((p for p in candidates if p.exists()), candidates[0])
    _write_json(
        out / "review-input-summary.json",
        {
            "input_mode": "rows_file" if rows_file is not None else ("flows_dir" if args.flows_dir else "run_root"),
            "rows_file": str(rows_file) if rows_file is not None else "",
            "flows_dir": str(Path(args.flows_dir).resolve()) if args.flows_dir else "",
            "run_root": str(Path(args.run_root).resolve()) if args.run_root else "",
            "materialized_flows_dir": str(materialized_dir) if materialized_dir is not None else "",
            "effective_flows_dir": str(flows_dir),
        },
    )
    run_id = args.run_id or (
        Path(args.run_root).name if args.run_root else (rows_file.stem if rows_file is not None else flows_dir.name)
    )

    files = _iter_flow_files(flows_dir)
    if not files:
        raise SystemExit(f"No flow JSON files found in {flows_dir}")

    default_methodology = Path(__file__).resolve().parents[1] / "references" / "tidas_flows.yaml"
    methodology_rule_source: Optional[str] = None
    methodology_path: Optional[Path] = None
    if args.methodology_file:
        mf = Path(args.methodology_file).expanduser().resolve()
        if not mf.exists():
            raise SystemExit(f"Methodology file not found: {mf}")
        methodology_path = mf
    elif default_methodology.exists():
        methodology_path = default_methodology

    if methodology_path is not None:
        methodology_rule_source = args.methodology_id or methodology_path.name

    registry_ctx: Optional[_FlowPropertyRegistryFacade] = None
    fp_registry_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    if args.with_reference_context:
        registry_ctx = _FlowPropertyRegistryFacade()
        registry_ctx.open()

    try:
        rule_findings: List[Dict[str, Any]] = []
        flow_summaries: List[Dict[str, Any]] = []

        for fp in files:
            obj = _read_json(fp)
            if not isinstance(obj, dict):
                continue
            summary, finds = _flow_summary_and_rule_findings(
                obj,
                registry_ctx=registry_ctx,
                fp_registry_cache=fp_registry_cache,
                methodology_rule_source=methodology_rule_source,
            )
            summary["source_file"] = fp.name
            flow_summaries.append(summary)
            rule_findings.extend(finds)

        similarity_pairs, candidates_by_flow = _build_similarity(flow_summaries, threshold=args.similarity_threshold)

        # Attach similarity candidates to summaries and rule-based findings.
        for s in flow_summaries:
            flow_uuid = _coerce_text(s.get("flow_uuid"))
            s["similarity_candidates"] = candidates_by_flow.get(flow_uuid, [])
            if s["similarity_candidates"]:
                rule_findings.append(
                    _rule_finding(
                        flow_uuid,
                        _coerce_text(s.get("base_version")),
                        "warning",
                        "same_category_high_similarity",
                        "Another flow in the same classification/flowProperty/unitgroup group is highly similar.",
                        fixability="review-needed",
                        evidence={"candidates": s["similarity_candidates"][:3]},
                    )
                )

        llm_result: Dict[str, Any] = {"enabled": False, "reason": "disabled"}
        if args.enable_llm:
            llm_result = _run_llm_review(
                flow_summaries,
                model=args.llm_model,
                batch_size=args.llm_batch_size,
                max_flows=args.llm_max_flows,
            )

        llm_findings = llm_result.get("llm_findings", []) if isinstance(llm_result, dict) else []
        llm_findings = [x for x in llm_findings if isinstance(x, dict)]
        merged_findings = rule_findings + llm_findings

        # machine-readable outputs
        _write_jsonl(out / "rule_findings.jsonl", rule_findings)
        _write_jsonl(out / "llm_findings.jsonl", llm_findings)
        _write_jsonl(out / "findings.jsonl", merged_findings)
        # strip internal fingerprints before save
        saved_summaries = []
        for s in flow_summaries:
            c = copy.deepcopy(s)
            c.pop("_name_fingerprint", None)
            saved_summaries.append(c)
        _write_jsonl(out / "flow_summaries.jsonl", saved_summaries)
        _write_jsonl(out / "similarity_pairs.jsonl", similarity_pairs)

        summary_json = {
            "run_id": run_id,
            "logic_version": args.logic_version,
            "flow_count": len(flow_summaries),
            "with_reference_context": bool(args.with_reference_context),
            "reference_context_mode": "process_automated_builder_registry" if args.with_reference_context else "disabled",
            "similarity_threshold": args.similarity_threshold,
            "methodology_rule_source": methodology_rule_source or "disabled",
            "rule_finding_count": len(rule_findings),
            "llm_finding_count": len(llm_findings),
            "finding_count": len(merged_findings),
            "severity_counts": _severity_counts(merged_findings),
            "rule_counts": _rule_counts(merged_findings),
            "llm": {
                k: v
                for k, v in llm_result.items()
                if k not in {"llm_findings"}
            },
        }
        _write_json(out / "flow_review_summary.json", summary_json)

        # markdown outputs
        zh = [
            "# flow_review_zh\n",
            f"- run_id: `{run_id}`\n",
            f"- logic_version: `{args.logic_version}`\n",
            f"- flows_dir: `{flows_dir}`\n",
            f"- flow count: `{len(flow_summaries)}`\n",
            f"- with_reference_context (`--with-reference-context`): `{bool(args.with_reference_context)}`\n",
            f"- methodology_rule_source: `{methodology_rule_source or 'disabled'}`\n",
            "\n## 基础统计\n",
            f"- rule-based findings: **{len(rule_findings)}**\n",
            f"- LLM findings: **{len(llm_findings)}**\n",
            f"- merged findings: **{len(merged_findings)}**\n",
        ]
        if merged_findings:
            zh.append("\n### Severity 统计\n")
            for k, v in summary_json["severity_counts"].items():
                zh.append(f"- {k}: {v}\n")

        zh += [
            "\n## Flow 摘要（最多展示 100 条）\n",
            "|flow uuid|version|typeOfDataSet|name(en)|class leaf|flow property|unitgroup ref unit|规则信号数|相似候选数|\n",
            "|---|---|---|---|---|---|---|---:|---:|\n",
        ]
        for s in saved_summaries[:100]:
            leaf = s.get("classification", {}).get("leaf", {}) if isinstance(s.get("classification"), dict) else {}
            names = s.get("names", {}) if isinstance(s.get("names"), dict) else {}
            fp_ref = s.get("flow_property", {}) if isinstance(s.get("flow_property"), dict) else {}
            ug = s.get("unitgroup", {}) if isinstance(s.get("unitgroup"), dict) else {}
            zh.append(
                f"|{str(s.get('flow_uuid','')).replace('|','/')}|{str(s.get('base_version','')).replace('|','/')}|{str(s.get('type_of_dataset','')).replace('|','/')}|{str(names.get('primary_en','')).replace('|','/')}|{str(leaf.get('text','')).replace('|','/')}|{str(fp_ref.get('referenced_short_name_en') or fp_ref.get('referenced_uuid') or '').replace('|','/')}|{str(ug.get('reference_unit_name','')).replace('|','/')}|{len(s.get('rule_signals') or [])}|{len(s.get('similarity_candidates') or [])}|\n"
            )

        zh.append("\n## LLM 语义复审层（可选）\n")
        if llm_result.get("enabled") and llm_result.get("ok"):
            if llm_result.get("truncated"):
                zh.append(
                    f"- 注意：LLM 仅复审前 `{llm_result.get('reviewed_flow_count')}` 条（受 `--llm-max-flows` 限制）。\n"
                )
            if llm_findings:
                zh.append(
                    "\n|flow uuid|severity|fixability|evidence|action|\n|---|---|---|---|---|\n"
                )
                for f in llm_findings[:200]:
                    zh.append(
                        f"|{str(f.get('flow_uuid','')).replace('|','/')}|{str(f.get('severity','')).replace('|','/')}|{str(f.get('fixability','')).replace('|','/')}|{json.dumps(f.get('evidence', {}), ensure_ascii=False).replace('|','/')}|{str(f.get('action','')).replace('|','/')}|\n"
                    )
        else:
            zh.append(f"- 未启用或调用失败：`{llm_result.get('reason', 'unknown')}`\n")

        zh += [
            "\n## 规则抽取层限制（非最终语义判断）\n",
            "- 本脚本的 rule-based 部分主要负责抽取结构化证据和局部一致性信号，最终语义判断应以 LLM 复审层为主。\n",
            "- 若未启用 `--with-reference-context`（该标志当前启用的是 process-automated-builder 本地 registry 上下文），flow property / unitgroup 合理性判断证据会明显不足。\n",
        ]

        en = [
            "# flow_review_en\n",
            f"- run_id: `{run_id}`\n",
            f"- logic_version: `{args.logic_version}`\n",
            f"- flows_dir: `{flows_dir}`\n",
            f"- flow count: `{len(flow_summaries)}`\n",
            f"- with_reference_context (`--with-reference-context`): `{bool(args.with_reference_context)}`\n",
            f"- methodology_rule_source: `{methodology_rule_source or 'disabled'}`\n",
            "\n## Summary\n",
            f"- rule-based findings: **{len(rule_findings)}**\n",
            f"- llm findings: **{len(llm_findings)}**\n",
            f"- merged findings: **{len(merged_findings)}**\n",
        ]
        if llm_result.get("enabled") and llm_result.get("ok"):
            if llm_result.get("truncated"):
                en.append(
                    f"- LLM reviewed only the first `{llm_result.get('reviewed_flow_count')}` flows due to `--llm-max-flows`.\n"
                )
        else:
            en.append(f"- LLM disabled or failed: `{llm_result.get('reason', 'unknown')}`\n")

        timing = ["# flow_review_timing\n", f"- run_id: `{run_id}`\n"]
        if args.start_ts and args.end_ts:
            try:
                s = datetime.fromisoformat(args.start_ts)
                e = datetime.fromisoformat(args.end_ts)
                timing += [
                    f"- start: `{args.start_ts}`\n",
                    f"- end: `{args.end_ts}`\n",
                    f"- total elapsed: **{(e - s).total_seconds() / 60:.2f} min**\n",
                ]
            except Exception:
                timing += [f"- start: `{args.start_ts}`\n", f"- end: `{args.end_ts}`\n"]
        timing.append(f"- flow files reviewed: `{len(flow_summaries)}`\n")
        timing.append("- major time consumers: flow JSON parsing, similarity grouping, optional local registry lookups, LLM review batches.\n")

        # write files
        (out / "flow_review_zh.md").write_text("".join(zh), encoding="utf-8")
        (out / "flow_review_en.md").write_text("".join(en), encoding="utf-8")
        (out / "flow_review_timing.md").write_text("".join(timing), encoding="utf-8")
    finally:
        if registry_ctx is not None:
            registry_ctx.close()


if __name__ == "__main__":
    main()
