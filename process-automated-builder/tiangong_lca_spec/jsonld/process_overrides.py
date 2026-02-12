"""JSON-LD specific post-processing helpers for ILCD process datasets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tiangong_lca_spec.core.constants import build_dataset_format_reference
from tiangong_lca_spec.core.uris import build_local_dataset_uri

TIANGONG_CONTACT_UUID = "f4b4c314-8c4c-4c83-968f-5b3c7724f6a8"
TIANGONG_CONTACT_VERSION = "01.00.000"
TIANGONG_CONTACT_URI = build_local_dataset_uri("contact data set", TIANGONG_CONTACT_UUID, TIANGONG_CONTACT_VERSION)
MASS_FLOW_PROPERTY_UUID = "93a60a56-a3c8-11da-a746-0800200b9a66"
MASS_FLOW_PROPERTY_VERSION = "03.00.003"


def apply_jsonld_process_overrides(process_dataset: dict[str, Any]) -> None:
    """Ensure JSON-LD process datasets contain mandatory ILCD metadata."""

    node = process_dataset.get("processDataSet")
    if isinstance(node, dict):
        target = node
    else:
        target = process_dataset

    modelling = target.setdefault("modellingAndValidation", {})
    dsr = modelling.setdefault("dataSourcesTreatmentAndRepresentativeness", {})
    references = dsr.get("referenceToDataSource")
    reference_entries: list[dict[str, Any]] = []
    if isinstance(references, dict):
        reference_entries.append(references)
    elif isinstance(references, list):
        reference_entries.extend([entry for entry in references if isinstance(entry, dict)])
    if reference_entries:
        dsr["referenceToDataSource"] = reference_entries

    process_info = target.setdefault("processInformation", {})
    technology = process_info.get("technology")
    if isinstance(technology, dict):
        if not technology.get("technologyDescriptionAndIncludedProcesses"):
            technology.pop("technologyDescriptionAndIncludedProcesses", None)
        if not technology:
            process_info.pop("technology", None)

    admin = target.setdefault("administrativeInformation", {})
    data_entry = admin.setdefault("dataEntryBy", {})
    data_entry.setdefault("common:referenceToDataSetFormat", build_dataset_format_reference())
    data_entry.setdefault("common:referenceToPersonOrEntityEnteringTheData", _build_contact_reference())
    data_entry["common:timeStamp"] = _current_timestamp()

    commissioner = admin.setdefault("common:commissionerAndGoal", {})
    commissioner.setdefault("common:referenceToCommissioner", _build_contact_reference())

    exchanges_node = target.setdefault("exchanges", {})
    exchanges = exchanges_node.get("exchange")
    if isinstance(exchanges, dict):
        exchanges = [exchanges]
    elif not isinstance(exchanges, list):
        exchanges = []
    for idx, exchange in enumerate(exchanges, start=1):
        if not isinstance(exchange, dict):
            continue
        exchange["@dataSetInternalID"] = str(exchange.get("@dataSetInternalID") or idx)
        ref = exchange.setdefault("referenceToFlowDataSet", {})
        flow_id = ref.get("@refObjectId") or exchange.get("exchangeId") or exchange.get("flowId")
        if flow_id:
            ref["@refObjectId"] = flow_id
            version = ref.setdefault("@version", "01.01.000")
            ref["@type"] = ref.get("@type") or "flow data set"
            ref["@uri"] = ref.get("@uri") or f"../flows/{flow_id}_{version}.xml"
        else:
            ref["@type"] = ref.get("@type") or "flow data set"
        if "common:shortDescription" not in ref:
            ref["common:shortDescription"] = _language_entry(exchange.get("exchangeName") or "Referenced flow")

        prop_ref = exchange.get("referenceToFlowPropertyDataSet")
        if isinstance(prop_ref, dict) and prop_ref.get("@refObjectId"):
            prop_ref.setdefault("@type", "flow property data set")
            prop_version = prop_ref.setdefault("@version", "01.01.000")
            prop_id = prop_ref.get("@refObjectId")
            if prop_id:
                prop_ref.setdefault("@uri", f"../flowproperties/{prop_id}_{prop_version}.xml")
        elif prop_ref:
            exchange["referenceToFlowPropertyDataSet"] = {
                "@type": "flow property data set",
                "@refObjectId": MASS_FLOW_PROPERTY_UUID,
                "@uri": f"../flowproperties/{MASS_FLOW_PROPERTY_UUID}_{MASS_FLOW_PROPERTY_VERSION}.xml",
                "@version": MASS_FLOW_PROPERTY_VERSION,
                "common:shortDescription": _language_entry("Mass"),
            }
    exchanges_node["exchange"] = exchanges


def _language_entry(text: str, lang: str = "en") -> dict[str, str]:
    return {"@xml:lang": lang, "#text": text}


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_contact_reference() -> dict[str, Any]:
    return {
        "@refObjectId": TIANGONG_CONTACT_UUID,
        "@type": "contact data set",
        "@uri": TIANGONG_CONTACT_URI,
        "@version": TIANGONG_CONTACT_VERSION,
        "common:shortDescription": [
            _language_entry("Tiangong LCA Data Working Group"),
            _language_entry("天工LCA数据团队", "zh"),
        ],
    }


def auto_fix_from_validation(report_path: Path | str, artifact_root: Path | str) -> bool:
    """Re-open failing process datasets and apply overrides based on validation findings."""

    report_file = Path(report_path)
    if not report_file.exists():
        return False
    try:
        payload = json.loads(report_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    findings = payload.get("validation_report") or []
    if not findings:
        return False

    changed = False
    base_dir = Path(artifact_root)
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        if finding.get("severity") != "error":
            continue
        message = finding.get("message", "")
        dataset_path = finding.get("path")
        if not dataset_path:
            continue
        file_path = Path(dataset_path)
        if not file_path.is_absolute():
            candidate = Path.cwd() / file_path
            if candidate.exists():
                file_path = candidate
            else:
                file_path = (base_dir / file_path).resolve()
        if not file_path.exists():
            continue
        if "dataCutOffAndCompletenessPrinciples" in message or "referenceToDataSource" in message:
            changed |= _apply_overrides_to_file(file_path)
    return changed


def _apply_overrides_to_file(file_path: Path) -> bool:
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    apply_jsonld_process_overrides(payload)
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return True
