#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
FLOW_GOVERNANCE_ROOT = SCRIPT_DIR.parent
LCA_SKILLS_ROOT = FLOW_GOVERNANCE_ROOT.parent
PROCESS_AUTOMATED_BUILDER_ROOT = LCA_SKILLS_ROOT / "process-automated-builder"
FLOW_PROCESSING_ARTIFACT_ROOT = FLOW_GOVERNANCE_ROOT / "assets" / "artifacts" / "flow-processing"
FLOW_PROCESSING_DATASETS_DIR = FLOW_PROCESSING_ARTIFACT_ROOT / "datasets"
FLOW_PROCESSING_REMEDIATION_DIR = FLOW_PROCESSING_ARTIFACT_ROOT / "remediation" / "round1"
if str(PROCESS_AUTOMATED_BUILDER_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESS_AUTOMATED_BUILDER_ROOT))

try:
    from tidas_sdk import create_flow
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        "tidas_sdk is required. Run this script with process-automated-builder/.venv/bin/python."
    ) from exc

from tiangong_lca_spec.core.constants import (  # noqa: E402
    ILCD_FORMAT_SOURCE_UUID,
    ILCD_FORMAT_SOURCE_VERSION,
    build_dataset_format_reference,
)
from tiangong_lca_spec.core.uris import build_local_dataset_uri  # noqa: E402
from tiangong_lca_spec.tidas.flow_property_registry import get_default_registry  # noqa: E402


CONTACT_UUID = "f4b4c314-8c4c-4c83-968f-5b3c7724f6a8"
CONTACT_VERSION = "01.00.000"
COMPLIANCE_SOURCE_UUID = "d92a1a12-2545-49e2-a585-55c259997756"
COMPLIANCE_SOURCE_VERSION = "20.20.002"
DEFAULT_MASS_FLOW_PROPERTY_UUID = "93a60a56-a3c8-11da-a746-0800200b9a66"
GENERIC_MASS_HINTS = {
    "flow property for kg",
    "flow property for kilogram",
    "flow property for kilograms",
    "kg",
    "kilogram",
    "kilograms",
    "mass",
    "mass flow",
    "mass flow kg",
}

KNOWN_SOURCE_VERSIONS = {
    ILCD_FORMAT_SOURCE_UUID: ILCD_FORMAT_SOURCE_VERSION,
    COMPLIANCE_SOURCE_UUID: COMPLIANCE_SOURCE_VERSION,
}

VERSION_QUERY_RE = re.compile(r"([?&]version=)(\d+\.\d+\.\d+)")
LOCAL_URI_VERSION_RE = re.compile(r"_(\d+\.\d+\.\d+)\.xml$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically remediate invalid flow json_ordered rows, revalidate them with "
            "tidas_sdk.create_flow(validate=True), and split ready-for-MCP rows from residual manual queue rows."
        )
    )
    parser.add_argument(
        "--input-file",
        default=str(FLOW_PROCESSING_DATASETS_DIR / "flows_tidas_sdk_plus_classification_invalid.jsonl"),
        help="Input invalid flow rows JSONL file.",
    )
    parser.add_argument(
        "--out-all-file",
        default=str(FLOW_PROCESSING_REMEDIATION_DIR / "flows_tidas_sdk_plus_classification_remediated_all.jsonl"),
        help="Output JSONL containing all remediated rows with post-remediation reason lists.",
    )
    parser.add_argument(
        "--out-valid-file",
        default=str(FLOW_PROCESSING_REMEDIATION_DIR / "flows_tidas_sdk_plus_classification_remediated_ready_for_mcp.jsonl"),
        help="Output JSONL containing only rows that pass tidas_sdk after deterministic remediation.",
    )
    parser.add_argument(
        "--out-manual-file",
        default=str(FLOW_PROCESSING_REMEDIATION_DIR / "flows_tidas_sdk_plus_classification_residual_manual_queue.jsonl"),
        help="Output JSONL containing only rows that still need manual remediation.",
    )
    parser.add_argument(
        "--out-report-file",
        default=str(FLOW_PROCESSING_REMEDIATION_DIR / "flows_tidas_sdk_plus_classification_remediation_report.json"),
        help="Output JSON summary report path.",
    )
    parser.add_argument(
        "--out-audit-file",
        default=str(FLOW_PROCESSING_REMEDIATION_DIR / "flows_tidas_sdk_plus_classification_remediation_audit.jsonl"),
        help="Output JSONL audit trail path.",
    )
    parser.add_argument(
        "--out-prompt-file",
        default=str(FLOW_PROCESSING_REMEDIATION_DIR / "flows_tidas_sdk_plus_classification_residual_manual_queue_prompt.md"),
        help="Output Markdown prompt path for OpenClaw residual manual remediation.",
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


def bump_ilcd_version(version: str) -> str:
    parts = str(version or "").split(".")
    numbers: list[int] = []
    for part in parts[:3]:
        try:
            numbers.append(int(part))
        except ValueError:
            numbers.append(0)
    while len(numbers) < 3:
        numbers.append(0)
    numbers[2] += 1
    return f"{numbers[0]:02d}.{numbers[1]:02d}.{numbers[2]:03d}"


def version_from_uri(uri: str) -> str:
    text = str(uri or "").strip()
    if not text:
        return ""
    match = LOCAL_URI_VERSION_RE.search(text)
    if match:
        return match.group(1)
    query_match = VERSION_QUERY_RE.search(text)
    if query_match:
        return query_match.group(2)
    return ""


def normalize_multilang_entries(value: Any) -> list[dict[str, str]] | None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return [{"@xml:lang": "en", "#text": text}]
        return None
    if isinstance(value, dict):
        text = str(value.get("#text") or "").strip()
        if not text:
            return None
        lang = str(value.get("@xml:lang") or "en").strip() or "en"
        return [{"@xml:lang": lang, "#text": text}]
    if isinstance(value, list):
        entries: list[dict[str, str]] = []
        for item in value:
            normalized = normalize_multilang_entries(item)
            if normalized:
                entries.extend(normalized)
        return entries or None
    return None


def first_text(value: Any, fallback: str = "") -> str:
    normalized = normalize_multilang_entries(value)
    if normalized:
        return str(normalized[0].get("#text") or "").strip()
    return fallback


def normalized_hint_text(value: Any) -> str:
    text = first_text(value) if not isinstance(value, str) else str(value)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def canonical_contact_reference() -> dict[str, Any]:
    return {
        "@type": "contact data set",
        "@refObjectId": CONTACT_UUID,
        "@uri": build_local_dataset_uri("contact data set", CONTACT_UUID, CONTACT_VERSION),
        "@version": CONTACT_VERSION,
        "common:shortDescription": [
            {"@xml:lang": "en", "#text": "Tiangong LCA Data Working Group"},
            {"@xml:lang": "zh", "#text": "天工LCA数据团队"},
        ],
    }


def canonical_compliance_block() -> dict[str, Any]:
    return {
        "common:referenceToComplianceSystem": {
            "@refObjectId": COMPLIANCE_SOURCE_UUID,
            "@type": "source data set",
            "@uri": build_local_dataset_uri("source", COMPLIANCE_SOURCE_UUID, COMPLIANCE_SOURCE_VERSION),
            "@version": COMPLIANCE_SOURCE_VERSION,
            "common:shortDescription": [
                {"@xml:lang": "en", "#text": "ILCD Data Network - Entry-level"},
            ],
        },
        "common:approvalOfOverallCompliance": "Fully compliant",
    }


def canonical_dataset_format_reference() -> dict[str, Any]:
    reference = build_dataset_format_reference()
    short_description = reference.get("common:shortDescription")
    if isinstance(short_description, dict):
        reference["common:shortDescription"] = [short_description]
    return reference


def normalize_reference_block(
    ref: Any,
    *,
    default_type: str = "",
    short_description_fallback: str = "",
) -> dict[str, Any] | None:
    if not isinstance(ref, dict):
        return None
    ref_object_id = str(ref.get("@refObjectId") or "").strip()
    if not ref_object_id:
        return None
    ref_type = str(ref.get("@type") or default_type or "").strip()
    version = str(ref.get("@version") or "").strip()
    if not version:
        version = KNOWN_SOURCE_VERSIONS.get(ref_object_id, "") or version_from_uri(str(ref.get("@uri") or ""))
    uri = str(ref.get("@uri") or "").strip()
    if version and ref_type:
        uri = build_local_dataset_uri(ref_type, ref_object_id, version)
    short_description = normalize_multilang_entries(ref.get("common:shortDescription"))
    if short_description is None and short_description_fallback:
        short_description = [{"@xml:lang": "en", "#text": short_description_fallback}]
    result = {
        "@type": ref_type,
        "@refObjectId": ref_object_id,
        "@uri": uri,
    }
    if version:
        result["@version"] = version
    if short_description:
        result["common:shortDescription"] = short_description
    return result


def normalize_multilang_field(container: dict[str, Any], key: str, fixes: list[str]) -> None:
    if key not in container:
        return
    normalized = normalize_multilang_entries(container.get(key))
    if normalized is None:
        container.pop(key, None)
        fixes.append(f"remove_empty_multilang:{key}")
        return
    if container.get(key) != normalized:
        container[key] = normalized
        fixes.append(f"normalize_multilang:{key}")


def normalize_name_block(data_set_info: dict[str, Any], fixes: list[str]) -> None:
    name_block = data_set_info.get("name")
    if not isinstance(name_block, dict):
        return
    for key in ("baseName", "flowProperties", "mixAndLocationTypes", "treatmentStandardsRoutes"):
        if key in name_block:
            normalized = normalize_multilang_entries(name_block.get(key))
            if normalized is None:
                name_block.pop(key, None)
                fixes.append(f"remove_empty_name_field:{key}")
            elif name_block.get(key) != normalized:
                name_block[key] = normalized
                fixes.append(f"normalize_name_field:{key}")


def build_flow_property_item(
    descriptor: Any,
    *,
    idx: int,
    source_item: dict[str, Any] | None = None,
) -> dict[str, Any]:
    version = descriptor.version
    data_set_internal_id = str((source_item or {}).get("@dataSetInternalID") or idx)
    mean_value = str((source_item or {}).get("meanValue") or "1.0")
    return {
        "@dataSetInternalID": data_set_internal_id,
        "meanValue": mean_value,
        "referenceToFlowPropertyDataSet": {
            "@type": "flow property data set",
            "@refObjectId": descriptor.uuid,
            "@uri": build_local_dataset_uri("flow property data set", descriptor.uuid, version),
            "@version": version,
            "common:shortDescription": [
                {"@xml:lang": "en", "#text": descriptor.name},
            ],
        },
    }


def default_mass_descriptor(registry: Any) -> Any:
    return registry.get(DEFAULT_MASS_FLOW_PROPERTY_UUID)


def infer_flow_property_descriptor_from_text(text: str, registry: Any) -> Any | None:
    if not text:
        return None
    exact = registry.find(text)
    if exact is not None:
        return exact

    normalized = normalized_hint_text(text)
    if not normalized:
        return None

    if normalized in GENERIC_MASS_HINTS:
        return default_mass_descriptor(registry)
    if normalized.startswith("flow property for ") and any(
        token in normalized.split() for token in ("kg", "kilogram", "kilograms")
    ):
        return default_mass_descriptor(registry)
    if normalized.startswith("flow property"):
        return None

    return registry.fuzzy_match(text)


def normalize_flow_properties(
    dataset: dict[str, Any],
    *,
    fixes: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registry = get_default_registry()
    flow_properties_block = dataset.setdefault("flowProperties", {})
    if not isinstance(flow_properties_block, dict):
        flow_properties_block = {}
        dataset["flowProperties"] = flow_properties_block
    raw_items = flow_properties_block.get("flowProperty")
    items = raw_items if isinstance(raw_items, list) else ([raw_items] if isinstance(raw_items, dict) else [])
    normalized_items: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    repaired_unknown_uuid = False
    filled_missing_uuid = False
    used_default_mass_fallback = False

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            unresolved.append(
                {
                    "path": "flowDataSet.flowProperties.flowProperty",
                    "code": "unsupported_item_type",
                    "message": "flowProperty item is not an object",
                }
            )
            continue
        reference = item.get("referenceToFlowPropertyDataSet")
        if not isinstance(reference, dict):
            unresolved.append(
                {
                    "path": "flowDataSet.flowProperties.flowProperty.referenceToFlowPropertyDataSet",
                    "code": "missing_reference",
                    "message": "referenceToFlowPropertyDataSet is missing",
                }
            )
            continue
        flow_property_uuid = str(reference.get("@refObjectId") or "").strip()
        hint_text = first_text(reference.get("common:shortDescription"))
        descriptor = None
        if flow_property_uuid:
            try:
                descriptor = registry.get(flow_property_uuid)
            except KeyError:
                descriptor = infer_flow_property_descriptor_from_text(hint_text, registry)
                if descriptor is None:
                    unresolved.append(
                        {
                            "path": "flowDataSet.flowProperties.flowProperty.referenceToFlowPropertyDataSet.@refObjectId",
                            "code": "unknown_flow_property_uuid",
                            "message": f"Unknown flow property UUID: {flow_property_uuid}",
                        }
                    )
                    continue
                repaired_unknown_uuid = True
        else:
            descriptor = infer_flow_property_descriptor_from_text(hint_text, registry)
            if descriptor is None:
                unresolved.append(
                    {
                        "path": "flowDataSet.flowProperties.flowProperty.referenceToFlowPropertyDataSet.@refObjectId",
                        "code": "unknown_flow_property_uuid",
                        "message": "Unable to infer flow property UUID from the reference block",
                    }
                )
                continue
            filled_missing_uuid = True

        normalized_items.append(build_flow_property_item(descriptor, idx=idx, source_item=item))

    if not normalized_items:
        if not items:
            normalized_items = [build_flow_property_item(default_mass_descriptor(registry), idx=0)]
            used_default_mass_fallback = True
        else:
            if isinstance(raw_items, dict) and not raw_items:
                fixes.append("remove_empty_flow_properties_block")
            if isinstance(raw_items, list) and not raw_items:
                fixes.append("remove_empty_flow_properties_block")
            flow_properties_block.pop("flowProperty", None)
            return [], unresolved

    if len(normalized_items) == 1:
        flow_properties_block["flowProperty"] = normalized_items[0]
    else:
        flow_properties_block["flowProperty"] = normalized_items
    fixes.append("normalize_flow_properties")
    if repaired_unknown_uuid:
        fixes.append("repair_unknown_flow_property_uuid_from_short_description")
    if filled_missing_uuid:
        fixes.append("fill_missing_flow_property_uuid_from_short_description")
    if used_default_mass_fallback:
        fixes.append("set_default_mass_flow_property")
    return normalized_items, unresolved


def normalize_quantitative_reference(
    dataset: dict[str, Any],
    normalized_flow_properties: list[dict[str, Any]],
    *,
    fixes: list[str],
    unresolved: list[dict[str, Any]],
) -> None:
    info = dataset.setdefault("flowInformation", {})
    quantitative_reference = info.setdefault("quantitativeReference", {})
    if not isinstance(quantitative_reference, dict):
        quantitative_reference = {}
        info["quantitativeReference"] = quantitative_reference

    if not normalized_flow_properties:
        unresolved.append(
            {
                "path": "flowDataSet.flowInformation.quantitativeReference.referenceToReferenceFlowProperty",
                "code": "missing_flow_properties",
                "message": "Cannot infer referenceToReferenceFlowProperty without any valid flowProperty entries",
            }
        )
        return

    allowed_ids = {str(item.get("@dataSetInternalID") or "").strip() for item in normalized_flow_properties}
    current = str(quantitative_reference.get("referenceToReferenceFlowProperty") or "").strip()
    if current not in allowed_ids:
        quantitative_reference["referenceToReferenceFlowProperty"] = str(
            normalized_flow_properties[0].get("@dataSetInternalID") or "0"
        )
        fixes.append("set_reference_to_reference_flow_property")


def normalize_technology_multilang_fields(dataset: dict[str, Any], fixes: list[str]) -> None:
    technology = dataset.get("flowInformation", {}).get("technology")
    if not isinstance(technology, dict):
        return
    normalize_multilang_field(technology, "technologicalApplicability", fixes)


def normalize_technical_specification(dataset: dict[str, Any], fixes: list[str]) -> None:
    technology = dataset.get("flowInformation", {}).get("technology")
    if not isinstance(technology, dict):
        return
    key = "referenceToTechnicalSpecification"
    if key not in technology:
        return
    normalized = normalize_reference_block(
        technology.get(key),
        default_type="source data set",
        short_description_fallback=first_text(technology.get(key)),
    )
    if normalized is None:
        technology.pop(key, None)
        fixes.append("remove_invalid_reference_to_technical_specification")
        return
    if technology.get(key) != normalized:
        technology[key] = normalized
        fixes.append("normalize_reference_to_technical_specification")


def update_permanent_dataset_uri(uri_value: Any, target_version: str) -> Any:
    uri = str(uri_value or "").strip()
    if not uri:
        return uri_value
    if VERSION_QUERY_RE.search(uri):
        return VERSION_QUERY_RE.sub(rf"\g<1>{target_version}", uri)
    return uri_value


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


def remediate_row(row: dict[str, Any]) -> dict[str, Any]:
    working = deepcopy(row)
    payload = working.get("json_ordered")
    if not isinstance(payload, dict) or not isinstance(payload.get("flowDataSet"), dict):
        return {
            "row": working,
            "valid": False,
            "applied_fixes": [],
            "final_reasons": [
                {
                    "validator": "tidas_sdk",
                    "path": "json_ordered.flowDataSet",
                    "message": "Missing flowDataSet payload",
                    "code": "missing_flow_dataset",
                }
            ],
            "version_before": "",
            "version_after": "",
        }

    fixes: list[str] = []
    residual_unresolved: list[dict[str, str]] = []
    dataset = payload["flowDataSet"]
    flow_info = dataset.setdefault("flowInformation", {})
    data_set_info = flow_info.setdefault("dataSetInformation", {})
    normalize_name_block(data_set_info, fixes)
    for key in ("common:synonyms", "common:generalComment", "common:shortDescription", "common:name", "common:shortName"):
        normalize_multilang_field(data_set_info, key, fixes)
    if isinstance(data_set_info.get("common:other"), dict):
        data_set_info.pop("common:other", None)
        fixes.append("remove_common_other_object")
    if data_set_info.get("CASNumber") == "":
        data_set_info.pop("CASNumber", None)
        fixes.append("remove_empty_cas_number")

    administrative_information = dataset.setdefault("administrativeInformation", {})
    data_entry_by = administrative_information.setdefault("dataEntryBy", {})
    publication_and_ownership = administrative_information.setdefault("publicationAndOwnership", {})

    format_reference = canonical_dataset_format_reference()
    if data_entry_by.get("common:referenceToDataSetFormat") != format_reference:
        data_entry_by["common:referenceToDataSetFormat"] = format_reference
        fixes.append("set_reference_to_dataset_format")

    contact_reference = canonical_contact_reference()
    if data_entry_by.get("common:referenceToPersonOrEntityEnteringTheData") != contact_reference:
        data_entry_by["common:referenceToPersonOrEntityEnteringTheData"] = contact_reference
        fixes.append("set_reference_to_person_or_entity_entering_the_data")
    if publication_and_ownership.get("common:referenceToOwnershipOfDataSet") != contact_reference:
        publication_and_ownership["common:referenceToOwnershipOfDataSet"] = contact_reference
        fixes.append("set_reference_to_ownership_of_dataset")

    compliance_block = canonical_compliance_block()
    compliance_declarations = dataset.setdefault("modellingAndValidation", {}).setdefault("complianceDeclarations", {})
    if compliance_declarations.get("compliance") != compliance_block:
        compliance_declarations["compliance"] = compliance_block
        fixes.append("set_compliance_block")

    normalized_flow_properties, property_unresolved = normalize_flow_properties(dataset, fixes=fixes)
    residual_unresolved.extend(property_unresolved)
    normalize_quantitative_reference(
        dataset,
        normalized_flow_properties,
        fixes=fixes,
        unresolved=residual_unresolved,
    )
    normalize_technology_multilang_fields(dataset, fixes)
    normalize_technical_specification(dataset, fixes)

    flow_uuid = str(data_set_info.get("common:UUID") or working.get("id") or "").strip()
    publication_and_ownership.setdefault("common:dataSetVersion", "01.01.000")
    version_before = str(publication_and_ownership.get("common:dataSetVersion") or "").strip() or "01.01.000"
    version_after = version_before
    state_code = int(working.get("state_code") or 0)
    if state_code == 100:
        version_after = bump_ilcd_version(version_before)
        if version_after != version_before:
            publication_and_ownership["common:dataSetVersion"] = version_after
            fixes.append("bump_dataset_version")
            publication_and_ownership["common:referenceToPrecedingDataSetVersion"] = {
                "@type": "flow data set",
                "@refObjectId": flow_uuid,
                "@uri": build_local_dataset_uri("flow data set", flow_uuid, version_before),
                "@version": version_before,
                "common:shortDescription": [
                    {
                        "@xml:lang": "en",
                        "#text": first_text(data_set_info.get("name", {}).get("baseName"), flow_uuid),
                    }
                ],
            }
            fixes.append("set_reference_to_preceding_dataset_version")
            permanent_uri = update_permanent_dataset_uri(
                publication_and_ownership.get("common:permanentDataSetURI"),
                version_after,
            )
            if permanent_uri != publication_and_ownership.get("common:permanentDataSetURI"):
                publication_and_ownership["common:permanentDataSetURI"] = permanent_uri
                fixes.append("update_permanent_dataset_uri_version")
            if "version" in working:
                working["version"] = version_after
    else:
        preceding = publication_and_ownership.get("common:referenceToPrecedingDataSetVersion")
        if isinstance(preceding, dict) and not preceding:
            publication_and_ownership.pop("common:referenceToPrecedingDataSetVersion", None)
            fixes.append("remove_empty_reference_to_preceding_dataset_version")

    try:
        create_flow(payload, validate=True)
        final_reasons: list[dict[str, str]] = []
        valid = True
    except Exception as exc:  # noqa: BLE001
        final_reasons = residual_unresolved + parse_validation_error(exc)
        valid = False

    working["json_ordered"] = payload
    working["reason"] = final_reasons
    return {
        "row": working,
        "valid": valid,
        "applied_fixes": fixes,
        "final_reasons": final_reasons,
        "version_before": version_before,
        "version_after": version_after,
    }


def build_prompt(
    *,
    manual_rows: list[dict[str, Any]],
    manual_file: Path,
    output_file: Path,
    prompt_file: Path,
) -> str:
    if not manual_rows:
        return f"""本轮 deterministic remediation 之后，residual manual queue 为 0。

输入文件：
`{manual_file}`

结果：
1. 当前没有需要再交给 OpenClaw 手工修复的 flow。
2. 可直接对 ready-for-MCP 文件执行后续批量处理。
3. 如果后续换了新的输入批次，再重新生成 residual manual queue prompt。
"""

    queue_ids = [str(row.get("id") or "").strip() for row in manual_rows]
    id_lines = "\n".join(f"- `{value}`" for value in queue_ids) if queue_ids else "- None"
    prompt = f"""你现在要处理 residual manual queue 中 deterministic remediator 之后仍未通过 `tidas_sdk.create_flow(validate=True)` 的 flow。

输入文件：
`{manual_file}`

输出文件：
`{output_file}`

执行要求：
1. 只修改每条记录里的 `json_ordered.flowDataSet`，保留外层 envelope key：`id`、`user_id`、`json_ordered`、`reason`、`state_code`。
2. 保持相同 UUID，不要改 `id`。
3. `state_code=0` 的 flow 保持当前 `common:dataSetVersion` 不变。
4. `state_code=100` 的 flow 已经完成 deterministic version bump，并已补 `common:referenceToPrecedingDataSetVersion`；除非绝对必要，不要再次 bump version。
5. 优先补齐合法的 `flowProperties.flowProperty` 和 `flowInformation.quantitativeReference.referenceToReferenceFlowProperty`，让数据先通过本地 TIDAS SDK 校验。
6. 不要无故重写 classification、命名、comment；如果必须改，改动最小，并在结果里保留必要说明。
7. 每条输出仍按一行一个 JSON object 的 JSONL 形式写入。
8. 修复成功的行把 `reason` 置为 `[]`；如果仍无法修复，单独另存并写明残留原因。

当前 residual manual queue 共 `{len(manual_rows)}` 条：
{id_lines}

建议步骤：
1. 逐条读取 `reason` 和 `json_ordered.flowDataSet`。
2. 先判断缺失的是 `flowProperties` 本体，还是只有 quantitative reference 丢失。
3. 能从现有 `classificationInformation`、`baseName`、`flow type`、`name.flowProperties` 推断合法 flow property 的，补成最小合法块。
4. 每改完一条都本地验证一次 `tidas_sdk.create_flow(validate=True)`。
5. 把通过校验的 patched row 写到 `{output_file}`。

如果你还要保留未解决项，另写一个旁路文件：
`{prompt_file.with_name(prompt_file.stem.replace('_prompt', '_still_invalid') + '.jsonl')}`
"""
    return prompt


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file).expanduser().resolve()
    out_all_file = Path(args.out_all_file).expanduser().resolve()
    out_valid_file = Path(args.out_valid_file).expanduser().resolve()
    out_manual_file = Path(args.out_manual_file).expanduser().resolve()
    out_report_file = Path(args.out_report_file).expanduser().resolve()
    out_audit_file = Path(args.out_audit_file).expanduser().resolve()
    out_prompt_file = Path(args.out_prompt_file).expanduser().resolve()
    for path in (
        out_all_file,
        out_valid_file,
        out_manual_file,
        out_report_file,
        out_audit_file,
        out_prompt_file,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl_rows(input_file)
    remediated_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    applied_fix_counts: dict[str, int] = {}

    for row in rows:
        result = remediate_row(row)
        remediated_row = result["row"]
        remediated_rows.append(remediated_row)
        if result["valid"]:
            valid_rows.append(remediated_row)
        else:
            manual_rows.append(remediated_row)
        for fix in result["applied_fixes"]:
            applied_fix_counts[fix] = applied_fix_counts.get(fix, 0) + 1
        audit_rows.append(
            {
                "id": row.get("id"),
                "user_id": row.get("user_id"),
                "state_code": row.get("state_code"),
                "version_before": result["version_before"],
                "version_after": result["version_after"],
                "valid_after_remediation": result["valid"],
                "applied_fixes": result["applied_fixes"],
                "original_reason": row.get("reason") or [],
                "final_reason": result["final_reasons"],
            }
        )

    dump_jsonl(out_all_file, remediated_rows)
    dump_jsonl(out_valid_file, valid_rows)
    dump_jsonl(out_manual_file, manual_rows)
    dump_jsonl(out_audit_file, audit_rows)

    prompt_text = build_prompt(
        manual_rows=manual_rows,
        manual_file=out_manual_file,
        output_file=out_manual_file.with_name("flows_tidas_sdk_plus_classification_residual_manual_fixed.jsonl"),
        prompt_file=out_prompt_file,
    )
    out_prompt_file.write_text(prompt_text, encoding="utf-8")

    report = {
        "input_file": str(input_file),
        "counts": {
            "input_rows": len(rows),
            "state_code_0_rows": sum(1 for row in rows if int(row.get("state_code") or 0) == 0),
            "state_code_100_rows": sum(1 for row in rows if int(row.get("state_code") or 0) == 100),
            "remediated_rows": len(remediated_rows),
            "ready_for_mcp_rows": len(valid_rows),
            "residual_manual_rows": len(manual_rows),
        },
        "applied_fix_counts": dict(sorted(applied_fix_counts.items())),
        "residual_manual_ids": [row.get("id") for row in manual_rows],
        "files": {
            "all_remediated": str(out_all_file),
            "ready_for_mcp": str(out_valid_file),
            "residual_manual_queue": str(out_manual_file),
            "audit": str(out_audit_file),
            "prompt": str(out_prompt_file),
        },
    }
    dump_json(out_report_file, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
