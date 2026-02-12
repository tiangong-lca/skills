"""Unified builder for product flow creation (plan + validated payload)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from tidas_sdk import create_flow

from tiangong_lca_spec.core.constants import build_dataset_format_reference
from tiangong_lca_spec.core.uris import build_local_dataset_uri

DEFAULT_FLOW_VERSION = "01.01.000"
DEFAULT_TREATMENT_EN = "Unspecified treatment"
DEFAULT_MIX_EN = "Production mix, at plant"
DEFAULT_FLOW_TYPE = "Product flow"

DEFAULT_MASS_FLOW_PROPERTY_UUID = "93a60a56-a3c8-11da-a746-0800200b9a66"
DEFAULT_MASS_FLOW_PROPERTY_VERSION = "03.00.003"
DEFAULT_MASS_FLOW_PROPERTY_NAME = "Mass"


def _lang_entry(text: str, lang: str) -> dict[str, str]:
    return {"@xml:lang": lang, "#text": text}


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().replace("；", "，").replace(";", ",")


def _normalize_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    normalized: list[str] = []
    for item in values:
        text = _normalize_text(item)
        if text:
            normalized.append(text)
    return normalized


def _normalize_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:  # noqa: BLE001
            return value
    try:
        dt = datetime.fromisoformat(str(value))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:  # noqa: BLE001
        return str(value)


def _contact_reference() -> dict[str, Any]:
    uuid_value = "f4b4c314-8c4c-4c83-968f-5b3c7724f6a8"
    version = "01.00.000"
    return {
        "@type": "contact data set",
        "@refObjectId": uuid_value,
        "@uri": build_local_dataset_uri("contact data set", uuid_value, version),
        "@version": version,
        "common:shortDescription": [
            _lang_entry("Tiangong LCA Data Working Group", "en"),
            _lang_entry("天工LCA数据团队", "zh"),
        ],
    }


def _compliance_block() -> dict[str, Any]:
    uuid_value = "d92a1a12-2545-49e2-a585-55c259997756"
    version = "20.20.002"
    return {
        "compliance": {
            "common:referenceToComplianceSystem": {
                "@refObjectId": uuid_value,
                "@type": "source data set",
                "@uri": build_local_dataset_uri("source", uuid_value, version),
                "@version": version,
                "common:shortDescription": [_lang_entry("ILCD Data Network - Entry-level", "en")],
            },
            "common:approvalOfOverallCompliance": "Fully compliant",
        }
    }


def _dataset_format_reference() -> dict[str, Any]:
    reference = build_dataset_format_reference()
    short_description = reference.get("common:shortDescription")
    if isinstance(short_description, dict):
        reference["common:shortDescription"] = [short_description]
    return reference


@dataclass(slots=True)
class ProductFlowCreateRequest:
    class_id: str
    classification: list[dict[str, str]]
    base_name_en: str
    base_name_zh: str
    treatment_en: str = DEFAULT_TREATMENT_EN
    treatment_zh: str | None = None
    mix_en: str = DEFAULT_MIX_EN
    mix_zh: str | None = None
    comment_en: str | None = None
    comment_zh: str | None = None
    synonyms_en: list[str] = field(default_factory=list)
    synonyms_zh: list[str] = field(default_factory=list)
    flow_type: str = DEFAULT_FLOW_TYPE
    flow_uuid: str | None = None
    version: str = DEFAULT_FLOW_VERSION
    timestamp: str | None = None
    mean_value: str = "1.0"
    flow_property_uuid: str = DEFAULT_MASS_FLOW_PROPERTY_UUID
    flow_property_version: str = DEFAULT_MASS_FLOW_PROPERTY_VERSION
    flow_property_name_en: str = DEFAULT_MASS_FLOW_PROPERTY_NAME


@dataclass(slots=True)
class ProductFlowBuildResult:
    flow_uuid: str
    version: str
    payload: dict[str, Any]
    dataset: dict[str, Any]
    xml: str


class ProductFlowCreationService:
    """Single entrypoint to build + validate product flow payloads."""

    def build(
        self,
        request: ProductFlowCreateRequest,
        *,
        allow_validation_fallback: bool = False,
    ) -> ProductFlowBuildResult:
        if not request.classification:
            raise ValueError(f"classification is required for class_id={request.class_id}")

        base_name_en = _normalize_text(request.base_name_en) or request.class_id
        base_name_zh = _normalize_text(request.base_name_zh) or base_name_en
        treatment_en = _normalize_text(request.treatment_en) or DEFAULT_TREATMENT_EN
        treatment_zh = _normalize_text(request.treatment_zh)
        mix_en = _normalize_text(request.mix_en) or DEFAULT_MIX_EN
        mix_zh = _normalize_text(request.mix_zh)

        comment_en = _normalize_text(request.comment_en) or f"Flow for {base_name_en}"
        comment_zh = _normalize_text(request.comment_zh) or comment_en

        synonyms_en = _normalize_list(request.synonyms_en) or [base_name_en]
        synonyms_zh = _normalize_list(request.synonyms_zh) or [base_name_zh]

        flow_uuid = (request.flow_uuid or "").strip() or str(uuid4())
        version = _normalize_text(request.version) or DEFAULT_FLOW_VERSION
        timestamp_raw = _normalize_text(request.timestamp)
        if timestamp_raw:
            timestamp = _normalize_timestamp(timestamp_raw)
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        flow_type = _normalize_text(request.flow_type) or DEFAULT_FLOW_TYPE

        name_block: dict[str, Any] = {
            "baseName": [
                _lang_entry(base_name_en, "en"),
                _lang_entry(base_name_zh, "zh"),
            ],
            "treatmentStandardsRoutes": [_lang_entry(treatment_en, "en")],
            "mixAndLocationTypes": [_lang_entry(mix_en, "en")],
        }
        if treatment_zh:
            name_block["treatmentStandardsRoutes"].append(_lang_entry(treatment_zh, "zh"))
        if mix_zh:
            name_block["mixAndLocationTypes"].append(_lang_entry(mix_zh, "zh"))

        dataset = {
            "@xmlns": "http://lca.jrc.it/ILCD/Flow",
            "@xmlns:common": "http://lca.jrc.it/ILCD/Common",
            "@xmlns:ecn": "http://eplca.jrc.ec.europa.eu/ILCD/Extensions/2018/ECNumber",
            "@xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "@locations": "../ILCDLocations.xml",
            "@version": "1.1",
            "@xsi:schemaLocation": "http://lca.jrc.it/ILCD/Flow ../../schemas/ILCD_FlowDataSet.xsd",
            "flowInformation": {
                "dataSetInformation": {
                    "common:UUID": flow_uuid,
                    "name": name_block,
                    "common:synonyms": [
                        _lang_entry("; ".join(synonyms_en), "en"),
                        _lang_entry("; ".join(synonyms_zh), "zh"),
                    ],
                    "common:generalComment": [
                        _lang_entry(comment_en, "en"),
                        _lang_entry(comment_zh, "zh"),
                    ],
                    "classificationInformation": {"common:classification": {"common:class": request.classification}},
                },
                "quantitativeReference": {
                    "referenceToReferenceFlowProperty": "0",
                },
            },
            "modellingAndValidation": {
                "LCIMethod": {"typeOfDataSet": flow_type},
                "complianceDeclarations": _compliance_block(),
            },
            "administrativeInformation": {
                "dataEntryBy": {
                    "common:timeStamp": timestamp,
                    "common:referenceToDataSetFormat": _dataset_format_reference(),
                    "common:referenceToPersonOrEntityEnteringTheData": _contact_reference(),
                },
                "publicationAndOwnership": {
                    "common:dataSetVersion": version,
                    "common:referenceToOwnershipOfDataSet": _contact_reference(),
                },
            },
            "flowProperties": {
                "flowProperty": {
                    "@dataSetInternalID": "0",
                    "meanValue": _normalize_text(request.mean_value) or "1.0",
                    "referenceToFlowPropertyDataSet": {
                        "@type": "flow property data set",
                        "@refObjectId": request.flow_property_uuid,
                        "@uri": f"../flowproperties/{request.flow_property_uuid}_{request.flow_property_version}.xml",
                        "@version": request.flow_property_version,
                        "common:shortDescription": [_lang_entry(request.flow_property_name_en, "en")],
                    },
                }
            },
        }

        try:
            entity = create_flow({"flowDataSet": dataset}, validate=True)
        except Exception:  # noqa: BLE001
            if not allow_validation_fallback:
                raise
            entity = create_flow({"flowDataSet": dataset}, validate=False)
        payload = entity.to_json(by_alias=True, exclude_none=True)

        ts_value = payload["flowDataSet"]["administrativeInformation"]["dataEntryBy"]["common:timeStamp"]
        payload["flowDataSet"]["administrativeInformation"]["dataEntryBy"]["common:timeStamp"] = _normalize_timestamp(ts_value)

        flow_uuid = str(payload["flowDataSet"]["flowInformation"]["dataSetInformation"]["common:UUID"]).strip()
        version = str(payload["flowDataSet"]["administrativeInformation"]["publicationAndOwnership"]["common:dataSetVersion"]).strip() or version
        return ProductFlowBuildResult(
            flow_uuid=flow_uuid,
            version=version,
            payload=payload,
            dataset=payload["flowDataSet"],
            xml=entity.to_xml(),
        )
