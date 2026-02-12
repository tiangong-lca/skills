"""Shared data models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


@dataclass(slots=True, frozen=True)
class FlowQuery:
    exchange_name: str
    description: str | None = None


@dataclass(slots=True)
class FlowCandidate:
    uuid: str | None
    base_name: str
    treatment_standards_routes: str | None = None
    mix_and_location_types: str | None = None
    flow_properties: str | None = None
    flow_type: str | None = None
    version: str | None = None
    general_comment: str | None = None
    geography: Mapping[str, Any] | None = None
    classification: list[Mapping[str, Any]] | None = None
    cas: str | None = None
    category_path: str | None = None
    reasoning: str = ""


@dataclass(slots=True)
class UnmatchedFlow:
    base_name: str
    general_comment: str | None = None
    status: Literal["requires_creation"] = "requires_creation"
    process_name: str | None = None


@dataclass(slots=True)
class ProcessDataset:
    process_information: dict[str, Any]
    modelling_and_validation: dict[str, Any]
    administrative_information: dict[str, Any]
    exchanges: list[Mapping[str, Any]] = field(default_factory=list)
    process_data_set: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        from tiangong_lca_spec.process_extraction.tidas_mapping import (
            build_tidas_process_dataset,
        )

        if self.process_data_set:
            base_dataset = deepcopy(self.process_data_set)
        else:
            base_dataset = {}

        base_dataset["processInformation"] = self.process_information
        base_dataset["modellingAndValidation"] = self.modelling_and_validation
        base_dataset["administrativeInformation"] = self.administrative_information
        base_dataset["exchanges"] = {"exchange": [dict(exchange) for exchange in self.exchanges]}

        return build_tidas_process_dataset(base_dataset)


@dataclass(slots=True)
class TidasValidationFinding:
    severity: Literal["info", "warning", "error"]
    message: str
    path: str | None = None
    suggestion: str | None = None


@dataclass(slots=True)
class SettingsProfile:
    concurrency: int
    retry_attempts: int
    cache_results: bool
    profile_name: str


@dataclass(slots=True)
class WorkflowResult:
    process_datasets: list[ProcessDataset]
    alignment: list[dict[str, Any]]
    validation_report: list[TidasValidationFinding]
