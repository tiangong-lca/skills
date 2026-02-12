"""Sequential orchestration of the full Tiangong LCA spec coding pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from tiangong_lca_spec.core.config import Settings, get_settings
from tiangong_lca_spec.core.exceptions import SpecCodingError
from tiangong_lca_spec.core.logging import get_logger
from tiangong_lca_spec.core.models import (
    FlowCandidate,
    ProcessDataset,
    TidasValidationFinding,
    WorkflowResult,
)
from tiangong_lca_spec.flow_alignment import FlowAlignmentService
from tiangong_lca_spec.process_extraction import ProcessExtractionService, preprocess_paper
from tiangong_lca_spec.process_extraction.extractors import LanguageModelProtocol
from tiangong_lca_spec.process_extraction.merge import determine_functional_unit, merge_results
from tiangong_lca_spec.tidas_validation import TidasValidationService

LOGGER = get_logger(__name__)


class WorkflowState(TypedDict, total=False):
    paper_md_json: str
    clean_text: str
    process_blocks: list[dict[str, Any]]
    alignment_results: list[dict[str, Any]]
    matched_lookup: dict[str, list[FlowCandidate]]
    origin_exchanges: dict[str, list[dict[str, Any]]]
    process_datasets: list[ProcessDataset]
    validation_report: list[TidasValidationFinding]
    result: WorkflowResult


class WorkflowOrchestrator:
    """Compose all pipeline stages using straightforward sequential execution."""

    def __init__(
        self,
        llm: LanguageModelProtocol,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._process_extraction = ProcessExtractionService(llm, self._settings)
        self._flow_alignment = FlowAlignmentService(self._settings)
        self._tidas = TidasValidationService(self._settings)

    def run_from_path(self, paper_path: Path | str) -> WorkflowResult:
        path = Path(paper_path)
        paper_md_json = path.read_text(encoding="utf-8")
        return self.run(paper_md_json)

    def run(self, paper_md_json: str) -> WorkflowResult:
        state: WorkflowState = {"paper_md_json": paper_md_json}
        state = self._preprocess(state)
        state = self._extract_processes(state)
        state = self._align_flows(state)
        state = self._merge_datasets(state)
        state = self._validate(state)
        state = self._finalize(state)
        result = state.get("result")
        if not result:
            raise SpecCodingError("Workflow did not produce a result")
        return result

    def close(self) -> None:
        self._flow_alignment.close()
        self._tidas.close()

    def __enter__(self) -> "WorkflowOrchestrator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _preprocess(self, state: WorkflowState) -> WorkflowState:
        paper_md_json = state.get("paper_md_json")
        if not paper_md_json:
            raise SpecCodingError("Missing paper JSON content")
        clean_text = preprocess_paper(paper_md_json)
        state["clean_text"] = clean_text
        return state

    def _extract_processes(self, state: WorkflowState) -> WorkflowState:
        clean_text = state.get("clean_text")
        if not clean_text:
            raise SpecCodingError("Preprocessing did not yield clean text")
        blocks = self._process_extraction.extract(clean_text)
        state["process_blocks"] = blocks
        return state

    def _align_flows(self, state: WorkflowState) -> WorkflowState:
        blocks = state.get("process_blocks") or []
        clean_text = state.get("clean_text")
        alignment_results: list[dict[str, Any]] = []
        matched_lookup: dict[str, list[FlowCandidate]] = {}
        origin_exchanges: dict[str, list[dict[str, Any]]] = {}
        for block in blocks:
            process_dataset = block.get("processDataSet", {})
            alignment = self._flow_alignment.align_exchanges(process_dataset, clean_text)
            alignment_results.append(alignment)
            process_name = alignment.get("process_name") or _extract_process_name_from_block(block)
            matched_lookup[process_name] = alignment.get("matched_flows", [])
            origin = []
            for exchange_list in alignment.get("origin_exchanges", {}).values():
                origin.extend(exchange_list)
            origin_exchanges[process_name] = origin
        state["alignment_results"] = alignment_results
        state["matched_lookup"] = matched_lookup
        state["origin_exchanges"] = origin_exchanges
        return state

    def _merge_datasets(self, state: WorkflowState) -> WorkflowState:
        process_blocks = state.get("process_blocks") or []
        matched_lookup = state.get("matched_lookup") or {}
        origin_exchanges = state.get("origin_exchanges") or {}
        datasets = merge_results(process_blocks, matched_lookup, origin_exchanges)
        for dataset in datasets:
            fu = determine_functional_unit(dataset.exchanges)
            if fu:
                info = dict(dataset.process_information)
                info.setdefault("processes", {})["functionalUnit"] = fu
                dataset.process_information = info
        state["process_datasets"] = datasets
        return state

    def _validate(self, state: WorkflowState) -> WorkflowState:
        datasets = state.get("process_datasets") or []
        findings = self._tidas.validate(datasets)
        state["validation_report"] = findings
        return state

    def _finalize(self, state: WorkflowState) -> WorkflowState:
        result = WorkflowResult(
            process_datasets=state.get("process_datasets") or [],
            alignment=state.get("alignment_results") or [],
            validation_report=state.get("validation_report") or [],
        )
        state["result"] = result
        return state


def _extract_process_name_from_block(block: dict[str, Any]) -> str:
    if "processDataSet" in block and isinstance(block["processDataSet"], dict):
        process_info = block["processDataSet"].get("processInformation", {})
    else:
        process_info = block.get("process_information", {})
    data_info = process_info.get("dataSetInformation", {})
    name_block = data_info.get("name")
    resolved = _resolve_base_name(name_block)
    if resolved:
        return resolved
    return block.get("process_name") or "unknown_process"


def _resolve_base_name(name_block: Any) -> str | None:
    if isinstance(name_block, dict):
        base = name_block.get("baseName")
        if isinstance(base, dict):
            text = base.get("#text") or base.get("text")
            if text:
                return text
            for value in base.values():
                if isinstance(value, str):
                    return value
        elif base:
            return str(base)
        text = name_block.get("#text") or name_block.get("text")
        if text:
            return str(text)
        for value in name_block.values():
            if isinstance(value, str):
                return value
    elif isinstance(name_block, list) and name_block:
        return _resolve_base_name(name_block[0])
    elif isinstance(name_block, str):
        return name_block
    return None
