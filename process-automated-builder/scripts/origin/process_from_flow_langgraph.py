#!/usr/bin/env python
"""Build ILCD process dataset(s) from a reference flow using LangGraph.

This command is a human-in-the-loop helper:
- Step 1: list plausible technology/process routes from the reference flow.
- Step 2: for each route, split into 1..N unit processes (ordered; last process produces/treats the reference flow;
  structured fields include inputs/outputs, exchange keywords, and standardized name_parts with quantitative_reference).
- Step 3: derive per-process input/output exchanges.
- Step 4: match exchanges to Tiangong flows via MCP flow_search.
- Step 5: generate TIDAS/ILCD process datasets via tidas-sdk.

Usage:
  See references/process-from-flow-workflow.md for workflow details.

Outputs (by default):
  - artifacts/process_from_flow/<run_id>/exports/processes

Manual cleanup (keep latest 3 runs):
  uv run python scripts/origin/process_from_flow_langgraph.py --cleanup-only --retain-runs 3

Publish latest run (commit to DB):
  uv run python scripts/origin/process_from_flow_langgraph.py --publish-only --commit

Checkpoint flow (edit cache JSON, then resume):
  uv run python scripts/origin/process_from_flow_langgraph.py --stop-after exchanges
  # edit artifacts/process_from_flow/<run_id>/cache/process_from_flow_state.json
  uv run python scripts/origin/process_from_flow_langgraph.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent
for path in (SCRIPTS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

try:
    from scripts.md._workflow_common import (  # type: ignore
        OpenAIResponsesLLM,
        dump_json,
        generate_run_id,
        load_openai_from_env,
    )
except ModuleNotFoundError:  # pragma: no cover
    from _workflow_common import (  # type: ignore
        OpenAIResponsesLLM,
        dump_json,
        generate_run_id,
        load_openai_from_env,
    )

PROCESS_FROM_FLOW_ARTIFACTS_ROOT = Path("artifacts/process_from_flow")
LATEST_RUN_ID_PATH = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / ".latest_run_id"
DATABASE_TOOL_NAME = "Database_CRUD_Tool"

_RUN_ID_SAFE_TOKEN_PATTERN = re.compile(r"[^0-9A-Za-z._-]+")


def _run_id_token(value: str, fallback: str) -> str:
    token = _RUN_ID_SAFE_TOKEN_PATTERN.sub("-", str(value or "").strip()).strip("-._")
    return token or fallback


def build_process_from_flow_run_id(flow_path: Path, operation: str = "produce") -> str:
    """Build a stable run-id pattern for process_from_flow runs.

    Pattern:
      pfw_<flow_code>_<flow_uuid8>_<operation>_<UTC_TIMESTAMP>
    Example:
      pfw_01211_3a8d74d8_produce_20260211T184500Z
    """

    stem = flow_path.stem
    parts = [part for part in stem.split("_") if str(part).strip()]
    flow_code = _run_id_token(parts[0] if parts else "", "flow")
    flow_uuid_short = _run_id_token(parts[1] if len(parts) > 1 else "", "unknown")[:8]
    operation_token = "treat" if str(operation or "").strip().lower() == "treat" else "produce"
    return f"pfw_{flow_code}_{flow_uuid_short}_{operation_token}_{generate_run_id()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flow",
        type=Path,
        help="Path to the reference flow JSON (ILCD flowDataSet wrapper). Required for new runs.",
    )
    parser.add_argument("--operation", choices=("produce", "treat"), default="produce", help="Whether the process produces or treats/disposes the reference flow.")
    parser.add_argument(
        "--run-id",
        help="Run identifier under artifacts/process_from_flow/<run_id>. Defaults to a new id when not resuming.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from artifacts/process_from_flow/<run_id>/cache/process_from_flow_state.json.")
    parser.add_argument(
        "--stop-after",
        choices=("references", "tech", "processes", "exchanges", "matches", "sources", "datasets"),
        help="Stop after a stage, writing state to cache for manual editing.",
    )
    parser.add_argument("--no-llm", action="store_true", help="Run without an LLM (uses minimal deterministic fallbacks).")
    parser.add_argument("--no-translate-zh", action="store_true", help="Skip adding Chinese translations to multi-language fields.")
    parser.add_argument(
        "--allow-density-conversion",
        action="store_true",
        help="Allow LLM-based density estimates for mass<->volume conversions (product/waste flows only).",
    )
    parser.add_argument(
        "--auto-balance-revise",
        action="store_true",
        help=("After the first balance review, auto-revise severe core-mass imbalances " "on non-reference exchanges, then recompute balance review."),
    )
    parser.add_argument(
        "--retain-runs",
        type=int,
        help="Manually clean process_from_flow run directories, keeping only the most recent N runs under artifacts/process_from_flow/.",
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Only perform cleanup (requires --retain-runs), skip running the pipeline.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish generated process datasets via Database_CRUD_Tool after the pipeline completes.",
    )
    parser.add_argument(
        "--publish-only",
        action="store_true",
        help="Publish process datasets from an existing run and skip the pipeline.",
    )
    parser.add_argument(
        "--publish-flows",
        action="store_true",
        help="Also publish placeholder flow datasets, rewrite process references, and export flow JSONs (disabled by default).",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Actually invoke Database_CRUD_Tool (default: dry-run).",
    )
    parser.add_argument(
        "--skip-balance-check",
        action="store_true",
        help="Skip balance quality check entirely before publish (no warning).",
    )
    parser.add_argument(
        "--strict-balance-check",
        action="store_true",
        help="Treat balance quality issues as blocking errors during publish.",
    )
    return parser.parse_args()


def _load_state(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"State file must contain an object: {path}")
    return payload


def _extract_process_uuid(process_payload: dict[str, Any]) -> str:
    dataset = process_payload.get("processDataSet") if isinstance(process_payload.get("processDataSet"), dict) else {}
    info = dataset.get("processInformation") if isinstance(dataset.get("processInformation"), dict) else {}
    data_info = info.get("dataSetInformation") if isinstance(info.get("dataSetInformation"), dict) else {}
    uuid_value = data_info.get("common:UUID")
    if isinstance(uuid_value, str) and uuid_value.strip():
        return uuid_value.strip()
    raise SystemExit("Generated process payload missing processInformation.dataSetInformation.common:UUID")


def _extract_process_version(process_payload: dict[str, Any]) -> str:
    dataset = process_payload.get("processDataSet") if isinstance(process_payload.get("processDataSet"), dict) else {}
    admin = dataset.get("administrativeInformation") if isinstance(dataset.get("administrativeInformation"), dict) else {}
    pub = admin.get("publicationAndOwnership") if isinstance(admin.get("publicationAndOwnership"), dict) else {}
    version = pub.get("common:dataSetVersion")
    if isinstance(version, str) and version.strip():
        return version.strip()
    return "01.01.000"


def _extract_source_uuid(source_payload: dict[str, Any]) -> str:
    dataset = source_payload.get("sourceDataSet") if isinstance(source_payload.get("sourceDataSet"), dict) else {}
    info = dataset.get("sourceInformation") if isinstance(dataset.get("sourceInformation"), dict) else {}
    data_info = info.get("dataSetInformation") if isinstance(info.get("dataSetInformation"), dict) else {}
    uuid_value = data_info.get("common:UUID")
    if isinstance(uuid_value, str) and uuid_value.strip():
        return uuid_value.strip()
    raise SystemExit("Generated source payload missing sourceInformation.dataSetInformation.common:UUID")


def _extract_source_version(source_payload: dict[str, Any]) -> str:
    dataset = source_payload.get("sourceDataSet") if isinstance(source_payload.get("sourceDataSet"), dict) else {}
    admin = dataset.get("administrativeInformation") if isinstance(dataset.get("administrativeInformation"), dict) else {}
    pub = admin.get("publicationAndOwnership") if isinstance(admin.get("publicationAndOwnership"), dict) else {}
    version = pub.get("common:dataSetVersion")
    if isinstance(version, str) and version.strip():
        return version.strip()
    return "01.01.000"


def _resolve_flow_dataset_version(dataset: Mapping[str, Any]) -> str:
    admin = dataset.get("administrativeInformation") if isinstance(dataset.get("administrativeInformation"), Mapping) else {}
    pub = admin.get("publicationAndOwnership") if isinstance(admin.get("publicationAndOwnership"), Mapping) else {}
    version = pub.get("common:dataSetVersion")
    if isinstance(version, str) and version.strip():
        return version.strip()
    return "01.01.000"


def _build_export_filename(uuid_value: str, dataset_version: str) -> str:
    safe_uuid = (uuid_value or "").strip()
    if not safe_uuid:
        raise ValueError("UUID required to build export filename.")
    version = (dataset_version or "").strip() or "01.01.000"
    safe_version = re.sub(r"[^0-9A-Za-z._-]", "_", version)
    if not safe_version:
        safe_version = "01.01.000"
    return f"{safe_uuid}_{safe_version}.json"


def _collect_source_ref_ids(refs: Any, used: set[str]) -> None:
    if not refs:
        return
    if isinstance(refs, Mapping):
        refs_iter = [refs]
    elif isinstance(refs, list):
        refs_iter = refs
    else:
        return
    for ref in refs_iter:
        if not isinstance(ref, Mapping):
            continue
        ref_id = ref.get("@refObjectId")
        if not isinstance(ref_id, str):
            continue
        ref_id = ref_id.strip()
        if not ref_id:
            continue
        ref_type = str(ref.get("@type") or "").strip().lower()
        if ref_type and ref_type != "source data set":
            continue
        used.add(ref_id)


def _collect_used_source_ids(process_datasets: list[dict[str, Any]]) -> set[str]:
    used: set[str] = set()
    for payload in process_datasets:
        dataset = payload.get("processDataSet") if isinstance(payload.get("processDataSet"), Mapping) else payload
        if not isinstance(dataset, Mapping):
            continue
        modelling = dataset.get("modellingAndValidation")
        if isinstance(modelling, Mapping):
            sources = modelling.get("dataSourcesTreatmentAndRepresentativeness")
            if isinstance(sources, Mapping):
                _collect_source_ref_ids(sources.get("referenceToDataSource"), used)
        exchanges_block = dataset.get("exchanges")
        if not isinstance(exchanges_block, Mapping):
            continue
        exchanges = exchanges_block.get("exchange", [])
        if isinstance(exchanges, Mapping):
            exchanges = [exchanges]
        if not isinstance(exchanges, list):
            continue
        for exchange in exchanges:
            if not isinstance(exchange, Mapping):
                continue
            refs_block = exchange.get("referencesToDataSource")
            if isinstance(refs_block, Mapping):
                _collect_source_ref_ids(refs_block.get("referenceToDataSource"), used)
    return used


def _filter_source_datasets_by_usage(
    source_datasets: list[dict[str, Any]],
    process_datasets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    used_ids = _collect_used_source_ids(process_datasets)
    if not used_ids:
        return []
    filtered: list[dict[str, Any]] = []
    for payload in source_datasets:
        if not isinstance(payload, dict):
            continue
        uuid_value = _extract_source_uuid(payload)
        if uuid_value in used_ids:
            filtered.append(payload)
    if len(filtered) != len(source_datasets):
        print(
            f"Filtered source datasets by usage: {len(filtered)}/{len(source_datasets)}",
            file=sys.stderr,
        )
    return filtered


def _ensure_run_root(run_id: str) -> Path:
    run_root = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def ensure_run_cache_dir(run_id: str) -> Path:
    run_root = _ensure_run_root(run_id)
    cache_dir = run_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def ensure_run_exports_dir(run_id: str, *, clean: bool = False) -> Path:
    run_root = _ensure_run_root(run_id)
    export_root = run_root / "exports"
    if clean and export_root.exists():
        shutil.rmtree(export_root)
    for name in ("processes", "flows", "sources"):
        (export_root / name).mkdir(parents=True, exist_ok=True)
    return export_root


def _ensure_run_input_dir(run_id: str) -> Path:
    run_root = _ensure_run_root(run_id)
    input_dir = run_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    return input_dir


def _find_runs(base_dir: Path, marker: Path) -> list[Path]:
    runs: list[Path] = []
    if not base_dir.exists():
        return runs
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        if (entry / marker).exists():
            runs.append(entry)
    return runs


def _parse_run_id(run_id: str) -> datetime | None:
    try:
        return datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _cleanup_runs(*, retain: int, current_run_id: str | None = None) -> None:
    if retain <= 0:
        raise SystemExit("--retain-runs must be >= 1")

    artifacts_root = PROCESS_FROM_FLOW_ARTIFACTS_ROOT
    artifacts_marker = Path("cache/process_from_flow_state.json")

    artifacts_runs = _find_runs(artifacts_root, artifacts_marker)
    artifacts_index = {path.name: path for path in artifacts_runs}
    all_run_ids = set(artifacts_index)
    if not all_run_ids:
        print("No process_from_flow runs found for cleanup.", file=sys.stderr)
        return

    def _sort_key(run_id: str) -> tuple[int, datetime]:
        parsed = _parse_run_id(run_id)
        if parsed is not None:
            return (0, parsed)
        entry = artifacts_index.get(run_id)
        if entry is None:
            return (2, datetime.min.replace(tzinfo=timezone.utc))
        try:
            ts = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc)
        except OSError:
            ts = datetime.min.replace(tzinfo=timezone.utc)
        return (1, ts)

    sorted_run_ids = sorted(all_run_ids, key=_sort_key, reverse=True)
    keep = set(sorted_run_ids[:retain])
    if current_run_id:
        keep.add(current_run_id)

    def _safe_remove(base_dir: Path, run_id: str) -> bool:
        target = base_dir / run_id
        try:
            if target.exists() and target.is_dir() and target.resolve().parent == base_dir.resolve():
                shutil.rmtree(target)
                return True
        except OSError as exc:
            print(f"Failed to remove {target}: {exc}", file=sys.stderr)
        return False

    removed = 0
    for run_id in sorted_run_ids:
        if run_id in keep:
            continue
        if run_id in artifacts_index and _safe_remove(artifacts_root, run_id):
            removed += 1

    print(
        f"Cleanup complete: kept {len(keep)} run(s), removed {removed} directory(s).",
        file=sys.stderr,
    )


def _resolve_run_id(run_id: str | None) -> str:
    if run_id:
        return run_id
    if LATEST_RUN_ID_PATH.exists():
        latest = LATEST_RUN_ID_PATH.read_text(encoding="utf-8").strip()
        if latest:
            return latest
    raise SystemExit("Missing --run-id and no latest run marker found in artifacts/process_from_flow.")


def _load_process_datasets(run_id: str) -> list[dict[str, Any]]:
    process_dir = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id / "exports" / "processes"
    if not process_dir.exists():
        raise SystemExit(f"Process output directory not found: {process_dir}")
    datasets: list[dict[str, Any]] = []
    for path in sorted(process_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            datasets.append(payload)
    if not datasets:
        raise SystemExit(f"No process datasets found under {process_dir}")
    return datasets


def _load_source_datasets(run_id: str) -> list[dict[str, Any]]:
    source_dir = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id / "exports" / "sources"
    if not source_dir.exists():
        return []
    datasets: list[dict[str, Any]] = []
    for path in sorted(source_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            datasets.append(payload)
    return datasets


def _load_run_state(run_id: str) -> dict[str, Any]:
    state_path = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id / "cache" / "process_from_flow_state.json"
    if not state_path.exists():
        raise SystemExit(f"State file not found for run {run_id}: {state_path}")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"State payload is not an object: {state_path}")
    return payload


def _enforce_balance_quality_gate(run_id: str, *, strict: bool = False) -> None:
    state = _load_run_state(run_id)
    review = state.get("balance_review")
    if not isinstance(review, list) or not review:
        message = (
            "Balance quality check unavailable: missing balance_review in run state. "
            "Publishing will continue because strict mode is disabled."
        )
        if strict:
            raise SystemExit("Balance quality gate failed: missing balance_review in run state. Re-run the workflow to generate balance review before publishing.")
        print(f"[warn] {message}", file=sys.stderr)
        return

    failures: list[str] = []
    for entry in review:
        if not isinstance(entry, Mapping):
            continue
        process_id = str(entry.get("process_id") or "").strip() or "unknown"
        process_name = str(entry.get("process_name") or "").strip() or process_id
        core_exchange_count = int(entry.get("core_exchange_count") or 0)
        mass_core = entry.get("mass_core") if isinstance(entry.get("mass_core"), Mapping) else None
        if mass_core is None and isinstance(entry.get("mass"), Mapping):
            mass_core = entry.get("mass")
        core_status = str((mass_core or {}).get("status") or "").strip().lower()
        core_ratio = (mass_core or {}).get("ratio")
        unit_mismatch_count = int(entry.get("unit_mismatch_count") or 0)
        mapping_conflict_count = int(entry.get("mapping_conflict_count") or 0)
        overall_status = str(entry.get("status") or "").strip().lower()

        reasons: list[str] = []
        if core_exchange_count > 0 and core_status != "ok":
            reasons.append(f"core_mass_status={core_status or 'unknown'} ratio={core_ratio}")
        if unit_mismatch_count > 0:
            reasons.append(f"unit_mismatch_count={unit_mismatch_count}")
        if mapping_conflict_count > 0:
            reasons.append(f"mapping_conflict_count={mapping_conflict_count}")
        # Keep existing status as an additional signal for backward compatibility.
        if overall_status in {"check", "insufficient"}:
            reasons.append(f"overall_status={overall_status}")

        if reasons:
            failures.append(f"{process_id} ({process_name}): " + ", ".join(reasons))

    if failures:
        preview = "\n".join(f"- {item}" for item in failures[:10])
        extra = ""
        if len(failures) > 10:
            extra = f"\n... and {len(failures) - 10} more process(es)."
        message = (
            "Balance quality gate found issues.\n"
            f"run_id={run_id}\n{preview}{extra}"
        )
        if strict:
            raise SystemExit(
                f"{message}\n"
                "Strict mode is enabled. Fix exchanges or rerun publish without --strict-balance-check."
            )
        print(
            "[warn] Balance quality gate found issues; continuing publish because strict mode is disabled.\n"
            f"{message}",
            file=sys.stderr,
        )
        return
    print(f"Balance quality gate passed for run {run_id}.", file=sys.stderr)


def _extract_lang_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        text = value.get("#text")
        if isinstance(text, str):
            return text.strip()
    if isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping) and item.get("@xml:lang") == "en":
                text = item.get("#text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        for item in value:
            text = _extract_lang_text(item)
            if text:
                return text
    return ""


def _compose_process_name(process_payload: Mapping[str, Any]) -> str:
    info = process_payload.get("processInformation", {})
    if not isinstance(info, Mapping):
        return "Unknown process"
    data_info = info.get("dataSetInformation", {})
    if not isinstance(data_info, Mapping):
        return "Unknown process"
    name_block = data_info.get("name", {})
    if not isinstance(name_block, Mapping):
        name_block = {}
    parts: list[str] = []
    for key in ("baseName", "treatmentStandardsRoutes", "mixAndLocationTypes", "functionalUnitFlowProperties"):
        text = _extract_lang_text(name_block.get(key))
        if text:
            parts.append(text)
    if parts:
        return " | ".join(parts)
    fallback = _extract_lang_text(data_info.get("common:generalComment"))
    return fallback or "Unknown process"


def _build_flow_alignment_from_process_datasets(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alignment: list[dict[str, Any]] = []
    for payload in datasets:
        process_payload = payload.get("processDataSet") if isinstance(payload.get("processDataSet"), Mapping) else payload
        if not isinstance(process_payload, Mapping):
            continue
        process_name = _compose_process_name(process_payload)
        exchanges_block = process_payload.get("exchanges", {})
        if not isinstance(exchanges_block, Mapping):
            continue
        exchanges = exchanges_block.get("exchange", [])
        if isinstance(exchanges, Mapping):
            exchanges = [exchanges]
        if not isinstance(exchanges, list):
            continue
        placeholder_exchanges: list[dict[str, Any]] = []
        for exchange in exchanges:
            if not isinstance(exchange, Mapping):
                continue
            ref = exchange.get("referenceToFlowDataSet")
            if not isinstance(ref, Mapping) or not ref.get("unmatched:placeholder"):
                continue
            exchange_name = _extract_lang_text(exchange.get("exchangeName"))
            if not exchange_name:
                exchange_name = _extract_lang_text(ref.get("common:shortDescription"))
            if not exchange_name:
                exchange_name = "Unnamed exchange"
            comment = _extract_lang_text(exchange.get("generalComment"))
            sanitized_ref = dict(ref)
            sanitized_ref.pop("@version", None)
            sanitized_ref.pop("@uri", None)
            placeholder_exchanges.append(
                {
                    "exchangeName": exchange_name,
                    "exchangeDirection": exchange.get("exchangeDirection"),
                    "unit": exchange.get("unit"),
                    "meanAmount": exchange.get("meanAmount") if exchange.get("meanAmount") is not None else exchange.get("resultingAmount", exchange.get("amount")),
                    "generalComment": comment,
                    "referenceToFlowDataSet": sanitized_ref,
                }
            )
        if placeholder_exchanges:
            alignment.append({"process_name": process_name, "origin_exchanges": {"placeholders": placeholder_exchanges}})
    return alignment


def _apply_flow_refs_to_processes(datasets: list[dict[str, Any]], plans: list[Any]) -> int:
    mapping: dict[str, dict[str, Any]] = {}
    for plan in plans:
        uuid_value = getattr(plan, "uuid", None)
        exchange_ref = getattr(plan, "exchange_ref", None)
        if isinstance(uuid_value, str) and uuid_value and isinstance(exchange_ref, Mapping):
            mapping[uuid_value] = dict(exchange_ref)
    if not mapping:
        return 0

    updated = 0
    for payload in datasets:
        process_payload = payload.get("processDataSet") if isinstance(payload.get("processDataSet"), Mapping) else payload
        if not isinstance(process_payload, Mapping):
            continue
        exchanges_block = process_payload.get("exchanges", {})
        if not isinstance(exchanges_block, Mapping):
            continue
        exchanges = exchanges_block.get("exchange", [])
        if isinstance(exchanges, Mapping):
            exchanges = [exchanges]
        if not isinstance(exchanges, list):
            continue
        for exchange in exchanges:
            if not isinstance(exchange, Mapping):
                continue
            ref = exchange.get("referenceToFlowDataSet")
            if not isinstance(ref, Mapping):
                continue
            uuid_value = ref.get("@refObjectId")
            if isinstance(uuid_value, str) and uuid_value in mapping:
                exchange["referenceToFlowDataSet"] = mapping[uuid_value]
                updated += 1
    return updated


def _publish_flows(
    datasets: list[dict[str, Any]],
    *,
    commit: bool,
    llm: Any | None = None,
    exports_dir: Path | None = None,
) -> list[Any]:
    from tiangong_lca_spec.publishing import FlowPublisher

    alignment = _build_flow_alignment_from_process_datasets(datasets)
    if not alignment:
        return []
    publisher = FlowPublisher(dry_run=not commit, llm=llm)
    try:
        plans = publisher.prepare_from_alignment(alignment)
        if not plans:
            return []
        if exports_dir is not None:
            _write_flow_exports(plans, exports_dir)
        try:
            publisher.publish()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Flow publish encountered errors but will continue: {exc}", file=sys.stderr)
        if commit:
            print(f"Attempted publish for {len(plans)} flow dataset(s) via Database_CRUD_Tool.", file=sys.stderr)
        else:
            print(f"Dry-run: prepared {len(plans)} flow dataset(s) for publish.", file=sys.stderr)
        return plans
    finally:
        publisher.close()


def _write_flow_exports(plans: list[Any], exports_dir: Path) -> list[Path]:
    written: list[Path] = []
    flows_dir = exports_dir / "flows"
    flows_dir.mkdir(parents=True, exist_ok=True)
    for plan in plans:
        dataset = getattr(plan, "dataset", None)
        uuid_value = getattr(plan, "uuid", None)
        if not isinstance(dataset, Mapping):
            continue
        if not isinstance(uuid_value, str) or not uuid_value.strip():
            continue
        dataset_version = _resolve_flow_dataset_version(dataset)
        filename = _build_export_filename(uuid_value, dataset_version)
        target = flows_dir / filename
        dump_json({"flowDataSet": dataset}, target)
        written.append(target)
    if written:
        print(f"Wrote {len(written)} flow dataset(s) to {flows_dir}", file=sys.stderr)
    return written


def _write_process_exports(datasets: list[dict[str, Any]], exports_dir: Path) -> list[Path]:
    written: list[Path] = []
    for payload in datasets:
        if not isinstance(payload, dict):
            continue
        uuid_value = _extract_process_uuid(payload)
        version = _extract_process_version(payload)
        filename = f"{uuid_value}_{version}.json"
        target = exports_dir / "processes" / filename
        dump_json(payload, target)
        written.append(target)
    return written


def _collect_referenced_flow_refs(process_datasets: list[dict[str, Any]]) -> list[tuple[str, str | None]]:
    refs: dict[tuple[str, str | None], None] = {}
    for payload in process_datasets:
        process_payload = payload.get("processDataSet") if isinstance(payload.get("processDataSet"), Mapping) else payload
        if not isinstance(process_payload, Mapping):
            continue
        exchanges_block = process_payload.get("exchanges")
        if not isinstance(exchanges_block, Mapping):
            continue
        exchanges = exchanges_block.get("exchange", [])
        if isinstance(exchanges, Mapping):
            exchanges = [exchanges]
        if not isinstance(exchanges, list):
            continue
        for exchange in exchanges:
            if not isinstance(exchange, Mapping):
                continue
            ref = exchange.get("referenceToFlowDataSet")
            if not isinstance(ref, Mapping):
                continue
            if ref.get("unmatched:placeholder"):
                continue
            flow_uuid = str(ref.get("@refObjectId") or "").strip()
            if not flow_uuid:
                continue
            version_text = str(ref.get("@version") or "").strip() or None
            refs[(flow_uuid, version_text)] = None
    return sorted(refs.keys())


def _export_referenced_flows_from_processes(
    process_datasets: list[dict[str, Any]],
    exports_dir: Path,
) -> list[Path]:
    refs = _collect_referenced_flow_refs(process_datasets)
    if not refs:
        return []

    from tiangong_lca_spec.publishing.crud import DatabaseCrudClient

    flows_dir = exports_dir / "flows"
    flows_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    failed: list[tuple[str, str | None, str]] = []

    client = DatabaseCrudClient()
    try:
        for flow_uuid, version in refs:
            try:
                dataset = client.select_flow(flow_uuid, version=version)
            except Exception as exc:  # noqa: BLE001
                failed.append((flow_uuid, version, str(exc)))
                continue
            if not isinstance(dataset, Mapping):
                failed.append((flow_uuid, version, "Flow payload not found"))
                continue

            dataset_version = _resolve_flow_dataset_version(dataset)
            if not dataset_version and version:
                dataset_version = version
            filename = _build_export_filename(flow_uuid, dataset_version)
            target = flows_dir / filename
            dump_json({"flowDataSet": dict(dataset)}, target)
            written.append(target)
    finally:
        client.close()

    if written:
        print(f"Fetched {len(written)} referenced flow dataset(s) via CRUD into {flows_dir}", file=sys.stderr)
    if failed:
        preview = "; ".join(f"{uuid}@{ver or '*'}: {error}" for uuid, ver, error in failed[:5])
        print(
            f"[warn] Failed to fetch {len(failed)} referenced flow dataset(s) via CRUD. {preview}",
            file=sys.stderr,
        )
    return written


def _publish_sources(
    datasets: list[dict[str, Any]],
    *,
    commit: bool,
    process_datasets: list[dict[str, Any]] | None = None,
) -> None:
    publishable = [item for item in datasets if isinstance(item, dict)]
    if process_datasets is not None:
        publishable = _filter_source_datasets_by_usage(publishable, process_datasets)
    if not publishable:
        return
    if not commit:
        print(f"Dry-run: prepared {len(publishable)} source dataset(s) for publish.", file=sys.stderr)
        return

    from tiangong_lca_spec.publishing.crud import DatabaseCrudClient

    published_count = 0
    reused_count = 0
    failed: list[tuple[str, str, str]] = []
    client = DatabaseCrudClient()
    try:
        for payload in publishable:
            uuid_value = _extract_source_uuid(payload)
            version = _extract_source_version(payload)
            existing = None
            try:
                existing = client.select_source(uuid_value, version=version) or client.select_source(uuid_value)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[warn] Source pre-check select failed, fallback to upsert attempts: {uuid_value} ({exc})",
                    file=sys.stderr,
                )
            try:
                if isinstance(existing, Mapping):
                    client.update_source(payload)
                else:
                    client.insert_source(payload)
                published_count += 1
            except Exception as primary_exc:  # noqa: BLE001
                try:
                    if isinstance(existing, Mapping):
                        client.insert_source(payload)
                    else:
                        client.update_source(payload)
                    published_count += 1
                except Exception as secondary_exc:  # noqa: BLE001
                    still_exists = existing
                    try:
                        if still_exists is None:
                            still_exists = client.select_source(uuid_value, version=version) or client.select_source(uuid_value)
                    except Exception:
                        still_exists = existing
                    if isinstance(still_exists, Mapping):
                        reused_count += 1
                        print(
                            f"[warn] Reusing existing source after publish conflict: {uuid_value} "
                            f"(primary_error={primary_exc}; secondary_error={secondary_exc})",
                            file=sys.stderr,
                        )
                        continue
                    failed.append((uuid_value, str(primary_exc), str(secondary_exc)))
    finally:
        client.close()

    if failed:
        details = "; ".join(f"{uuid_value} insert failed: {insert_error}; update failed: {update_error}" for uuid_value, insert_error, update_error in failed)
        print(
            f"[warn] Failed to publish {len(failed)} source dataset(s), continuing publish. details={details}",
            file=sys.stderr,
        )
    print(
        f"Published {published_count} source dataset(s) via Database_CRUD_Tool (reused_existing={reused_count}).",
        file=sys.stderr,
    )


def _publish_processes(datasets: list[dict[str, Any]], *, commit: bool) -> None:
    from tiangong_lca_spec.publishing import ProcessPublisher

    publishable = [item for item in datasets if isinstance(item, dict)]
    if not publishable:
        raise SystemExit("No valid process datasets found for publishing.")
    publisher = ProcessPublisher(dry_run=not commit)
    try:
        success = 0
        failed: list[tuple[str, str]] = []
        for payload in publishable:
            uuid_value = "unknown"
            try:
                uuid_value = _extract_process_uuid(payload)
                results = publisher.publish([payload])
                if commit:
                    success += len(results)
            except Exception as exc:  # noqa: BLE001
                failed.append((uuid_value, str(exc)))
        if failed:
            details = "; ".join(f"{uuid_value}: {error}" for uuid_value, error in failed)
            print(
                f"[warn] Failed to publish {len(failed)} process dataset(s), continuing publish. details={details}",
                file=sys.stderr,
            )
        if commit:
            print(f"Published {success} process dataset(s) via Database_CRUD_Tool.", file=sys.stderr)
        else:
            print(f"Dry-run: prepared {len(publishable)} process dataset(s) for publish.", file=sys.stderr)
    finally:
        publisher.close()


def main() -> None:
    args = parse_args()
    if args.cleanup_only:
        if args.retain_runs is None:
            raise SystemExit("--cleanup-only requires --retain-runs")
        _cleanup_runs(retain=args.retain_runs)
        return
    if args.publish_only:
        run_id = _resolve_run_id(args.run_id)
        cache_dir = ensure_run_cache_dir(run_id)
        os.environ["TIANGONG_PFF_RUN_ID"] = run_id
        os.environ["TIANGONG_PFF_STATE_PATH"] = str(cache_dir / "process_from_flow_state.json")
        os.environ.setdefault("TIANGONG_PFF_FLOW_CACHE_PATH", str(cache_dir / "flow_select_cache.json"))
        exports_dir = ensure_run_exports_dir(run_id)
        if not args.skip_balance_check:
            _enforce_balance_quality_gate(run_id, strict=args.strict_balance_check)
        llm = None
        if args.publish_flows and not args.no_llm:
            api_key, model, base_url = load_openai_from_env()
            llm = OpenAIResponsesLLM(api_key=api_key, model=model, base_url=base_url)
        source_datasets = _load_source_datasets(run_id)
        datasets = _load_process_datasets(run_id)
        _export_referenced_flows_from_processes(datasets, exports_dir)
        if args.publish_flows:
            flow_plans = _publish_flows(datasets, commit=args.commit, llm=llm, exports_dir=exports_dir)
            if args.commit and flow_plans:
                updated = _apply_flow_refs_to_processes(datasets, flow_plans)
                if updated:
                    _write_process_exports(datasets, exports_dir)
                    _export_referenced_flows_from_processes(datasets, exports_dir)
        if source_datasets:
            _publish_sources(source_datasets, commit=args.commit, process_datasets=datasets)
        _publish_processes(datasets, commit=args.commit)
        return

    if args.resume:
        if args.run_id:
            run_id = args.run_id
        elif LATEST_RUN_ID_PATH.exists():
            run_id = LATEST_RUN_ID_PATH.read_text(encoding="utf-8").strip()
        else:
            raise SystemExit("Missing --run-id and no cached run marker found for --resume.")
    else:
        if args.flow is None:
            raise SystemExit("Missing --flow for a new run. Pass an agent-provided reference flow JSON path.")
        run_id = args.run_id or build_process_from_flow_run_id(args.flow, args.operation)

    cache_dir = ensure_run_cache_dir(run_id)
    exports_dir = ensure_run_exports_dir(run_id)
    state_path = cache_dir / "process_from_flow_state.json"
    snapshot_dir = cache_dir / "mcp_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TIANGONG_PFF_MCP_SNAPSHOT_DIR"] = str(snapshot_dir)
    os.environ["TIANGONG_PFF_RUN_ID"] = run_id
    os.environ["TIANGONG_PFF_STATE_PATH"] = str(state_path)
    initial_state: dict[str, Any] | None = None
    flow_path: Path
    if args.resume:
        if not state_path.exists():
            raise SystemExit(f"Missing cached state file for --resume: {state_path}")
        initial_state = _load_state(state_path)
        if args.flow is not None:
            flow_path = args.flow
        else:
            cached_flow_path = str(initial_state.get("flow_path") or "").strip()
            if not cached_flow_path:
                raise SystemExit("Missing --flow and cached state has no flow_path. Provide --flow explicitly for --resume.")
            flow_path = Path(cached_flow_path)
    else:
        flow_path = args.flow  # checked above
        input_dir = _ensure_run_input_dir(run_id)
        try:
            shutil.copy2(flow_path, input_dir / flow_path.name)
        except FileNotFoundError:
            raise SystemExit(f"Reference flow file not found: {flow_path}")
        dump_json(
            {
                "run_id": run_id,
                "flow_path": str(flow_path),
                "operation": args.operation,
            },
            input_dir / "input_manifest.json",
        )

    if not flow_path.exists():
        raise SystemExit(f"Reference flow file not found: {flow_path}")

    llm = None
    if not args.no_llm:
        api_key, model, base_url = load_openai_from_env()
        llm = OpenAIResponsesLLM(api_key=api_key, model=model, base_url=base_url)

    from tiangong_lca_spec.process_from_flow import ProcessFromFlowService
    from tiangong_lca_spec.utils.translate import Translator

    translator = None
    if llm is not None and not args.no_translate_zh:
        translator = Translator(llm=llm)

    if args.allow_density_conversion:
        if initial_state is None:
            initial_state = {}
        initial_state["allow_density_conversion"] = True
    if args.auto_balance_revise:
        if initial_state is None:
            initial_state = {}
        initial_state["auto_balance_revise"] = True

    service = ProcessFromFlowService(llm=llm, translator=translator)
    stop_after = None if args.stop_after == "datasets" else args.stop_after
    result_state = service.run(
        flow_path=flow_path,
        operation=args.operation,
        initial_state=initial_state,
        stop_after=stop_after,
    )

    dump_json(result_state, state_path, lock_reason="langgraph.final_state_write")

    if args.stop_after and args.stop_after != "datasets":
        print(f"Stopped after stage '{args.stop_after}'. Edit state and resume with: --resume --run-id {run_id}", file=sys.stderr)
        LATEST_RUN_ID_PATH.write_text(run_id, encoding="utf-8")
        return

    datasets = result_state.get("process_datasets") or []
    if not isinstance(datasets, list) or not datasets:
        print("No process datasets generated.", file=sys.stderr)
        LATEST_RUN_ID_PATH.write_text(run_id, encoding="utf-8")
        return

    written = _write_process_exports(datasets, exports_dir)
    _export_referenced_flows_from_processes(datasets, exports_dir)

    LATEST_RUN_ID_PATH.write_text(run_id, encoding="utf-8")
    print(f"Wrote {len(written)} process dataset(s) to {exports_dir / 'processes'}", file=sys.stderr)

    source_payloads = result_state.get("source_datasets") or []
    source_written: list[Path] = []
    if isinstance(source_payloads, list) and source_payloads:
        for payload in source_payloads:
            if not isinstance(payload, dict):
                continue
            uuid_value = _extract_source_uuid(payload)
            version = _extract_source_version(payload)
            filename = f"{uuid_value}_{version}.json"
            target = exports_dir / "sources" / filename
            dump_json(payload, target)
            source_written.append(target)
        print(f"Wrote {len(source_written)} source dataset(s) to {exports_dir / 'sources'}", file=sys.stderr)
    if args.publish:
        if not args.skip_balance_check:
            _enforce_balance_quality_gate(run_id, strict=args.strict_balance_check)
        if args.publish_flows:
            flow_plans = _publish_flows(datasets, commit=args.commit, llm=llm, exports_dir=exports_dir)
            if args.commit and flow_plans:
                updated = _apply_flow_refs_to_processes(datasets, flow_plans)
                if updated:
                    _write_process_exports(datasets, exports_dir)
                    _export_referenced_flows_from_processes(datasets, exports_dir)
        if isinstance(source_payloads, list) and source_payloads:
            _publish_sources(source_payloads, commit=args.commit, process_datasets=datasets)
        _publish_processes(datasets, commit=args.commit)
    if args.retain_runs:
        _cleanup_runs(retain=args.retain_runs, current_run_id=run_id)


if __name__ == "__main__":  # pragma: no cover
    main()
