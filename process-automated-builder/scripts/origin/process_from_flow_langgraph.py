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

Build placeholder flows only for an existing run:
  uv run python scripts/origin/process_from_flow_langgraph.py flow-auto-build --run-id <run_id>

Rewrite process placeholder references only:
  uv run python scripts/origin/process_from_flow_langgraph.py process-update --run-id <run_id>

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
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, urlparse

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

try:
    from scripts.origin.process_from_flow_cost_report import (  # type: ignore
        DEFAULT_INPUT_PRICE_PER_1M,
        DEFAULT_OUTPUT_PRICE_PER_1M,
        generate_cost_report,
    )
except ModuleNotFoundError:  # pragma: no cover
    from process_from_flow_cost_report import (  # type: ignore
        DEFAULT_INPUT_PRICE_PER_1M,
        DEFAULT_OUTPUT_PRICE_PER_1M,
        generate_cost_report,
    )

from tiangong_lca_spec.state_lock import hold_state_file_lock

PROCESS_FROM_FLOW_ARTIFACTS_ROOT = Path("artifacts/process_from_flow")
LATEST_RUN_ID_PATH = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / ".latest_run_id"
DATABASE_TOOL_NAME = "Database_CRUD_Tool"
FLOW_AUTO_BUILD_MANIFEST = "flow_auto_build_manifest.jsonl"
PROCESS_UPDATE_REPORT = "process_update_report.json"
FLOW_PUBLISH_RESULTS = "flow_publish_results.jsonl"
FLOW_PUBLISH_FAILURES = "flow_publish_failures.jsonl"
FLOW_PUBLISH_READY = "flow_publish_ready.json"
PUBLISH_SUMMARY = "publish_summary.json"
METHOD_POLICY_AUTOFIX_REPORT = "method_policy_autofix_report.json"
DEFAULT_DATASET_VERSION = "01.01.000"
DEFAULT_PLACEHOLDER_FLOW_VERSION = "00.00.000"
DEFAULT_COST_INPUT_PRICE_PER_1M = float(DEFAULT_INPUT_PRICE_PER_1M)
DEFAULT_COST_OUTPUT_PRICE_PER_1M = float(DEFAULT_OUTPUT_PRICE_PER_1M)
ENV_COST_INPUT_PRICE_PER_1M = "TIANGONG_PFF_COST_INPUT_PRICE_PER_1M"
ENV_COST_OUTPUT_PRICE_PER_1M = "TIANGONG_PFF_COST_OUTPUT_PRICE_PER_1M"
DEFAULT_SI_SUBDIR = Path("input/si")
MINERU_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".tsv", ".xlsx", ".docx"}
WORKFLOW_LOG_SUBDIR = Path("cache/workflow_logs")
WORKFLOW_TIMING_REPORT = Path("cache/workflow_timing_report.json")
WORKFLOW_TIMING_HEARTBEAT_SECONDS = 5.0
AGENT_HANDOFF_SUMMARY = "agent_handoff_summary.json"

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


def _read_cost_price_from_env(env_name: str, default_value: float) -> float:
    raw = (os.getenv(env_name) or "").strip()
    if not raw:
        return default_value
    try:
        value = float(raw)
    except ValueError:
        return default_value
    if value < 0:
        return default_value
    return value


def _build_main_arg_parser() -> argparse.ArgumentParser:
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
        dest="allow_density_conversion",
        action="store_true",
        help="Enable LLM-based density estimates for mass<->volume conversions (product/waste flows only; default: enabled).",
    )
    parser.add_argument(
        "--no-allow-density-conversion",
        dest="allow_density_conversion",
        action="store_false",
        help="Disable LLM-based density estimates for mass<->volume conversions.",
    )
    parser.add_argument(
        "--auto-balance-revise",
        dest="auto_balance_revise",
        action="store_true",
        help=("Enable auto-revision after the first balance review: revise severe core-mass " "imbalances on non-reference exchanges, then recompute balance review (default: enabled)."),
    )
    parser.add_argument(
        "--no-auto-balance-revise",
        dest="auto_balance_revise",
        action="store_false",
        help="Disable auto-revision after balance review.",
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
        help="Publish flow/process/source datasets via Database_CRUD_Tool after the pipeline completes.",
    )
    parser.add_argument(
        "--publish-only",
        action="store_true",
        help="Publish flow/process/source datasets from an existing run and skip the pipeline.",
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
    parser.add_argument(
        "--strict-flow-property-check",
        action="store_true",
        help="Treat unresolved placeholder flow-property selection as a blocking error during flow auto build.",
    )
    parser.add_argument(
        "--skip-flow-auto-build",
        action="store_true",
        help="Skip flow-auto-build during --publish/--publish-only (debug only).",
    )
    parser.add_argument(
        "--skip-process-update",
        action="store_true",
        help="Skip process-update during --publish/--publish-only (debug only).",
    )
    parser.add_argument(
        "--cost-report",
        dest="cost_report",
        action="store_true",
        help="Generate cache/llm_cost_report.json after run/publish (default: enabled).",
    )
    parser.add_argument(
        "--no-cost-report",
        dest="cost_report",
        action="store_false",
        help="Disable automatic LLM cost report generation.",
    )
    parser.add_argument(
        "--cost-input-price-per-1m",
        type=float,
        default=_read_cost_price_from_env(ENV_COST_INPUT_PRICE_PER_1M, DEFAULT_COST_INPUT_PRICE_PER_1M),
        help=f"USD per 1M input tokens for cost report (default: {DEFAULT_COST_INPUT_PRICE_PER_1M}).",
    )
    parser.add_argument(
        "--cost-output-price-per-1m",
        type=float,
        default=_read_cost_price_from_env(ENV_COST_OUTPUT_PRICE_PER_1M, DEFAULT_COST_OUTPUT_PRICE_PER_1M),
        help=f"USD per 1M output tokens for cost report (default: {DEFAULT_COST_OUTPUT_PRICE_PER_1M}).",
    )
    parser.set_defaults(auto_balance_revise=True, allow_density_conversion=True, cost_report=True)
    return parser


def _build_workflow_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="process_from_flow_langgraph.py workflow",
        description="Run the SI-enhanced process_from_flow recipe before resuming the main pipeline.",
    )
    parser.add_argument(
        "--flow",
        type=Path,
        required=True,
        help="Path to the reference flow JSON (ILCD flowDataSet wrapper).",
    )
    parser.add_argument(
        "--operation",
        choices=("produce", "treat"),
        default="produce",
        help="Whether the process produces or treats the reference flow.",
    )
    parser.add_argument(
        "--run-id",
        help="Run identifier under artifacts/process_from_flow/<run_id>.",
    )
    parser.add_argument("--no-translate-zh", action="store_true", help="Skip adding Chinese translations.")
    parser.add_argument(
        "--allow-density-conversion",
        dest="allow_density_conversion",
        action="store_true",
        help="Enable LLM-based density conversion for mass/volume mismatches (default: enabled).",
    )
    parser.add_argument(
        "--no-allow-density-conversion",
        dest="allow_density_conversion",
        action="store_false",
        help="Disable LLM-based density conversion for mass/volume mismatches.",
    )
    parser.add_argument(
        "--auto-balance-revise",
        dest="auto_balance_revise",
        action="store_true",
        help=(
            "Enable auto-revision after the first balance review: revise severe core-mass "
            "imbalances on non-reference exchanges, then recompute balance review (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-auto-balance-revise",
        dest="auto_balance_revise",
        action="store_false",
        help="Disable auto-revision after balance review.",
    )
    parser.add_argument("--min-si-hint", default="possible", help="Min si_hint to download (none|possible|likely).")
    parser.add_argument("--si-max-links", type=int, help="Max SI links per DOI.")
    parser.add_argument("--si-timeout", type=float, help="HTTP timeout for SI download.")
    parser.add_argument(
        "--publish",
        dest="publish",
        action="store_true",
        help="Publish generated flow/process/source datasets after completion (default: enabled).",
    )
    parser.add_argument(
        "--no-publish",
        dest="publish",
        action="store_false",
        help="Disable publishing and keep outputs local under exports/.",
    )
    parser.add_argument(
        "--commit",
        dest="commit",
        action="store_true",
        help="Commit publish actions to remote CRUD service (default: enabled).",
    )
    parser.add_argument(
        "--no-commit",
        dest="commit",
        action="store_false",
        help="Publish in dry-run mode without remote commit.",
    )
    parser.add_argument(
        "--skip-flow-auto-build",
        action="store_true",
        help="Skip flow-auto-build during publish (debug only).",
    )
    parser.add_argument(
        "--skip-process-update",
        action="store_true",
        help="Skip process-update during publish (debug only).",
    )
    parser.add_argument(
        "--cost-report",
        dest="cost_report",
        action="store_true",
        help="Generate cache/llm_cost_report.json after run/publish (default: enabled).",
    )
    parser.add_argument(
        "--no-cost-report",
        dest="cost_report",
        action="store_false",
        help="Disable automatic LLM cost report generation.",
    )
    parser.add_argument(
        "--cost-input-price-per-1m",
        type=float,
        default=_read_cost_price_from_env(ENV_COST_INPUT_PRICE_PER_1M, DEFAULT_COST_INPUT_PRICE_PER_1M),
        help=f"USD per 1M input tokens for cost report (default: {DEFAULT_COST_INPUT_PRICE_PER_1M}).",
    )
    parser.add_argument(
        "--cost-output-price-per-1m",
        type=float,
        default=_read_cost_price_from_env(ENV_COST_OUTPUT_PRICE_PER_1M, DEFAULT_COST_OUTPUT_PRICE_PER_1M),
        help=f"USD per 1M output tokens for cost report (default: {DEFAULT_COST_OUTPUT_PRICE_PER_1M}).",
    )
    parser.add_argument(
        "--stop-after",
        choices=("references", "tech", "processes", "exchanges", "matches", "sources", "datasets"),
        help="Stop after a stage (debug only).",
    )
    parser.set_defaults(
        publish=True,
        commit=True,
        auto_balance_revise=True,
        allow_density_conversion=True,
        cost_report=True,
    )
    return parser


def _build_flow_auto_build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="process_from_flow_langgraph.py flow-auto-build",
        description="Prepare flow plans/datasets for placeholder references and write flow auto-build manifests.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run identifier under artifacts/process_from_flow/<run_id>.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run without an LLM (uses deterministic fallbacks).",
    )
    parser.add_argument(
        "--strict-flow-property-check",
        action="store_true",
        help="Treat unresolved placeholder flow-property selection as a blocking error.",
    )
    return parser


def _build_process_update_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="process_from_flow_langgraph.py process-update",
        description="Rewrite process placeholder references to target flow references and export updated datasets.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run identifier under artifacts/process_from_flow/<run_id>.",
    )
    parser.add_argument(
        "--flow-publish-results",
        type=Path,
        help="Optional flow publish result JSONL containing `target_ref` mappings.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cli_args = list(argv) if argv is not None else sys.argv[1:]
    if cli_args and cli_args[0] == "workflow":
        parser = _build_workflow_arg_parser()
        namespace = parser.parse_args(cli_args[1:])
        setattr(namespace, "command", "workflow")
        return namespace
    if cli_args and cli_args[0] == "flow-auto-build":
        parser = _build_flow_auto_build_arg_parser()
        namespace = parser.parse_args(cli_args[1:])
        setattr(namespace, "command", "flow-auto-build")
        return namespace
    if cli_args and cli_args[0] == "process-update":
        parser = _build_process_update_arg_parser()
        namespace = parser.parse_args(cli_args[1:])
        setattr(namespace, "command", "process-update")
        return namespace
    parser = _build_main_arg_parser()
    namespace = parser.parse_args(cli_args)
    setattr(namespace, "command", "run")
    return namespace


def _maybe_generate_cost_report(
    *,
    run_id: str,
    enabled: bool,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> None:
    if not enabled:
        return
    log_path = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id / "cache" / "llm_log.jsonl"
    if not log_path.exists():
        return
    try:
        report, output_path = generate_cost_report(
            run_id=run_id,
            log_path=log_path,
            input_price_per_1m=input_price_per_1m,
            output_price_per_1m=output_price_per_1m,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Failed to generate LLM cost report for run {run_id}: {exc}", file=sys.stderr)
        return
    totals = report.get("totals") if isinstance(report.get("totals"), Mapping) else {}
    total_cost = totals.get("total_cost_usd")
    input_tokens = totals.get("input_tokens")
    output_tokens = totals.get("output_tokens")
    print(
        (
            f"[progress] llm_cost_report={output_path} "
            f"input_tokens={input_tokens} output_tokens={output_tokens} total_cost_usd={total_cost}"
        ),
        file=sys.stderr,
    )


def _load_state(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"State file must contain an object: {path}")
    return payload


def _persist_flow_publish_diagnostics_to_state(
    *,
    flow_property_decisions: list[dict[str, Any]],
    held_flows: list[dict[str, Any]],
) -> None:
    state_path_raw = (os.getenv("TIANGONG_PFF_STATE_PATH") or "").strip()
    if not state_path_raw:
        return
    state_path = Path(state_path_raw)
    if not state_path.exists():
        return
    try:
        state = _load_state(state_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Failed to load run state for flow publish diagnostics: {exc}", file=sys.stderr)
        return
    state["flow_property_decisions"] = flow_property_decisions
    state["flow_property_decision_summary"] = {
        "total": len(flow_property_decisions),
        "held_total": len(held_flows),
        "held_exchanges": [
            {
                "process_name": item.get("process_name"),
                "exchange_name": item.get("exchange_name"),
                "reason": item.get("reason"),
                "decision_mode": item.get("decision_mode"),
            }
            for item in held_flows[:200]
            if isinstance(item, Mapping)
        ],
    }
    try:
        dump_json(state, state_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Failed to persist flow publish diagnostics to run state: {exc}", file=sys.stderr)


def _count_placeholder_flow_refs(process_datasets: list[dict[str, Any]]) -> int:
    count = 0
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
            if isinstance(ref, Mapping) and ref.get("unmatched:placeholder"):
                count += 1
    return count


def _collect_placeholder_flow_refs(
    process_datasets: list[dict[str, Any]],
    *,
    limit: int = 10,
) -> list[dict[str, str]]:
    def _process_name_from_payload(process_payload: Mapping[str, Any]) -> str:
        process_info = process_payload.get("processInformation")
        if not isinstance(process_info, Mapping):
            return "Unknown process"
        dsi = process_info.get("dataSetInformation")
        if not isinstance(dsi, Mapping):
            return "Unknown process"
        name_block = dsi.get("name")
        if not isinstance(name_block, Mapping):
            return "Unknown process"
        base_name = name_block.get("baseName")
        if isinstance(base_name, Mapping):
            text = base_name.get("#text")
            return str(text) if isinstance(text, str) and text.strip() else "Unknown process"
        if isinstance(base_name, list):
            for item in base_name:
                if not isinstance(item, Mapping):
                    continue
                if item.get("@xml:lang") != "en":
                    continue
                text = item.get("#text")
                if isinstance(text, str) and text.strip():
                    return text
            for item in base_name:
                if not isinstance(item, Mapping):
                    continue
                text = item.get("#text")
                if isinstance(text, str) and text.strip():
                    return text
        return "Unknown process"

    def _short_desc_from_ref(ref: Mapping[str, Any]) -> str:
        desc = ref.get("common:shortDescription")
        if isinstance(desc, Mapping):
            text = desc.get("#text")
            return str(text) if isinstance(text, str) and text.strip() else "Unnamed exchange"
        if isinstance(desc, list):
            for item in desc:
                if not isinstance(item, Mapping):
                    continue
                if item.get("@xml:lang") != "en":
                    continue
                text = item.get("#text")
                if isinstance(text, str) and text.strip():
                    return text
            for item in desc:
                if not isinstance(item, Mapping):
                    continue
                text = item.get("#text")
                if isinstance(text, str) and text.strip():
                    return text
        return "Unnamed exchange"

    results: list[dict[str, str]] = []
    for payload in process_datasets:
        process_payload = payload.get("processDataSet") if isinstance(payload.get("processDataSet"), Mapping) else payload
        if not isinstance(process_payload, Mapping):
            continue
        process_name = _process_name_from_payload(process_payload)
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
            if not (isinstance(ref, Mapping) and ref.get("unmatched:placeholder")):
                continue
            exchange_name = _short_desc_from_ref(ref)
            results.append(
                {
                    "process_name": process_name or "Unknown process",
                    "exchange_name": exchange_name or "Unnamed exchange",
                    "uuid": str(ref.get("@refObjectId") or ""),
                    "version": str(ref.get("@version") or ""),
                }
            )
            if len(results) >= limit:
                return results
    return results


def _format_placeholder_flow_refs(refs: list[dict[str, str]]) -> str:
    if not refs:
        return ""
    details = "; ".join(
        (
            f"{item.get('exchange_name') or 'Unnamed exchange'}"
            f" ({item.get('uuid') or 'unknown-uuid'}@{item.get('version') or '*'})"
            f" in {item.get('process_name') or 'Unknown process'}"
        )
        for item in refs
    )
    return details


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remain = seconds - (minutes * 60)
    return f"{minutes}m{remain:04.1f}s"


def _tail_log(path: Path, *, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _run_python(script: Path, args: list[str], *, log_path: Path | None = None) -> None:
    cmd = [sys.executable, str(script), *args]
    if log_path is None:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now(timezone.utc).isoformat()}] CMD: {' '.join(cmd)}\n")
        handle.flush()
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def _run_reference_stage(args: argparse.Namespace, run_id: str, *, log_path: Path | None = None) -> None:
    script = SCRIPT_DIR / "process_from_flow_langgraph.py"
    cmd = [
        "--flow",
        str(args.flow),
        "--operation",
        args.operation,
        "--run-id",
        run_id,
        "--stop-after",
        "references",
    ]
    if args.no_translate_zh:
        cmd.append("--no-translate-zh")
    if args.allow_density_conversion:
        cmd.append("--allow-density-conversion")
    else:
        cmd.append("--no-allow-density-conversion")
    if args.auto_balance_revise:
        cmd.append("--auto-balance-revise")
    else:
        cmd.append("--no-auto-balance-revise")
    if args.cost_report:
        cmd.append("--cost-report")
    else:
        cmd.append("--no-cost-report")
    cmd.extend(
        [
            "--cost-input-price-per-1m",
            str(args.cost_input_price_per_1m),
            "--cost-output-price-per-1m",
            str(args.cost_output_price_per_1m),
        ]
    )
    _run_python(script, cmd, log_path=log_path)


def _run_usability(_args: argparse.Namespace, run_id: str, *, log_path: Path | None = None) -> None:
    script = SCRIPT_DIR / "process_from_flow_reference_usability.py"
    _run_python(script, ["--run-id", run_id], log_path=log_path)


def _run_si_download(args: argparse.Namespace, run_id: str, *, log_path: Path | None = None) -> None:
    script = SCRIPT_DIR / "process_from_flow_download_si.py"
    cmd = [
        "--run-id",
        run_id,
        "--min-si-hint",
        args.min_si_hint,
    ]
    if args.si_max_links is not None:
        cmd.extend(["--max-links", str(args.si_max_links)])
    if args.si_timeout is not None:
        cmd.extend(["--timeout", str(args.si_timeout)])
    _run_python(script, cmd, log_path=log_path)


def _run_usage_tagging(_args: argparse.Namespace, run_id: str, *, log_path: Path | None = None) -> None:
    script = SCRIPT_DIR / "process_from_flow_reference_usage_tagging.py"
    _run_python(script, ["--run-id", run_id], log_path=log_path)


def _iter_si_files(run_id: str) -> list[Path]:
    si_root = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id / DEFAULT_SI_SUBDIR
    if not si_root.exists():
        return []
    files: list[Path] = []
    for path in sorted(si_root.rglob("*")):
        if path.is_file():
            files.append(path)
    return files


def _run_mineru_for_si(args: argparse.Namespace, run_id: str, *, log_path: Path | None = None) -> None:
    script = SCRIPT_DIR / "mineru_for_process_si.py"
    failures: list[Path] = []
    for path in _iter_si_files(run_id):
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            continue
        if suffix not in MINERU_SUFFIXES:
            print(f"[warn] Skip unsupported SI file: {path}", file=sys.stderr)
            continue
        cmd = [str(path), "--run-id", run_id]
        try:
            _run_python(script, cmd, log_path=log_path)
        except subprocess.CalledProcessError:
            failures.append(path)
    if failures:
        print(f"[warn] MinerU failed for {len(failures)} SI file(s).", file=sys.stderr)


def _run_main_pipeline(args: argparse.Namespace, run_id: str, *, log_path: Path | None = None) -> None:
    script = SCRIPT_DIR / "process_from_flow_langgraph.py"
    cmd = [
        "--flow",
        str(args.flow),
        "--operation",
        args.operation,
        "--run-id",
        run_id,
        "--resume",
    ]
    if args.no_translate_zh:
        cmd.append("--no-translate-zh")
    if args.allow_density_conversion:
        cmd.append("--allow-density-conversion")
    else:
        cmd.append("--no-allow-density-conversion")
    if args.auto_balance_revise:
        cmd.append("--auto-balance-revise")
    else:
        cmd.append("--no-auto-balance-revise")
    if args.stop_after:
        cmd.extend(["--stop-after", args.stop_after])
    if args.publish:
        cmd.append("--publish")
        if args.skip_flow_auto_build:
            cmd.append("--skip-flow-auto-build")
        if args.skip_process_update:
            cmd.append("--skip-process-update")
    if args.commit:
        cmd.append("--commit")
    if args.cost_report:
        cmd.append("--cost-report")
    else:
        cmd.append("--no-cost-report")
    cmd.extend(
        [
            "--cost-input-price-per-1m",
            str(args.cost_input_price_per_1m),
            "--cost-output-price-per-1m",
            str(args.cost_output_price_per_1m),
        ]
    )
    _run_python(script, cmd, log_path=log_path)


def _clear_stop_after(run_id: str) -> None:
    state_path = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id / "cache" / "process_from_flow_state.json"
    if not state_path.exists():
        return
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "stop_after" not in payload:
        return
    payload.pop("stop_after", None)
    with hold_state_file_lock(state_path, reason="workflow.clear_stop_after"):
        state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] Cleared stop_after in {state_path}", file=sys.stderr)


def _write_timing_report(
    *,
    report_path: Path,
    run_id: str,
    flow_path: Path,
    operation: str,
    started_at: datetime,
    stages: list[dict[str, object]],
) -> None:
    total_seconds = float(sum(float(item.get("elapsed_seconds") or 0.0) for item in stages))
    payload = {
        "run_id": run_id,
        "flow_path": str(flow_path),
        "operation": operation,
        "started_at": started_at.isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_seconds": round(total_seconds, 3),
        "stages": stages,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _start_timing_heartbeat(
    *,
    report_lock: threading.Lock,
    report_path: Path,
    run_id: str,
    flow_path: Path,
    operation: str,
    workflow_started_at: datetime,
    stage_records: list[dict[str, object]],
    stage_record: dict[str, object],
    stage_started_perf: float,
    interval_seconds: float = WORKFLOW_TIMING_HEARTBEAT_SECONDS,
) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def _heartbeat_loop() -> None:
        while not stop_event.wait(interval_seconds):
            with report_lock:
                if stage_record.get("status") != "running":
                    return
                stage_record["elapsed_seconds"] = round(time.perf_counter() - stage_started_perf, 3)
                stage_record["updated_at"] = datetime.now(timezone.utc).isoformat()
                _write_timing_report(
                    report_path=report_path,
                    run_id=run_id,
                    flow_path=flow_path,
                    operation=operation,
                    started_at=workflow_started_at,
                    stages=stage_records,
                )

    thread = threading.Thread(target=_heartbeat_loop, name="workflow-timing-heartbeat", daemon=True)
    thread.start()
    return stop_event, thread


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _flow_auto_build_manifest_path(cache_dir: Path) -> Path:
    return cache_dir / FLOW_AUTO_BUILD_MANIFEST


def _process_update_report_path(cache_dir: Path) -> Path:
    return cache_dir / PROCESS_UPDATE_REPORT


def _flow_publish_results_path(cache_dir: Path) -> Path:
    return cache_dir / FLOW_PUBLISH_RESULTS


def _flow_publish_failures_path(cache_dir: Path) -> Path:
    return cache_dir / FLOW_PUBLISH_FAILURES


def _flow_publish_ready_path(cache_dir: Path) -> Path:
    return cache_dir / FLOW_PUBLISH_READY


def _publish_summary_path(cache_dir: Path) -> Path:
    return cache_dir / PUBLISH_SUMMARY


def _method_policy_autofix_report_path(cache_dir: Path) -> Path:
    return cache_dir / METHOD_POLICY_AUTOFIX_REPORT


def _flow_publish_plan_cache_path(cache_dir: Path) -> Path:
    return cache_dir / "flow_publish_plans.json"


def _extract_flow_name_entries(dataset: Mapping[str, Any] | None) -> Any:
    if not isinstance(dataset, Mapping):
        return None
    flow_root = dataset.get("flowDataSet")
    if not isinstance(flow_root, Mapping):
        flow_root = dataset
    flow_info = flow_root.get("flowInformation")
    if not isinstance(flow_info, Mapping):
        return None
    data_info = flow_info.get("dataSetInformation")
    if not isinstance(data_info, Mapping):
        return None
    name_block = data_info.get("name")
    if not isinstance(name_block, Mapping):
        return None
    return name_block.get("baseName")


def _ensure_bilingual_short_description(
    ref: Mapping[str, Any],
    *,
    dataset: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = dict(ref)
    desc_node = normalized.get("common:shortDescription")
    en_text = _extract_lang_text_with_preference(desc_node, "en")
    zh_text = _extract_lang_text_with_preference(desc_node, "zh")

    name_entries = _extract_flow_name_entries(dataset)
    if not en_text:
        en_text = _extract_lang_text_with_preference(name_entries, "en")
    if not zh_text:
        zh_text = _extract_lang_text_with_preference(name_entries, "zh")

    if not en_text:
        ref_uuid = normalized.get("@refObjectId")
        en_text = ref_uuid.strip() if isinstance(ref_uuid, str) and ref_uuid.strip() else "Unnamed flow"
    if not zh_text:
        zh_text = en_text

    normalized["common:shortDescription"] = [
        {"@xml:lang": "en", "#text": en_text},
        {"@xml:lang": "zh", "#text": zh_text},
    ]
    return normalized


def _serialize_flow_publish_plans(
    plans: list[Any],
    *,
    flow_property_decisions: list[dict[str, Any]] | None = None,
    held_flows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    serialized_plans: list[dict[str, Any]] = []
    for plan in plans:
        uuid_value = getattr(plan, "uuid", None)
        exchange_name = getattr(plan, "exchange_name", None)
        process_name = getattr(plan, "process_name", None)
        dataset = getattr(plan, "dataset", None)
        exchange_ref = getattr(plan, "exchange_ref", None)
        mode = getattr(plan, "mode", None)
        flow_property_uuid = getattr(plan, "flow_property_uuid", None)
        if not isinstance(uuid_value, str) or not uuid_value.strip():
            continue
        if not isinstance(exchange_name, str):
            exchange_name = "Unnamed exchange"
        if not isinstance(process_name, str):
            process_name = "Unknown process"
        if not isinstance(dataset, Mapping) or not isinstance(exchange_ref, Mapping):
            continue
        serialized_plans.append(
            {
                "uuid": uuid_value,
                "exchange_name": exchange_name,
                "process_name": process_name,
                "dataset": dict(dataset),
                "exchange_ref": _ensure_bilingual_short_description(exchange_ref, dataset=dataset),
                "mode": mode if isinstance(mode, str) and mode else "insert",
                "flow_property_uuid": flow_property_uuid if isinstance(flow_property_uuid, str) and flow_property_uuid else None,
            }
        )
    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now_iso(),
        "plan_count": len(serialized_plans),
        "plans": serialized_plans,
        "flow_property_decisions": flow_property_decisions or [],
        "held_flows": held_flows or [],
    }


def _write_flow_publish_plan_cache(
    cache_dir: Path,
    plans: list[Any],
    *,
    flow_property_decisions: list[dict[str, Any]] | None = None,
    held_flows: list[dict[str, Any]] | None = None,
) -> Path:
    payload = _serialize_flow_publish_plans(
        plans,
        flow_property_decisions=flow_property_decisions,
        held_flows=held_flows,
    )
    target = _flow_publish_plan_cache_path(cache_dir)
    dump_json(payload, target)
    return target


def _load_flow_publish_plan_cache(cache_dir: Path) -> tuple[list[Any], dict[str, Any]]:
    target = _flow_publish_plan_cache_path(cache_dir)
    if not target.exists():
        raise FileNotFoundError(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Flow publish plan cache must be a JSON object: {target}")
    plans_raw = payload.get("plans")
    if not isinstance(plans_raw, list):
        raise SystemExit(f"Flow publish plan cache missing 'plans' list: {target}")

    from tiangong_lca_spec.publishing import FlowPublishPlan

    plans: list[Any] = []
    for item in plans_raw:
        if not isinstance(item, Mapping):
            continue
        uuid_value = item.get("uuid")
        exchange_name = item.get("exchange_name")
        process_name = item.get("process_name")
        dataset = item.get("dataset")
        exchange_ref = item.get("exchange_ref")
        mode = item.get("mode")
        flow_property_uuid = item.get("flow_property_uuid")
        if not isinstance(uuid_value, str) or not uuid_value.strip():
            continue
        if not isinstance(dataset, Mapping) or not isinstance(exchange_ref, Mapping):
            continue
        plans.append(
            FlowPublishPlan(
                uuid=uuid_value.strip(),
                exchange_name=(exchange_name if isinstance(exchange_name, str) else "Unnamed exchange"),
                process_name=(process_name if isinstance(process_name, str) else "Unknown process"),
                dataset=dict(dataset),
                exchange_ref=_ensure_bilingual_short_description(exchange_ref, dataset=dataset),
                mode=(mode if isinstance(mode, str) and mode else "insert"),
                flow_property_uuid=(flow_property_uuid if isinstance(flow_property_uuid, str) and flow_property_uuid else None),
            )
        )
    return plans, payload


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


def _extract_lang_text_with_preference(value: Any, preferred_lang: str = "en") -> str:
    preferred = (preferred_lang or "").strip().lower()
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        item_lang = str(value.get("@xml:lang") or "").strip().lower()
        text = value.get("#text")
        if isinstance(text, str):
            normalized = text.strip()
            if not normalized:
                return ""
            if preferred and item_lang and item_lang != preferred:
                return ""
            return normalized
    if isinstance(value, list):
        if preferred:
            for item in value:
                if isinstance(item, Mapping) and str(item.get("@xml:lang") or "").strip().lower() == preferred:
                    text = item.get("#text")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
        for item in value:
            text = _extract_lang_text_with_preference(item, "")
            if text:
                return text
    return ""


def _extract_lang_text(value: Any) -> str:
    text = _extract_lang_text_with_preference(value, "en")
    if text:
        return text
    return _extract_lang_text_with_preference(value, "")


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


def _resolve_run_operation(run_id: str) -> str:
    try:
        state = _load_run_state(run_id)
    except SystemExit:
        return "produce"
    operation = str(state.get("operation") or "produce").strip().lower()
    return "treat" if operation == "treat" else "produce"


def _normalize_exchange_direction(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("in"):
        return "input"
    if text.startswith("out"):
        return "output"
    return text


def _extract_version_from_uri(uri_value: Any) -> str | None:
    uri = str(uri_value or "").strip()
    if not uri:
        return None
    try:
        parsed = urlparse(uri)
    except ValueError:
        parsed = None
    if parsed is not None:
        query_version = parse_qs(parsed.query).get("version")
        if query_version:
            candidate = str(query_version[0] or "").strip()
            if candidate:
                return candidate
        path_value = parsed.path
    else:
        path_value = uri
    match = re.search(r"_(\d+\.\d+\.\d+)\.(?:xml|json)$", path_value, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _ensure_reference_version(ref: dict[str, Any], *, placeholder_default: bool = False) -> bool:
    current = str(ref.get("@version") or "").strip()
    if current:
        return False
    inferred = _extract_version_from_uri(ref.get("@uri"))
    if not inferred:
        if placeholder_default or bool(ref.get("unmatched:placeholder")):
            inferred = DEFAULT_PLACEHOLDER_FLOW_VERSION
        else:
            inferred = DEFAULT_DATASET_VERSION
    ref["@version"] = inferred
    return True


def _iter_reference_entries(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, list):
        refs: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                refs.append(item)
        return refs
    return []


def _select_reference_exchange(exchanges: list[dict[str, Any]], *, operation: str) -> dict[str, Any] | None:
    target_direction = "input" if operation == "treat" else "output"

    def _score(exchange: dict[str, Any]) -> tuple[int, int]:
        direction = _normalize_exchange_direction(exchange.get("exchangeDirection"))
        ref = exchange.get("referenceToFlowDataSet")
        has_non_placeholder_ref = int(isinstance(ref, Mapping) and not ref.get("unmatched:placeholder"))
        direction_score = 1 if direction == target_direction else 0
        return (direction_score, has_non_placeholder_ref)

    valid = [item for item in exchanges if str(item.get("@dataSetInternalID") or "").strip()]
    if not valid:
        return None
    valid.sort(key=_score, reverse=True)
    return valid[0]


def _synchronize_reference_amounts(exchange: dict[str, Any]) -> int:
    changes = 0
    mean_amount = str(exchange.get("meanAmount") or "").strip()
    resulting_amount = str(exchange.get("resultingAmount") or "").strip()
    if mean_amount and not resulting_amount:
        exchange["resultingAmount"] = mean_amount
        changes += 1
    elif resulting_amount and not mean_amount:
        exchange["meanAmount"] = resulting_amount
        changes += 1
    elif not mean_amount and not resulting_amount:
        exchange["meanAmount"] = "1"
        exchange["resultingAmount"] = "1"
        changes += 2
    return changes


def _apply_method_policy_deterministic_fixes(
    run_id: str,
    datasets: list[dict[str, Any]],
) -> dict[str, Any]:
    operation = _resolve_run_operation(run_id)
    fix_counts = {
        "process_dataset_version_filled": 0,
        "flow_ref_version_filled": 0,
        "source_ref_version_filled": 0,
        "quantitative_reference_aligned": 0,
        "quantitative_reference_amount_synced": 0,
    }

    for payload in datasets:
        process_payload = payload.get("processDataSet") if isinstance(payload.get("processDataSet"), Mapping) else payload
        if not isinstance(process_payload, dict):
            continue

        admin = process_payload.get("administrativeInformation")
        if not isinstance(admin, dict):
            admin = {}
            process_payload["administrativeInformation"] = admin
        publication = admin.get("publicationAndOwnership")
        if not isinstance(publication, dict):
            publication = {}
            admin["publicationAndOwnership"] = publication
        dataset_version = str(publication.get("common:dataSetVersion") or "").strip()
        if not dataset_version:
            publication["common:dataSetVersion"] = DEFAULT_DATASET_VERSION
            fix_counts["process_dataset_version_filled"] += 1

        exchanges_block = process_payload.get("exchanges")
        exchanges_raw = exchanges_block.get("exchange") if isinstance(exchanges_block, Mapping) else []
        exchanges = [item for item in (exchanges_raw if isinstance(exchanges_raw, list) else [exchanges_raw]) if isinstance(item, dict)]

        for exchange in exchanges:
            flow_ref = exchange.get("referenceToFlowDataSet")
            if isinstance(flow_ref, dict) and _ensure_reference_version(flow_ref):
                fix_counts["flow_ref_version_filled"] += 1
            refs_block = exchange.get("referencesToDataSource")
            refs_raw = refs_block.get("referenceToDataSource") if isinstance(refs_block, Mapping) else None
            for source_ref in _iter_reference_entries(refs_raw):
                if _ensure_reference_version(source_ref):
                    fix_counts["source_ref_version_filled"] += 1

        modelling = process_payload.get("modellingAndValidation")
        sources = modelling.get("dataSourcesTreatmentAndRepresentativeness") if isinstance(modelling, Mapping) else None
        source_refs = sources.get("referenceToDataSource") if isinstance(sources, Mapping) else None
        for source_ref in _iter_reference_entries(source_refs):
            if _ensure_reference_version(source_ref):
                fix_counts["source_ref_version_filled"] += 1

        process_info = process_payload.get("processInformation")
        if not isinstance(process_info, dict):
            continue
        quant_ref = process_info.get("quantitativeReference")
        if not isinstance(quant_ref, dict):
            quant_ref = {"@type": "Reference flow(s)"}
            process_info["quantitativeReference"] = quant_ref
        selected_exchange = _select_reference_exchange(exchanges, operation=operation)
        if selected_exchange is None:
            continue
        selected_id = str(selected_exchange.get("@dataSetInternalID") or "").strip()
        if not selected_id:
            continue
        current_id = str(quant_ref.get("referenceToReferenceFlow") or "").strip()
        if current_id != selected_id:
            quant_ref["referenceToReferenceFlow"] = selected_id
            fix_counts["quantitative_reference_aligned"] += 1
        fix_counts["quantitative_reference_amount_synced"] += _synchronize_reference_amounts(selected_exchange)

    return {
        "run_id": run_id,
        "updated_at_utc": _utc_now_iso(),
        "operation": operation,
        "fix_counts": fix_counts,
        "fix_total": int(sum(int(value) for value in fix_counts.values())),
    }


def _load_flow_property_holds(run_id: str) -> list[dict[str, Any]]:
    try:
        state = _load_run_state(run_id)
    except SystemExit:
        return []
    summary = state.get("flow_property_decision_summary")
    held = summary.get("held_exchanges") if isinstance(summary, Mapping) else None
    if not isinstance(held, list):
        return []
    return [item for item in held if isinstance(item, dict)]


def _load_reference_output_low_confidence(run_id: str) -> list[dict[str, Any]]:
    try:
        state = _load_run_state(run_id)
    except SystemExit:
        return []
    summary = state.get("reference_output_decision_summary")
    rows = summary.get("low_confidence_processes") if isinstance(summary, Mapping) else None
    if not isinstance(rows, list):
        return []
    return [item for item in rows if isinstance(item, dict)]


def _load_chain_conflict_holds(run_id: str) -> list[dict[str, Any]]:
    try:
        state = _load_run_state(run_id)
    except SystemExit:
        return []
    preflight = state.get("chain_preflight")
    if not isinstance(preflight, Mapping):
        return []
    status = str(preflight.get("status") or "").strip().lower()
    errors = preflight.get("errors")
    if status != "failed" or not isinstance(errors, list):
        return []
    return [item for item in errors if isinstance(item, dict)]


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


def _build_flow_ref_mapping_from_plans(plans: list[Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for plan in plans:
        uuid_value = getattr(plan, "uuid", None)
        dataset = getattr(plan, "dataset", None)
        exchange_ref = getattr(plan, "exchange_ref", None)
        if isinstance(uuid_value, str) and uuid_value.strip() and isinstance(exchange_ref, Mapping):
            mapping[uuid_value.strip()] = _ensure_bilingual_short_description(exchange_ref, dataset=dataset)
    return mapping


def _build_flow_ref_mapping_from_publish_results(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        flow_uuid = row.get("flow_uuid")
        target_ref = row.get("target_ref")
        if isinstance(flow_uuid, str) and flow_uuid.strip() and isinstance(target_ref, Mapping):
            mapping[flow_uuid.strip()] = _ensure_bilingual_short_description(target_ref)
    return mapping


def _resolve_process_update_mapping(
    *,
    cache_dir: Path,
    flow_publish_results_path: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], str]:
    if flow_publish_results_path is not None and flow_publish_results_path.exists():
        rows = _load_jsonl(flow_publish_results_path)
        mapping = _build_flow_ref_mapping_from_publish_results(rows)
        if mapping:
            return mapping, f"flow_publish_results:{flow_publish_results_path}"
    try:
        plans, _ = _load_flow_publish_plan_cache(cache_dir)
    except FileNotFoundError:
        return {}, "none"
    mapping = _build_flow_ref_mapping_from_plans(plans)
    return mapping, "flow_publish_plans"


def _apply_flow_ref_mapping_to_processes(
    datasets: list[dict[str, Any]],
    mapping: dict[str, dict[str, Any]],
    *,
    placeholder_only: bool = True,
) -> int:
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
            if placeholder_only and not ref.get("unmatched:placeholder"):
                continue
            uuid_value = ref.get("@refObjectId")
            if isinstance(uuid_value, str):
                mapped_ref = mapping.get(uuid_value.strip())
                if mapped_ref:
                    exchange["referenceToFlowDataSet"] = dict(mapped_ref)
                    updated += 1
    return updated


def _prepare_flow_publish_plans(
    datasets: list[dict[str, Any]],
    *,
    llm: Any | None = None,
    cache_dir: Path | None = None,
    exports_dir: Path | None = None,
    strict_flow_property_check: bool = False,
) -> list[Any]:
    from tiangong_lca_spec.publishing import FlowPublisher

    alignment = _build_flow_alignment_from_process_datasets(datasets)
    if not alignment:
        return []
    publisher = FlowPublisher(
        dry_run=True,
        llm=llm,
        strict_flow_property_check=strict_flow_property_check,
    )
    try:
        plans = publisher.prepare_from_alignment(alignment)
        held_flows = publisher.held_flows
        diagnostics = publisher.flow_property_decisions
        _persist_flow_publish_diagnostics_to_state(
            flow_property_decisions=diagnostics,
            held_flows=held_flows,
        )
        if cache_dir is not None:
            cache_path = _write_flow_publish_plan_cache(
                cache_dir,
                plans,
                flow_property_decisions=diagnostics,
                held_flows=held_flows,
            )
            print(f"Cached {len(plans)} flow publish plan(s) at {cache_path}", file=sys.stderr)
        if held_flows:
            preview = "; ".join(
                f"{item.get('exchange_name')} ({item.get('reason')})"
                for item in held_flows[:5]
                if isinstance(item, Mapping)
            )
            print(
                f"[warn] Held {len(held_flows)} flow(s) for manual review due to unresolved flow-property selection."
                + (f" {preview}" if preview else ""),
                file=sys.stderr,
            )
        if not plans:
            return []
        if exports_dir is not None:
            _write_flow_exports(plans, exports_dir)
        return plans
    finally:
        publisher.close()


def _build_flow_auto_build_manifest(
    *,
    datasets: list[dict[str, Any]],
    plans: list[Any],
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for flow_uuid, version in _collect_referenced_flow_refs(datasets):
        manifest.append(
            {
                "origin": "selected",
                "flow_uuid": flow_uuid,
                "target_uuid": flow_uuid,
                "target_version": version,
                "needs_publish": False,
                "mode": "selected",
                "generated_at_utc": _utc_now_iso(),
            }
        )
    for plan in plans:
        flow_uuid = getattr(plan, "uuid", None)
        dataset = getattr(plan, "dataset", None)
        mode = getattr(plan, "mode", None)
        process_name = getattr(plan, "process_name", None)
        exchange_name = getattr(plan, "exchange_name", None)
        if not isinstance(flow_uuid, str) or not flow_uuid.strip():
            continue
        if not isinstance(dataset, Mapping):
            continue
        intended_version = _resolve_flow_dataset_version(dataset)
        mode_text = mode if isinstance(mode, str) and mode.strip() else "insert"
        manifest.append(
            {
                "origin": "generated",
                "flow_uuid": flow_uuid.strip(),
                "target_uuid": flow_uuid.strip(),
                "target_version": intended_version,
                "needs_publish": mode_text == "insert",
                "mode": mode_text,
                "process_name": process_name if isinstance(process_name, str) else "Unknown process",
                "exchange_name": exchange_name if isinstance(exchange_name, str) else "Unnamed exchange",
                "generated_at_utc": _utc_now_iso(),
            }
        )
    return manifest


def _run_flow_auto_build(
    run_id: str,
    datasets: list[dict[str, Any]],
    *,
    cache_dir: Path,
    exports_dir: Path,
    llm: Any | None = None,
    strict_flow_property_check: bool = False,
) -> dict[str, Any]:
    plans = _prepare_flow_publish_plans(
        datasets,
        llm=llm,
        cache_dir=cache_dir,
        exports_dir=exports_dir,
        strict_flow_property_check=strict_flow_property_check,
    )
    _export_referenced_flows_from_processes(datasets, exports_dir)
    manifest = _build_flow_auto_build_manifest(datasets=datasets, plans=plans)
    manifest_path = _write_jsonl(_flow_auto_build_manifest_path(cache_dir), manifest)

    publish_ready = [entry for entry in manifest if entry.get("origin") == "generated" and entry.get("needs_publish")]
    dump_json(
        {
            "run_id": run_id,
            "generated_at_utc": _utc_now_iso(),
            "publish_ready_count": len(publish_ready),
            "flows": publish_ready,
        },
        _flow_publish_ready_path(cache_dir),
    )

    print(
        (
            f"Flow auto-build complete: generated={len([item for item in manifest if item.get('origin') == 'generated'])}, "
            f"selected={len([item for item in manifest if item.get('origin') == 'selected'])}, "
            f"publish_ready={len(publish_ready)} manifest={manifest_path}"
        ),
        file=sys.stderr,
    )
    return {
        "plans": plans,
        "manifest": manifest,
        "publish_ready": publish_ready,
    }


def _classify_flow_publish_error(exc: Exception) -> tuple[str, bool]:
    message = f"{type(exc).__name__}: {exc}".lower()
    if "timeout" in message:
        return "timeout", True
    if any(token in message for token in ("network", "connection", "temporarily", "unreachable", "reset by peer")):
        return "network", True
    if any(token in message for token in ("conflict", "duplicate", "already exists", "already present")):
        return "conflict", False
    if any(token in message for token in ("schema", "validation", "invalid", "required field")):
        return "schema", False
    return "unknown", True


def _publish_prepared_flow_plans(
    plans: list[Any],
    *,
    commit: bool,
    cache_dir: Path,
    manifest: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    from tiangong_lca_spec.publishing import FlowPublisher

    manifest_rows = manifest or _load_jsonl(_flow_auto_build_manifest_path(cache_dir))
    publishable_uuids = {
        str(item.get("flow_uuid") or "").strip()
        for item in manifest_rows
        if isinstance(item, Mapping) and item.get("origin") == "generated" and item.get("needs_publish")
    }

    publishable: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for plan in plans:
        flow_uuid = getattr(plan, "uuid", None)
        dataset = getattr(plan, "dataset", None)
        if not isinstance(flow_uuid, str) or not flow_uuid.strip():
            continue
        if publishable_uuids and flow_uuid.strip() not in publishable_uuids:
            continue
        if not isinstance(dataset, Mapping):
            continue
        intended_version = _resolve_flow_dataset_version(dataset)
        key = (flow_uuid.strip(), intended_version)
        if key in seen:
            continue
        seen.add(key)
        publishable.append(plan)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not publishable:
        _write_jsonl(_flow_publish_results_path(cache_dir), rows)
        _write_jsonl(_flow_publish_failures_path(cache_dir), failures)
        return {"generated_flows_total": 0, "flow_insert_success": 0, "flow_insert_failed": 0}

    publisher = FlowPublisher(dry_run=not commit)
    try:
        for plan in publishable:
            flow_uuid = str(getattr(plan, "uuid", "") or "").strip()
            dataset = getattr(plan, "dataset", None)
            exchange_ref = getattr(plan, "exchange_ref", None)
            intended_version = _resolve_flow_dataset_version(dataset) if isinstance(dataset, Mapping) else "01.01.000"
            row: dict[str, Any] = {
                "flow_uuid": flow_uuid,
                "intended_version": intended_version,
                "origin": "generated",
                "process_name": getattr(plan, "process_name", "Unknown process"),
                "exchange_name": getattr(plan, "exchange_name", "Unnamed exchange"),
                "target_ref": dict(exchange_ref) if isinstance(exchange_ref, Mapping) else {},
                "updated_at_utc": _utc_now_iso(),
            }
            if not commit:
                row["status"] = "skipped"
                row["reason"] = "dry_run"
                rows.append(row)
                continue
            try:
                publish_results = publisher.publish_prepared([plan])
                status = "inserted"
                reason = ""
                if publish_results:
                    first = publish_results[0]
                    if isinstance(first, Mapping):
                        action = str(first.get("action") or "").strip().lower()
                        if action == "reuse":
                            status = "conflict"
                            reason = "reused_existing"
                else:
                    status = "conflict"
                    reason = "reused_existing_or_noop"
                row["status"] = status
                if reason:
                    row["reason"] = reason
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                error_type, retryable = _classify_flow_publish_error(exc)
                row["status"] = "error"
                row["reason"] = str(exc)
                row["error_type"] = error_type
                row["retryable"] = retryable
                rows.append(row)
                failures.append(
                    {
                        "flow_uuid": flow_uuid,
                        "intended_version": intended_version,
                        "reason": str(exc),
                        "error_type": error_type,
                        "retryable": retryable,
                        "updated_at_utc": _utc_now_iso(),
                    }
                )
                print(
                    f"[warn] Flow publish failed for {flow_uuid}@{intended_version}; continuing. error={exc}",
                    file=sys.stderr,
                )
    finally:
        publisher.close()

    _write_jsonl(_flow_publish_results_path(cache_dir), rows)
    _write_jsonl(_flow_publish_failures_path(cache_dir), failures)
    return {
        "generated_flows_total": len(publishable),
        "flow_insert_success": sum(1 for item in rows if item.get("status") == "inserted"),
        "flow_insert_failed": len(failures),
    }


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


def _run_process_update(
    run_id: str,
    datasets: list[dict[str, Any]],
    *,
    cache_dir: Path,
    exports_dir: Path,
    flow_ref_mapping: dict[str, dict[str, Any]],
    mapping_source: str,
) -> dict[str, Any]:
    replaced_count = _apply_flow_ref_mapping_to_processes(datasets, flow_ref_mapping, placeholder_only=True)
    _write_process_exports(datasets, exports_dir)
    remaining_placeholders = _count_placeholder_flow_refs(datasets)
    unresolved_refs = _collect_placeholder_flow_refs(datasets, limit=20) if remaining_placeholders else []
    report = {
        "run_id": run_id,
        "updated_at_utc": _utc_now_iso(),
        "mapping_source": mapping_source,
        "flow_mapping_size": len(flow_ref_mapping),
        "replaced_reference_count": replaced_count,
        "remaining_placeholder_refs": remaining_placeholders,
        "process_update_incomplete": remaining_placeholders > 0,
        "unresolved_placeholder_examples": unresolved_refs,
    }
    dump_json(report, _process_update_report_path(cache_dir))
    if remaining_placeholders:
        detail = _format_placeholder_flow_refs(unresolved_refs)
        print(
            (
                f"[warn] process-update incomplete: remaining_placeholder_refs={remaining_placeholders}"
                + (f" unresolved={detail}" if detail else "")
            ),
            file=sys.stderr,
        )
    else:
        print(
            f"process-update complete: replaced_reference_count={replaced_count}, remaining_placeholder_refs=0",
            file=sys.stderr,
        )
    return report


def _build_llm_for_flow_ops(*, no_llm: bool, run_id: str | None = None, stage: str = "flow_ops") -> Any | None:
    if no_llm:
        return None
    api_key, model, base_url = load_openai_from_env()
    return OpenAIResponsesLLM(
        api_key=api_key,
        model=model,
        base_url=base_url,
        run_id=run_id,
        module="process_from_flow_langgraph",
        stage=stage,
    )


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
) -> dict[str, int]:
    publishable = [item for item in datasets if isinstance(item, dict)]
    if process_datasets is not None:
        publishable = _filter_source_datasets_by_usage(publishable, process_datasets)
    if not publishable:
        return {"publishable": 0, "published": 0, "failed": 0, "reused": 0}
    if not commit:
        print(f"Dry-run: prepared {len(publishable)} source dataset(s) for publish.", file=sys.stderr)
        return {"publishable": len(publishable), "published": 0, "failed": 0, "reused": 0}

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
    return {
        "publishable": len(publishable),
        "published": published_count,
        "failed": len(failed),
        "reused": reused_count,
    }


def _publish_processes(datasets: list[dict[str, Any]], *, commit: bool) -> dict[str, int]:
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
        return {
            "publishable": len(publishable),
            "published": success if commit else 0,
            "failed": len(failed),
        }
    finally:
        publisher.close()


def _run_publish_sequence(
    *,
    run_id: str,
    datasets: list[dict[str, Any]],
    source_datasets: list[dict[str, Any]],
    cache_dir: Path,
    exports_dir: Path,
    commit: bool,
    no_llm: bool,
    strict_flow_property_check: bool,
    skip_flow_auto_build: bool,
    skip_process_update: bool,
) -> dict[str, Any]:
    flow_plans: list[Any] = []
    manifest: list[dict[str, Any]] = []
    method_policy_report: dict[str, Any] = {
        "run_id": run_id,
        "updated_at_utc": _utc_now_iso(),
        "deterministic_fixes": _apply_method_policy_deterministic_fixes(run_id, datasets),
        "rebuild_attempts": [],
        "manual_required": [],
    }

    def _append_manual_required(entry: dict[str, Any]) -> None:
        code = str(entry.get("code") or "").strip()
        reason = str(entry.get("reason") or "").strip()
        items = method_policy_report.get("manual_required")
        if not isinstance(items, list):
            items = []
            method_policy_report["manual_required"] = items
        for item in items:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("code") or "").strip() == code and str(item.get("reason") or "").strip() == reason:
                return
        items.append(entry)

    if int(method_policy_report["deterministic_fixes"].get("fix_total") or 0) > 0:
        _write_process_exports(datasets, exports_dir)
        print("[info] Applied deterministic method-policy auto-fixes before publish sequence.", file=sys.stderr)

    def _collect_rebuild_signals(report: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        signals: list[dict[str, Any]] = []
        remaining_refs = int(report.get("remaining_placeholder_refs") or 0)
        if remaining_refs > 0:
            signals.append(
                {
                    "code": "remaining_placeholder_refs",
                    "reason": f"{remaining_refs} placeholder flow reference(s) remain after process-update.",
                    "remaining_placeholder_refs": remaining_refs,
                }
            )
        low_conf_outputs = _load_reference_output_low_confidence(run_id)
        if low_conf_outputs:
            signals.append(
                {
                    "code": "reference_output_low_confidence",
                    "reason": (
                        f"{len(low_conf_outputs)} process(es) have low-confidence reference-output unit decisions."
                    ),
                    "low_confidence_count": len(low_conf_outputs),
                    "examples": low_conf_outputs[:5],
                }
            )
        chain_conflicts = _load_chain_conflict_holds(run_id)
        if chain_conflicts:
            signals.append(
                {
                    "code": "chain_conflict_hold",
                    "reason": (
                        f"{len(chain_conflicts)} severe chain conflict(s) remain in chain_preflight."
                    ),
                    "chain_conflict_count": len(chain_conflicts),
                    "examples": chain_conflicts[:5],
                }
            )
        held_flows = _load_flow_property_holds(run_id)
        if held_flows:
            signals.append(
                {
                    "code": "flow_property_semantic_conflict",
                    "reason": f"{len(held_flows)} flow(s) are held because automatic flow-property selection remains unresolved.",
                    "held_count": len(held_flows),
                    "examples": held_flows[:5],
                }
            )
        return signals, held_flows

    if skip_flow_auto_build:
        try:
            flow_plans, _ = _load_flow_publish_plan_cache(cache_dir)
        except FileNotFoundError:
            flow_plans = []
        manifest = _load_jsonl(_flow_auto_build_manifest_path(cache_dir))
        if not manifest:
            manifest = _build_flow_auto_build_manifest(datasets=datasets, plans=flow_plans)
            _write_jsonl(_flow_auto_build_manifest_path(cache_dir), manifest)
        publish_ready = [item for item in manifest if item.get("origin") == "generated" and item.get("needs_publish")]
        dump_json(
            {
                "run_id": run_id,
                "generated_at_utc": _utc_now_iso(),
                "publish_ready_count": len(publish_ready),
                "flows": publish_ready,
            },
            _flow_publish_ready_path(cache_dir),
        )
        print("[warn] --skip-flow-auto-build enabled; using cached/derived flow publish plans.", file=sys.stderr)
    else:
        llm = _build_llm_for_flow_ops(no_llm=no_llm, run_id=run_id, stage="publish_flow_ops")
        flow_auto_build = _run_flow_auto_build(
            run_id,
            datasets,
            cache_dir=cache_dir,
            exports_dir=exports_dir,
            llm=llm,
            strict_flow_property_check=strict_flow_property_check,
        )
        flow_plans = flow_auto_build.get("plans") if isinstance(flow_auto_build.get("plans"), list) else []
        manifest = flow_auto_build.get("manifest") if isinstance(flow_auto_build.get("manifest"), list) else []

    if skip_process_update:
        remaining_placeholders = _count_placeholder_flow_refs(datasets)
        unresolved_refs = _collect_placeholder_flow_refs(datasets, limit=20) if remaining_placeholders else []
        process_update_report = {
            "run_id": run_id,
            "updated_at_utc": _utc_now_iso(),
            "mapping_source": "skipped",
            "flow_mapping_size": 0,
            "replaced_reference_count": 0,
            "remaining_placeholder_refs": remaining_placeholders,
            "process_update_incomplete": remaining_placeholders > 0,
            "unresolved_placeholder_examples": unresolved_refs,
            "skipped": True,
        }
        dump_json(process_update_report, _process_update_report_path(cache_dir))
        print("[warn] --skip-process-update enabled; process placeholders were not rewritten.", file=sys.stderr)
    else:
        flow_mapping, mapping_source = _resolve_process_update_mapping(cache_dir=cache_dir)
        process_update_report = _run_process_update(
            run_id,
            datasets,
            cache_dir=cache_dir,
            exports_dir=exports_dir,
            flow_ref_mapping=flow_mapping,
            mapping_source=mapping_source,
        )

    rebuild_signals, held_after_first = _collect_rebuild_signals(process_update_report)
    if rebuild_signals:
        if skip_flow_auto_build or skip_process_update:
            _append_manual_required(
                {
                    "code": "auto_rebuild_skipped",
                    "reason": "Method-policy rebuild path was skipped by flags (--skip-flow-auto-build/--skip-process-update).",
                    "minimum_action": f"Re-run publish without skip flags for run_id={run_id}.",
                    "signals": rebuild_signals,
                }
            )
        else:
            print(
                "[info] Method-policy repair: detected unresolved violations; triggering one automatic flow-auto-build/process-update retry.",
                file=sys.stderr,
            )
            retry_record: dict[str, Any] = {
                "attempt": 1,
                "started_at_utc": _utc_now_iso(),
                "trigger_signals": rebuild_signals,
                "held_before_retry": len(held_after_first),
                "remaining_placeholder_refs_before_retry": int(process_update_report.get("remaining_placeholder_refs") or 0),
            }
            llm = _build_llm_for_flow_ops(no_llm=no_llm, run_id=run_id, stage="publish_flow_ops")
            flow_auto_build = _run_flow_auto_build(
                run_id,
                datasets,
                cache_dir=cache_dir,
                exports_dir=exports_dir,
                llm=llm,
                strict_flow_property_check=strict_flow_property_check,
            )
            flow_plans = flow_auto_build.get("plans") if isinstance(flow_auto_build.get("plans"), list) else []
            manifest = flow_auto_build.get("manifest") if isinstance(flow_auto_build.get("manifest"), list) else []
            flow_mapping, mapping_source = _resolve_process_update_mapping(cache_dir=cache_dir)
            process_update_report = _run_process_update(
                run_id,
                datasets,
                cache_dir=cache_dir,
                exports_dir=exports_dir,
                flow_ref_mapping=flow_mapping,
                mapping_source=mapping_source,
            )
            post_retry_signals, held_after_retry = _collect_rebuild_signals(process_update_report)
            retry_record["finished_at_utc"] = _utc_now_iso()
            retry_record["held_after_retry"] = len(held_after_retry)
            retry_record["remaining_placeholder_refs_after_retry"] = int(process_update_report.get("remaining_placeholder_refs") or 0)
            retry_record["result"] = "resolved" if not post_retry_signals else "unresolved"
            retry_record["post_retry_signals"] = post_retry_signals
            method_policy_report["rebuild_attempts"].append(retry_record)
            if post_retry_signals:
                for signal in post_retry_signals:
                    code = str(signal.get("code") or "manual_required")
                    if code == "remaining_placeholder_refs":
                        _append_manual_required(
                            {
                                "code": code,
                                "reason": signal.get("reason"),
                                "minimum_action": (
                                    "Inspect cache/process_update_report.json unresolved_placeholder_examples, "
                                    "set explicit flow mappings/overrides, then rerun --publish-only."
                                ),
                            }
                        )
                    elif code == "flow_property_semantic_conflict":
                        _append_manual_required(
                            {
                                "code": code,
                                "reason": signal.get("reason"),
                                "minimum_action": (
                                    "Review state.flow_property_decision_summary.held_exchanges, "
                                    "adjust exchange naming/classification or add flow-property overrides, then rerun --publish-only."
                                ),
                            }
                        )
                    elif code == "reference_output_low_confidence":
                        _append_manual_required(
                            {
                                "code": code,
                                "reason": signal.get("reason"),
                                "minimum_action": (
                                    "Review state.reference_output_decision_summary.low_confidence_processes, "
                                    "add evidence/overrides for reference-output units, then rerun --publish-only."
                                ),
                            }
                        )
                    elif code == "chain_conflict_hold":
                        _append_manual_required(
                            {
                                "code": code,
                                "reason": signal.get("reason"),
                                "minimum_action": (
                                    "Resolve chain_preflight errors in state.chain_preflight.errors "
                                    "before publish."
                                ),
                            }
                        )

    final_signals, _final_held = _collect_rebuild_signals(process_update_report)
    low_confidence_signals = [item for item in final_signals if str(item.get("code") or "") == "reference_output_low_confidence"]
    chain_conflict_signals = [item for item in final_signals if str(item.get("code") or "") == "chain_conflict_hold"]
    if low_confidence_signals:
        _append_manual_required(
            {
                "code": "reference_output_low_confidence",
                "reason": low_confidence_signals[0].get("reason"),
                "minimum_action": (
                    "Fix low-confidence reference-output unit decisions before publish "
                    "(state.reference_output_decision_summary.low_confidence_processes)."
                ),
            }
        )
    if chain_conflict_signals:
        _append_manual_required(
            {
                "code": "chain_conflict_hold",
                "reason": chain_conflict_signals[0].get("reason"),
                "minimum_action": (
                    "Fix severe chain conflicts first (state.chain_preflight.errors)."
                ),
            }
        )
    blocking_signals = low_confidence_signals + chain_conflict_signals
    if blocking_signals and commit:
        method_policy_report["updated_at_utc"] = _utc_now_iso()
        dump_json(method_policy_report, _method_policy_autofix_report_path(cache_dir))
        raise SystemExit(
            "Publish blocked: unresolved hard-rule holds remain "
            "(reference_output_low_confidence / chain_conflict_hold)."
        )

    flow_publish_stats = _publish_prepared_flow_plans(
        flow_plans,
        commit=commit,
        cache_dir=cache_dir,
        manifest=manifest,
    )
    process_stats = _publish_processes(datasets, commit=commit)
    source_stats = _publish_sources(source_datasets, commit=commit, process_datasets=datasets)
    remaining_placeholder_refs = _count_placeholder_flow_refs(datasets)

    summary = {
        "run_id": run_id,
        "updated_at_utc": _utc_now_iso(),
        "generated_flows_total": int(flow_publish_stats.get("generated_flows_total") or 0),
        "flow_insert_success": int(flow_publish_stats.get("flow_insert_success") or 0),
        "flow_insert_failed": int(flow_publish_stats.get("flow_insert_failed") or 0),
        "process_published": int(process_stats.get("published") or 0),
        "source_published": int(source_stats.get("published") or 0),
        "flow_publish_failures_count": len(_load_jsonl(_flow_publish_failures_path(cache_dir))),
        "remaining_placeholder_refs": remaining_placeholder_refs,
        "process_update_incomplete": bool(process_update_report.get("process_update_incomplete")),
        "method_policy_fix_total": int(method_policy_report["deterministic_fixes"].get("fix_total") or 0),
        "method_policy_manual_required_count": len(method_policy_report.get("manual_required") or []),
        "method_policy_report": str(_method_policy_autofix_report_path(cache_dir)),
    }
    method_policy_report["updated_at_utc"] = _utc_now_iso()
    dump_json(method_policy_report, _method_policy_autofix_report_path(cache_dir))
    dump_json(summary, _publish_summary_path(cache_dir))
    if remaining_placeholder_refs:
        print(
            f"[warn] Publish acceptance check: remaining_placeholder_refs={remaining_placeholder_refs}",
            file=sys.stderr,
        )
    else:
        print("[acceptance] remaining_placeholder_refs=0", file=sys.stderr)
    if method_policy_report.get("manual_required"):
        print(
            (
                "[warn] Method-policy auto-repair could not safely resolve all violations. "
                f"manual_required={len(method_policy_report['manual_required'])} "
                f"report={_method_policy_autofix_report_path(cache_dir)}"
            ),
            file=sys.stderr,
        )
    return summary


def _run_flow_auto_build_command(args: argparse.Namespace) -> None:
    run_id = _resolve_run_id(args.run_id)
    cache_dir = ensure_run_cache_dir(run_id)
    exports_dir = ensure_run_exports_dir(run_id)
    os.environ["TIANGONG_PFF_RUN_ID"] = run_id
    os.environ["TIANGONG_PFF_STATE_PATH"] = str(cache_dir / "process_from_flow_state.json")
    os.environ.setdefault("TIANGONG_PFF_FLOW_CACHE_PATH", str(cache_dir / "flow_select_cache.json"))
    datasets = _load_process_datasets(run_id)
    method_policy_report = {
        "run_id": run_id,
        "updated_at_utc": _utc_now_iso(),
        "deterministic_fixes": _apply_method_policy_deterministic_fixes(run_id, datasets),
        "rebuild_attempts": [],
        "manual_required": [],
    }
    if int(method_policy_report["deterministic_fixes"].get("fix_total") or 0) > 0:
        _write_process_exports(datasets, exports_dir)
        print("[info] Applied deterministic method-policy auto-fixes before flow-auto-build.", file=sys.stderr)
    dump_json(method_policy_report, _method_policy_autofix_report_path(cache_dir))
    llm = _build_llm_for_flow_ops(no_llm=args.no_llm, run_id=run_id, stage="flow_auto_build")
    _run_flow_auto_build(
        run_id,
        datasets,
        cache_dir=cache_dir,
        exports_dir=exports_dir,
        llm=llm,
        strict_flow_property_check=args.strict_flow_property_check,
    )
    handoff_path = _write_agent_handoff_summary(run_id=run_id, command="flow-auto-build", cache_dir=cache_dir)
    print(f"[progress] agent_handoff_summary={handoff_path}", file=sys.stderr)


def _run_process_update_command(args: argparse.Namespace) -> None:
    run_id = _resolve_run_id(args.run_id)
    cache_dir = ensure_run_cache_dir(run_id)
    exports_dir = ensure_run_exports_dir(run_id)
    os.environ["TIANGONG_PFF_RUN_ID"] = run_id
    os.environ["TIANGONG_PFF_STATE_PATH"] = str(cache_dir / "process_from_flow_state.json")
    os.environ.setdefault("TIANGONG_PFF_FLOW_CACHE_PATH", str(cache_dir / "flow_select_cache.json"))
    datasets = _load_process_datasets(run_id)
    method_policy_report = {
        "run_id": run_id,
        "updated_at_utc": _utc_now_iso(),
        "deterministic_fixes": _apply_method_policy_deterministic_fixes(run_id, datasets),
        "rebuild_attempts": [],
        "manual_required": [],
    }
    if int(method_policy_report["deterministic_fixes"].get("fix_total") or 0) > 0:
        _write_process_exports(datasets, exports_dir)
        print("[info] Applied deterministic method-policy auto-fixes before process-update.", file=sys.stderr)
    dump_json(method_policy_report, _method_policy_autofix_report_path(cache_dir))
    flow_publish_results_path = args.flow_publish_results
    if flow_publish_results_path is not None and not flow_publish_results_path.is_absolute():
        flow_publish_results_path = (Path.cwd() / flow_publish_results_path).resolve()
    flow_mapping, mapping_source = _resolve_process_update_mapping(
        cache_dir=cache_dir,
        flow_publish_results_path=flow_publish_results_path,
    )
    _run_process_update(
        run_id,
        datasets,
        cache_dir=cache_dir,
        exports_dir=exports_dir,
        flow_ref_mapping=flow_mapping,
        mapping_source=mapping_source,
    )
    handoff_path = _write_agent_handoff_summary(run_id=run_id, command="process-update", cache_dir=cache_dir)
    print(f"[progress] agent_handoff_summary={handoff_path}", file=sys.stderr)


def _build_agent_handoff_summary(
    *,
    run_id: str,
    command: str,
    cache_dir: Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state_path = cache_dir / "process_from_flow_state.json"
    state = _load_state(state_path) if state_path.exists() else {}
    process_datasets = state.get("process_datasets") if isinstance(state.get("process_datasets"), list) else []
    source_datasets = state.get("source_datasets") if isinstance(state.get("source_datasets"), list) else []
    placeholder_examples = _collect_placeholder_flow_refs(process_datasets, limit=5) if process_datasets else []

    publish_summary_path = _publish_summary_path(cache_dir)
    publish_summary = {}
    if publish_summary_path.exists():
        try:
            publish_summary = _load_state(publish_summary_path)
        except Exception:
            publish_summary = {}

    llm_cost_report_path = cache_dir / "llm_cost_report.json"
    timing_report_path = cache_dir / WORKFLOW_TIMING_REPORT.name

    next_actions: list[str] = []
    stop_after = str(state.get("stop_after") or "").strip()
    if stop_after:
        next_actions.append(f"resume: process_from_flow_langgraph.py --resume --run-id {run_id}")
    elif process_datasets and not publish_summary_path.exists():
        next_actions.append(f"publish: process_from_flow_langgraph.py --publish-only --run-id {run_id}")
    if placeholder_examples:
        next_actions.append("inspect: cache/process_update_report.json or cache/placeholder_report.json")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": _utc_now_iso(),
        "run_id": run_id,
        "command": command,
        "flow_path": state.get("flow_path"),
        "operation": state.get("operation"),
        "selected_route_id": state.get("selected_route_id"),
        "stop_after": stop_after or None,
        "process_count": len(state.get("processes") or []) if isinstance(state.get("processes"), list) else 0,
        "matched_exchange_count": (
            len(state.get("matched_process_exchanges") or [])
            if isinstance(state.get("matched_process_exchanges"), list)
            else 0
        ),
        "process_dataset_count": len(process_datasets),
        "source_dataset_count": len(source_datasets),
        "remaining_placeholder_refs": _count_placeholder_flow_refs(process_datasets) if process_datasets else 0,
        "placeholder_examples": placeholder_examples,
        "publish_summary": publish_summary if isinstance(publish_summary, dict) else {},
        "artifacts": {
            "state_path": str(state_path),
            "timing_report": str(timing_report_path) if timing_report_path.exists() else None,
            "publish_summary": str(publish_summary_path) if publish_summary_path.exists() else None,
            "llm_cost_report": str(llm_cost_report_path) if llm_cost_report_path.exists() else None,
            "process_update_report": str(_process_update_report_path(cache_dir)) if _process_update_report_path(cache_dir).exists() else None,
            "flow_auto_build_manifest": str(_flow_auto_build_manifest_path(cache_dir)) if _flow_auto_build_manifest_path(cache_dir).exists() else None,
        },
        "next_actions": next_actions,
    }
    if extra:
        summary["extra"] = dict(extra)
    return summary


def _write_agent_handoff_summary(
    *,
    run_id: str,
    command: str,
    cache_dir: Path,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    target = cache_dir / AGENT_HANDOFF_SUMMARY
    dump_json(_build_agent_handoff_summary(run_id=run_id, command=command, cache_dir=cache_dir, extra=extra), target)
    return target


def _run_workflow_command(args: argparse.Namespace) -> None:
    if not args.flow.exists():
        raise SystemExit(f"Reference flow file not found: {args.flow}")

    run_id = args.run_id or build_process_from_flow_run_id(args.flow, args.operation)
    run_root = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id
    workflow_log_dir = run_root / WORKFLOW_LOG_SUBDIR
    workflow_log_dir.mkdir(parents=True, exist_ok=True)
    timing_report_path = run_root / WORKFLOW_TIMING_REPORT
    started_at = datetime.now(timezone.utc)

    print(f"[progress] run_id={run_id}", file=sys.stderr)
    print(f"[progress] logs={workflow_log_dir}", file=sys.stderr)

    stage_plan: list[tuple[str, Callable[[Path], None]]] = [
        ("01_references", lambda log: _run_reference_stage(args, run_id, log_path=log)),
        ("02_usability", lambda log: _run_usability(args, run_id, log_path=log)),
        ("03_si_download", lambda log: _run_si_download(args, run_id, log_path=log)),
        ("04_mineru", lambda log: _run_mineru_for_si(args, run_id, log_path=log)),
        ("05_usage_tagging", lambda log: _run_usage_tagging(args, run_id, log_path=log)),
        ("06_clear_stop_after", lambda _log: _clear_stop_after(run_id)),
        ("07_main_pipeline", lambda log: _run_main_pipeline(args, run_id, log_path=log)),
    ]

    stage_records: list[dict[str, object]] = []
    timing_report_lock = threading.Lock()
    total_start = time.perf_counter()
    total_stages = len(stage_plan)

    def _write_timing_snapshot() -> None:
        with timing_report_lock:
            _write_timing_report(
                report_path=timing_report_path,
                run_id=run_id,
                flow_path=args.flow,
                operation=args.operation,
                started_at=started_at,
                stages=stage_records,
            )

    for index, (stage_name, runner) in enumerate(stage_plan, start=1):
        stage_log_path = workflow_log_dir / f"{stage_name}.log"
        elapsed_before = time.perf_counter() - total_start
        avg_stage = (
            sum(float(item.get("elapsed_seconds") or 0.0) for item in stage_records) / len(stage_records)
            if stage_records
            else None
        )
        remaining_including_current = total_stages - index + 1
        eta_before = (avg_stage * remaining_including_current) if avg_stage is not None else None
        print(
            (
                f"[progress] stage {index}/{total_stages} start={stage_name} "
                f"elapsed={_format_seconds(elapsed_before)} eta={_format_seconds(eta_before)} "
                f"log={stage_log_path}"
            ),
            file=sys.stderr,
        )

        stage_started_at = datetime.now(timezone.utc)
        stage_start = time.perf_counter()
        record = {
            "index": index,
            "name": stage_name,
            "status": "running",
            "started_at": stage_started_at.isoformat(),
            "updated_at": stage_started_at.isoformat(),
            "elapsed_seconds": 0.0,
            "log_path": str(stage_log_path),
        }
        stage_records.append(record)
        _write_timing_snapshot()
        heartbeat_stop, heartbeat_thread = _start_timing_heartbeat(
            report_lock=timing_report_lock,
            report_path=timing_report_path,
            run_id=run_id,
            flow_path=args.flow,
            operation=args.operation,
            workflow_started_at=started_at,
            stage_records=stage_records,
            stage_record=record,
            stage_started_perf=stage_start,
        )
        try:
            runner(stage_log_path)
        except BaseException as exc:  # noqa: BLE001
            heartbeat_stop.set()
            heartbeat_thread.join()
            stage_elapsed = time.perf_counter() - stage_start
            record["status"] = "failed"
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            record["updated_at"] = record["finished_at"]
            record["elapsed_seconds"] = round(stage_elapsed, 3)
            record["error_type"] = exc.__class__.__name__
            record["error"] = str(exc)
            _write_timing_snapshot()
            tail = _tail_log(stage_log_path)
            if tail:
                print(f"[progress] stage {stage_name} failed. log tail:\n{tail}", file=sys.stderr)
            raise

        heartbeat_stop.set()
        heartbeat_thread.join()
        stage_elapsed = time.perf_counter() - stage_start
        record["status"] = "ok"
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["updated_at"] = record["finished_at"]
        record["elapsed_seconds"] = round(stage_elapsed, 3)
        _write_timing_snapshot()

        elapsed_after = time.perf_counter() - total_start
        remaining = total_stages - index
        avg_after = sum(float(item.get("elapsed_seconds") or 0.0) for item in stage_records) / len(stage_records)
        eta_after = avg_after * remaining if remaining > 0 else 0.0
        print(
            (
                f"[progress] stage {index}/{total_stages} done={stage_name} "
                f"stage_elapsed={_format_seconds(stage_elapsed)} "
                f"total_elapsed={_format_seconds(elapsed_after)} "
                f"eta={_format_seconds(eta_after)}"
            ),
            file=sys.stderr,
        )

    total_elapsed = time.perf_counter() - total_start
    cache_dir = ensure_run_cache_dir(run_id)
    handoff_path = _write_agent_handoff_summary(
        run_id=run_id,
        command="workflow",
        cache_dir=cache_dir,
        extra={
            "workflow_timing_report": str(timing_report_path),
            "total_elapsed_seconds": round(total_elapsed, 3),
        },
    )
    print(f"[progress] workflow completed run_id={run_id} total={_format_seconds(total_elapsed)}", file=sys.stderr)
    print(f"[progress] timing_report={timing_report_path}", file=sys.stderr)
    print(f"[progress] agent_handoff_summary={handoff_path}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    if args.command == "workflow":
        _run_workflow_command(args)
        return
    if args.command == "flow-auto-build":
        _run_flow_auto_build_command(args)
        return
    if args.command == "process-update":
        _run_process_update_command(args)
        return
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
        source_datasets = _load_source_datasets(run_id)
        datasets = _load_process_datasets(run_id)
        _run_publish_sequence(
            run_id=run_id,
            datasets=datasets,
            source_datasets=source_datasets,
            cache_dir=cache_dir,
            exports_dir=exports_dir,
            commit=args.commit,
            no_llm=args.no_llm,
            strict_flow_property_check=args.strict_flow_property_check,
            skip_flow_auto_build=args.skip_flow_auto_build,
            skip_process_update=args.skip_process_update,
        )
        _maybe_generate_cost_report(
            run_id=run_id,
            enabled=bool(args.cost_report),
            input_price_per_1m=float(args.cost_input_price_per_1m),
            output_price_per_1m=float(args.cost_output_price_per_1m),
        )
        handoff_path = _write_agent_handoff_summary(run_id=run_id, command="publish-only", cache_dir=cache_dir)
        print(f"[progress] agent_handoff_summary={handoff_path}", file=sys.stderr)
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
        llm = OpenAIResponsesLLM(
            api_key=api_key,
            model=model,
            base_url=base_url,
            run_id=run_id,
            module="process_from_flow_langgraph",
            stage="main_pipeline",
        )

    from tiangong_lca_spec.process_from_flow import ProcessFromFlowService
    from tiangong_lca_spec.utils.translate import Translator

    translator = None
    if llm is not None and not args.no_translate_zh:
        translator = Translator(llm=llm)

    if initial_state is None:
        initial_state = {}
    initial_state["allow_density_conversion"] = bool(args.allow_density_conversion)
    initial_state["auto_balance_revise"] = bool(args.auto_balance_revise)

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
        _maybe_generate_cost_report(
            run_id=run_id,
            enabled=bool(args.cost_report),
            input_price_per_1m=float(args.cost_input_price_per_1m),
            output_price_per_1m=float(args.cost_output_price_per_1m),
        )
        handoff_path = _write_agent_handoff_summary(run_id=run_id, command="run", cache_dir=cache_dir)
        print(f"[progress] agent_handoff_summary={handoff_path}", file=sys.stderr)
        return

    datasets = result_state.get("process_datasets") or []
    if not isinstance(datasets, list) or not datasets:
        print("No process datasets generated.", file=sys.stderr)
        LATEST_RUN_ID_PATH.write_text(run_id, encoding="utf-8")
        _maybe_generate_cost_report(
            run_id=run_id,
            enabled=bool(args.cost_report),
            input_price_per_1m=float(args.cost_input_price_per_1m),
            output_price_per_1m=float(args.cost_output_price_per_1m),
        )
        handoff_path = _write_agent_handoff_summary(run_id=run_id, command="run", cache_dir=cache_dir)
        print(f"[progress] agent_handoff_summary={handoff_path}", file=sys.stderr)
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
        source_datasets = source_payloads if isinstance(source_payloads, list) else []
        _run_publish_sequence(
            run_id=run_id,
            datasets=datasets,
            source_datasets=source_datasets,
            cache_dir=cache_dir,
            exports_dir=exports_dir,
            commit=args.commit,
            no_llm=args.no_llm,
            strict_flow_property_check=args.strict_flow_property_check,
            skip_flow_auto_build=args.skip_flow_auto_build,
            skip_process_update=args.skip_process_update,
        )
    _maybe_generate_cost_report(
        run_id=run_id,
        enabled=bool(args.cost_report),
        input_price_per_1m=float(args.cost_input_price_per_1m),
        output_price_per_1m=float(args.cost_output_price_per_1m),
    )
    handoff_path = _write_agent_handoff_summary(run_id=run_id, command="run", cache_dir=cache_dir)
    print(f"[progress] agent_handoff_summary={handoff_path}", file=sys.stderr)
    if args.retain_runs:
        _cleanup_runs(retain=args.retain_runs, current_run_id=run_id)


if __name__ == "__main__":  # pragma: no cover
    main()
