#!/usr/bin/env python
# ruff: noqa: E402
"""Screen scientific references for process_from_flow usability."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent
for path in (SCRIPTS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

try:
    from scripts.md._workflow_common import OpenAIResponsesLLM, dump_json, load_secrets  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from _workflow_common import OpenAIResponsesLLM, dump_json, load_secrets  # type: ignore

from tiangong_lca_spec.core.json_utils import parse_json_response
from tiangong_lca_spec.process_from_flow.prompts import REFERENCE_USABILITY_PROMPT

PROCESS_FROM_FLOW_ARTIFACTS_ROOT = Path("artifacts/process_from_flow")
DEFAULT_STEP_KEY = "step_1b_reference_fulltext"
DEFAULT_MAX_CHARS = 12000
DEFAULT_MAX_RECORDS = 6
PROMPT_VERSION = "v2"
SI_STRONG_HINTS = (
    "supporting information",
    "supplementary information",
    "supplementary material",
    "supplementary data",
    "supplemental material",
    "supplemental data",
    "electronic supplementary",
    "supplementary table",
    "supplementary tables",
    "supporting data",
)
SI_WEAK_HINTS = (
    "appendix",
    "appendices",
    "additional file",
    "online resource",
    "supplementary",
    "supplemental",
    "esm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Run ID under artifacts/process_from_flow.")
    parser.add_argument("--state-path", type=Path, help="Explicit state JSON path to load.")
    parser.add_argument("--secrets", type=Path, default=Path(".secrets/secrets.toml"), help="Secrets file with OpenAI credentials.")
    parser.add_argument("--step-key", default=DEFAULT_STEP_KEY, help="scientific_references key to screen (default: step_1b_reference_fulltext).")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Max characters per DOI to send to LLM.")
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS, help="Max records per DOI to include.")
    return parser.parse_args()


def _resolve_state_paths(run_id: str, state_path: Path | None) -> list[Path]:
    if state_path:
        return [state_path]
    cache_state = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id / "cache" / "process_from_flow_state.json"
    paths: list[Path] = []
    if cache_state.exists():
        paths.append(cache_state)
    if not paths:
        raise SystemExit(f"No state file found for run {run_id}.")
    return paths


def _load_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_record_text(record: dict[str, Any]) -> str:
    for key in ("content", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    segment = record.get("segment")
    if isinstance(segment, dict):
        for key in ("content", "text"):
            value = segment.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _collect_article_text(records: list[dict[str, Any]], *, max_chars: int, max_records: int) -> str:
    chunks: list[str] = []
    for record in records:
        if max_records and len(chunks) >= max_records:
            break
        if not isinstance(record, dict):
            continue
        text = _extract_record_text(record)
        if text:
            chunks.append(text)
    combined = "\n\n".join(chunks).strip()
    if max_chars and len(combined) > max_chars:
        combined = combined[:max_chars]
    return combined


def _normalize_decision(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"usable", "useful", "yes", "y", "\u53ef\u7528"}:
            return "usable"
        if text in {"unusable", "not usable", "no", "n", "\u4e0d\u53ef\u7528"}:
            return "unusable"
    return "unusable"


def _normalize_si_hint(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"likely", "yes", "y", "true", "high", "\u6709", "\u5f88\u53ef\u80fd"}:
            return "likely"
        if text in {"possible", "maybe", "unclear", "medium", "low", "\u53ef\u80fd"}:
            return "possible"
        if text in {"none", "no", "n", "false", "absent", "\u65e0", "\u4e0d\u786e\u5b9a"}:
            return "none"
    return "none"


def _detect_si_hint(text: str) -> tuple[str, str]:
    lower = text.lower()
    for phrase in SI_STRONG_HINTS:
        if phrase in lower:
            return "likely", f"Mentions '{phrase}'."
    for phrase in SI_WEAK_HINTS:
        if phrase in lower:
            return "possible", f"Mentions '{phrase}'."
    return "none", ""


def _normalize_steps(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    normalized: list[str] = []
    for item in items:
        text = str(item).strip().lower()
        text = text.replace(" ", "").replace("-", "")
        if text in {"step1", "s1", "1"}:
            normalized.append("step1")
        elif text in {"step2", "s2", "2"}:
            normalized.append("step2")
        elif text in {"step3", "s3", "3"}:
            normalized.append("step3")
    return sorted(set(normalized))


def main() -> None:
    args = parse_args()
    state_paths = _resolve_state_paths(args.run_id, args.state_path)
    state = _load_state(state_paths[0])

    scientific_references = state.get("scientific_references", {})
    references = scientific_references.get(args.step_key, {}).get("references", [])
    if not isinstance(references, list) or not references:
        raise SystemExit(f"No references found under scientific_references.{args.step_key}.references")

    if not args.secrets.exists():
        raise SystemExit(f"Secrets file not found: {args.secrets}")
    api_key, model, base_url = load_secrets(args.secrets)
    llm = OpenAIResponsesLLM(api_key=api_key, model=model, base_url=base_url)

    flow_summary = state.get("flow_summary") or {}
    operation = state.get("operation") or "produce"

    results: list[dict[str, Any]] = []
    for entry in references:
        if not isinstance(entry, dict):
            continue
        doi = str(entry.get("doi") or "").strip() or "unknown"
        records = entry.get("records") or []
        records_list = [item for item in records if isinstance(item, dict)]
        content = _collect_article_text(records_list, max_chars=args.max_chars, max_records=args.max_records)
        if not content:
            results.append(
                {
                    "doi": doi,
                    "decision": "unusable",
                    "supported_steps": [],
                    "reason": "No usable content returned from kb search.",
                    "evidence": [],
                    "si_hint": "none",
                    "si_reason": "",
                    "records_count": len(records_list),
                    "content_chars": 0,
                    "source_refs": entry.get("source_refs") or [],
                }
            )
            continue

        payload = {
            "prompt": REFERENCE_USABILITY_PROMPT,
            "context": {
                "reference_flow": flow_summary,
                "operation": operation,
                "doi": doi,
                "article": {
                    "content": content,
                },
            },
            "response_format": {"type": "json_object"},
        }
        raw = llm.invoke(payload)
        try:
            parsed = parse_json_response(raw)
        except Exception:
            parsed = {}

        decision = _normalize_decision(parsed.get("decision"))
        supported_steps = _normalize_steps(parsed.get("supported_steps"))
        reason = str(parsed.get("reason") or "").strip()
        evidence = parsed.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = [str(evidence)]
        evidence = [str(item).strip() for item in evidence if str(item).strip()]
        si_hint = _normalize_si_hint(parsed.get("si_hint") or parsed.get("siHint"))
        si_reason = str(parsed.get("si_reason") or parsed.get("siReason") or "").strip()
        if si_hint == "none" and not si_reason:
            auto_hint, auto_reason = _detect_si_hint(content)
            if auto_hint != "none":
                si_hint = auto_hint
                si_reason = auto_reason
        results.append(
            {
                "doi": doi,
                "decision": decision,
                "supported_steps": supported_steps,
                "reason": reason,
                "evidence": evidence,
                "si_hint": si_hint,
                "si_reason": si_reason,
                "records_count": len(records_list),
                "content_chars": len(content),
                "source_refs": entry.get("source_refs") or [],
            }
        )

    scientific_references = state.get("scientific_references")
    if not isinstance(scientific_references, dict):
        scientific_references = {}
    scientific_references["usability"] = {
        "source_step": args.step_key,
        "prompt_version": PROMPT_VERSION,
        "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "max_chars": args.max_chars,
        "max_records": args.max_records,
        "results": results,
    }
    state["scientific_references"] = scientific_references

    for path in state_paths:
        dump_json(state, path)
    print(f"Updated {len(state_paths)} state file(s) with usability results.", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
