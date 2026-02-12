#!/usr/bin/env python
# ruff: noqa: E402
"""Tag scientific references for process_from_flow usage (Step 1e)."""

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
from tiangong_lca_spec.process_from_flow.prompts import REFERENCE_USAGE_TAGGING_PROMPT
from tiangong_lca_spec.process_from_flow.service import (  # type: ignore
    REFERENCE_FULLTEXT_KEY,
    _collect_article_text,
    _collect_reference_infos,
    _load_si_snippets,
    _normalize_doi,
)

PROCESS_FROM_FLOW_ARTIFACTS_ROOT = Path("artifacts/process_from_flow")
DEFAULT_MAX_CHARS = 8000
DEFAULT_MAX_RECORDS = 4
DEFAULT_MAX_SI_SNIPPETS = 3
DEFAULT_MAX_SI_CHARS = 1200
PROMPT_VERSION = "v1"
ALLOWED_TAGS = {"tech_route", "process_split", "exchange_values", "background_only"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Run ID under artifacts/process_from_flow.")
    parser.add_argument("--state-path", type=Path, help="Explicit state JSON path to load.")
    parser.add_argument("--secrets", type=Path, default=Path(".secrets/secrets.toml"), help="Secrets file with OpenAI credentials.")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Max fulltext chars per DOI.")
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS, help="Max fulltext records per DOI.")
    parser.add_argument("--max-si-snippets", type=int, default=DEFAULT_MAX_SI_SNIPPETS, help="Max SI snippets per DOI.")
    parser.add_argument("--max-si-chars", type=int, default=DEFAULT_MAX_SI_CHARS, help="Max chars per SI snippet.")
    return parser.parse_args()


def _resolve_state_paths(run_id: str, state_path: Path | None) -> list[Path]:
    if state_path:
        return [state_path]
    cache_state = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id / "cache" / "process_from_flow_state.json"
    if cache_state.exists():
        return [cache_state]
    raise SystemExit(f"No state file found for run {run_id}.")


def _load_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_tags(value: Any) -> list[str]:
    items = value if isinstance(value, list) else [value]
    tags: list[str] = []
    for item in items:
        text = str(item).strip().lower()
        if text in ALLOWED_TAGS and text not in tags:
            tags.append(text)
    if not tags:
        tags = ["background_only"]
    if "background_only" in tags and len(tags) > 1:
        tags = [tag for tag in tags if tag != "background_only"]
    return tags


def _build_fulltext_map(scientific_references: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = scientific_references.get(REFERENCE_FULLTEXT_KEY, {}).get("references", [])
    fulltext_map: dict[str, dict[str, Any]] = {}
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            doi = _normalize_doi(entry.get("doi"))
            if doi:
                fulltext_map[doi] = entry
    return fulltext_map


def _collect_si_for_doi(
    si_snippets: list[dict[str, Any]],
    doi: str,
    *,
    max_snippets: int,
    max_chars: int,
) -> list[str]:
    if not doi:
        return []
    items: list[str] = []
    for entry in si_snippets:
        if not isinstance(entry, dict):
            continue
        entry_doi = _normalize_doi(entry.get("doi"))
        if entry_doi != doi:
            continue
        snippet = entry.get("snippet")
        if not isinstance(snippet, str) or not snippet.strip():
            continue
        snippet = snippet.strip()
        if max_chars and len(snippet) > max_chars:
            snippet = snippet[:max_chars]
        items.append(snippet)
        if max_snippets and len(items) >= max_snippets:
            break
    return items


def _format_json_list(values: list[str]) -> list[str]:
    return [value for value in values if isinstance(value, str) and value.strip()]


def main() -> None:
    args = parse_args()
    state_paths = _resolve_state_paths(args.run_id, args.state_path)
    state = _load_state(state_paths[0])

    scientific_references = state.get("scientific_references")
    if not isinstance(scientific_references, dict):
        raise SystemExit("Missing scientific_references in state.")

    if not args.secrets.exists():
        raise SystemExit(f"Secrets file not found: {args.secrets}")
    api_key, model, base_url = load_secrets(args.secrets)
    llm = OpenAIResponsesLLM(api_key=api_key, model=model, base_url=base_url)

    fulltext_map = _build_fulltext_map(scientific_references)
    si_snippets = _load_si_snippets(scientific_references)
    if si_snippets:
        scientific_references = dict(scientific_references)
        scientific_references["si_snippets"] = si_snippets

    infos = _collect_reference_infos(scientific_references)
    if not infos:
        raise SystemExit("No references available for usage tagging.")

    results: list[dict[str, Any]] = []
    for info in infos:
        if not isinstance(info, dict):
            continue
        doi = _normalize_doi(info.get("doi"))
        key = str(info.get("key") or "").strip()
        title = str(info.get("title") or info.get("short_name") or "").strip()
        citation = str(info.get("citation") or "").strip()
        origin_steps = _format_json_list(info.get("origin_steps") or [])

        fulltext_snippet = ""
        fulltext_entry = fulltext_map.get(doi) if doi else None
        if isinstance(fulltext_entry, dict):
            records = fulltext_entry.get("records") or []
            records_list = [item for item in records if isinstance(item, dict)]
            fulltext_snippet = _collect_article_text(records_list, max_chars=args.max_chars, max_records=args.max_records)

        si_texts = _collect_si_for_doi(
            si_snippets,
            doi,
            max_snippets=args.max_si_snippets,
            max_chars=args.max_si_chars,
        )

        if not fulltext_snippet and not si_texts and not title and not citation:
            results.append(
                {
                    "key": key or None,
                    "doi": doi or None,
                    "usage_tags": ["background_only"],
                    "reason": "No usable content for tagging.",
                }
            )
            continue

        payload = {
            "prompt": REFERENCE_USAGE_TAGGING_PROMPT,
            "context": {
                "title": title,
                "citation": citation,
                "origin_steps": origin_steps,
                "fulltext_snippet": fulltext_snippet,
                "si_snippets": si_texts,
            },
            "response_format": {"type": "json_object"},
        }
        raw = llm.invoke(payload)
        data = raw if isinstance(raw, dict) else parse_json_response(raw)
        if not isinstance(data, dict):
            data = {}
        usage_tags = _normalize_tags(data.get("usage_tags") or data.get("usageTags"))
        reason = str(data.get("reason") or "").strip()
        results.append(
            {
                "key": key or None,
                "doi": doi or None,
                "usage_tags": usage_tags,
                "reason": reason,
            }
        )

    tag_map: dict[str, list[str]] = {}
    for item in results:
        doi = _normalize_doi(item.get("doi"))
        tags = item.get("usage_tags")
        if doi and isinstance(tags, list):
            tag_map[doi] = tags

    clusters = scientific_references.get("step_1c_reference_clusters")
    if isinstance(clusters, dict):
        summaries = clusters.get("reference_summaries")
        if isinstance(summaries, list):
            updated_summaries: list[dict[str, Any]] = []
            for summary in summaries:
                if not isinstance(summary, dict):
                    continue
                doi = _normalize_doi(summary.get("doi"))
                tags = tag_map.get(doi)
                if tags:
                    summary = dict(summary)
                    summary["usage_tags"] = tags
                updated_summaries.append(summary)
            clusters = dict(clusters)
            clusters["reference_summaries"] = updated_summaries
            scientific_references["step_1c_reference_clusters"] = clusters

    scientific_references["usage_tagging"] = {
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results,
    }

    state["scientific_references"] = scientific_references
    for path in state_paths:
        dump_json(state, path)


if __name__ == "__main__":  # pragma: no cover
    main()
