#!/usr/bin/env python
"""Split process_from_flow SI files via mineru_with_images.

Defaults to artifacts/process_from_flow/<run_id>/input/si_mineru when output is not provided.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from tiangong_lca_spec.utils.mineru_with_images import (
    DEFAULT_MINERU_SECTION,
    MineruWithImagesClient,
    load_mineru_with_images_config,
)

PROCESS_FROM_FLOW_ARTIFACTS_ROOT = Path("artifacts/process_from_flow")
LATEST_RUN_ID_PATH = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / ".latest_run_id"
DEFAULT_OUTPUT_SUBDIR = Path("input/si_mineru")
DEFAULT_SI_SUBDIR = Path("input/si")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Path to the document file.")
    parser.add_argument(
        "--run-id",
        help="Run ID under artifacts/process_from_flow (inferred from input path or .latest_run_id if omitted).",
    )
    parser.add_argument("--output", type=Path, help="Write the response to this file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory when --output is not set (default: run_id/input/si_mineru).",
    )
    parser.add_argument(
        "--secrets-path",
        type=Path,
        default=Path(".secrets/secrets.toml"),
        help="Secrets TOML path.",
    )
    parser.add_argument(
        "--section",
        default=DEFAULT_MINERU_SECTION,
        help="Secrets section name for the mineru_with_images config.",
    )
    parser.add_argument("--url", help="Override the service URL.")
    parser.add_argument("--api-key", help="Override the service API key.")
    parser.add_argument("--provider", help="Override the provider form field.")
    parser.add_argument("--model", help="Override the model form field.")
    chunk_group = parser.add_mutually_exclusive_group()
    chunk_group.add_argument("--chunk-type", dest="chunk_type", action="store_true", help="Send chunk_type=true.")
    chunk_group.add_argument("--no-chunk-type", dest="chunk_type", action="store_false", help="Send chunk_type=false.")
    parser.set_defaults(chunk_type=None)
    ssl_group = parser.add_mutually_exclusive_group()
    ssl_group.add_argument("--verify-ssl", dest="verify_ssl", action="store_true", help="Verify TLS certificates.")
    ssl_group.add_argument("--no-verify-ssl", dest="verify_ssl", action="store_false", help="Disable TLS verification.")
    parser.set_defaults(verify_ssl=None)
    parser.add_argument("--timeout", type=float, help="Request timeout in seconds.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument("--no-update-state", action="store_true", help="Skip writing mineru metadata back to the state file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise SystemExit(f"Input file not found: {args.input_path}")

    run_id = args.run_id or _infer_run_id_from_path(args.input_path) or _load_latest_run_id()
    if args.output is None and args.output_dir is None and run_id is None:
        raise SystemExit("Provide --run-id/--output/--output-dir, or place input under artifacts/process_from_flow/<run_id>/input/si.")

    config = load_mineru_with_images_config(args.secrets_path, section_name=args.section)
    overrides = {}
    if args.url:
        overrides["url"] = args.url
    api_key = _normalize_optional_text(args.api_key)
    if api_key is not None:
        overrides["api_key"] = _sanitize_api_key(api_key, config.api_key_prefix)
    if args.timeout is not None:
        overrides["timeout"] = args.timeout
    if args.verify_ssl is not None:
        overrides["verify_ssl"] = args.verify_ssl
    if overrides:
        config = replace(config, **overrides)

    client = MineruWithImagesClient(config)
    try:
        payload = client.split_document(
            args.input_path,
            provider=_normalize_optional_text(args.provider),
            model=_normalize_optional_text(args.model),
            chunk_type=args.chunk_type,
            timeout=args.timeout,
        )
    except httpx.HTTPError as exc:
        raise SystemExit(_format_http_error(exc)) from exc

    output_text = _format_payload(payload, pretty=args.pretty)
    if args.output is not None:
        output_path = args.output
    else:
        output_dir = args.output_dir or _default_output_dir(run_id)
        output_path = _default_output_path(output_dir, args.input_path, run_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8")
    print(f"Saved response to {output_path}")

    if not args.no_update_state and run_id:
        _update_state_with_mineru_output(
            run_id=run_id,
            input_path=args.input_path,
            output_path=output_path,
        )


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _sanitize_api_key(value: str, prefix: str | None) -> str:
    token = value.strip()
    if not token:
        return token
    prefix_text = prefix.strip() if isinstance(prefix, str) else ""
    if prefix_text and token.lower().startswith(f"{prefix_text.lower()} "):
        token = token[len(prefix_text) + 1 :].strip()
    return token


def _format_payload(payload: Any, *, pretty: bool) -> str:
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    if isinstance(payload, str):
        return payload
    indent = 2 if pretty else None
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def _format_http_error(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        response = exc.response
        status = response.status_code
        detail = response.text.strip()
        return f"Request failed ({status}): {detail or response.reason_phrase}"
    return f"Request failed: {exc}"


def _default_output_dir(run_id: str | None) -> Path:
    if not run_id:
        raise SystemExit("Missing run_id for default output directory.")
    return PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id / DEFAULT_OUTPUT_SUBDIR


def _default_output_path(output_dir: Path, input_path: Path, run_id: str | None) -> Path:
    if run_id:
        si_root = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id / DEFAULT_SI_SUBDIR
        try:
            relative = input_path.resolve().relative_to(si_root.resolve())
            output_path = output_dir / relative
            return output_path.with_suffix(".json")
        except ValueError:
            pass
    stem = input_path.stem or "mineru_output"
    return output_dir / f"{stem}.json"


def _infer_doi_from_path(input_path: Path, run_id: str) -> str | None:
    si_root = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id / DEFAULT_SI_SUBDIR
    try:
        relative = input_path.resolve().relative_to(si_root.resolve())
    except ValueError:
        return None
    if not relative.parts:
        return None
    return relative.parts[0]


def _infer_run_id_from_path(input_path: Path) -> str | None:
    parts = list(input_path.parts)
    for idx, part in enumerate(parts):
        if part == "process_from_flow" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _load_latest_run_id() -> str | None:
    if not LATEST_RUN_ID_PATH.exists():
        return None
    text = LATEST_RUN_ID_PATH.read_text(encoding="utf-8").strip()
    return text or None


def _update_state_with_mineru_output(*, run_id: str, input_path: Path, output_path: Path) -> None:
    state_path = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id / "cache" / "process_from_flow_state.json"
    if not state_path.exists():
        return
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return
    scientific_references = payload.get("scientific_references")
    if not isinstance(scientific_references, dict):
        scientific_references = {}
    entries = scientific_references.get("si_mineru_outputs")
    if not isinstance(entries, list):
        entries = []
    doi_sanitized = _infer_doi_from_path(input_path, run_id)
    doi = _lookup_original_doi(scientific_references, input_path) or doi_sanitized
    entries.append(
        {
            "doi": doi,
            "doi_sanitized": doi_sanitized,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "status": "ok",
        }
    )
    scientific_references["si_mineru_outputs"] = entries
    payload["scientific_references"] = scientific_references
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _lookup_original_doi(scientific_references: dict[str, Any], input_path: Path) -> str | None:
    downloads = scientific_references.get("si_downloads")
    entries = downloads.get("entries") if isinstance(downloads, dict) else None
    if not isinstance(entries, list):
        return None
    input_resolved = input_path.resolve()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path_text = entry.get("path")
        if not isinstance(path_text, str) or not path_text:
            continue
        try:
            candidate_path = Path(path_text).resolve()
        except OSError:
            candidate_path = None
        if candidate_path and candidate_path == input_resolved:
            doi_value = entry.get("doi")
            if isinstance(doi_value, str) and doi_value.strip():
                return doi_value.strip()
        if path_text == str(input_path):
            doi_value = entry.get("doi")
            if isinstance(doi_value, str) and doi_value.strip():
                return doi_value.strip()
    return None


if __name__ == "__main__":
    main()
