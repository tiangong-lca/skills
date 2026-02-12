#!/usr/bin/env python
"""Download supporting information files by DOI for process_from_flow."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

PROCESS_FROM_FLOW_ARTIFACTS_ROOT = Path("artifacts/process_from_flow")
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_LINKS = 10

KEYWORDS = (
    "supporting information",
    "supplementary information",
    "supplementary material",
    "supplementary data",
    "supplemental material",
    "supplemental data",
    "supporting data",
    "appendix",
    "appendices",
    "additional file",
    "electronic supplementary",
    "supplementary",
    "supplemental",
    "esm",
)
FILE_EXTS = (".pdf", ".zip", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".txt", ".ppt", ".pptx")


@dataclass(frozen=True)
class LinkCandidate:
    url: str
    text: str
    reason: str


class _AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._current_href: str | None = None
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = None
        for key, value in attrs:
            if key.lower() == "href" and value:
                href = value
                break
        if href:
            self._current_href = href
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._current_href is None:
            return
        text = "".join(self._current_text).strip()
        self.links.append((self._current_href, text))
        self._current_href = None
        self._current_text = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Run ID under artifacts/process_from_flow.")
    parser.add_argument("--state-path", type=Path, help="Explicit state JSON path to load.")
    parser.add_argument("--doi", action="append", help="DOI to fetch (repeatable).")
    parser.add_argument("--min-si-hint", default="possible", help="Filter usability results by si_hint (none|possible|likely).")
    parser.add_argument("--cluster", help="Filter by step_1c_reference_clusters cluster_id (or 'primary').")
    parser.add_argument(
        "--recommendation",
        help="Filter by step_1c_reference_clusters recommendation (primary|supplement|exclude).",
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory for SI downloads.")
    parser.add_argument("--max-links", type=int, default=DEFAULT_MAX_LINKS, help="Max candidate links per DOI.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="List candidate SI links without downloading.")
    parser.add_argument("--no-update-state", action="store_true", help="Skip writing SI metadata back to the state file.")
    parser.add_argument("--user-agent", default="Mozilla/5.0", help="User-Agent header.")
    return parser.parse_args()


def _resolve_state_path(run_id: str | None, state_path: Path | None) -> Path | None:
    if state_path:
        return state_path
    if not run_id:
        return None
    path = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id / "cache" / "process_from_flow_state.json"
    return path if path.exists() else None


def _load_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hint_rank(value: str) -> int:
    normalized = value.strip().lower()
    if normalized == "likely":
        return 2
    if normalized == "possible":
        return 1
    return 0


def _collect_usability_hint_map(state: dict[str, Any]) -> dict[str, str]:
    refs = state.get("scientific_references") or {}
    usability = refs.get("usability")
    results = usability.get("results") if isinstance(usability, dict) else None
    hints: dict[str, str] = {}
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            doi = str(item.get("doi") or "").strip()
            if doi:
                hints[doi] = str(item.get("si_hint") or "none").strip().lower()
    return hints


def _apply_si_hint_filter(
    dois: list[str],
    *,
    state: dict[str, Any],
    min_hint: str,
    keep_unknown: bool,
) -> list[str]:
    min_rank = _hint_rank(min_hint)
    if min_rank <= 0:
        return sorted(set(dois))
    hint_map = _collect_usability_hint_map(state)
    filtered: list[str] = []
    for doi in dois:
        hint = hint_map.get(doi)
        if hint is None:
            if keep_unknown:
                filtered.append(doi)
            continue
        if _hint_rank(hint) >= min_rank:
            filtered.append(doi)
    return sorted(set(filtered))


def _collect_dois_from_clusters(
    state: dict[str, Any],
    *,
    cluster_id: str | None,
    recommendation: str | None,
) -> list[str]:
    refs = state.get("scientific_references") or {}
    clusters_block = refs.get("step_1c_reference_clusters")
    if not isinstance(clusters_block, dict):
        return []
    clusters = clusters_block.get("clusters")
    if not isinstance(clusters, list):
        return []
    cluster_id = (cluster_id or "").strip()
    recommendation = (recommendation or "").strip().lower()
    primary_id = str(clusters_block.get("primary_cluster_id") or "").strip()
    if cluster_id.lower() == "primary" and primary_id:
        cluster_id = primary_id
    selected: list[str] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cid = str(cluster.get("cluster_id") or "").strip()
        rec = str(cluster.get("recommendation") or "").strip().lower()
        if cluster_id and cid != cluster_id:
            continue
        if recommendation and rec != recommendation:
            continue
        dois = [str(item).strip() for item in (cluster.get("dois") or []) if str(item).strip()]
        selected.extend(dois)
    return sorted(set(selected))


def _collect_dois_from_state(
    state: dict[str, Any],
    *,
    min_hint: str,
    cluster_id: str | None,
    recommendation: str | None,
) -> list[str]:
    if cluster_id or recommendation:
        base = _collect_dois_from_clusters(state, cluster_id=cluster_id, recommendation=recommendation)
        if not base:
            return []
        return _apply_si_hint_filter(base, state=state, min_hint=min_hint, keep_unknown=True)

    min_rank = _hint_rank(min_hint)
    refs = state.get("scientific_references") or {}
    usability = refs.get("usability")
    results = usability.get("results") if isinstance(usability, dict) else None
    dois: list[str] = []
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            doi = str(item.get("doi") or "").strip()
            if not doi:
                continue
            hint = str(item.get("si_hint") or "none")
            if _hint_rank(hint) >= min_rank:
                dois.append(doi)
    if dois:
        return sorted(set(dois))
    step1b = refs.get("step_1b_reference_fulltext") if isinstance(refs, dict) else None
    references = step1b.get("references") if isinstance(step1b, dict) else None
    if isinstance(references, list):
        for item in references:
            if isinstance(item, dict):
                doi = str(item.get("doi") or "").strip()
                if doi:
                    dois.append(doi)
    return sorted(set(dois))


def _sanitize_doi(doi: str) -> str:
    cleaned = doi.strip().lower()
    cleaned = re.sub(r"[^a-z0-9._-]+", "_", cleaned)
    return cleaned.strip("_") or "unknown"


def _extract_links(html: str) -> list[tuple[str, str]]:
    parser = _AnchorParser()
    parser.feed(html)
    return parser.links


def _is_candidate(href: str, text: str) -> bool:
    combined = f"{href} {text}".lower()
    if any(keyword in combined for keyword in KEYWORDS):
        return True
    if any(href.lower().endswith(ext) for ext in FILE_EXTS) and "supp" in combined:
        return True
    return False


def _guess_extension(url: str, content_type: str | None) -> str:
    path = urlparse(url).path
    for ext in FILE_EXTS:
        if path.lower().endswith(ext):
            return ext
    if content_type:
        lowered = content_type.lower()
        if "pdf" in lowered:
            return ".pdf"
        if "zip" in lowered:
            return ".zip"
        if "excel" in lowered or "spreadsheet" in lowered:
            return ".xlsx"
        if "word" in lowered:
            return ".docx"
        if "csv" in lowered:
            return ".csv"
        if "text" in lowered:
            return ".txt"
    return ".bin"


def _is_same_page_anchor(url: str, base_url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.fragment:
        return False
    base = urlparse(base_url)
    if parsed.netloc != base.netloc or parsed.path != base.path:
        return False
    return parsed.fragment.lower().startswith("moesm")


def _dedupe_candidates(candidates: list[LinkCandidate]) -> list[LinkCandidate]:
    seen: set[str] = set()
    deduped: list[LinkCandidate] = []
    for item in candidates:
        if item.url in seen:
            continue
        seen.add(item.url)
        deduped.append(item)
    return deduped


def _find_candidates(html: str, base_url: str) -> list[LinkCandidate]:
    candidates: list[LinkCandidate] = []
    for href, text in _extract_links(html):
        if not href:
            continue
        url = urljoin(base_url, href)
        if _is_candidate(href, text):
            reason = text.strip() or "keyword match"
            candidates.append(LinkCandidate(url=url, text=text.strip(), reason=reason))
    filtered = [item for item in candidates if not _is_same_page_anchor(item.url, base_url)]
    return _dedupe_candidates(filtered)


def main() -> None:
    args = parse_args()
    state_path = _resolve_state_path(args.run_id, args.state_path)
    dois: list[str] = []
    if args.doi:
        dois = [item.strip() for item in args.doi if item and item.strip()]
    elif state_path:
        state = _load_state(state_path)
        dois = _collect_dois_from_state(
            state,
            min_hint=args.min_si_hint,
            cluster_id=args.cluster,
            recommendation=args.recommendation,
        )
    if not dois:
        raise SystemExit("No DOIs found. Provide --doi or a valid --run-id/--state-path with references.")

    output_dir = args.output_dir
    if output_dir is None:
        if not args.run_id:
            raise SystemExit("Provide --output-dir when --run-id is omitted.")
        output_dir = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / args.run_id / "input" / "si"
    output_dir.mkdir(parents=True, exist_ok=True)

    report: list[dict[str, Any]] = []
    headers = {"User-Agent": args.user_agent}
    with httpx.Client(follow_redirects=True, headers=headers, timeout=args.timeout) as client:
        for doi in dois:
            doi_entry: dict[str, Any] = {"doi": doi, "candidates": [], "downloads": []}
            landing_url = f"https://doi.org/{doi}"
            try:
                response = client.get(landing_url)
            except Exception as exc:  # pylint: disable=broad-except
                doi_entry["error"] = f"Landing page fetch failed: {exc}"
                report.append(doi_entry)
                continue
            doi_entry["landing_url"] = str(response.url)
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type.lower():
                doi_entry["error"] = f"Landing page is not HTML (content-type={content_type})."
                report.append(doi_entry)
                continue
            candidates = _find_candidates(response.text, str(response.url))
            if args.max_links:
                candidates = candidates[: args.max_links]
            doi_entry["candidates"] = [{"url": item.url, "text": item.text, "reason": item.reason} for item in candidates]

            if args.dry_run:
                report.append(doi_entry)
                continue

            doi_dir = output_dir / _sanitize_doi(doi)
            doi_dir.mkdir(parents=True, exist_ok=True)
            for idx, candidate in enumerate(candidates, start=1):
                try:
                    download = client.get(candidate.url)
                except Exception as exc:  # pylint: disable=broad-except
                    doi_entry["downloads"].append({"url": candidate.url, "status": "error", "error": str(exc)})
                    continue
                if download.status_code >= 400:
                    doi_entry["downloads"].append({"url": candidate.url, "status": "error", "error": f"HTTP {download.status_code}"})
                    continue
                download_type = download.headers.get("content-type")
                ext = _guess_extension(str(download.url), download_type)
                filename = doi_dir / f"si_{idx:02d}{ext}"
                filename.write_bytes(download.content)
                doi_entry["downloads"].append(
                    {
                        "url": str(download.url),
                        "status": "ok",
                        "path": str(filename),
                        "content_type": download_type,
                    }
                )
            report.append(doi_entry)

    report_path = output_dir / "si_download_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote report to {report_path}")

    if state_path and not args.no_update_state and not args.dry_run:
        state = _load_state(state_path)
        scientific_references = state.get("scientific_references")
        if not isinstance(scientific_references, dict):
            scientific_references = {}
        flat_entries: list[dict[str, Any]] = []
        for entry in report:
            if not isinstance(entry, dict):
                continue
            doi = str(entry.get("doi") or "").strip()
            downloads = entry.get("downloads") or []
            if not isinstance(downloads, list):
                continue
            for download in downloads:
                if not isinstance(download, dict):
                    continue
                flat_entries.append(
                    {
                        "doi": doi,
                        "url": download.get("url"),
                        "status": download.get("status"),
                        "path": download.get("path"),
                        "content_type": download.get("content_type"),
                        "error": download.get("error"),
                    }
                )
        scientific_references["si_downloads"] = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "report_path": str(report_path),
            "entries": flat_entries,
        }
        state["scientific_references"] = scientific_references
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Updated state with SI metadata at {state_path}")


if __name__ == "__main__":
    main()
