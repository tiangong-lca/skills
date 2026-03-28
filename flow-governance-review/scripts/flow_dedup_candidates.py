#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from flow_governance_common import (
    PUBLIC_TARGET_FLOW_TYPES,
    deep_get,
    dump_json,
    dump_jsonl,
    ensure_dir,
    extract_flow_record,
    filter_emergy_named_flows,
    flow_dataset_from_row,
    lang_entries,
    lang_text,
    listify,
    load_json_or_jsonl,
    normalize_text,
    postgrest_auth_password,
    postgrest_select_page,
    select_reference_flow_property,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ILCD_FLOW_DIR = (
    REPO_ROOT.parent / "EF-reference-package-3.1-Correction" / "EF-Reference-Package" / "ILCD" / "flows"
)
GENERIC_TOKENS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "other",
    "the",
    "to",
    "with",
}


@dataclass
class FlowPropertyMeta:
    uuid: str
    version: str
    name: str
    unitgroup_uuid: str
    unitgroup_name: str


@dataclass
class FlowSummary:
    flow_key: str
    flow_id: str
    version: str
    snapshot_roles: tuple[str, ...]
    flow_type: str
    state_code: int | None
    user_id: str
    public_origin: str
    is_ilcd_reference_flow: bool
    primary_name: str
    base_name_entries: list[str]
    flow_property_entries: list[str]
    mix_entries: list[str]
    route_entries: list[str]
    full_name_entries: list[str]
    base_name_fingerprint: str
    full_name_fingerprint: str
    concept_tokens: list[str]
    qualifier_tokens: list[str]
    qualifier_signature: str
    classification_leaf_key: str
    classification_leaf_text: str
    classification_path: list[str]
    flow_property_uuid: str
    flow_property_version: str
    flow_property_name: str
    unitgroup_uuid: str
    unitgroup_name: str
    reference_process_count: int
    process_ref_count: int
    exchange_ref_count: int
    uuid_reference_process_count: int
    uuid_process_ref_count: int
    uuid_exchange_ref_count: int


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self._parent.setdefault(value, value)

    def find(self, value: str) -> str:
        parent = self._parent.setdefault(value, value)
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root

    def groups(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for value in self._parent:
            grouped[self.find(value)].append(value)
        return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate flow dedup candidate clusters and canonical suggestions.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--flows-file", help="Legacy explicit full-snapshot flow rows JSON/JSONL file (all-vs-all comparison).")
    parser.add_argument(
        "--subject-flows-file",
        help="Local subject flow rows JSON/JSONL file. Candidate pairs are restricted to flows touching this subject set.",
    )
    parser.add_argument(
        "--candidate-flows-file",
        help="Optional local candidate-pool flow rows JSON/JSONL file. If omitted in subject mode, fetch the candidate pool live.",
    )
    parser.add_argument("--processes-file", help="Optional local process rows JSON/JSONL file for ref counting.")
    parser.add_argument(
        "--live-ref-counts",
        action="store_true",
        help="Enable live visible-process reference counting when --processes-file is not provided.",
    )
    parser.add_argument("--user-id", default=os.getenv("FLOW_GOVERNANCE_USER_ID", ""))
    parser.add_argument("--flow-id", action="append", dest="flow_ids", default=[])
    parser.add_argument("--no-user-0", action="store_true", help="Skip user-owned state_code=0 flow fetch when running live.")
    parser.add_argument("--no-public-100", action="store_true", help="Skip public state_code=100 flow fetch when running live.")
    parser.add_argument(
        "--exclude-emergy",
        action="store_true",
        help="Exclude emergy-named flows before generating duplicate candidates.",
    )
    parser.add_argument("--same-property-threshold", type=float, default=0.93)
    parser.add_argument("--cross-property-threshold", type=float, default=0.96)
    parser.add_argument("--near-threshold", type=float, default=0.84)
    parser.add_argument("--bucket-token-limit", type=int, default=6)
    parser.add_argument("--bucket-max-size", type=int, default=120)
    parser.add_argument("--process-page-size", type=int, default=100)
    parser.add_argument("--flow-page-size", type=int, default=200)
    parser.add_argument("--lookup-chunk-size", type=int, default=50)
    parser.add_argument(
        "--ref-count-mode",
        choices=("auto", "candidate_lookup", "full_scan"),
        default="auto",
        help="How to compute process reference counts used for canonical selection.",
    )
    parser.add_argument(
        "--full-scan-threshold",
        type=int,
        default=200,
        help="When ref-count-mode=auto and candidate flow ids exceed this threshold, switch to one full process scan.",
    )
    parser.add_argument(
        "--public-origin-filter",
        choices=("all", "tiangong", "ilcd"),
        default="all",
        help="When public state_code=100 rows are included, optionally keep only TianGong public or only ILCD rows.",
    )
    parser.add_argument(
        "--ilcd-flow-dir",
        default=os.getenv("FLOW_GOVERNANCE_ILCD_FLOW_DIR", str(DEFAULT_ILCD_FLOW_DIR)),
    )
    parser.add_argument("--supabase-url", default=os.getenv("SUPABASE_URL", ""))
    parser.add_argument("--supabase-publishable-key", default=os.getenv("SUPABASE_PUBLISHABLE_KEY", ""))
    parser.add_argument("--access-token", default=os.getenv("SUPABASE_ACCESS_TOKEN", ""))
    parser.add_argument("--email", default=os.getenv("SUPABASE_EMAIL", ""))
    parser.add_argument("--password", default=os.getenv("SUPABASE_PASSWORD", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    ilcd_ids = load_ilcd_flow_ids(args.ilcd_flow_dir)
    access_token = resolve_access_token(args)

    flow_input = resolve_flow_input_rows(args, access_token, out_dir, ilcd_ids)
    flow_rows = flow_input["merged_rows"]
    subject_keys = flow_input["subject_keys"]
    snapshot_role_map = flow_input["snapshot_role_map"]
    if not flow_rows:
        raise SystemExit("No flow rows were loaded for dedup candidate analysis.")

    flow_property_meta = fetch_flow_property_meta(args, access_token, collect_flow_property_ids(flow_rows))
    if args.exclude_emergy:
        flow_rows, emergy_excluded_flows = filter_emergy_named_flows(flow_rows, flow_property_meta=flow_property_meta)
        if not flow_rows:
            raise SystemExit("All loaded flow rows were excluded by the emergy-named scope rule.")
        subject_keys = {flow_row_key(row) for row in flow_rows if flow_row_key(row) in subject_keys}
    else:
        emergy_excluded_flows = []
    provisional_ref_counts = empty_reference_counts()
    provisional_summaries = build_flow_summaries(flow_rows, flow_property_meta, provisional_ref_counts, ilcd_ids, snapshot_role_map)
    provisional_pairs = build_candidate_pairs(provisional_summaries, args, subject_keys=subject_keys)
    candidate_flow_ids = collect_candidate_flow_ids(provisional_pairs)
    ref_counts = build_reference_counts(args, access_token, candidate_flow_ids)
    summaries = build_flow_summaries(flow_rows, flow_property_meta, ref_counts, ilcd_ids, snapshot_role_map)
    pair_rows = build_candidate_pairs(summaries, args, subject_keys=subject_keys)
    clusters, canonical_map, manual_review_rows = build_clusters_and_maps(summaries, pair_rows)
    rewrite_plan = build_rewrite_plan(clusters, canonical_map)
    version_bump_plan = build_version_bump_plan(clusters)
    rewrite_plan_next_version = project_rewrite_plan_to_next_versions(rewrite_plan, version_bump_plan)

    summary_payload = {
        "input_mode": flow_input["input_mode"],
        "subject_flow_count": flow_input["subject_flow_count"],
        "candidate_pool_flow_count": flow_input["candidate_pool_flow_count"],
        "merged_snapshot_flow_count": flow_input["merged_snapshot_flow_count"],
        "candidate_pool_source": flow_input["candidate_pool_source"],
        "flow_rows_before_emergy_exclusion": len(flow_rows) + len(emergy_excluded_flows),
        "flow_rows_analyzed": len(summaries),
        "emergy_excluded_flow_count": len(emergy_excluded_flows),
        "public_origin_filter": args.public_origin_filter,
        "candidate_pair_count": len(pair_rows),
        "pair_label_counts": dict(Counter(item["pair_label"] for item in pair_rows)),
        "cluster_count": len(clusters),
        "same_property_auto_merge_candidates": sum(
            1
            for entry in canonical_map["property_safe_rewrite_map"].values()
            if entry.get("rewrite_scope") == "same_property"
        ),
        "cross_property_family_candidates": sum(1 for cluster in clusters if cluster.get("cross_property_member_count", 0) > 0),
        "manual_review_count": len(manual_review_rows),
        "rewrite_plan_count": len(rewrite_plan),
        "rewrite_plan_next_version_count": len(rewrite_plan_next_version),
        "canonical_version_bump_count": len(version_bump_plan),
        "processes_scanned_for_ref_count": ref_counts["meta"]["process_count_scanned"],
        "flow_ids_counted_for_ref_count": len(candidate_flow_ids),
        "ref_count_source": ref_counts["meta"]["source_mode"],
        "visible_scope_note": ref_counts["meta"]["visibility_scope_note"],
        "notes": [
            "family canonical collapses same-concept duplicate families for governance review",
            "property safe rewrite map is the only auto-merge-safe output; it never rewrites across different flow properties",
            "same-UUID version lineage is outside semantic dedup scope and is not auto-merged here",
            "process reference counts are visibility-bounded by current Supabase/PostgREST permissions",
            "emergy-named flows are excluded only when --exclude-emergy is set",
        ],
    }

    dump_json(out_dir / "flow-dedup-summary.json", summary_payload)
    dump_json(out_dir / "flow-dedup-emergy-excluded.json", emergy_excluded_flows)
    dump_json(out_dir / "flow-dedup-candidate-clusters.json", clusters)
    dump_json(out_dir / "flow-dedup-canonical-map.json", canonical_map)
    dump_json(out_dir / "flow-dedup-rewrite-plan.json", rewrite_plan)
    dump_json(out_dir / "flow-dedup-rewrite-plan-next-version.json", rewrite_plan_next_version)
    dump_json(out_dir / "flow-dedup-version-bump-plan.json", version_bump_plan)
    dump_jsonl(out_dir / "flow-dedup-manual-review.jsonl", manual_review_rows)
    (out_dir / "flow-dedup-rewrite-plan.tsv").write_text(render_rewrite_plan_tsv(rewrite_plan), encoding="utf-8")
    (out_dir / "flow-dedup-rewrite-plan-next-version.tsv").write_text(
        render_rewrite_plan_tsv(rewrite_plan_next_version),
        encoding="utf-8",
    )
    (out_dir / "flow-dedup-sql-rewrite-template.sql").write_text(
        render_sql_rewrite_template(rewrite_plan, rewrite_plan_next_version, version_bump_plan),
        encoding="utf-8",
    )
    (out_dir / "flow-dedup-candidate-clusters.md").write_text(
        render_markdown(summary_payload, clusters),
        encoding="utf-8",
    )


def resolve_flow_input_rows(
    args: argparse.Namespace,
    access_token: str | None,
    out_dir: Path,
    ilcd_ids: set[str],
) -> dict[str, Any]:
    if args.flows_file and (args.subject_flows_file or args.candidate_flows_file):
        raise SystemExit("Use either --flows-file or --subject-flows-file/--candidate-flows-file, not both.")
    if args.candidate_flows_file and not args.subject_flows_file:
        raise SystemExit("--candidate-flows-file requires --subject-flows-file.")

    if args.flows_file:
        merged_rows = [row for row in load_json_or_jsonl(args.flows_file) if isinstance(row, dict)]
        dump_jsonl(out_dir / "merged-flow-snapshot.jsonl", merged_rows)
        dump_json(
            out_dir / "flow-snapshot-manifest.json",
            {
                "input_mode": "explicit_full_snapshot",
                "subject_flow_count": len(merged_rows),
                "candidate_pool_flow_count": len(merged_rows),
                "merged_snapshot_flow_count": len(merged_rows),
                "candidate_pool_source": "same_as_flows_file",
                "files": {
                    "merged_flow_snapshot": str((out_dir / "merged-flow-snapshot.jsonl").resolve()),
                },
            },
        )
        return {
            "input_mode": "explicit_full_snapshot",
            "subject_keys": set(),
            "snapshot_role_map": {flow_row_key(row): ("snapshot",) for row in merged_rows},
            "subject_flow_count": len(merged_rows),
            "candidate_pool_flow_count": len(merged_rows),
            "merged_snapshot_flow_count": len(merged_rows),
            "candidate_pool_source": "same_as_flows_file",
            "merged_rows": merged_rows,
        }

    if args.subject_flows_file:
        subject_rows = [row for row in load_json_or_jsonl(args.subject_flows_file) if isinstance(row, dict)]
        if args.candidate_flows_file:
            candidate_rows = [row for row in load_json_or_jsonl(args.candidate_flows_file) if isinstance(row, dict)]
            candidate_source = "candidate_flows_file"
        else:
            candidate_rows = fetch_candidate_pool_rows(args, access_token, ilcd_ids)
            candidate_source = "live_candidate_pool"
        merged_rows, snapshot_role_map = merge_snapshot_rows(subject_rows, candidate_rows)
        subject_keys = {flow_row_key(row) for row in subject_rows}
        dump_jsonl(out_dir / "subject-flows.snapshot.jsonl", subject_rows)
        dump_jsonl(out_dir / "candidate-pool-flows.snapshot.jsonl", candidate_rows)
        dump_jsonl(out_dir / "merged-flow-snapshot.jsonl", merged_rows)
        dump_json(
            out_dir / "flow-snapshot-manifest.json",
            {
                "input_mode": "subject_candidate_snapshot",
                "subject_flow_count": len(subject_rows),
                "candidate_pool_flow_count": len(candidate_rows),
                "merged_snapshot_flow_count": len(merged_rows),
                "candidate_pool_source": candidate_source,
                "files": {
                    "subject_snapshot": str((out_dir / "subject-flows.snapshot.jsonl").resolve()),
                    "candidate_pool_snapshot": str((out_dir / "candidate-pool-flows.snapshot.jsonl").resolve()),
                    "merged_flow_snapshot": str((out_dir / "merged-flow-snapshot.jsonl").resolve()),
                },
            },
        )
        return {
            "input_mode": "subject_candidate_snapshot",
            "subject_keys": subject_keys,
            "snapshot_role_map": snapshot_role_map,
            "subject_flow_count": len(subject_rows),
            "candidate_pool_flow_count": len(candidate_rows),
            "merged_snapshot_flow_count": len(merged_rows),
            "candidate_pool_source": candidate_source,
            "merged_rows": merged_rows,
        }

    merged_rows = fetch_candidate_pool_rows(args, access_token, ilcd_ids)
    dump_jsonl(out_dir / "merged-flow-snapshot.jsonl", merged_rows)
    dump_json(
        out_dir / "flow-snapshot-manifest.json",
        {
            "input_mode": "live_full_snapshot",
            "subject_flow_count": len(merged_rows),
            "candidate_pool_flow_count": len(merged_rows),
            "merged_snapshot_flow_count": len(merged_rows),
            "candidate_pool_source": "live_candidate_pool",
            "files": {
                "merged_flow_snapshot": str((out_dir / "merged-flow-snapshot.jsonl").resolve()),
            },
        },
    )
    return {
        "input_mode": "live_full_snapshot",
        "subject_keys": set(),
        "snapshot_role_map": {flow_row_key(row): ("snapshot",) for row in merged_rows},
        "subject_flow_count": len(merged_rows),
        "candidate_pool_flow_count": len(merged_rows),
        "merged_snapshot_flow_count": len(merged_rows),
        "candidate_pool_source": "live_candidate_pool",
        "merged_rows": merged_rows,
    }


def merge_snapshot_rows(
    subject_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, ...]]]:
    merged: dict[str, dict[str, Any]] = {}
    roles: dict[str, set[str]] = {}
    for row in candidate_rows:
        key = flow_row_key(row)
        if not key:
            continue
        merged.setdefault(key, row)
        roles.setdefault(key, set()).add("candidate")
    for row in subject_rows:
        key = flow_row_key(row)
        if not key:
            continue
        merged[key] = row
        roles.setdefault(key, set()).add("subject")
    sorted_rows = sorted(merged.values(), key=lambda row: (extract_flow_record(row).name, flow_row_key(row)))
    return sorted_rows, {key: tuple(sorted(value)) for key, value in roles.items()}


def flow_row_key(row: dict[str, Any]) -> str:
    record = extract_flow_record(row)
    if not record.id:
        return ""
    return f"{record.id}@{record.version}"


def resolve_access_token(args: argparse.Namespace) -> str | None:
    needs_live_candidate_pool = (not args.flows_file and not args.subject_flows_file) or (
        bool(args.subject_flows_file) and not bool(args.candidate_flows_file)
    )
    needs_live_ref_counts = bool(args.live_ref_counts)
    if not (needs_live_candidate_pool or needs_live_ref_counts):
        return None
    if not args.supabase_url or not args.supabase_publishable_key:
        if needs_live_candidate_pool:
            raise SystemExit("SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY are required for live candidate-pool analysis.")
        return None
    access_token = str(args.access_token or "").strip() or None
    if not access_token and args.email and args.password:
        access_token = postgrest_auth_password(
            base_url=args.supabase_url,
            apikey=args.supabase_publishable_key,
            email=args.email,
            password=args.password,
        )
    return access_token


def fetch_candidate_pool_rows(
    args: argparse.Namespace,
    access_token: str | None,
    ilcd_ids: set[str],
) -> list[dict[str, Any]]:
    if not access_token:
        raise SystemExit("Authenticated access token is required for live flow analysis.")

    if args.flow_ids:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for chunk in chunked(sorted(set(args.flow_ids)), args.lookup_chunk_size):
            batch = postgrest_select_page(
                base_url=args.supabase_url,
                apikey=args.supabase_publishable_key,
                access_token=access_token,
                table="flows",
                raw_filters={"id": "in.(" + ",".join(chunk) + ")"},
                columns="id,version,state_code,user_id,json",
                limit=max(len(chunk), 1),
                offset=0,
            )
            for row in batch:
                key = f"{row.get('id')}@{row.get('version')}"
                if key not in seen:
                    rows.append(row)
                    seen.add(key)
        return filter_rows_by_public_origin(filter_target_flow_rows(rows), ilcd_ids, args.public_origin_filter)

    rows: list[dict[str, Any]] = []
    if not args.no_public_100:
        rows.extend(
            fetch_flow_pages(
                base_url=args.supabase_url,
                apikey=args.supabase_publishable_key,
                access_token=access_token,
                raw_filters={
                    "state_code": "eq.100",
                    "json->flowDataSet->modellingAndValidation->LCIMethod->>typeOfDataSet": "in.(Product flow,Waste flow)",
                },
                page_size=args.flow_page_size,
            )
        )
    if not args.no_user_0:
        if not args.user_id:
            raise SystemExit("Pass --user-id or set --no-user-0 when loading live state_code=0 flows.")
        rows.extend(
            fetch_flow_pages(
                base_url=args.supabase_url,
                apikey=args.supabase_publishable_key,
                access_token=access_token,
                raw_filters={
                    "state_code": "eq.0",
                    "user_id": f"eq.{args.user_id}",
                    "json->flowDataSet->modellingAndValidation->LCIMethod->>typeOfDataSet": "in.(Product flow,Waste flow)",
                },
                page_size=args.flow_page_size,
            )
        )
    return filter_rows_by_public_origin(filter_target_flow_rows(rows), ilcd_ids, args.public_origin_filter)


def fetch_flow_pages(
    *,
    base_url: str,
    apikey: str,
    access_token: str,
    raw_filters: dict[str, str],
    page_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        batch = postgrest_select_page(
            base_url=base_url,
            apikey=apikey,
            access_token=access_token,
            table="flows",
            raw_filters=raw_filters,
            columns="id,version,state_code,user_id,json",
            limit=page_size,
            offset=offset,
            order="id.asc",
        )
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def filter_target_flow_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        dataset = flow_dataset_from_row(row)
        flow_type = str(
            deep_get(dataset, ["modellingAndValidation", "LCIMethodAndAllocation", "typeOfDataSet"])
            or deep_get(dataset, ["modellingAndValidation", "LCIMethod", "typeOfDataSet"])
            or row.get("typeOfDataSet")
            or ""
        ).strip()
        if flow_type not in PUBLIC_TARGET_FLOW_TYPES:
            continue
        if row.get("state_code") == 200:
            continue
        key = f"{row.get('id')}@{row.get('version')}"
        if key in seen:
            continue
        filtered.append(row)
        seen.add(key)
    return filtered


def filter_rows_by_public_origin(
    rows: list[dict[str, Any]],
    ilcd_ids: set[str],
    public_origin_filter: str,
) -> list[dict[str, Any]]:
    if public_origin_filter == "all":
        return rows
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if row.get("state_code") != 100:
            filtered.append(row)
            continue
        flow_id = str(row.get("id") or "").strip()
        is_ilcd = flow_id in ilcd_ids
        if public_origin_filter == "tiangong" and not is_ilcd:
            filtered.append(row)
        elif public_origin_filter == "ilcd" and is_ilcd:
            filtered.append(row)
    return filtered


def collect_flow_property_ids(flow_rows: list[dict[str, Any]]) -> set[str]:
    property_ids: set[str] = set()
    for row in flow_rows:
        prop_ref = select_reference_flow_property(flow_dataset_from_row(row))
        flow_property_uuid = str(prop_ref.get("@refObjectId") or "").strip()
        if flow_property_uuid:
            property_ids.add(flow_property_uuid)
    return property_ids


def fetch_flow_property_meta(
    args: argparse.Namespace,
    access_token: str | None,
    property_ids: set[str],
) -> dict[str, FlowPropertyMeta]:
    if not property_ids or not access_token or not args.supabase_url or not args.supabase_publishable_key:
        return {}
    meta: dict[str, FlowPropertyMeta] = {}
    for chunk in chunked(sorted(property_ids), args.lookup_chunk_size):
        batch = postgrest_select_page(
            base_url=args.supabase_url,
            apikey=args.supabase_publishable_key,
            access_token=access_token,
            table="flowproperties",
            raw_filters={"id": "in.(" + ",".join(chunk) + ")"},
            columns="id,version,json",
            limit=max(len(chunk), 1),
            offset=0,
        )
        for row in batch:
            data = row.get("json", {}).get("flowPropertyDataSet", row.get("json", {}))
            info = deep_get(data, ["flowPropertiesInformation", "dataSetInformation"], {})
            quant = deep_get(data, ["flowPropertiesInformation", "quantitativeReference"], {})
            ref_unit_group = quant.get("referenceToReferenceUnitGroup") if isinstance(quant, dict) else {}
            prop_id = str(row.get("id") or deep_get(info, ["common:UUID"]) or "").strip()
            if not prop_id:
                continue
            candidate = FlowPropertyMeta(
                uuid=prop_id,
                version=str(row.get("version") or "").strip(),
                name=lang_text(info.get("common:name")) or prop_id,
                unitgroup_uuid=str(ref_unit_group.get("@refObjectId") or "").strip() if isinstance(ref_unit_group, dict) else "",
                unitgroup_name=lang_text(ref_unit_group.get("common:shortDescription")) if isinstance(ref_unit_group, dict) else "",
            )
            current = meta.get(prop_id)
            if not current or version_key(candidate.version) >= version_key(current.version):
                meta[prop_id] = candidate
    return meta


def empty_reference_counts() -> dict[str, Any]:
    return {
        "reference_process_count_by_flow_key": {},
        "process_count_by_flow_key": {},
        "exchange_count_by_flow_key": {},
        "reference_process_count_by_uuid": {},
        "process_count_by_uuid": {},
        "exchange_count_by_uuid": {},
        "meta": {
            "process_count_scanned": 0,
            "source_mode": "none",
            "visibility_scope_note": "Reference counts not computed yet.",
        },
    }


def build_reference_counts(
    args: argparse.Namespace,
    access_token: str | None,
    flow_ids: set[str],
) -> dict[str, Any]:
    reference_process_count_by_flow_key: Counter[str] = Counter()
    process_count_by_flow_key: Counter[str] = Counter()
    exchange_count_by_flow_key: Counter[str] = Counter()
    reference_process_count_by_uuid: Counter[str] = Counter()
    process_count_by_uuid: Counter[str] = Counter()
    exchange_count_by_uuid: Counter[str] = Counter()
    process_keys_scanned: set[str] = set()

    source_mode = "none"
    visibility_scope_note = "Reference counts not computed yet."

    if args.processes_file:
        source_mode = "local_process_snapshot"
        visibility_scope_note = "Counts come from the explicit local processes snapshot passed via --processes-file."
        process_rows = [row for row in load_json_or_jsonl(args.processes_file) if isinstance(row, dict)]
        for row in process_rows:
            process_keys_scanned.add(process_identity_key(row))
            accumulate_process_counts(
                row=row,
                reference_process_count_by_flow_key=reference_process_count_by_flow_key,
                process_count_by_flow_key=process_count_by_flow_key,
                exchange_count_by_flow_key=exchange_count_by_flow_key,
                reference_process_count_by_uuid=reference_process_count_by_uuid,
                process_count_by_uuid=process_count_by_uuid,
                exchange_count_by_uuid=exchange_count_by_uuid,
            )
    elif access_token and args.live_ref_counts:
        source_mode = "live_visible_processes"
        visibility_scope_note = "Counts use indexed visible-process lookups on state_code in (0,100) only; other users' private state_code=0 rows remain invisible under current RLS."
        mode = args.ref_count_mode
        if mode == "auto":
            mode = "full_scan" if len(flow_ids) > args.full_scan_threshold else "candidate_lookup"
        if mode == "full_scan":
            process_keys_scanned = full_scan_reference_counts(
                args=args,
                access_token=access_token,
                reference_process_count_by_flow_key=reference_process_count_by_flow_key,
                process_count_by_flow_key=process_count_by_flow_key,
                exchange_count_by_flow_key=exchange_count_by_flow_key,
                reference_process_count_by_uuid=reference_process_count_by_uuid,
                process_count_by_uuid=process_count_by_uuid,
                exchange_count_by_uuid=exchange_count_by_uuid,
            )
        elif flow_ids:
            process_keys_scanned = count_matching_process_refs(
                args=args,
                access_token=access_token,
                flow_ids=flow_ids,
                reference_process_count_by_flow_key=reference_process_count_by_flow_key,
                process_count_by_flow_key=process_count_by_flow_key,
                exchange_count_by_flow_key=exchange_count_by_flow_key,
                reference_process_count_by_uuid=reference_process_count_by_uuid,
                process_count_by_uuid=process_count_by_uuid,
                exchange_count_by_uuid=exchange_count_by_uuid,
            )

    return {
        "reference_process_count_by_flow_key": dict(reference_process_count_by_flow_key),
        "process_count_by_flow_key": dict(process_count_by_flow_key),
        "exchange_count_by_flow_key": dict(exchange_count_by_flow_key),
        "reference_process_count_by_uuid": dict(reference_process_count_by_uuid),
        "process_count_by_uuid": dict(process_count_by_uuid),
        "exchange_count_by_uuid": dict(exchange_count_by_uuid),
        "meta": {
            "process_count_scanned": len(process_keys_scanned),
            "source_mode": source_mode,
            "visibility_scope_note": visibility_scope_note,
        },
    }


def count_matching_process_refs(
    *,
    args: argparse.Namespace,
    access_token: str,
    flow_ids: set[str],
    reference_process_count_by_flow_key: Counter[str],
    process_count_by_flow_key: Counter[str],
    exchange_count_by_flow_key: Counter[str],
    reference_process_count_by_uuid: Counter[str],
    process_count_by_uuid: Counter[str],
    exchange_count_by_uuid: Counter[str],
) -> set[str]:
    process_keys_scanned: set[str] = set()
    for flow_id in sorted(flow_ids):
        offset = 0
        while True:
            batch = postgrest_select_page(
                base_url=args.supabase_url,
                apikey=args.supabase_publishable_key,
                access_token=access_token,
                table="processes",
                raw_filters={
                    "state_code": "in.(0,100)",
                    "json->processDataSet->exchanges->exchange": "cs."
                    + json_contains_flow_id(flow_id),
                },
                columns="id,version,state_code,json->processDataSet->processInformation->quantitativeReference->>referenceToReferenceFlow,json->processDataSet->exchanges->exchange",
                limit=args.process_page_size,
                offset=offset,
                order="id.asc",
            )
            if not batch:
                break
            for row in batch:
                process_keys_scanned.add(process_identity_key(row))
                accumulate_process_counts_for_flow_id(
                    row=row,
                    target_flow_id=flow_id,
                    reference_process_count_by_flow_key=reference_process_count_by_flow_key,
                    process_count_by_flow_key=process_count_by_flow_key,
                    exchange_count_by_flow_key=exchange_count_by_flow_key,
                    reference_process_count_by_uuid=reference_process_count_by_uuid,
                    process_count_by_uuid=process_count_by_uuid,
                    exchange_count_by_uuid=exchange_count_by_uuid,
                )
            if len(batch) < args.process_page_size:
                break
            offset += args.process_page_size
    return process_keys_scanned


def full_scan_reference_counts(
    *,
    args: argparse.Namespace,
    access_token: str,
    reference_process_count_by_flow_key: Counter[str],
    process_count_by_flow_key: Counter[str],
    exchange_count_by_flow_key: Counter[str],
    reference_process_count_by_uuid: Counter[str],
    process_count_by_uuid: Counter[str],
    exchange_count_by_uuid: Counter[str],
) -> set[str]:
    process_keys_scanned: set[str] = set()
    offset = 0
    while True:
        batch = postgrest_select_page(
            base_url=args.supabase_url,
            apikey=args.supabase_publishable_key,
            access_token=access_token,
            table="processes",
            raw_filters={"state_code": "in.(0,100)"},
            columns="id,version,state_code,json->processDataSet->processInformation->quantitativeReference->>referenceToReferenceFlow,json->processDataSet->exchanges->exchange",
            limit=args.process_page_size,
            offset=offset,
            order="id.asc",
        )
        if not batch:
            break
        for row in batch:
            process_keys_scanned.add(process_identity_key(row))
            accumulate_process_counts(
                row=row,
                reference_process_count_by_flow_key=reference_process_count_by_flow_key,
                process_count_by_flow_key=process_count_by_flow_key,
                exchange_count_by_flow_key=exchange_count_by_flow_key,
                reference_process_count_by_uuid=reference_process_count_by_uuid,
                process_count_by_uuid=process_count_by_uuid,
                exchange_count_by_uuid=exchange_count_by_uuid,
            )
        if len(batch) < args.process_page_size:
            break
        offset += args.process_page_size
    return process_keys_scanned


def accumulate_process_counts(
    *,
    row: dict[str, Any],
    reference_process_count_by_flow_key: Counter[str],
    process_count_by_flow_key: Counter[str],
    exchange_count_by_flow_key: Counter[str],
    reference_process_count_by_uuid: Counter[str],
    process_count_by_uuid: Counter[str],
    exchange_count_by_uuid: Counter[str],
) -> None:
    seen_flow_keys: set[str] = set()
    seen_flow_ids: set[str] = set()
    seen_reference_flow_keys: set[str] = set()
    seen_reference_flow_ids: set[str] = set()
    reference_internal_id = process_reference_internal_id(row)
    for exchange in process_exchange_rows(row):
        ref = exchange.get("referenceToFlowDataSet") or {}
        flow_id = str(ref.get("@refObjectId") or "").strip()
        flow_version = str(ref.get("@version") or "").strip()
        if not flow_id or not flow_version:
            continue
        flow_key = f"{flow_id}@{flow_version}"
        exchange_count_by_flow_key[flow_key] += 1
        exchange_count_by_uuid[flow_id] += 1
        if flow_key not in seen_flow_keys:
            process_count_by_flow_key[flow_key] += 1
            seen_flow_keys.add(flow_key)
        if flow_id not in seen_flow_ids:
            process_count_by_uuid[flow_id] += 1
            seen_flow_ids.add(flow_id)
        if reference_internal_id and str(exchange.get("@dataSetInternalID") or "").strip() == reference_internal_id:
            if flow_key not in seen_reference_flow_keys:
                reference_process_count_by_flow_key[flow_key] += 1
                seen_reference_flow_keys.add(flow_key)
            if flow_id not in seen_reference_flow_ids:
                reference_process_count_by_uuid[flow_id] += 1
                seen_reference_flow_ids.add(flow_id)


def accumulate_process_counts_for_flow_id(
    *,
    row: dict[str, Any],
    target_flow_id: str,
    reference_process_count_by_flow_key: Counter[str],
    process_count_by_flow_key: Counter[str],
    exchange_count_by_flow_key: Counter[str],
    reference_process_count_by_uuid: Counter[str],
    process_count_by_uuid: Counter[str],
    exchange_count_by_uuid: Counter[str],
) -> None:
    seen_flow_keys: set[str] = set()
    seen_reference_flow_keys: set[str] = set()
    saw_target_uuid = False
    saw_target_reference_uuid = False
    reference_internal_id = process_reference_internal_id(row)
    for exchange in process_exchange_rows(row):
        ref = exchange.get("referenceToFlowDataSet") or {}
        flow_id = str(ref.get("@refObjectId") or "").strip()
        flow_version = str(ref.get("@version") or "").strip()
        if flow_id != target_flow_id or not flow_version:
            continue
        flow_key = f"{flow_id}@{flow_version}"
        exchange_count_by_flow_key[flow_key] += 1
        exchange_count_by_uuid[flow_id] += 1
        saw_target_uuid = True
        if flow_key not in seen_flow_keys:
            process_count_by_flow_key[flow_key] += 1
            seen_flow_keys.add(flow_key)
        if reference_internal_id and str(exchange.get("@dataSetInternalID") or "").strip() == reference_internal_id:
            if flow_key not in seen_reference_flow_keys:
                reference_process_count_by_flow_key[flow_key] += 1
                seen_reference_flow_keys.add(flow_key)
            saw_target_reference_uuid = True
    if saw_target_uuid:
        process_count_by_uuid[target_flow_id] += 1
    if saw_target_reference_uuid:
        reference_process_count_by_uuid[target_flow_id] += 1


def process_exchange_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    if "exchange" in row:
        value = row.get("exchange")
    else:
        dataset = row.get("json", {}).get("processDataSet", row.get("processDataSet", {}))
        value = deep_get(dataset, ["exchanges", "exchange"], [])
    return [item for item in listify(value) if isinstance(item, dict)]


def process_identity_key(row: dict[str, Any]) -> str:
    process_id = str(row.get("id") or "").strip()
    version = str(row.get("version") or "").strip()
    if process_id:
        return f"{process_id}@{version}"
    dataset = row.get("json", {}).get("processDataSet", row.get("processDataSet", {}))
    info = deep_get(dataset, ["processInformation", "dataSetInformation"], {})
    fallback_id = str(info.get("common:UUID") or "").strip()
    return f"{fallback_id}@{version}" if fallback_id else "@"


def process_reference_internal_id(row: dict[str, Any]) -> str:
    direct = str(
        row.get("referenceToReferenceFlow")
        or row.get("ref_internal_id")
        or row.get("reference_flow_internal_id")
        or ""
    ).strip()
    if direct:
        return direct
    dataset = row.get("json", {}).get("processDataSet", row.get("processDataSet", {}))
    return str(
        deep_get(dataset, ["processInformation", "quantitativeReference", "referenceToReferenceFlow"]) or ""
    ).strip()


def build_flow_summaries(
    flow_rows: list[dict[str, Any]],
    flow_property_meta: dict[str, FlowPropertyMeta],
    ref_counts: dict[str, Any],
    ilcd_ids: set[str],
    snapshot_role_map: dict[str, tuple[str, ...]],
) -> list[FlowSummary]:
    reference_process_counts = ref_counts["reference_process_count_by_flow_key"]
    process_counts = ref_counts["process_count_by_flow_key"]
    exchange_counts = ref_counts["exchange_count_by_flow_key"]
    uuid_reference_process_counts = ref_counts["reference_process_count_by_uuid"]
    uuid_process_counts = ref_counts["process_count_by_uuid"]
    uuid_exchange_counts = ref_counts["exchange_count_by_uuid"]
    summaries: list[FlowSummary] = []
    for row in flow_rows:
        dataset = flow_dataset_from_row(row)
        info = deep_get(dataset, ["flowInformation", "dataSetInformation"], {})
        flow_id = str(row.get("id") or info.get("common:UUID") or "").strip()
        version = str(
            row.get("version")
            or deep_get(dataset, ["administrativeInformation", "publicationAndOwnership", "common:dataSetVersion"])
            or "01.00.000"
        ).strip()
        flow_key = f"{flow_id}@{version}"
        flow_type = str(
            deep_get(dataset, ["modellingAndValidation", "LCIMethodAndAllocation", "typeOfDataSet"])
            or deep_get(dataset, ["modellingAndValidation", "LCIMethod", "typeOfDataSet"])
            or ""
        ).strip()
        name_parts = extract_name_parts(info.get("name"))
        prop_ref = select_reference_flow_property(dataset)
        flow_property_uuid = str(prop_ref.get("@refObjectId") or "").strip()
        flow_property_version = str(prop_ref.get("@version") or "").strip()
        meta = flow_property_meta.get(flow_property_uuid)
        public_origin, is_ilcd_reference_flow = classify_public_origin(row, flow_id, ilcd_ids)
        summaries.append(
            FlowSummary(
                flow_key=flow_key,
                flow_id=flow_id,
                version=version,
                snapshot_roles=snapshot_role_map.get(flow_key, ()),
                flow_type=flow_type,
                state_code=row.get("state_code"),
                user_id=str(row.get("user_id") or "").strip(),
                public_origin=public_origin,
                is_ilcd_reference_flow=is_ilcd_reference_flow,
                primary_name=name_parts["primary_name"] or flow_id,
                base_name_entries=name_parts["base_name_entries"],
                flow_property_entries=name_parts["flow_property_entries"],
                mix_entries=name_parts["mix_entries"],
                route_entries=name_parts["route_entries"],
                full_name_entries=name_parts["full_name_entries"],
                base_name_fingerprint=name_parts["base_name_fingerprint"],
                full_name_fingerprint=name_parts["full_name_fingerprint"],
                concept_tokens=name_parts["concept_tokens"],
                qualifier_tokens=name_parts["qualifier_tokens"],
                qualifier_signature=name_parts["qualifier_signature"],
                classification_leaf_key=classification_leaf_key(info),
                classification_leaf_text=classification_leaf_text(info),
                classification_path=classification_path(info),
                flow_property_uuid=flow_property_uuid,
                flow_property_version=flow_property_version,
                flow_property_name=(
                    meta.name
                    if meta and meta.name
                    else lang_text(prop_ref.get("common:shortDescription")) or flow_property_uuid
                ),
                unitgroup_uuid=meta.unitgroup_uuid if meta else "",
                unitgroup_name=meta.unitgroup_name if meta else "",
                reference_process_count=int(reference_process_counts.get(flow_key, 0)),
                process_ref_count=int(process_counts.get(flow_key, 0)),
                exchange_ref_count=int(exchange_counts.get(flow_key, 0)),
                uuid_reference_process_count=int(uuid_reference_process_counts.get(flow_id, 0)),
                uuid_process_ref_count=int(uuid_process_counts.get(flow_id, 0)),
                uuid_exchange_ref_count=int(uuid_exchange_counts.get(flow_id, 0)),
            )
        )
    summaries.sort(key=lambda item: (item.flow_type, item.primary_name, item.flow_id, item.version))
    return summaries


def build_candidate_pairs(
    summaries: list[FlowSummary],
    args: argparse.Namespace,
    *,
    subject_keys: set[str],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[FlowSummary]] = defaultdict(list)
    for summary in summaries:
        class_key = summary.classification_leaf_key or "__missing__"
        tokens = summary.concept_tokens[: args.bucket_token_limit] or [summary.base_name_fingerprint or summary.flow_key]
        for token in tokens:
            buckets[(summary.flow_type, class_key, token)].append(summary)
        for entry_key in normalized_base_entry_keys(summary):
            buckets[(summary.flow_type, "__shared_base__", entry_key)].append(summary)

    seen_pairs: set[tuple[str, str]] = set()
    pair_rows: list[dict[str, Any]] = []
    for group_rows in buckets.values():
        for chunk_rows in subdivide_bucket(group_rows, args.bucket_max_size):
            if len(chunk_rows) < 2:
                continue
            for index, left in enumerate(chunk_rows):
                for right in chunk_rows[index + 1 :]:
                    if subject_keys and left.flow_key not in subject_keys and right.flow_key not in subject_keys:
                        continue
                    pair_key = tuple(sorted((left.flow_key, right.flow_key)))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    candidate = classify_pair(left, right, args)
                    if candidate:
                        if subject_keys:
                            left_in_subject = left.flow_key in subject_keys
                            right_in_subject = right.flow_key in subject_keys
                            candidate["left_in_subject_scope"] = left_in_subject
                            candidate["right_in_subject_scope"] = right_in_subject
                            candidate["pair_scope"] = (
                                "subject-vs-subject"
                                if left_in_subject and right_in_subject
                                else "subject-vs-candidate"
                            )
                        else:
                            candidate["pair_scope"] = "all-vs-all"
                        pair_rows.append(candidate)
    pair_rows.sort(
        key=lambda item: (
            item["pair_label"],
            -item["signal_strength"],
            item["left"]["primary_name"],
            item["right"]["primary_name"],
        )
    )
    return pair_rows


def subdivide_bucket(rows: list[FlowSummary], max_size: int) -> list[list[FlowSummary]]:
    if len(rows) <= max_size:
        return [rows]
    by_prefix: dict[str, list[FlowSummary]] = defaultdict(list)
    for row in rows:
        prefix = (row.base_name_fingerprint or row.full_name_fingerprint or row.flow_key)[:8]
        by_prefix[prefix].append(row)
    groups = [group for group in by_prefix.values() if len(group) >= 2]
    return groups or [rows]


def classify_pair(left: FlowSummary, right: FlowSummary, args: argparse.Namespace) -> dict[str, Any] | None:
    if left.flow_type != right.flow_type:
        return None
    same_uuid = left.flow_id == right.flow_id
    if same_uuid:
        return None
    metrics = similarity_metrics(left, right)
    same_property = property_signature(left) == property_signature(right) and property_signature(left) != ""
    class_relation = classification_relation(left, right)
    classification_compatible = class_relation in {"same_leaf", "missing_leaf"} or bool(metrics["shared_base_name"])
    qualifier_relation = metrics["qualifier_relation"]
    pair_label: str | None = None
    if same_property and classification_compatible:
        if qualifier_relation in {"exact", "both_missing"} and (
            metrics["shared_base_name"] or metrics["max_base_entry_similarity"] >= args.same_property_threshold
        ):
            pair_label = "same_property_semantic_review"
        elif qualifier_relation in {"exact", "both_missing"} and metrics["token_jaccard"] >= 0.80 and metrics["base_ratio"] >= 0.86:
            pair_label = "same_property_semantic_review"
    if not pair_label and classification_compatible and not same_property:
        if qualifier_relation in {"exact", "both_missing"} and (
            metrics["shared_base_name"] or metrics["max_base_entry_similarity"] >= args.cross_property_threshold
        ):
            pair_label = "cross_property_semantic_review"
        elif qualifier_relation in {"exact", "both_missing"} and metrics["token_jaccard"] >= 0.70 and metrics["base_ratio"] >= 0.88:
            pair_label = "cross_property_semantic_review"
    if not pair_label:
        if qualifier_relation == "conflict":
            if same_property and metrics["shared_base_name"]:
                pair_label = None
            elif metrics["base_entry_contains"] and metrics["token_jaccard"] >= 0.20:
                pair_label = "near_duplicate_manual_review"
        elif metrics["shared_base_name"] or metrics["max_base_entry_similarity"] >= args.near_threshold:
            pair_label = "near_duplicate_manual_review"
        elif metrics["base_entry_contains"] and (
            class_relation != "different_leaf" or metrics["token_jaccard"] >= 0.20
        ):
            pair_label = "near_duplicate_manual_review"
        elif metrics["base_ratio"] >= args.near_threshold and class_relation != "different_leaf":
            pair_label = "near_duplicate_manual_review"
    if not pair_label:
        return None
    signal_strength = round(
        max(
            metrics["max_base_entry_similarity"],
            metrics["base_ratio"],
            metrics["full_ratio"],
            metrics["token_jaccard"],
        ),
        6,
    )
    return {
        "pair_label": pair_label,
        "signal_strength": signal_strength,
        "same_property": same_property,
        "same_uuid": same_uuid,
        "classification_relation": class_relation,
        "qualifier_relation": qualifier_relation,
        "metrics": metrics,
        "left": summary_view(left),
        "right": summary_view(right),
    }


def build_clusters_and_maps(
    summaries: list[FlowSummary],
    pair_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    summary_by_key = {item.flow_key: item for item in summaries}
    union_find = UnionFind()
    strong_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manual_review_rows: list[dict[str, Any]] = []

    for summary in summaries:
        union_find.add(summary.flow_key)

    for pair in pair_rows:
        left_key = pair["left"]["flow_key"]
        right_key = pair["right"]["flow_key"]
        if pair["pair_label"] == "same_property_duplicate_candidate":
            union_find.union(left_key, right_key)
            strong_edges[left_key].append(pair)
            strong_edges[right_key].append(pair)
        else:
            review_type = "near_duplicate_pair"
            reason = "Near duplicate candidate needs manual semantic review before any merge decision."
            if pair["pair_label"] in {"same_property_semantic_review", "cross_property_semantic_review"}:
                review_type = "semantic_duplicate_pair"
                reason = "High-similarity pair is not auto-merge-safe; require semantic review before any merge decision."
            manual_review_rows.append(
                {
                    "review_type": review_type,
                    "pair_label": pair["pair_label"],
                    "left_flow_key": left_key,
                    "right_flow_key": right_key,
                    "signal_strength": pair["signal_strength"],
                    "same_uuid": pair["same_uuid"],
                    "same_property": pair["same_property"],
                    "pair_scope": pair.get("pair_scope", ""),
                    "left_in_subject_scope": pair.get("left_in_subject_scope"),
                    "right_in_subject_scope": pair.get("right_in_subject_scope"),
                    "classification_relation": pair["classification_relation"],
                    "metrics": pair["metrics"],
                    "left": pair["left"],
                    "right": pair["right"],
                    "reason": reason,
                }
            )

    clusters: list[dict[str, Any]] = []
    family_canonical_map: dict[str, dict[str, Any]] = {}
    property_safe_rewrite_map: dict[str, dict[str, Any]] = {}

    groups = union_find.groups()
    for cluster_index, member_keys in enumerate(sorted(groups.values(), key=lambda values: (len(values), values), reverse=True), start=1):
        if len(member_keys) < 2:
            continue
        members = [summary_by_key[key] for key in sorted(member_keys)]
        cluster_pairs = [
            pair
            for pair in pair_rows
            if pair["pair_label"] == "same_property_duplicate_candidate"
            and pair["left"]["flow_key"] in member_keys
            and pair["right"]["flow_key"] in member_keys
        ]
        family_canonical = pick_canonical(members)
        property_groups: dict[str, list[FlowSummary]] = defaultdict(list)
        for member in members:
            property_groups[property_signature(member) or f"missing:{member.flow_key}"].append(member)
        property_canonicals = {
            signature: pick_canonical(group_members) for signature, group_members in property_groups.items()
        }

        member_rows: list[dict[str, Any]] = []
        cross_property_count = 0
        for member in members:
            prop_signature = property_signature(member) or f"missing:{member.flow_key}"
            property_canonical = property_canonicals[prop_signature]
            family_decision = "keep_family_canonical" if member.flow_key == family_canonical.flow_key else "merge_into_family"
            property_rewrite_safe = prop_signature == property_signature(property_canonical) and member.flow_key != property_canonical.flow_key
            if property_signature(member) != property_signature(family_canonical):
                cross_property_count += 1
            if property_rewrite_safe:
                property_safe_rewrite_map[member.flow_key] = {
                    "source_flow_key": member.flow_key,
                    "target_flow_key": property_canonical.flow_key,
                    "target_flow_id": property_canonical.flow_id,
                    "target_version": property_canonical.version,
                    "rewrite_scope": "same_property",
                    "reason": "same_property_duplicate_highest_visible_process_refs",
                }
            family_canonical_map[member.flow_key] = {
                "source_flow_key": member.flow_key,
                "family_canonical_flow_key": family_canonical.flow_key,
                "family_canonical_flow_id": family_canonical.flow_id,
                "family_canonical_version": family_canonical.version,
                "reason": "same_concept_family_highest_visible_process_refs",
                "cross_property": property_signature(member) != property_signature(family_canonical),
            }
            member_rows.append(
                {
                    **summary_view(member),
                    "family_canonical_flow_key": family_canonical.flow_key,
                    "property_canonical_flow_key": property_canonical.flow_key,
                    "family_decision": family_decision,
                    "property_rewrite_safe": property_rewrite_safe,
                    "requires_manual_conversion_review": property_signature(member) != property_signature(family_canonical),
                }
            )
            if property_signature(member) != property_signature(family_canonical):
                manual_review_rows.append(
                    {
                        "review_type": "cross_property_family",
                        "source_flow_key": member.flow_key,
                        "family_canonical_flow_key": family_canonical.flow_key,
                        "property_canonical_flow_key": property_canonical.flow_key,
                        "reason": "Same concept family spans different flow properties; governance merge is possible, but automatic process ref rewrite is unsafe.",
                        "source": summary_view(member),
                        "family_canonical": summary_view(family_canonical),
                        "property_canonical": summary_view(property_canonical),
                    }
                )

        clusters.append(
            {
                "cluster_id": f"cluster-{cluster_index:04d}",
                "cluster_label": family_canonical.primary_name,
                "flow_type": family_canonical.flow_type,
                "classification_leaf_text": family_canonical.classification_leaf_text,
                "family_canonical": summary_view(family_canonical),
                "property_canonicals": [
                    {
                        "property_signature": signature,
                        "canonical": summary_view(canonical),
                    }
                    for signature, canonical in sorted(
                        property_canonicals.items(),
                        key=lambda item: (
                            item[1].flow_property_name,
                            item[1].unitgroup_name,
                            item[1].primary_name,
                            item[1].flow_key,
                        ),
                    )
                ],
                "pair_label_counts": dict(Counter(item["pair_label"] for item in cluster_pairs)),
                "cross_property_member_count": cross_property_count,
                "member_count": len(member_rows),
                "members": sorted(
                    member_rows,
                    key=lambda item: (
                        item["family_decision"] != "keep_family_canonical",
                        item["flow_property_name"],
                        item["primary_name"],
                        item["flow_key"],
                    ),
                ),
                "pair_evidence": cluster_pairs[:20],
            }
        )

    clusters.sort(key=lambda item: (item["member_count"], item["cross_property_member_count"], item["cluster_label"]), reverse=True)
    canonical_map = {
        "family_canonical_map": family_canonical_map,
        "property_safe_rewrite_map": property_safe_rewrite_map,
        "notes": [
            "family_canonical_map is for concept-level governance review",
            "property_safe_rewrite_map is the only merge output safe for automatic ref rewrites",
            "cross-property families still require manual conversion and review before any process patch",
        ],
    }
    return clusters, canonical_map, manual_review_rows


def build_rewrite_plan(clusters: list[dict[str, Any]], canonical_map: dict[str, Any]) -> list[dict[str, Any]]:
    cluster_by_member: dict[str, str] = {}
    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        for member in cluster.get("members", []):
            cluster_by_member[member["flow_key"]] = cluster_id
    rows: list[dict[str, Any]] = []
    for source_flow_key, mapping in canonical_map.get("property_safe_rewrite_map", {}).items():
        rows.append(
            {
                "cluster_id": cluster_by_member.get(source_flow_key, ""),
                "source_flow_key": source_flow_key,
                "source_flow_id": source_flow_key.split("@", 1)[0],
                "source_version": source_flow_key.split("@", 1)[1] if "@" in source_flow_key else "",
                "target_flow_key": mapping.get("target_flow_key", ""),
                "target_flow_id": mapping.get("target_flow_id", ""),
                "target_version": mapping.get("target_version", ""),
                "rewrite_scope": mapping.get("rewrite_scope", ""),
                "reason": mapping.get("reason", ""),
            }
        )
    rows.sort(key=lambda item: (item["cluster_id"], item["target_flow_id"], item["source_flow_id"], item["source_version"]))
    return rows


def build_version_bump_plan(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        family = cluster["family_canonical"]
        add_bump_plan_row(
            rows_by_key,
            cluster_id=cluster_id,
            role="family_canonical",
            canonical=family,
        )
        for item in cluster.get("property_canonicals", []):
            canonical = item.get("canonical") or {}
            add_bump_plan_row(
                rows_by_key,
                cluster_id=cluster_id,
                role="property_canonical",
                canonical=canonical,
            )
    rows = list(rows_by_key.values())
    rows.sort(
        key=lambda item: (
            -item["process_ref_count"],
            ",".join(item.get("roles", [])),
            item["flow_id"],
            item["current_version"],
        )
    )
    return rows


def project_rewrite_plan_to_next_versions(
    rewrite_plan: list[dict[str, Any]],
    version_bump_plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    next_version_by_target = {
        (str(item.get("flow_id") or ""), str(item.get("current_version") or "")): str(item.get("recommended_new_version") or "")
        for item in version_bump_plan
    }
    rows: list[dict[str, Any]] = []
    for row in rewrite_plan:
        target_flow_id = str(row.get("target_flow_id") or "")
        target_version = str(row.get("target_version") or "")
        projected_target_version = next_version_by_target.get((target_flow_id, target_version), target_version)
        rows.append(
            {
                **row,
                "target_version": projected_target_version,
                "target_version_source": "recommended_new_version"
                if projected_target_version != target_version
                else "current_version",
            }
        )
    return rows


def add_bump_plan_row(
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    *,
    cluster_id: str,
    role: str,
    canonical: dict[str, Any],
) -> None:
    flow_id = str(canonical.get("flow_id") or "").strip()
    current_version = str(canonical.get("version") or "").strip()
    if not flow_id or not current_version:
        return
    key = (flow_id, current_version)
    row = rows_by_key.get(key)
    if not row:
        row = {
            "flow_id": flow_id,
            "current_version": current_version,
            "recommended_new_version": bump_ilcd_version(current_version),
            "roles": [],
            "cluster_ids": [],
            "flow_property_uuid": canonical.get("flow_property_uuid", ""),
            "flow_property_name": canonical.get("flow_property_name", ""),
            "unitgroup_uuid": canonical.get("unitgroup_uuid", ""),
            "unitgroup_name": canonical.get("unitgroup_name", ""),
            "primary_name": canonical.get("primary_name", ""),
            "state_code": canonical.get("state_code"),
            "public_origin": canonical.get("public_origin", ""),
            "reference_process_count": int(canonical.get("reference_process_count", 0)),
            "process_ref_count": int(canonical.get("process_ref_count", 0)),
            "exchange_ref_count": int(canonical.get("exchange_ref_count", 0)),
            "publish_note": "Prepare a full flow payload with the same UUID, bump common:dataSetVersion, publish append-only through MCP insert, then rewrite references to the new version.",
        }
        rows_by_key[key] = row
    if role not in row["roles"]:
        row["roles"].append(role)
    if cluster_id and cluster_id not in row["cluster_ids"]:
        row["cluster_ids"].append(cluster_id)


def pick_canonical(members: list[FlowSummary]) -> FlowSummary:
    return max(members, key=canonical_score)


def canonical_score(item: FlowSummary) -> tuple[Any, ...]:
    return (
        item.reference_process_count,
        item.process_ref_count,
        item.exchange_ref_count,
        item.uuid_reference_process_count,
        item.uuid_process_ref_count,
        item.uuid_exchange_ref_count,
        1 if item.is_ilcd_reference_flow else 0,
        1 if item.state_code == 100 else 0,
        len(item.base_name_entries),
        version_key(item.version),
        item.flow_key,
    )


def summary_view(item: FlowSummary) -> dict[str, Any]:
    return {
        "flow_key": item.flow_key,
        "flow_id": item.flow_id,
        "version": item.version,
        "snapshot_roles": list(item.snapshot_roles),
        "primary_name": item.primary_name,
        "flow_type": item.flow_type,
        "state_code": item.state_code,
        "user_id": item.user_id,
        "public_origin": item.public_origin,
        "flow_property_uuid": item.flow_property_uuid,
        "flow_property_name": item.flow_property_name,
        "unitgroup_uuid": item.unitgroup_uuid,
        "unitgroup_name": item.unitgroup_name,
        "classification_leaf_text": item.classification_leaf_text,
        "qualifier_signature": item.qualifier_signature,
        "reference_process_count": item.reference_process_count,
        "process_ref_count": item.process_ref_count,
        "exchange_ref_count": item.exchange_ref_count,
        "uuid_reference_process_count": item.uuid_reference_process_count,
        "uuid_process_ref_count": item.uuid_process_ref_count,
        "uuid_exchange_ref_count": item.uuid_exchange_ref_count,
    }


def collect_candidate_flow_ids(pair_rows: list[dict[str, Any]]) -> set[str]:
    flow_ids: set[str] = set()
    for pair in pair_rows:
        flow_ids.add(pair["left"]["flow_id"])
        flow_ids.add(pair["right"]["flow_id"])
    return flow_ids


def similarity_metrics(left: FlowSummary, right: FlowSummary) -> dict[str, Any]:
    left_tokens = set(left.concept_tokens)
    right_tokens = set(right.concept_tokens)
    union = left_tokens | right_tokens
    token_jaccard = (len(left_tokens & right_tokens) / len(union)) if union else 0.0
    left_qualifier_tokens = set(left.qualifier_tokens)
    right_qualifier_tokens = set(right.qualifier_tokens)
    qualifier_union = left_qualifier_tokens | right_qualifier_tokens
    qualifier_token_jaccard = (len(left_qualifier_tokens & right_qualifier_tokens) / len(qualifier_union)) if qualifier_union else 0.0
    left_entries = [normalize_text(item) for item in left.base_name_entries if normalize_text(item)]
    right_entries = [normalize_text(item) for item in right.base_name_entries if normalize_text(item)]
    shared_base_name = sorted(set(left_entries) & set(right_entries))
    max_base_entry_similarity = 0.0
    base_entry_contains = False
    for left_entry in left_entries:
        for right_entry in right_entries:
            max_base_entry_similarity = max(max_base_entry_similarity, SequenceMatcher(None, left_entry, right_entry).ratio())
            if left_entry != right_entry and left_entry and right_entry and (
                left_entry in right_entry or right_entry in left_entry
            ):
                base_entry_contains = True
    return {
        "base_ratio": round(SequenceMatcher(None, left.base_name_fingerprint, right.base_name_fingerprint).ratio(), 6),
        "full_ratio": round(SequenceMatcher(None, left.full_name_fingerprint, right.full_name_fingerprint).ratio(), 6),
        "token_jaccard": round(token_jaccard, 6),
        "qualifier_token_jaccard": round(qualifier_token_jaccard, 6),
        "qualifier_relation": qualifier_relation(left, right),
        "shared_base_name": shared_base_name,
        "max_base_entry_similarity": round(max_base_entry_similarity, 6),
        "base_entry_contains": base_entry_contains,
    }


def property_signature(item: FlowSummary) -> str:
    if item.flow_property_uuid:
        return f"{item.flow_property_uuid}|{item.unitgroup_uuid}"
    return ""


def classification_relation(left: FlowSummary, right: FlowSummary) -> str:
    if left.classification_leaf_key and right.classification_leaf_key:
        if left.classification_leaf_key == right.classification_leaf_key:
            return "same_leaf"
        return "different_leaf"
    return "missing_leaf"


def extract_name_parts(name_node: Any) -> dict[str, Any]:
    if not isinstance(name_node, dict):
        return {
            "primary_name": "",
            "base_name_entries": [],
            "full_name_entries": [],
            "base_name_fingerprint": "",
            "full_name_fingerprint": "",
            "concept_tokens": [],
        }
    base_entries = [item["text"] for item in lang_entries(name_node.get("baseName")) if item.get("text")]
    flow_property_entries = [item["text"] for item in lang_entries(name_node.get("flowProperties")) if item.get("text")]
    route_entries = [item["text"] for item in lang_entries(name_node.get("treatmentStandardsRoutes")) if item.get("text")]
    mix_entries = [item["text"] for item in lang_entries(name_node.get("mixAndLocationTypes")) if item.get("text")]
    primary_name = base_entries[0] if base_entries else ""
    full_entries = base_entries + flow_property_entries + route_entries + mix_entries
    qualifier_entries = flow_property_entries + route_entries + mix_entries
    return {
        "primary_name": primary_name,
        "base_name_entries": base_entries,
        "flow_property_entries": flow_property_entries,
        "mix_entries": mix_entries,
        "route_entries": route_entries,
        "full_name_entries": full_entries,
        "base_name_fingerprint": normalize_text(" || ".join(base_entries)),
        "full_name_fingerprint": normalize_text(" || ".join(full_entries or base_entries)),
        "concept_tokens": concept_tokens_from_entries(base_entries),
        "qualifier_tokens": qualifier_tokens_from_entries(qualifier_entries),
        "qualifier_signature": normalize_qualifier_signature(qualifier_entries),
    }


def concept_tokens_from_entries(entries: list[str]) -> list[str]:
    tokens: list[str] = []
    for entry in entries:
        normalized = normalize_text(entry)
        if not normalized:
            continue
        for token in normalized.split():
            if token in GENERIC_TOKENS:
                continue
            if token not in tokens:
                tokens.append(token)
        if normalized and normalized not in tokens:
            tokens.append(normalized)
    return tokens


def qualifier_tokens_from_entries(entries: list[str]) -> list[str]:
    tokens: list[str] = []
    for entry in entries:
        normalized = normalize_text(entry)
        if not normalized:
            continue
        if normalized not in tokens:
            tokens.append(normalized)
        for token in normalized.split():
            if token in GENERIC_TOKENS:
                continue
            if token not in tokens:
                tokens.append(token)
    return tokens


def normalize_qualifier_signature(entries: list[str]) -> str:
    normalized = sorted({normalize_text(entry) for entry in entries if normalize_text(entry)})
    return " || ".join(normalized)


def qualifier_relation(left: FlowSummary, right: FlowSummary) -> str:
    if not left.qualifier_signature and not right.qualifier_signature:
        return "both_missing"
    if left.qualifier_signature and right.qualifier_signature:
        return "exact" if left.qualifier_signature == right.qualifier_signature else "conflict"
    return "partial"


def normalized_base_entry_keys(summary: FlowSummary) -> list[str]:
    keys: list[str] = []
    for entry in summary.base_name_entries:
        normalized = normalize_text(entry)
        if normalized and normalized not in keys:
            keys.append(normalized)
    return keys
def classification_leaf_key(info: dict[str, Any]) -> str:
    entries = classification_entries(info)
    if not entries:
        return ""
    leaf = entries[-1]
    return (leaf.get("class_id", "") + "|" + leaf.get("text", "")).strip("|")


def classification_leaf_text(info: dict[str, Any]) -> str:
    entries = classification_entries(info)
    if not entries:
        return ""
    return entries[-1].get("text", "")


def classification_path(info: dict[str, Any]) -> list[str]:
    return [item.get("text", "") for item in classification_entries(info) if item.get("text")]


def classification_entries(info: dict[str, Any]) -> list[dict[str, str]]:
    raw = deep_get(info, ["classificationInformation", "common:classification", "common:class"], [])
    rows: list[dict[str, str]] = []
    for item in listify(raw):
        if isinstance(item, dict):
            rows.append(
                {
                    "level": str(item.get("@level") or "").strip(),
                    "class_id": str(item.get("@classId") or "").strip(),
                    "text": str(item.get("#text") or "").strip(),
                }
            )
    return rows


def render_markdown(summary_payload: dict[str, Any], clusters: list[dict[str, Any]]) -> str:
    lines = [
        "# Flow Dedup Candidates",
        "",
        f"- flow rows analyzed: {summary_payload['flow_rows_analyzed']}",
        f"- candidate pairs: {summary_payload['candidate_pair_count']}",
        f"- clusters: {summary_payload['cluster_count']}",
        f"- same-property auto merge candidates: {summary_payload['same_property_auto_merge_candidates']}",
        f"- cross-property family candidates: {summary_payload['cross_property_family_candidates']}",
        f"- manual review rows: {summary_payload['manual_review_count']}",
        f"- SQL-safe rewrite rows: {summary_payload['rewrite_plan_count']}",
        f"- next-version rewrite rows: {summary_payload['rewrite_plan_next_version_count']}",
        f"- canonical version bump rows: {summary_payload['canonical_version_bump_count']}",
        f"- processes scanned for ref counts: {summary_payload['processes_scanned_for_ref_count']}",
        f"- public origin filter: {summary_payload['public_origin_filter']}",
        "",
        "## Pair Labels",
    ]
    for label, count in sorted(summary_payload["pair_label_counts"].items()):
        lines.append(f"- {label}: {count}")
    lines.extend(
        [
            "",
            "## Top Clusters",
            "",
            "| cluster_id | members | cross_property | family_canonical | property | reference_processes | processes |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for cluster in clusters[:50]:
        canonical = cluster["family_canonical"]
        lines.append(
            f"| {cluster['cluster_id']} | {cluster['member_count']} | {cluster['cross_property_member_count']} | "
            f"{canonical['primary_name'].replace('|', '/')} | {canonical['flow_property_name'].replace('|', '/')} | "
            f"{canonical['reference_process_count']} | {canonical['process_ref_count']} |"
        )
    if len(clusters) > 50:
        lines.extend(["", "_Truncated cluster table preview at 50 rows. Full payload is in `flow-dedup-candidate-clusters.json`._"])
    lines.append("")
    return "\n".join(lines)


def classify_public_origin(row: dict[str, Any], flow_id: str, ilcd_ids: set[str]) -> tuple[str, bool]:
    if row.get("state_code") != 100:
        return "", False
    is_ilcd = flow_id in ilcd_ids
    return ("ILCD" if is_ilcd else "TianGong public", is_ilcd)


def load_ilcd_flow_ids(flow_dir: str) -> set[str]:
    target = Path(flow_dir)
    if not target.exists() or not target.is_dir():
        return set()
    return {item.stem for item in target.glob("*.xml")}


def version_key(version: str) -> tuple[int, ...]:
    parts = []
    for part in str(version or "").split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def json_contains_flow_id(flow_id: str) -> str:
    return '[{"referenceToFlowDataSet":{"@refObjectId":"' + flow_id + '"}}]'


def bump_ilcd_version(version: str) -> str:
    text = str(version or "").strip()
    parts = text.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return "01.01.001"
    head, middle, tail = parts
    return f"{int(head):0{len(head)}d}.{int(middle):0{len(middle)}d}.{int(tail) + 1:0{len(tail)}d}"


def render_rewrite_plan_tsv(rows: list[dict[str, Any]]) -> str:
    header = "\t".join(
        [
            "cluster_id",
            "source_flow_id",
            "source_version",
            "target_flow_id",
            "target_version",
            "rewrite_scope",
            "reason",
        ]
    )
    body = [
        "\t".join(
            [
                str(row.get("cluster_id", "")),
                str(row.get("source_flow_id", "")),
                str(row.get("source_version", "")),
                str(row.get("target_flow_id", "")),
                str(row.get("target_version", "")),
                str(row.get("rewrite_scope", "")),
                str(row.get("reason", "")),
            ]
        )
        for row in rows
    ]
    return "\n".join([header] + body) + "\n"


def render_sql_rewrite_template(
    rewrite_plan: list[dict[str, Any]],
    rewrite_plan_next_version: list[dict[str, Any]],
    version_bump_plan: list[dict[str, Any]],
) -> str:
    lines = [
        "-- Flow dedup SQL rewrite template",
        "-- 1. Publish the canonical target flows first using the recommended new version plan below.",
        "-- 2. Load flow-dedup-rewrite-plan.tsv into a temp mapping table with source_flow_id/source_version/target_flow_id/target_version.",
        "-- 3. Run a jsonb rewrite function over processes/lifecyclemodels only after validating the new target rows exist.",
        "",
        "-- Recommended canonical publish rows:",
    ]
    for row in version_bump_plan[:200]:
        lines.append(
            "-- "
            + f"{row['flow_id']} {row['current_version']} -> {row['recommended_new_version']} "
            + f"roles={','.join(row['roles'])} refs={row['process_ref_count']}"
        )
    if len(version_bump_plan) > 200:
        lines.append("-- truncated preview; full list is in flow-dedup-version-bump-plan.json")
    lines.extend(
        [
            "",
            "create temporary table dedup_flow_map (",
            "  source_flow_id uuid not null,",
            "  source_version text not null,",
            "  target_flow_id uuid not null,",
            "  target_version text not null",
            ");",
            "",
            "-- Example load path: copy dedup_flow_map from '/abs/path/flow-dedup-rewrite-plan.tsv' with (format csv, delimiter E'\\t', header true);",
            "-- If you publish the canonical rows to their recommended_new_version first, prefer flow-dedup-rewrite-plan-next-version.tsv instead.",
            "",
            "-- Then implement a controlled jsonb rewrite over processDataSet.exchanges.exchange[*].referenceToFlowDataSet",
            "-- using source_flow_id + source_version -> target_flow_id + target_version from dedup_flow_map.",
            "-- Do not rewrite across different flow properties unless the mapping came from property_safe_rewrite_map.",
            "",
            f"-- rewrite rows available: {len(rewrite_plan)}",
            f"-- next-version rewrite rows available: {len(rewrite_plan_next_version)}",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
