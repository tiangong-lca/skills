#!/usr/bin/env python3
"""Publish LCA artifact bundles through the approved local publish paths."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

SCRIPT_PATH = Path(__file__).resolve()
SKILL_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SKILL_DIR.parent
PROCESS_AUTOMATED_BUILDER_WRAPPER = REPO_ROOT / "process-automated-builder" / "scripts" / "run-process-automated-builder.sh"

DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "lca_publish_executor"
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_RETRY_DELAY_SECONDS = 2.0
DEFAULT_DATASET_VERSION = "01.01.000"
DATABASE_TOOL_NAME = "Database_CRUD_Tool"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def first_non_empty(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def jsonrpc_payload(method: str, params: dict[str, Any], request_id: int) -> str:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        },
        ensure_ascii=False,
    )


def parse_mcp_response(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise ValueError("empty MCP response")
    data_lines = [
        line[len("data: ") :].strip()
        for line in text.splitlines()
        if line.startswith("data: ")
    ]
    payload_text = "\n".join(data_lines).strip() if data_lines else text
    return json.loads(payload_text)


def run_curl_mcp_request(
    *,
    api_key: str,
    url: str,
    method: str,
    params: dict[str, Any],
    request_id: int,
    max_attempts: int,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    body = jsonrpc_payload(method, params, request_id)
    command = [
        "curl",
        "-sS",
        "--max-time",
        "30",
        "-X",
        "POST",
        url,
        "-H",
        "Content-Type: application/json",
        "-H",
        "Accept: application/json, text/event-stream",
        "-H",
        f"Authorization: Bearer {api_key}",
        "--data",
        body,
    ]
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            return parse_mcp_response(result.stdout)
        stderr_text = result.stderr.strip() or result.stdout.strip() or f"curl exit {result.returncode}"
        last_error = f"attempt {attempt}/{max_attempts}: {stderr_text}"
        if attempt < max_attempts:
            time.sleep(retry_delay_seconds)
    raise RuntimeError(f"MCP request failed: {last_error}")


def mcp_call_tool(
    *,
    api_key: str,
    url: str,
    tool_name: str,
    arguments: dict[str, Any],
    request_id: int,
    max_attempts: int,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    payload = run_curl_mcp_request(
        api_key=api_key,
        url=url,
        method="tools/call",
        params={"name": tool_name, "arguments": arguments},
        request_id=request_id,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    return payload


def parse_tool_text_json(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result") or {}
    content = result.get("content")
    if not isinstance(content, list):
        return {}
    text_blocks = [
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    joined = "\n".join(block for block in text_blocks if block).strip()
    if not joined:
        return {}
    return json.loads(joined)


class PublishSession:
    def __init__(self, *, commit: bool, max_attempts: int, retry_delay_seconds: float) -> None:
        self.commit = commit
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.request_id = 1000
        self.url = os.getenv("TIANGONG_LCA_REMOTE_URL", "https://lcamcp.tiangong.earth/mcp").strip()
        self.api_key = os.getenv("TIANGONG_LCA_REMOTE_API_KEY", "").strip()
        if self.commit and not self.api_key:
            raise ValueError("TIANGONG_LCA_REMOTE_API_KEY is required when publish.commit=true")

    def call(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = mcp_call_tool(
            api_key=self.api_key,
            url=self.url,
            tool_name=DATABASE_TOOL_NAME,
            arguments=arguments,
            request_id=self.request_id,
            max_attempts=self.max_attempts,
            retry_delay_seconds=self.retry_delay_seconds,
        )
        self.request_id += 1
        return parse_tool_text_json(payload)

    def select(self, *, table: str, dataset_id: str, version: str | None = None) -> dict[str, Any]:
        args = {"operation": "select", "table": table, "id": dataset_id}
        if version:
            args["version"] = version
        return self.call(args)

    def insert(self, *, table: str, dataset_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.call(
            {
                "operation": "insert",
                "table": table,
                "id": dataset_id,
                "jsonOrdered": payload,
            }
        )

    def update(self, *, table: str, dataset_id: str, version: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.call(
            {
                "operation": "update",
                "table": table,
                "id": dataset_id,
                "version": version,
                "jsonOrdered": payload,
            }
        )


def extract_lifecyclemodel_identity(payload: Mapping[str, Any]) -> tuple[str, str]:
    dataset_id = first_non_empty(payload.get("@id"), payload.get("id"))
    version = first_non_empty(payload.get("@version"), payload.get("version"), DEFAULT_DATASET_VERSION)
    if not dataset_id:
        raise ValueError("Lifecycle model payload missing @id/id")
    return dataset_id, version or DEFAULT_DATASET_VERSION


def extract_process_identity(payload: Mapping[str, Any]) -> tuple[str, str]:
    root = payload.get("processDataSet") if isinstance(payload.get("processDataSet"), Mapping) else payload
    info = root.get("processInformation") if isinstance(root.get("processInformation"), Mapping) else {}
    data_info = info.get("dataSetInformation") if isinstance(info.get("dataSetInformation"), Mapping) else {}
    admin = root.get("administrativeInformation") if isinstance(root.get("administrativeInformation"), Mapping) else {}
    publication = admin.get("publicationAndOwnership") if isinstance(admin.get("publicationAndOwnership"), Mapping) else {}
    dataset_id = first_non_empty(data_info.get("common:UUID"))
    version = first_non_empty(publication.get("common:dataSetVersion"), DEFAULT_DATASET_VERSION)
    if not dataset_id:
        raise ValueError("Process payload missing processInformation.dataSetInformation.common:UUID")
    return dataset_id, version or DEFAULT_DATASET_VERSION


def extract_source_identity(payload: Mapping[str, Any]) -> tuple[str, str]:
    root = payload.get("sourceDataSet") if isinstance(payload.get("sourceDataSet"), Mapping) else payload
    info = root.get("sourceInformation") if isinstance(root.get("sourceInformation"), Mapping) else {}
    data_info = info.get("dataSetInformation") if isinstance(info.get("dataSetInformation"), Mapping) else {}
    admin = root.get("administrativeInformation") if isinstance(root.get("administrativeInformation"), Mapping) else {}
    publication = admin.get("publicationAndOwnership") if isinstance(admin.get("publicationAndOwnership"), Mapping) else {}
    dataset_id = first_non_empty(data_info.get("common:UUID"))
    version = first_non_empty(publication.get("common:dataSetVersion"), DEFAULT_DATASET_VERSION)
    if not dataset_id:
        raise ValueError("Source payload missing sourceInformation.dataSetInformation.common:UUID")
    return dataset_id, version or DEFAULT_DATASET_VERSION


def select_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def load_dataset_entry(entry: Any, *, key_candidates: tuple[str, ...]) -> dict[str, Any]:
    if isinstance(entry, Mapping):
        inline = None
        for key in key_candidates:
            candidate = entry.get(key)
            if isinstance(candidate, Mapping):
                inline = dict(candidate)
                break
        if inline is not None:
            return inline
        file_value = first_non_empty(entry.get("file"), entry.get("path"))
        if file_value:
            return load_json(Path(file_value).expanduser().resolve())
    if isinstance(entry, str) and entry.strip():
        return load_json(Path(entry).expanduser().resolve())
    raise ValueError(f"Unsupported dataset entry: {entry!r}")


def normalize_request(raw: Mapping[str, Any], *, out_dir_override: str | None = None) -> dict[str, Any]:
    inputs = raw.get("inputs") if isinstance(raw.get("inputs"), Mapping) else {}
    publish = raw.get("publish") if isinstance(raw.get("publish"), Mapping) else {}
    out_dir_value = out_dir_override or raw.get("out_dir") or raw.get("output_dir")
    normalized = {
        "inputs": {
            "bundle_paths": [str(Path(item).expanduser().resolve()) for item in ensure_list(inputs.get("bundle_paths")) if str(item).strip()],
            "lifecyclemodels": ensure_list(inputs.get("lifecyclemodels")),
            "processes": ensure_list(inputs.get("processes")),
            "sources": ensure_list(inputs.get("sources")),
            "relations": ensure_list(inputs.get("relations")),
            "process_build_runs": ensure_list(inputs.get("process_build_runs")),
        },
        "publish": {
            "commit": bool(publish.get("commit", False)),
            "publish_lifecyclemodels": bool(publish.get("publish_lifecyclemodels", True)),
            "publish_processes": bool(publish.get("publish_processes", True)),
            "publish_sources": bool(publish.get("publish_sources", True)),
            "publish_relations": bool(publish.get("publish_relations", True)),
            "publish_process_build_runs": bool(publish.get("publish_process_build_runs", True)),
            "relation_mode": str(publish.get("relation_mode") or "local_manifest_only"),
            "max_attempts": int(publish.get("max_attempts") or DEFAULT_MAX_ATTEMPTS),
            "retry_delay_seconds": float(publish.get("retry_delay_seconds") or DEFAULT_RETRY_DELAY_SECONDS),
            "process_build_forward_args": [
                str(item)
                for item in ensure_list(publish.get("process_build_forward_args"))
                if str(item).strip()
            ],
        },
        "out_dir": str(Path(str(out_dir_value or DEFAULT_OUT_DIR)).expanduser().resolve()),
    }
    if normalized["publish"]["relation_mode"] != "local_manifest_only":
        raise ValueError("publish.relation_mode only supports 'local_manifest_only'")
    return normalized


def collect_inputs(normalized: Mapping[str, Any]) -> dict[str, Any]:
    inputs = normalized["inputs"]
    collected = {
        "bundle_paths": list(inputs.get("bundle_paths") or []),
        "lifecyclemodels": [],
        "processes": [],
        "sources": [],
        "relations": [],
        "process_build_runs": [],
    }

    for bundle_path_raw in inputs.get("bundle_paths") or []:
        bundle_path = Path(str(bundle_path_raw)).expanduser().resolve()
        bundle = load_json(bundle_path)
        if not isinstance(bundle, Mapping):
            raise ValueError(f"Publish bundle must be a JSON object: {bundle_path}")
        collected["lifecyclemodels"].extend(ensure_list(bundle.get("lifecyclemodels")))
        collected["processes"].extend(ensure_list(bundle.get("projected_processes")))
        collected["processes"].extend(ensure_list(bundle.get("processes")))
        collected["sources"].extend(ensure_list(bundle.get("sources")))
        collected["relations"].extend(ensure_list(bundle.get("resulting_process_relations")))
        collected["relations"].extend(ensure_list(bundle.get("relations")))
        collected["process_build_runs"].extend(ensure_list(bundle.get("process_build_runs")))

    for key in ("lifecyclemodels", "processes", "sources", "relations", "process_build_runs"):
        collected[key].extend(ensure_list(inputs.get(key)))
    return collected


def publish_lifecyclemodels(
    entries: list[Any],
    *,
    session: PublishSession,
    commit: bool,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for entry in entries:
        payload = load_dataset_entry(entry, key_candidates=("json_ordered", "jsonOrdered", "payload"))
        dataset_id, version = extract_lifecyclemodel_identity(payload)
        report: dict[str, Any] = {
            "table": "lifecyclemodels",
            "id": dataset_id,
            "version": version,
            "status": "prepared" if not commit else "pending",
        }
        if not commit:
            reports.append(report)
            continue
        existing = session.select(table="lifecyclemodels", dataset_id=dataset_id)
        if select_rows(existing):
            report["status"] = "skipped_existing"
            report["select_before"] = existing
            reports.append(report)
            continue
        insert_result = session.insert(table="lifecyclemodels", dataset_id=dataset_id, payload=payload)
        report["status"] = "inserted"
        report["insert"] = insert_result
        reports.append(report)
    return reports


def publish_processes(
    entries: list[Any],
    *,
    session: PublishSession,
    commit: bool,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for entry in entries:
        payload = load_dataset_entry(entry, key_candidates=("json_ordered", "jsonOrdered", "payload"))
        try:
            dataset_id, version = extract_process_identity(payload)
        except ValueError:
            projection_id = first_non_empty(payload.get("@id"), payload.get("id"))
            projection_version = first_non_empty(payload.get("@version"), payload.get("version"), DEFAULT_DATASET_VERSION)
            reports.append(
                {
                    "table": "processes",
                    "id": projection_id,
                    "version": projection_version,
                    "status": "deferred_projection_payload",
                    "reason": "Payload is not a canonical processDataSet wrapper; keep it in the local publish bundle until a projection-to-process publish adapter exists.",
                }
            )
            continue
        report: dict[str, Any] = {
            "table": "processes",
            "id": dataset_id,
            "version": version,
            "status": "prepared" if not commit else "pending",
        }
        if not commit:
            reports.append(report)
            continue
        existing = session.select(table="processes", dataset_id=dataset_id, version=version)
        try:
            if select_rows(existing):
                report["select_before"] = existing
                report["update"] = session.update(table="processes", dataset_id=dataset_id, version=version, payload=payload)
                report["status"] = "updated"
            else:
                report["insert"] = session.insert(table="processes", dataset_id=dataset_id, payload=payload)
                report["status"] = "inserted"
        except Exception as primary_exc:  # noqa: BLE001
            fallback_result = None
            try:
                if select_rows(existing):
                    fallback_result = session.insert(table="processes", dataset_id=dataset_id, payload=payload)
                    report["status"] = "inserted_after_update_failure"
                    report["insert"] = fallback_result
                else:
                    fallback_result = session.update(table="processes", dataset_id=dataset_id, version=version, payload=payload)
                    report["status"] = "updated_after_insert_failure"
                    report["update"] = fallback_result
            except Exception as secondary_exc:  # noqa: BLE001
                report["status"] = "failed"
                report["error"] = {
                    "primary": str(primary_exc),
                    "secondary": str(secondary_exc),
                }
        reports.append(report)
    return reports


def publish_sources(
    entries: list[Any],
    *,
    session: PublishSession,
    commit: bool,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for entry in entries:
        payload = load_dataset_entry(entry, key_candidates=("json_ordered", "jsonOrdered", "payload"))
        dataset_id, version = extract_source_identity(payload)
        report: dict[str, Any] = {
            "table": "sources",
            "id": dataset_id,
            "version": version,
            "status": "prepared" if not commit else "pending",
        }
        if not commit:
            reports.append(report)
            continue
        existing = session.select(table="sources", dataset_id=dataset_id, version=version)
        try:
            if select_rows(existing):
                report["select_before"] = existing
                report["update"] = session.update(table="sources", dataset_id=dataset_id, version=version, payload=payload)
                report["status"] = "updated"
            else:
                report["insert"] = session.insert(table="sources", dataset_id=dataset_id, payload=payload)
                report["status"] = "inserted"
        except Exception as primary_exc:  # noqa: BLE001
            try:
                if select_rows(existing):
                    report["insert"] = session.insert(table="sources", dataset_id=dataset_id, payload=payload)
                    report["status"] = "inserted_after_update_failure"
                else:
                    report["update"] = session.update(table="sources", dataset_id=dataset_id, version=version, payload=payload)
                    report["status"] = "updated_after_insert_failure"
            except Exception as secondary_exc:  # noqa: BLE001
                report["status"] = "failed"
                report["error"] = {
                    "primary": str(primary_exc),
                    "secondary": str(secondary_exc),
                }
        reports.append(report)
    return reports


def delegate_process_build_runs(
    entries: list[Any],
    *,
    commit: bool,
    forward_args: list[str],
    out_dir: Path,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    if not entries:
        return reports
    if not PROCESS_AUTOMATED_BUILDER_WRAPPER.exists():
        raise FileNotFoundError(PROCESS_AUTOMATED_BUILDER_WRAPPER)
    for index, entry in enumerate(entries, start=1):
        run_id = None
        if isinstance(entry, Mapping):
            run_id = first_non_empty(entry.get("run_id"))
        elif isinstance(entry, str):
            run_id = entry.strip()
        if not run_id:
            raise ValueError(f"process_build_run missing run_id: {entry!r}")
        log_path = out_dir / "delegated-process-build-runs" / f"{run_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(PROCESS_AUTOMATED_BUILDER_WRAPPER),
            "--mode",
            "langgraph",
            "--",
            "--publish-only",
            "--run-id",
            run_id,
            *forward_args,
        ]
        if commit:
            cmd.append("--commit")
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"[{now_iso()}] CMD: {' '.join(cmd)}\n")
            handle.flush()
            result = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        reports.append(
            {
                "index": index,
                "run_id": run_id,
                "status": "ok" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "log_path": str(log_path),
                "command": cmd,
            }
        )
    return reports


def publish_request(request_path: Path, *, out_dir_override: str | None = None) -> dict[str, Any]:
    raw = load_json(request_path)
    if not isinstance(raw, Mapping):
        raise ValueError("request must be a JSON object")
    normalized = normalize_request(raw, out_dir_override=out_dir_override)
    out_dir = Path(normalized["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / "normalized-request.json", normalized)

    collected = collect_inputs(normalized)
    dump_json(out_dir / "collected-inputs.json", collected)

    publish_cfg = normalized["publish"]
    session = PublishSession(
        commit=bool(publish_cfg["commit"]),
        max_attempts=int(publish_cfg["max_attempts"]),
        retry_delay_seconds=float(publish_cfg["retry_delay_seconds"]),
    )

    lifecyclemodel_reports = []
    if publish_cfg["publish_lifecyclemodels"]:
        lifecyclemodel_reports = publish_lifecyclemodels(
            collected["lifecyclemodels"],
            session=session,
            commit=bool(publish_cfg["commit"]),
        )

    process_reports = []
    if publish_cfg["publish_processes"]:
        process_reports = publish_processes(
            collected["processes"],
            session=session,
            commit=bool(publish_cfg["commit"]),
        )

    source_reports = []
    if publish_cfg["publish_sources"]:
        source_reports = publish_sources(
            collected["sources"],
            session=session,
            commit=bool(publish_cfg["commit"]),
        )

    delegated_reports = []
    if publish_cfg["publish_process_build_runs"]:
        delegated_reports = delegate_process_build_runs(
            collected["process_build_runs"],
            commit=bool(publish_cfg["commit"]),
            forward_args=list(publish_cfg["process_build_forward_args"]),
            out_dir=out_dir,
        )

    relation_manifest = {
        "generated_at_utc": now_iso(),
        "relation_mode": publish_cfg["relation_mode"],
        "status": "prepared_local_relation_manifest",
        "relations": copy.deepcopy(collected["relations"]) if publish_cfg["publish_relations"] else [],
    }
    relation_manifest_path = dump_json(out_dir / "relation-manifest.json", relation_manifest)

    summary = {
        "generated_at_utc": now_iso(),
        "request_path": str(request_path.resolve()),
        "out_dir": str(out_dir),
        "commit": bool(publish_cfg["commit"]),
        "status": "completed_with_failures"
        if any(
            item.get("status") == "failed"
            for item in lifecyclemodel_reports + process_reports + source_reports + delegated_reports
        )
        else "completed",
        "counts": {
            "bundle_paths": len(collected["bundle_paths"]),
            "lifecyclemodels": len(lifecyclemodel_reports),
            "processes": len(process_reports),
            "sources": len(source_reports),
            "relations": len(relation_manifest["relations"]),
            "process_build_runs": len(delegated_reports),
        },
        "files": {
            "normalized_request": str(out_dir / "normalized-request.json"),
            "collected_inputs": str(out_dir / "collected-inputs.json"),
            "relation_manifest": str(relation_manifest_path),
        },
    }
    dump_json(out_dir / "publish-report.json", {
        "summary": summary,
        "lifecyclemodels": lifecyclemodel_reports,
        "processes": process_reports,
        "sources": source_reports,
        "delegated_process_build_runs": delegated_reports,
        "relation_manifest": relation_manifest,
    })
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    publish = sub.add_parser("publish", help="Publish a bundle/manifest request.")
    publish.add_argument("--request", required=True, help="Path to publish request JSON.")
    publish.add_argument("--out-dir", help="Override output directory for reports.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "publish":
        summary = publish_request(
            Path(args.request).expanduser().resolve(),
            out_dir_override=args.out_dir,
        )
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
