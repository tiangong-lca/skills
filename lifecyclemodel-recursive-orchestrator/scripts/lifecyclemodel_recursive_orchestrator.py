#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
REPO_ROOT = SKILL_DIR.parent

REQUEST_SCHEMA_PATH = SKILL_DIR / "assets" / "request.schema.json"
GRAPH_SCHEMA_PATH = SKILL_DIR / "assets" / "graph-manifest.schema.json"
LINEAGE_SCHEMA_PATH = SKILL_DIR / "assets" / "lineage-manifest.schema.json"

PROCESS_BUILDER_WRAPPER = REPO_ROOT / "process-automated-builder" / "scripts" / "run-process-automated-builder.sh"
LIFECYCLEMODEL_BUILDER_WRAPPER = REPO_ROOT / "lifecyclemodel-automated-builder" / "scripts" / "run-lifecyclemodel-automated-builder.sh"
PROJECTOR_WRAPPER = REPO_ROOT / "lifecyclemodel-resulting-process-builder" / "scripts" / "run-lifecyclemodel-resulting-process-builder.sh"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_non_empty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_slug(text: str) -> str:
    cleaned = []
    for ch in text.lower():
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("-")
    slug = "".join(cleaned)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "item"


def resolve_local_ref(root_schema: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"Only local schema refs are supported: {ref}")
    node: Any = root_schema
    for part in ref[2:].split("/"):
        node = node[part]
    if not isinstance(node, dict):
        raise ValueError(f"Schema ref does not resolve to an object: {ref}")
    return node


def _matches_type(value: Any, schema_type: str) -> bool:
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "null":
        return value is None
    return True


def _schema_is_valid(value: Any, schema: dict[str, Any], root_schema: dict[str, Any]) -> bool:
    return not validate_against_schema(value, schema, root_schema=root_schema)


def validate_against_schema(
    value: Any,
    schema: dict[str, Any],
    *,
    root_schema: dict[str, Any] | None = None,
    path: str = "$",
) -> list[str]:
    root = root_schema or schema
    errors: list[str] = []

    if "$ref" in schema:
        return validate_against_schema(
            value,
            resolve_local_ref(root, str(schema["$ref"])),
            root_schema=root,
            path=path,
        )

    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        if not any(_schema_is_valid(value, candidate, root) for candidate in any_of):
            errors.append(f"{path}: value does not satisfy anyOf")
        return errors

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for candidate in all_of:
            errors.extend(validate_against_schema(value, candidate, root_schema=root, path=path))

    if_schema = schema.get("if")
    then_schema = schema.get("then")
    if isinstance(if_schema, dict) and isinstance(then_schema, dict):
        if _schema_is_valid(value, if_schema, root):
            errors.extend(validate_against_schema(value, then_schema, root_schema=root, path=path))

    schema_type = schema.get("type")
    if isinstance(schema_type, str) and not _matches_type(value, schema_type):
        return [f"{path}: expected {schema_type}, got {type(value).__name__}"]

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"{path}: expected one of {enum_values}, got {value!r}")

    if "const" in schema and value != schema.get("const"):
        errors.append(f"{path}: expected const {schema.get('const')!r}, got {value!r}")

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            errors.append(f"{path}: string shorter than minLength={min_length}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{path}: number smaller than minimum={minimum}")

    if isinstance(value, list):
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for index, item in enumerate(value):
                errors.extend(
                    validate_against_schema(
                        item,
                        items_schema,
                        root_schema=root,
                        path=f"{path}[{index}]",
                    )
                )

    if isinstance(value, dict):
        required = schema.get("required") or []
        for field in required:
            if field not in value:
                errors.append(f"{path}: missing required field '{field}'")

        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, subschema in properties.items():
                if key in value and isinstance(subschema, dict):
                    errors.extend(
                        validate_against_schema(
                            value[key],
                            subschema,
                            root_schema=root,
                            path=f"{path}.{key}",
                        )
                    )

            additional = schema.get("additionalProperties", True)
            if additional is False:
                unexpected = sorted(set(value) - set(properties))
                for key in unexpected:
                    errors.append(f"{path}: unexpected property '{key}'")
            elif isinstance(additional, dict):
                for key in sorted(set(value) - set(properties)):
                    errors.extend(
                        validate_against_schema(
                            value[key],
                            additional,
                            root_schema=root,
                            path=f"{path}.{key}",
                        )
                    )

    return errors


def validate_schema_file(payload: dict[str, Any], schema_path: Path, *, label: str) -> None:
    schema = load_json(schema_path)
    errors = validate_against_schema(payload, schema)
    if errors:
        raise ValueError(f"{label} validation failed:\n- " + "\n- ".join(errors))


def resolve_path(base_dir: Path, raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if "://" in text or text.startswith("file:"):
        return text
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def default_candidate_sources() -> dict[str, bool]:
    return {
        "my_processes": True,
        "team_processes": True,
        "public_processes": True,
        "existing_lifecyclemodels": True,
        "existing_resulting_processes": True,
    }


def entity_from_root(root: dict[str, Any]) -> dict[str, Any]:
    kind = root.get("kind")
    if kind == "reference_flow":
        return copy.deepcopy(root.get("flow") or {})
    if kind == "process":
        return copy.deepcopy(root.get("process") or {})
    if kind == "lifecyclemodel":
        return copy.deepcopy(root.get("lifecyclemodel") or {})
    if kind == "resulting_process":
        return copy.deepcopy(root.get("resulting_process") or {})
    return {}


def derive_root_node(root: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
    entity = entity_from_root(root)
    label = first_non_empty(entity.get("name"), goal.get("name"), root.get("kind"), "root") or "root"
    derived = {
        "node_id": first_non_empty(root.get("node_id"), entity.get("id"), "root") or "root",
        "kind": root.get("kind"),
        "label": label,
        "entity": entity,
        "requested_action": root.get("requested_action", "auto"),
        "depends_on": ensure_list(root.get("depends_on")),
    }
    for key in (
        "existing_resulting_process_candidates",
        "existing_process_candidates",
        "existing_lifecyclemodel_candidates",
        "process_builder",
        "submodel_builder",
        "projector",
        "parent_node_id",
    ):
        if key in root:
            derived[key] = copy.deepcopy(root[key])
    return derived


def normalize_candidate(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, str):
        return {"id": candidate, "score": 1.0}
    if not isinstance(candidate, dict):
        raise ValueError(f"candidate must be an object or string, got {type(candidate).__name__}")
    normalized = copy.deepcopy(candidate)
    score = normalized.get("score")
    if score is None:
        normalized["score"] = 1.0
    else:
        normalized["score"] = float(score)
    return normalized


def normalize_node(
    raw_node: dict[str, Any],
    *,
    index: int,
    base_dir: Path,
) -> dict[str, Any]:
    node = copy.deepcopy(raw_node)
    node_id = first_non_empty(node.get("node_id"), node.get("id"), f"node-{index}") or f"node-{index}"
    kind = first_non_empty(node.get("kind"), "process") or "process"
    entity = copy.deepcopy(node.get("entity") or {})
    if not entity:
        entity = {
            key: copy.deepcopy(node[key])
            for key in ("flow", "process", "lifecyclemodel", "resulting_process")
            if key in node
        }
        if len(entity) == 1:
            entity = next(iter(entity.values()))
    label = first_non_empty(
        node.get("label"),
        entity.get("name") if isinstance(entity, dict) else None,
        node_id,
    ) or node_id
    normalized = {
        "node_id": node_id,
        "kind": kind,
        "label": label,
        "entity": entity if isinstance(entity, dict) else {},
        "requested_action": first_non_empty(node.get("requested_action"), "auto") or "auto",
        "depends_on": [str(item).strip() for item in ensure_list(node.get("depends_on")) if str(item).strip()],
        "parent_node_id": first_non_empty(node.get("parent_node_id")),
        "existing_resulting_process_candidates": sorted(
            [normalize_candidate(item) for item in ensure_list(node.get("existing_resulting_process_candidates"))],
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        ),
        "existing_process_candidates": sorted(
            [normalize_candidate(item) for item in ensure_list(node.get("existing_process_candidates"))],
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        ),
        "existing_lifecyclemodel_candidates": sorted(
            [normalize_candidate(item) for item in ensure_list(node.get("existing_lifecyclemodel_candidates"))],
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        ),
    }

    process_builder = node.get("process_builder")
    if isinstance(process_builder, dict):
        normalized["process_builder"] = {
            "mode": first_non_empty(process_builder.get("mode"), "workflow") or "workflow",
            "flow_file": resolve_path(base_dir, process_builder.get("flow_file")),
            "flow_json": process_builder.get("flow_json"),
            "run_id": first_non_empty(process_builder.get("run_id")),
            "python_bin": first_non_empty(process_builder.get("python_bin")),
            "publish": bool(process_builder.get("publish", False)),
            "commit": bool(process_builder.get("commit", False)),
            "forward_args": [str(item) for item in ensure_list(process_builder.get("forward_args"))],
        }

    submodel_builder = node.get("submodel_builder")
    if isinstance(submodel_builder, dict):
        normalized["submodel_builder"] = {
            "manifest": resolve_path(base_dir, submodel_builder.get("manifest")),
            "out_dir": resolve_path(base_dir, submodel_builder.get("out_dir")),
            "dry_run": bool(submodel_builder.get("dry_run", False)),
        }

    projector = node.get("projector")
    if isinstance(projector, dict):
        normalized["projector"] = {
            "command": first_non_empty(projector.get("command"), "project") or "project",
            "request": resolve_path(base_dir, projector.get("request")),
            "model_file": resolve_path(base_dir, projector.get("model_file")),
            "out_dir": resolve_path(base_dir, projector.get("out_dir")),
            "projection_role": first_non_empty(projector.get("projection_role"), "primary") or "primary",
            "run_always": bool(projector.get("run_always", False)),
            "publish_processes": bool(projector.get("publish_processes", False)),
            "publish_relations": bool(projector.get("publish_relations", False)),
        }

    if normalized["parent_node_id"]:
        normalized["depends_on"].append(normalized["parent_node_id"])
    normalized["depends_on"] = sorted(set(normalized["depends_on"]))
    return normalized


def build_edges(request_edges: list[Any], nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    known_pairs: set[tuple[str, str, str]] = set()
    edges: list[dict[str, Any]] = []
    for raw in request_edges:
        if not isinstance(raw, dict):
            continue
        src = first_non_empty(raw.get("from"))
        dst = first_non_empty(raw.get("to"))
        relation = first_non_empty(raw.get("relation"), "depends_on") or "depends_on"
        if not src or not dst:
            continue
        key = (src, dst, relation)
        if key in known_pairs:
            continue
        known_pairs.add(key)
        edges.append(
            {
                "from": src,
                "to": dst,
                "relation": relation,
            }
        )
    for node in nodes:
        for dep in node.get("depends_on", []):
            key = (node["node_id"], dep, "depends_on")
            if key in known_pairs:
                continue
            known_pairs.add(key)
            edges.append(
                {
                    "from": node["node_id"],
                    "to": dep,
                    "relation": "depends_on",
                }
            )
    return edges


def topo_sort_nodes(nodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    order = {node["node_id"]: index for index, node in enumerate(nodes)}
    node_map = {node["node_id"]: node for node in nodes}
    indegree = {node["node_id"]: 0 for node in nodes}
    adjacency: dict[str, list[str]] = {node["node_id"]: [] for node in nodes}
    warnings: list[str] = []

    for node in nodes:
        for dep in node.get("depends_on", []):
            if dep not in node_map:
                warnings.append(f"Node {node['node_id']} depends on unknown node {dep}; keeping it as metadata only.")
                continue
            indegree[node["node_id"]] += 1
            adjacency[dep].append(node["node_id"])

    queue = deque(sorted((node_id for node_id, degree in indegree.items() if degree == 0), key=lambda item: order[item]))
    result: list[dict[str, Any]] = []
    while queue:
        node_id = queue.popleft()
        result.append(node_map[node_id])
        for downstream in sorted(adjacency[node_id], key=lambda item: order[item]):
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                queue.append(downstream)

    if len(result) != len(nodes):
        warnings.append("Dependency cycle detected; preserving original order for cyclic remainder.")
        seen = {node["node_id"] for node in result}
        for node in nodes:
            if node["node_id"] not in seen:
                result.append(node)
    return result, warnings


def select_resolution(node: dict[str, Any], orchestration: dict[str, Any]) -> dict[str, Any]:
    requested_action = first_non_empty(node.get("requested_action"), "auto") or "auto"
    result_candidate = (node.get("existing_resulting_process_candidates") or [None])[0]
    process_candidate = (node.get("existing_process_candidates") or [None])[0]
    lifecyclemodel_candidate = (node.get("existing_lifecyclemodel_candidates") or [None])[0]
    allow_process_build = bool(orchestration.get("allow_process_build", False))
    allow_submodel_build = bool(orchestration.get("allow_submodel_build", False))
    reuse_resulting_process_first = bool(orchestration.get("reuse_resulting_process_first", True))

    def unresolved(reason: str) -> dict[str, Any]:
        return {
            "resolution": "unresolved",
            "selected_candidate": None,
            "reason": reason,
            "boundary_reason": "unresolved",
        }

    if requested_action == "cutoff":
        return {
            "resolution": "cutoff",
            "selected_candidate": None,
            "reason": "Explicit cutoff requested.",
            "boundary_reason": "explicit_cutoff",
        }
    if requested_action == "unresolved":
        return unresolved("Explicit unresolved marker provided.")
    if requested_action == "reuse_existing_resulting_process":
        if not result_candidate:
            return unresolved("requested reuse_existing_resulting_process but no candidate was provided")
        return {
            "resolution": "reused_existing_resulting_process",
            "selected_candidate": result_candidate,
            "reason": "Requested resulting-process reuse.",
            "boundary_reason": "collapsed_at_resulting_process",
        }
    if requested_action == "reuse_existing_process":
        if not process_candidate:
            return unresolved("requested reuse_existing_process but no candidate was provided")
        return {
            "resolution": "reused_existing_process",
            "selected_candidate": process_candidate,
            "reason": "Requested process reuse.",
            "boundary_reason": "collapsed_at_existing_process",
        }
    if requested_action == "reuse_existing_model":
        if not lifecyclemodel_candidate:
            return unresolved("requested reuse_existing_model but no lifecyclemodel candidate was provided")
        return {
            "resolution": "reused_existing_model",
            "selected_candidate": lifecyclemodel_candidate,
            "reason": "Requested lifecyclemodel reuse.",
            "boundary_reason": "collapsed_at_existing_model",
        }
    if requested_action == "build_process":
        if not node.get("process_builder"):
            return unresolved("requested build_process but process_builder config is missing")
        if not allow_process_build:
            return unresolved("requested build_process but orchestration.allow_process_build=false")
        return {
            "resolution": "build_via_process_automated_builder",
            "selected_candidate": None,
            "reason": "Requested process build.",
            "boundary_reason": None,
        }
    if requested_action == "build_submodel":
        if not node.get("submodel_builder"):
            return unresolved("requested build_submodel but submodel_builder config is missing")
        if not allow_submodel_build:
            return unresolved("requested build_submodel but orchestration.allow_submodel_build=false")
        return {
            "resolution": "build_via_lifecyclemodel_automated_builder",
            "selected_candidate": None,
            "reason": "Requested submodel build.",
            "boundary_reason": None,
        }

    if reuse_resulting_process_first and result_candidate:
        return {
            "resolution": "reused_existing_resulting_process",
            "selected_candidate": result_candidate,
            "reason": "Auto-selected highest scoring resulting process candidate.",
            "boundary_reason": "collapsed_at_resulting_process",
        }
    if process_candidate:
        return {
            "resolution": "reused_existing_process",
            "selected_candidate": process_candidate,
            "reason": "Auto-selected highest scoring existing process candidate.",
            "boundary_reason": "collapsed_at_existing_process",
        }
    if node.get("kind") in {"lifecyclemodel", "subsystem"} and lifecyclemodel_candidate:
        return {
            "resolution": "reused_existing_model",
            "selected_candidate": lifecyclemodel_candidate,
            "reason": "Auto-selected highest scoring lifecyclemodel candidate.",
            "boundary_reason": "collapsed_at_existing_model",
        }
    if node.get("kind") in {"lifecyclemodel", "subsystem"} and node.get("submodel_builder") and allow_submodel_build:
        return {
            "resolution": "build_via_lifecyclemodel_automated_builder",
            "selected_candidate": None,
            "reason": "No reusable model/process matched; scheduling lifecyclemodel builder.",
            "boundary_reason": None,
        }
    if node.get("process_builder") and allow_process_build:
        return {
            "resolution": "build_via_process_automated_builder",
            "selected_candidate": None,
            "reason": "No reusable process matched; scheduling process builder.",
            "boundary_reason": None,
        }
    if lifecyclemodel_candidate:
        return {
            "resolution": "reused_existing_model",
            "selected_candidate": lifecyclemodel_candidate,
            "reason": "Fallback to lifecyclemodel candidate after no process candidate matched.",
            "boundary_reason": "collapsed_at_existing_model",
        }
    return unresolved("No reusable candidate or build config satisfied the node policy")


def should_run_projector(node: dict[str, Any], resolution: str) -> bool:
    projector = node.get("projector")
    if not projector:
        return False
    if projector.get("run_always"):
        return True
    return resolution in {
        "build_via_lifecyclemodel_automated_builder",
        "reused_existing_model",
    }


def build_plan(
    normalized_request: dict[str, Any],
    *,
    request_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    validate_schema_file(normalized_request, REQUEST_SCHEMA_PATH, label="request")
    goal = copy.deepcopy(normalized_request.get("goal") or {})
    root = copy.deepcopy(normalized_request.get("root") or {})
    orchestration = copy.deepcopy(normalized_request.get("orchestration") or {})
    orchestration.setdefault("fail_fast", True)
    candidate_sources = default_candidate_sources()
    candidate_sources.update(copy.deepcopy(normalized_request.get("candidate_sources") or {}))
    publish = copy.deepcopy(normalized_request.get("publish") or {})
    publish.setdefault("intent", "dry_run")
    publish.setdefault("prepare_lifecyclemodel_payload", True)
    publish.setdefault("prepare_resulting_process_payload", True)
    publish.setdefault("prepare_relation_payload", True)

    root_node = derive_root_node(root, goal)
    requested_nodes = [item for item in ensure_list(normalized_request.get("nodes")) if isinstance(item, dict)]
    root_node_id = root_node["node_id"]
    raw_nodes: list[dict[str, Any]] = []
    if not any(first_non_empty(item.get("node_id"), item.get("id")) == root_node_id for item in requested_nodes):
        raw_nodes.append(root_node)
    raw_nodes.extend(requested_nodes)
    nodes: list[dict[str, Any]] = []
    seen_node_ids: set[str] = set()
    for index, raw_node in enumerate(raw_nodes, start=1):
        if not isinstance(raw_node, dict):
            continue
        normalized_node = normalize_node(raw_node, index=index, base_dir=request_path.parent)
        if normalized_node["node_id"] in seen_node_ids:
            if normalized_node["node_id"] == "root":
                continue
            raise ValueError(f"Duplicate node_id: {normalized_node['node_id']}")
        seen_node_ids.add(normalized_node["node_id"])
        nodes.append(normalized_node)

    ordered_nodes, warnings = topo_sort_nodes(nodes)
    edges = build_edges(ensure_list(normalized_request.get("edges")), ordered_nodes)
    invocations: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []

    for node in ordered_nodes:
        resolution_info = select_resolution(node, orchestration)
        node["resolution"] = resolution_info["resolution"]
        node["resolution_reason"] = resolution_info["reason"]
        node["selected_candidate"] = resolution_info["selected_candidate"]
        node["boundary_reason"] = resolution_info["boundary_reason"]
        node["planned_invocations"] = []
        if node["boundary_reason"]:
            boundaries.append(
                {
                    "node_id": node["node_id"],
                    "reason": node["boundary_reason"],
                }
            )
        if node["resolution"] == "unresolved":
            unresolved.append(
                {
                    "node_id": node["node_id"],
                    "label": node["label"],
                    "reason": node["resolution_reason"],
                }
            )
            continue

        if node["resolution"] == "build_via_process_automated_builder":
            invocation_id = f"{node['node_id']}:process-builder"
            artifact_dir = REPO_ROOT / "artifacts" / "process_from_flow"
            invocations.append(
                {
                    "invocation_id": invocation_id,
                    "node_id": node["node_id"],
                    "kind": "process_builder",
                    "config": copy.deepcopy(node.get("process_builder") or {}),
                    "artifact_dir": str(artifact_dir),
                }
            )
            node["planned_invocations"].append(invocation_id)

        if node["resolution"] == "build_via_lifecyclemodel_automated_builder":
            invocation_id = f"{node['node_id']}:lifecyclemodel-builder"
            artifact_dir = Path(
                (node.get("submodel_builder") or {}).get("out_dir")
                or out_dir / "downstream" / safe_slug(node["node_id"]) / "lifecyclemodel-builder"
            ).resolve()
            invocations.append(
                {
                    "invocation_id": invocation_id,
                    "node_id": node["node_id"],
                    "kind": "lifecyclemodel_builder",
                    "config": copy.deepcopy(node.get("submodel_builder") or {}),
                    "artifact_dir": str(artifact_dir),
                }
            )
            node["planned_invocations"].append(invocation_id)

        if should_run_projector(node, node["resolution"]):
            invocation_id = f"{node['node_id']}:projector"
            depends_on_invocation_id = None
            if node["resolution"] == "build_via_lifecyclemodel_automated_builder":
                depends_on_invocation_id = f"{node['node_id']}:lifecyclemodel-builder"
            artifact_dir = Path(
                (node.get("projector") or {}).get("out_dir")
                or out_dir / "downstream" / safe_slug(node["node_id"]) / "projector"
            ).resolve()
            invocations.append(
                {
                    "invocation_id": invocation_id,
                    "node_id": node["node_id"],
                    "kind": "projector",
                    "depends_on_invocation_id": depends_on_invocation_id,
                    "config": copy.deepcopy(node.get("projector") or {}),
                    "artifact_dir": str(artifact_dir),
                }
            )
            node["planned_invocations"].append(invocation_id)

    plan = {
        "skill": "lifecyclemodel-recursive-orchestrator",
        "request_id": normalized_request.get("request_id") or f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": now_iso(),
        "request_file": str(request_path),
        "goal": goal,
        "root": root,
        "orchestration": orchestration,
        "candidate_sources": candidate_sources,
        "publish": publish,
        "notes": ensure_list(normalized_request.get("notes")),
        "nodes": ordered_nodes,
        "edges": edges,
        "invocations": invocations,
        "planner_summary": {
            "status": "planned",
            "message": "Request validated, nodes normalized, and downstream invocations scheduled.",
        },
        "warnings": warnings,
        "unresolved": unresolved,
        "boundaries": boundaries,
        "artifacts": {
            "root": str(out_dir),
            "request_normalized": str(out_dir / "request.normalized.json"),
            "assembly_plan": str(out_dir / "assembly-plan.json"),
            "graph_manifest": str(out_dir / "graph-manifest.json"),
            "lineage_manifest": str(out_dir / "lineage-manifest.json"),
            "boundary_report": str(out_dir / "boundary-report.json"),
            "invocations_dir": str(out_dir / "invocations"),
            "publish_bundle": str(out_dir / "publish-bundle.json"),
            "publish_summary": str(out_dir / "publish-summary.json"),
        },
        "summary": {
            "node_count": len(ordered_nodes),
            "edge_count": len(edges),
            "invocation_count": len(invocations),
            "unresolved_count": len(unresolved),
        },
    }
    return plan


def execution_status_by_node(plan: dict[str, Any], execution_results: list[dict[str, Any]]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    results_by_node: dict[str, list[dict[str, Any]]] = {}
    for result in execution_results:
        results_by_node.setdefault(result["node_id"], []).append(result)

    for node in plan.get("nodes", []):
        node_id = node["node_id"]
        node_results = results_by_node.get(node_id, [])
        if node["resolution"] == "unresolved":
            statuses[node_id] = "unresolved"
        elif node["resolution"] == "cutoff":
            statuses[node_id] = "cutoff"
        elif node["resolution"].startswith("reused_"):
            statuses[node_id] = "reused"
        elif not node_results:
            statuses[node_id] = "planned"
        elif any(result["status"] == "failed" for result in node_results):
            statuses[node_id] = "failed"
        elif any(result["status"].startswith("skipped") for result in node_results):
            statuses[node_id] = "blocked"
        elif all(result["status"] == "success" for result in node_results):
            statuses[node_id] = "completed"
        else:
            statuses[node_id] = "incomplete"
    return statuses


def collect_resulting_process_relations(execution_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for result in execution_results:
        artifacts = result.get("artifacts") or {}
        bundle_path = projector_bundle_path_from_artifacts(artifacts) if isinstance(artifacts, dict) else None
        if not bundle_path:
            continue
        path = Path(str(bundle_path))
        if not path.exists():
            continue
        try:
            bundle = load_json(path)
        except Exception:
            continue
        for relation in ensure_list(bundle.get("relations")):
            if isinstance(relation, dict):
                relation_copy = copy.deepcopy(relation)
                relation_copy.setdefault("node_id", result.get("node_id"))
                relations.append(relation_copy)
    return relations


def build_graph_manifest(plan: dict[str, Any], execution_results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    execution_results = execution_results or []
    node_statuses = execution_status_by_node(plan, execution_results)
    graph = {
        "root": {
            "request_id": plan["request_id"],
            "goal": plan.get("goal") or {},
            "mode": (plan.get("orchestration") or {}).get("mode"),
            "max_depth": (plan.get("orchestration") or {}).get("max_depth"),
        },
        "nodes": [],
        "edges": copy.deepcopy(plan.get("edges") or []),
        "boundaries": copy.deepcopy(plan.get("boundaries") or []),
        "unresolved": copy.deepcopy(plan.get("unresolved") or []),
        "stats": {
            "node_count": len(plan.get("nodes") or []),
            "edge_count": len(plan.get("edges") or []),
            "invocation_count": len(plan.get("invocations") or []),
            "unresolved_count": len(plan.get("unresolved") or []),
            "completed_invocation_count": sum(1 for result in execution_results if result["status"] == "success"),
        },
    }
    for node in plan.get("nodes", []):
        graph["nodes"].append(
            {
                "node_id": node["node_id"],
                "label": node["label"],
                "kind": node["kind"],
                "resolution": node["resolution"],
                "execution_status": node_statuses.get(node["node_id"], "planned"),
                "selected_candidate": copy.deepcopy(node.get("selected_candidate")),
                "depends_on": copy.deepcopy(node.get("depends_on") or []),
                "boundary_reason": node.get("boundary_reason"),
            }
        )
    validate_schema_file(graph, GRAPH_SCHEMA_PATH, label="graph manifest")
    return graph


def build_lineage_manifest(plan: dict[str, Any], execution_results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    execution_results = execution_results or []
    execution_by_invocation = {result["invocation_id"]: result for result in execution_results}
    node_statuses = execution_status_by_node(plan, execution_results)
    relations = collect_resulting_process_relations(execution_results)

    published_dependencies: list[dict[str, Any]] = []
    for node in plan.get("nodes", []):
        candidate = node.get("selected_candidate")
        if not isinstance(candidate, dict):
            continue
        published_dependencies.append(
            {
                "node_id": node["node_id"],
                "dependency_type": node["resolution"],
                "candidate_id": candidate.get("id"),
                "candidate_version": candidate.get("version"),
            }
        )

    lineage = {
        "root_request": {
            "request_id": plan["request_id"],
            "goal": copy.deepcopy(plan.get("goal") or {}),
            "root": copy.deepcopy(plan.get("root") or {}),
            "orchestration": copy.deepcopy(plan.get("orchestration") or {}),
            "publish": copy.deepcopy(plan.get("publish") or {}),
        },
        "builder_invocations": [],
        "node_resolution_log": [],
        "published_dependencies": published_dependencies,
        "resulting_process_relations": relations,
        "unresolved_history": copy.deepcopy(plan.get("unresolved") or []),
    }
    for invocation in plan.get("invocations", []):
        result = execution_by_invocation.get(invocation["invocation_id"], {})
        lineage["builder_invocations"].append(
            {
                "invocation_id": invocation["invocation_id"],
                "node_id": invocation["node_id"],
                "kind": invocation["kind"],
                "artifact_dir": invocation["artifact_dir"],
                "status": result.get("status", "planned"),
                "exit_code": result.get("exit_code"),
                "result_file": result.get("result_file"),
            }
        )
    for node in plan.get("nodes", []):
        lineage["node_resolution_log"].append(
            {
                "node_id": node["node_id"],
                "label": node["label"],
                "resolution": node["resolution"],
                "reason": node["resolution_reason"],
                "selected_candidate": copy.deepcopy(node.get("selected_candidate")),
                "execution_status": node_statuses.get(node["node_id"], "planned"),
            }
        )
    validate_schema_file(lineage, LINEAGE_SCHEMA_PATH, label="lineage manifest")
    return lineage


def build_boundary_report(plan: dict[str, Any], execution_results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    execution_results = execution_results or []
    return {
        "request_id": plan["request_id"],
        "generated_at": now_iso(),
        "boundaries": copy.deepcopy(plan.get("boundaries") or []),
        "unresolved": copy.deepcopy(plan.get("unresolved") or []),
        "execution_summary": {
            "successful_invocations": sum(1 for result in execution_results if result["status"] == "success"),
            "failed_invocations": sum(1 for result in execution_results if result["status"] == "failed"),
            "blocked_invocations": sum(1 for result in execution_results if result["status"].startswith("skipped")),
        },
    }


def write_plan_artifacts(
    *,
    normalized_request: dict[str, Any],
    plan: dict[str, Any],
    graph_manifest: dict[str, Any],
    lineage_manifest: dict[str, Any],
    boundary_report: dict[str, Any],
    out_dir: Path,
) -> None:
    dump_json(out_dir / "request.normalized.json", normalized_request)
    dump_json(out_dir / "assembly-plan.json", plan)
    dump_json(out_dir / "graph-manifest.json", graph_manifest)
    dump_json(out_dir / "lineage-manifest.json", lineage_manifest)
    dump_json(out_dir / "boundary-report.json", boundary_report)


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise ValueError(f"Missing {description}: {path}")


def build_process_builder_command(invocation: dict[str, Any], plan: dict[str, Any]) -> tuple[list[str], Path, dict[str, Any]]:
    require_file(PROCESS_BUILDER_WRAPPER, "process builder wrapper")
    config = invocation["config"]
    command = [str(PROCESS_BUILDER_WRAPPER), "--mode", first_non_empty(config.get("mode"), "workflow") or "workflow"]
    flow_file = config.get("flow_file")
    flow_json = config.get("flow_json")
    if flow_file:
        command.extend(["--flow-file", str(flow_file)])
    elif flow_json is not None:
        payload = flow_json if isinstance(flow_json, str) else canonical_json(flow_json)
        command.extend(["--flow-json", payload])
    else:
        raise ValueError(f"{invocation['invocation_id']} is missing process_builder.flow_file or flow_json")
    if config.get("python_bin"):
        command.extend(["--python-bin", str(config["python_bin"])])

    run_id = first_non_empty(
        config.get("run_id"),
        f"{safe_slug(plan['request_id'])}-{safe_slug(invocation['node_id'])}",
    ) or f"{safe_slug(plan['request_id'])}-{safe_slug(invocation['node_id'])}"
    forward_args = [str(item) for item in ensure_list(config.get("forward_args"))]
    if "--run-id" not in forward_args:
        forward_args.extend(["--run-id", run_id])
    if "--publish" not in forward_args and "--no-publish" not in forward_args and not config.get("publish", False):
        forward_args.append("--no-publish")
    if "--commit" not in forward_args and "--no-commit" not in forward_args and not config.get("commit", False):
        forward_args.append("--no-commit")
    if forward_args:
        command.extend(["--", *forward_args])
    artifacts = {
        "run_id": run_id,
        "run_root": str(REPO_ROOT / "artifacts" / "process_from_flow" / run_id),
    }
    return command, REPO_ROOT, artifacts


def build_lifecyclemodel_builder_command(invocation: dict[str, Any]) -> tuple[list[str], Path, dict[str, Any]]:
    require_file(LIFECYCLEMODEL_BUILDER_WRAPPER, "lifecyclemodel builder wrapper")
    config = invocation["config"]
    manifest = first_non_empty(config.get("manifest"))
    if not manifest:
        raise ValueError(f"{invocation['invocation_id']} is missing submodel_builder.manifest")
    require_file(Path(manifest), "lifecyclemodel builder manifest")
    artifact_dir = Path(invocation["artifact_dir"]).resolve()
    command = [
        str(LIFECYCLEMODEL_BUILDER_WRAPPER),
        "--manifest",
        manifest,
        "--out-dir",
        str(artifact_dir),
    ]
    if config.get("dry_run"):
        command.append("--dry-run")
    artifacts = {
        "out_dir": str(artifact_dir),
        "manifest": manifest,
    }
    return command, REPO_ROOT, artifacts


def discover_submodel_builder_artifacts(out_dir: Path) -> dict[str, Any]:
    model_files = sorted(str(path.resolve()) for path in out_dir.glob("models/**/tidas_bundle/lifecyclemodels/*.json"))
    report_files = sorted(str(path.resolve()) for path in out_dir.glob("reports/*.json"))
    return {
        "out_dir": str(out_dir),
        "run_plan": str((out_dir / "run-plan.json").resolve()) if (out_dir / "run-plan.json").exists() else None,
        "resolved_manifest": str((out_dir / "resolved-manifest.json").resolve()) if (out_dir / "resolved-manifest.json").exists() else None,
        "produced_model_files": model_files,
        "report_files": report_files,
    }


def discover_process_builder_artifacts(run_root: Path) -> dict[str, Any]:
    exports_dir = run_root / "exports" / "processes"
    return {
        "run_id": run_root.name,
        "run_root": str(run_root.resolve()),
        "state_file": str((run_root / "cache" / "process_from_flow_state.json").resolve()) if (run_root / "cache" / "process_from_flow_state.json").exists() else None,
        "exports_dir": str(exports_dir.resolve()) if exports_dir.exists() else None,
    }


def discover_projector_artifacts(out_dir: Path) -> dict[str, Any]:
    return {
        "out_dir": str(out_dir.resolve()),
        "projection_bundle": str((out_dir / "process-projection-bundle.json").resolve()) if (out_dir / "process-projection-bundle.json").exists() else None,
        "projection_report": str((out_dir / "projection-report.json").resolve()) if (out_dir / "projection-report.json").exists() else None,
        "request_normalized": str((out_dir / "request.normalized.json").resolve()) if (out_dir / "request.normalized.json").exists() else None,
    }


def projector_bundle_path_from_artifacts(artifacts: dict[str, Any]) -> str | None:
    for key in ("projection_bundle", "process_projection_bundle"):
        bundle_path = first_non_empty((artifacts or {}).get(key))
        if bundle_path:
            return bundle_path
    out_dir = first_non_empty((artifacts or {}).get("out_dir"))
    if not out_dir:
        return None
    candidate = Path(str(out_dir)) / "process-projection-bundle.json"
    if candidate.exists():
        return str(candidate.resolve())
    return None


def infer_projector_model_file(
    invocation: dict[str, Any],
    execution_map: dict[str, dict[str, Any]],
) -> str | None:
    config = invocation["config"]
    explicit = first_non_empty(config.get("model_file"))
    if explicit:
        return explicit
    dependency_id = invocation.get("depends_on_invocation_id")
    if not dependency_id:
        return None
    dependency = execution_map.get(dependency_id) or {}
    dependency_artifacts = dependency.get("artifacts") or {}
    model_files = ensure_list(dependency_artifacts.get("produced_model_files"))
    if not model_files:
        return None
    return str(model_files[0])


def build_projector_command(
    invocation: dict[str, Any],
    execution_map: dict[str, dict[str, Any]],
) -> tuple[list[str], Path, dict[str, Any]]:
    require_file(PROJECTOR_WRAPPER, "resulting process builder wrapper")
    config = invocation["config"]
    artifact_dir = Path(invocation["artifact_dir"]).resolve()
    command = [str(PROJECTOR_WRAPPER), first_non_empty(config.get("command"), "build") or "build"]
    request = first_non_empty(config.get("request"))
    if request:
        require_file(Path(request), "projector request")
        command.extend(["--request", request])
    else:
        model_file = infer_projector_model_file(invocation, execution_map)
        if not model_file:
            raise ValueError(
                f"{invocation['invocation_id']} needs projector.request or a previous lifecyclemodel-builder model artifact"
            )
        require_file(Path(model_file), "projector model file")
        command.extend(["--model-file", model_file])
        command.extend(["--projection-role", first_non_empty(config.get("projection_role"), "primary") or "primary"])
    command.extend(["--out-dir", str(artifact_dir)])
    artifacts = {
        "out_dir": str(artifact_dir),
        "request": request,
    }
    return command, REPO_ROOT, artifacts


def run_command(command: list[str], *, cwd: Path) -> dict[str, Any]:
    started_at = now_iso()
    started_monotonic = datetime.now(timezone.utc)
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    finished_at = now_iso()
    duration_seconds = (datetime.now(timezone.utc) - started_monotonic).total_seconds()
    return {
        "command": command,
        "cwd": str(cwd),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(duration_seconds, 3),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "status": "success" if completed.returncode == 0 else "failed",
    }


def execute_plan(plan: dict[str, Any], *, out_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    invocations_dir = out_dir / "invocations"
    invocations_dir.mkdir(parents=True, exist_ok=True)
    execution_results: list[dict[str, Any]] = []
    execution_map: dict[str, dict[str, Any]] = {}
    fail_fast = bool((plan.get("orchestration") or {}).get("fail_fast", True))
    stop_remaining = False

    for invocation in plan.get("invocations", []):
        result_file = invocations_dir / f"{safe_slug(invocation['invocation_id'])}.json"
        if stop_remaining:
            skipped = {
                "invocation_id": invocation["invocation_id"],
                "node_id": invocation["node_id"],
                "kind": invocation["kind"],
                "status": "skipped_due_to_fail_fast",
                "exit_code": None,
                "result_file": str(result_file.resolve()),
            }
            dump_json(result_file, skipped)
            execution_results.append(skipped)
            execution_map[invocation["invocation_id"]] = skipped
            continue

        dependency_id = invocation.get("depends_on_invocation_id")
        if dependency_id:
            dependency = execution_map.get(str(dependency_id))
            if dependency and dependency.get("status") != "success":
                skipped = {
                    "invocation_id": invocation["invocation_id"],
                    "node_id": invocation["node_id"],
                    "kind": invocation["kind"],
                    "status": f"skipped_due_to_dependency_{dependency.get('status', 'unknown')}",
                    "exit_code": None,
                    "depends_on_invocation_id": dependency_id,
                    "result_file": str(result_file.resolve()),
                }
                dump_json(result_file, skipped)
                execution_results.append(skipped)
                execution_map[invocation["invocation_id"]] = skipped
                continue

        if invocation["kind"] == "process_builder":
            command, cwd, artifacts = build_process_builder_command(invocation, plan)
        elif invocation["kind"] == "lifecyclemodel_builder":
            command, cwd, artifacts = build_lifecyclemodel_builder_command(invocation)
        elif invocation["kind"] == "projector":
            command, cwd, artifacts = build_projector_command(invocation, execution_map)
        else:
            raise ValueError(f"Unsupported invocation kind: {invocation['kind']}")

        result = run_command(command, cwd=cwd)
        result["invocation_id"] = invocation["invocation_id"]
        result["node_id"] = invocation["node_id"]
        result["kind"] = invocation["kind"]
        result["planned_artifacts"] = artifacts
        result["result_file"] = str(result_file.resolve())

        if invocation["kind"] == "process_builder":
            result["artifacts"] = discover_process_builder_artifacts(Path(artifacts["run_root"]))
        elif invocation["kind"] == "lifecyclemodel_builder":
            result["artifacts"] = discover_submodel_builder_artifacts(Path(artifacts["out_dir"]))
        elif invocation["kind"] == "projector":
            result["artifacts"] = discover_projector_artifacts(Path(artifacts["out_dir"]))

        dump_json(result_file, result)
        execution_results.append(result)
        execution_map[invocation["invocation_id"]] = result
        if result["status"] == "failed" and fail_fast:
            stop_remaining = True

    plan["execution_summary"] = {
        "executed_at": now_iso(),
        "successful_invocations": sum(1 for result in execution_results if result["status"] == "success"),
        "failed_invocations": sum(1 for result in execution_results if result["status"] == "failed"),
        "blocked_invocations": sum(1 for result in execution_results if result["status"].startswith("skipped")),
        "status": "failed" if any(result["status"] == "failed" for result in execution_results) else "completed",
    }
    plan["planner_summary"] = {
        "status": "executed",
        "message": "Scheduled downstream builders were executed and invocation artifacts were recorded.",
    }
    for invocation in plan.get("invocations", []):
        result = execution_map.get(invocation["invocation_id"])
        if not result:
            continue
        invocation["last_status"] = result["status"]
        invocation["last_exit_code"] = result.get("exit_code")
        invocation["last_result_file"] = result.get("result_file")
        invocation["artifacts"] = result.get("artifacts")
    return plan, execution_results


def load_invocation_results(run_dir: Path) -> list[dict[str, Any]]:
    invocations_dir = run_dir / "invocations"
    if not invocations_dir.exists():
        return []
    results: list[dict[str, Any]] = []
    for path in sorted(invocations_dir.glob("*.json")):
        try:
            results.append(load_json(path))
        except Exception:
            continue
    return results


def collect_publish_bundle(
    *,
    run_dir: Path,
    plan: dict[str, Any],
    graph_manifest: dict[str, Any],
    lineage_manifest: dict[str, Any],
    execution_results: list[dict[str, Any]],
    include_lifecyclemodels: bool,
    include_resulting_process_relations: bool,
) -> dict[str, Any]:
    lifecyclemodels: list[dict[str, Any]] = []
    projected_processes: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    process_build_runs: list[dict[str, Any]] = []

    for result in execution_results:
        artifacts = result.get("artifacts") or {}
        if result.get("kind") == "lifecyclemodel_builder" and include_lifecyclemodels:
            for model_file in ensure_list(artifacts.get("produced_model_files")):
                path = Path(str(model_file))
                if not path.exists():
                    continue
                try:
                    lifecyclemodels.append(
                        {
                            "node_id": result.get("node_id"),
                            "file": str(path.resolve()),
                            "json_ordered": load_json(path),
                        }
                    )
                except Exception:
                    continue
        if result.get("kind") == "projector" and include_resulting_process_relations:
            bundle_path = projector_bundle_path_from_artifacts(artifacts)
            if bundle_path and Path(str(bundle_path)).exists():
                try:
                    bundle = load_json(Path(str(bundle_path)))
                except Exception:
                    bundle = {}
                for process_payload in ensure_list(bundle.get("projected_processes")):
                    if isinstance(process_payload, dict):
                        payload = copy.deepcopy(process_payload)
                        payload.setdefault("node_id", result.get("node_id"))
                        projected_processes.append(payload)
                for relation in ensure_list(bundle.get("relations")):
                    if isinstance(relation, dict):
                        relation_copy = copy.deepcopy(relation)
                        relation_copy.setdefault("node_id", result.get("node_id"))
                        relations.append(relation_copy)
        if result.get("kind") == "process_builder":
            process_build_runs.append(
                {
                    "node_id": result.get("node_id"),
                    "run_id": (artifacts or {}).get("run_id") or (result.get("planned_artifacts") or {}).get("run_id"),
                    "run_root": (artifacts or {}).get("run_root"),
                    "exports_dir": (artifacts or {}).get("exports_dir"),
                }
            )

    return {
        "generated_at": now_iso(),
        "run_dir": str(run_dir.resolve()),
        "request_id": plan.get("request_id"),
        "status": "prepared_local_publish_bundle",
        "include_lifecyclemodels": include_lifecyclemodels,
        "include_resulting_process_relations": include_resulting_process_relations,
        "graph_manifest": graph_manifest,
        "lineage_manifest": lineage_manifest,
        "lifecyclemodels": lifecyclemodels,
        "projected_processes": projected_processes,
        "resulting_process_relations": relations,
        "process_build_runs": process_build_runs,
    }


def load_request_and_plan(
    request_arg: str,
    *,
    out_dir: Path,
    allow_process_build_override: bool = False,
    allow_submodel_build_override: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_path = Path(request_arg).expanduser().resolve()
    request = load_json(request_path)
    validate_schema_file(request, REQUEST_SCHEMA_PATH, label="request")
    request_for_plan = copy.deepcopy(request)
    if allow_process_build_override:
        request_for_plan.setdefault("orchestration", {})["allow_process_build"] = True
    if allow_submodel_build_override:
        request_for_plan.setdefault("orchestration", {})["allow_submodel_build"] = True
    plan = build_plan(request_for_plan, request_path=request_path, out_dir=out_dir)
    normalized_request = {
        "request_id": plan["request_id"],
        "goal": copy.deepcopy(plan.get("goal") or {}),
        "root": copy.deepcopy(plan.get("root") or {}),
        "orchestration": copy.deepcopy(plan.get("orchestration") or {}),
        "candidate_sources": copy.deepcopy(plan.get("candidate_sources") or {}),
        "publish": copy.deepcopy(plan.get("publish") or {}),
        "nodes": copy.deepcopy(plan.get("nodes") or []),
        "edges": copy.deepcopy(plan.get("edges") or []),
        "notes": copy.deepcopy(plan.get("notes") or []),
    }
    return normalized_request, plan


def cmd_plan(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).expanduser().resolve()
    normalized_request, plan = load_request_and_plan(args.request, out_dir=out_dir)
    graph_manifest = build_graph_manifest(plan)
    lineage_manifest = build_lineage_manifest(plan)
    boundary_report = build_boundary_report(plan)
    write_plan_artifacts(
        normalized_request=normalized_request,
        plan=plan,
        graph_manifest=graph_manifest,
        lineage_manifest=lineage_manifest,
        boundary_report=boundary_report,
        out_dir=out_dir,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "command": "plan",
                "out_dir": str(out_dir),
                "request_id": plan["request_id"],
                "node_count": plan["summary"]["node_count"],
                "invocation_count": plan["summary"]["invocation_count"],
                "unresolved_count": plan["summary"]["unresolved_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_execute(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).expanduser().resolve()
    normalized_request, plan = load_request_and_plan(
        args.request,
        out_dir=out_dir,
        allow_process_build_override=bool(args.allow_process_build),
        allow_submodel_build_override=bool(args.allow_submodel_build),
    )
    initial_graph_manifest = build_graph_manifest(plan)
    initial_lineage_manifest = build_lineage_manifest(plan)
    initial_boundary_report = build_boundary_report(plan)
    write_plan_artifacts(
        normalized_request=normalized_request,
        plan=plan,
        graph_manifest=initial_graph_manifest,
        lineage_manifest=initial_lineage_manifest,
        boundary_report=initial_boundary_report,
        out_dir=out_dir,
    )
    plan, execution_results = execute_plan(plan, out_dir=out_dir)
    graph_manifest = build_graph_manifest(plan, execution_results)
    lineage_manifest = build_lineage_manifest(plan, execution_results)
    boundary_report = build_boundary_report(plan, execution_results)
    write_plan_artifacts(
        normalized_request=normalized_request,
        plan=plan,
        graph_manifest=graph_manifest,
        lineage_manifest=lineage_manifest,
        boundary_report=boundary_report,
        out_dir=out_dir,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "command": "execute",
                "out_dir": str(out_dir),
                "request_id": plan["request_id"],
                "status": plan["execution_summary"]["status"],
                "successful_invocations": plan["execution_summary"]["successful_invocations"],
                "failed_invocations": plan["execution_summary"]["failed_invocations"],
                "blocked_invocations": plan["execution_summary"]["blocked_invocations"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if plan["execution_summary"]["status"] == "completed" else 1


def cmd_publish(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    plan_path = run_dir / "assembly-plan.json"
    graph_path = run_dir / "graph-manifest.json"
    lineage_path = run_dir / "lineage-manifest.json"
    require_file(plan_path, "assembly plan")
    require_file(graph_path, "graph manifest")
    require_file(lineage_path, "lineage manifest")
    plan = load_json(plan_path)
    graph_manifest = load_json(graph_path)
    lineage_manifest = load_json(lineage_path)
    execution_results = load_invocation_results(run_dir)
    publish_bundle = collect_publish_bundle(
        run_dir=run_dir,
        plan=plan,
        graph_manifest=graph_manifest,
        lineage_manifest=lineage_manifest,
        execution_results=execution_results,
        include_lifecyclemodels=bool(args.publish_lifecyclemodels),
        include_resulting_process_relations=bool(args.publish_resulting_process_relations),
    )
    dump_json(run_dir / "publish-bundle.json", publish_bundle)
    dump_json(
        run_dir / "publish-summary.json",
        {
            "ok": True,
            "command": "publish",
            "run_dir": str(run_dir),
            "publish_lifecyclemodels": bool(args.publish_lifecyclemodels),
            "publish_resulting_process_relations": bool(args.publish_resulting_process_relations),
            "lifecyclemodel_count": len(publish_bundle["lifecyclemodels"]),
            "projected_process_count": len(publish_bundle["projected_processes"]),
            "relation_count": len(publish_bundle["resulting_process_relations"]),
            "status": "prepared_local_publish_bundle",
        },
    )
    print(json.dumps(publish_bundle, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lifecycle model recursive orchestrator")
    sub = p.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--request", required=True)
    plan.add_argument("--out-dir", required=True)
    plan.set_defaults(func=cmd_plan)

    execute = sub.add_parser("execute")
    execute.add_argument("--request", required=True)
    execute.add_argument("--out-dir", required=True)
    execute.add_argument("--allow-process-build", action="store_true")
    execute.add_argument("--allow-submodel-build", action="store_true")
    execute.set_defaults(func=cmd_execute)

    publish = sub.add_parser("publish")
    publish.add_argument("--run-dir", required=True)
    publish.add_argument("--publish-lifecyclemodels", action="store_true")
    publish.add_argument("--publish-resulting-process-relations", action="store_true")
    publish.set_defaults(func=cmd_publish)
    return p


if __name__ == "__main__":
    parsed = parser().parse_args()
    raise SystemExit(parsed.func(parsed))
