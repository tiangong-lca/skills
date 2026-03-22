# Data Model and Relation Contract

## Core entities

### 1. Process
Represents an executable / referenceable process dataset.

Suggested metadata:
- `process_id`
- `version`
- `is_resulting_process` (boolean)
- `generated_from_lifecyclemodel_id` (nullable)
- `generated_from_lifecyclemodel_version` (nullable)
- `generation_mode` = `manual` | `process_builder` | `lifecyclemodel_projection`

### 2. Lifecycle model
Represents a composed model that may project to a resulting process.

Suggested metadata:
- `lifecyclemodel_id`
- `version`
- `resulting_process_id`
- `resulting_process_version`
- `projection_status`
- `graph_manifest_uri` (optional)
- `lineage_manifest_uri` (optional)

### 3. Model node
Represents a node inside a lifecycle model graph.

Suggested fields:
- `node_id`
- `reference_type` = `process` | `lifecyclemodel` | `resulting_process`
- `reference_id`
- `reference_version`
- `resolved_process_id` (nullable)
- `resolved_process_version` (nullable)
- `resolution_mode` = `reuse_process` | `reuse_model` | `build_process` | `build_submodel` | `unresolved`
- `boundary_reason` (nullable)

## Required relations

### lifecycle model -> resulting process
This must be explicit and queryable.

Minimum contract:
- `lifecyclemodel.resulting_process_id`
- `lifecyclemodel.resulting_process_version`

### resulting process -> source lifecycle model
Reverse lookup must also be possible.

Minimum contract:
- `process.generated_from_lifecyclemodel_id`
- `process.generated_from_lifecyclemodel_version`

### parent model -> child model
When a parent reuses a child model, it may do so through the child's resulting process at runtime, but lineage should still retain the child lifecycle model identity.

## Manifest outputs

### Graph manifest
Use for front-end graph rendering and debugging.

Recommended sections:
- `root`
- `nodes`
- `edges`
- `boundaries`
- `unresolved`
- `stats`

### Lineage manifest
Use for provenance and rebuild reproducibility.

Recommended sections:
- `root_request`
- `builder_invocations`
- `node_resolution_log`
- `published_dependencies`
- `resulting_process_relations`
- `unresolved_history`

## Version policy

- Parent references should pin child `model/version` or `resulting_process/version`.
- `latest`-style implicit dependency should be forbidden by default.
- A changed lifecycle model should project a new resulting process version when semantics change.

## Flattening note

Storage may remain hierarchical. Calculation/export workflows may later choose to recursively flatten the graph to elementary flows. That flattening step should not erase the original hierarchy or lineage metadata.
