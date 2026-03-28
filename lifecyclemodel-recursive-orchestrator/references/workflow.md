# Workflow

## Objective

Provide a planner/executor layer that can recursively assemble LCA systems by combining existing processes, generated processes, existing lifecycle models, and newly assembled submodels.

## Stages

### 1. Intake
Normalize the request into:
- root goal / product system target
- functional unit
- preferred recursion mode: `collapsed` | `expanded` | `hybrid`
- max recursion depth
- reuse policy
- publish intent

### 2. Discovery
Query for:
- existing account processes
- eligible public processes
- existing lifecycle models
- resulting processes projected from lifecycle models
- previously generated lineage manifests if available

### 3. Candidate scoring
For each required node, score candidates by:
- semantic match
- flow compatibility
- quantitative reference compatibility
- geography / technology fit
- version status / publication state
- confidence in reuse

### 4. Planning
For every node, choose one action:
- reuse existing resulting process
- reuse existing process
- build process
- build submodel
- unresolved / cutoff

The planner should emit a dry-run plan first.

### 5. Materialization
If allowed:
- call `process-automated-builder` for missing process nodes
- call `lifecyclemodel-automated-builder` for submodel assembly

### 6. Reconciliation
Re-read newly materialized outputs and resolve:
- resulting-process links
- parent/child model references
- unresolved node deltas

### 7. Validation
Validate:
- no illegal cycles
- version pin integrity
- resulting-process reverse linkage
- node reference integrity
- policy-compliant stopping boundaries

### 8. Publish handoff
Only when explicitly requested:
- publish lifecycle model payloads
- persist resulting-process relation metadata
- save manifests alongside build artifacts

## Policy knobs

- `mode`: `collapsed` | `expanded` | `hybrid`
- `max_depth`: recursion ceiling
- `reuse_resulting_process_first`: boolean
- `allow_process_build`: boolean
- `allow_submodel_build`: boolean
- `pin_child_versions`: default true
- `stop_at_elementary_flow`: boolean for analysis workflows
- `cutoff_policy`: named rule set

## Why dry-run first

Recursive assembly can explode combinatorially and hide dependency mistakes. Dry-run planning keeps the orchestration layer inspectable before downstream builders spend time or publish anything.
