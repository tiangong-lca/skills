# Workflow

## Objective

Provide a planner/executor layer that can recursively assemble LCA systems by combining existing processes, CLI-prepared process builds, existing lifecycle models, and newly assembled submodels.

Public entrypoint:

- `node scripts/run-lifecyclemodel-recursive-orchestrator.mjs <plan|execute|publish> ...`
- canonical command: `tiangong lifecyclemodel orchestrate <plan|execute|publish> ...`

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
Normalize the request-time candidate sets for:

- existing processes
- existing lifecycle models
- existing resulting processes
- previously generated lineage manifests if available

This skill does not perform its own remote search. Candidate discovery must happen before the orchestrator runs or be materialized into the request.

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
- execute the native process-builder slice used by `tiangong process auto-build`
- execute the native lifecyclemodel-builder slice used by `tiangong lifecyclemodel auto-build`
- execute the native projector slice used by `tiangong lifecyclemodel build-resulting-process` when projection is requested

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

This step prepares local handoff artifacts only. Commit execution belongs to downstream CLI publish flow.

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

## Request-surface guardrails

- `process_builder` only supports `flow_file`, `flow_json`, and `run_id`
- removed legacy fields such as `process_builder.mode=langgraph` and `process_builder.python_bin` are outside the supported path
- if a new builder control is truly needed, add it first as a native CLI capability instead of smuggling it back through skill-local runtime
