# Integration Notes Across lca-skills / tiangong-lca-mcp / tiangong-lca-next

## Key observation from MCP

`tiangong-lca-mcp/src/tools/life_cycle_model_file_tools.ts` already establishes an important contract:

- MCP accepts lifecycle model payloads centered on `jsonOrdered`
- MCP can derive `jsonTg.xflow` graph data for rendering
- Graph generation is driven from lifecycle model `processInstance[*].referenceToProcess` plus fetched process `json_ordered`
- MCP reads `lifeCycleModelInformation.dataSetInformation.referenceToResultingProcess`

Therefore the orchestrator should treat **graph JSON as a derived view**, not as the system of record.

## Recommended payload handoff to MCP

For model-oriented publish/preview workflows, hand off:

1. `lifecyclemodel.json_ordered`
2. optional `lifecyclemodel.json_tg` only when already available and intentionally preferred
3. the corresponding resulting process payload / relation metadata

The graph should be regenerated or merged by MCP rather than being authored independently inside the orchestrator.

## Relation semantics already visible in next

In `tiangong-lca-next/src/services/processes/api.ts`, process rows already expose `model_id` and UI flows already distinguish lifecycle-backed processes (`isFromLifeCycle`).

That means the platform already has an embryonic relation model:

- process can point back to model
- UI can detect lifecycle-derived processes

The new orchestrator should standardize and enrich this rather than invent a conflicting path.

## Resulting-process generation: dedicated builder skill

### Updated recommendation
Create a dedicated skill:
- `lifecyclemodel-resulting-process-builder`

### Why the recommendation changed
After checking `tiangong-lca-next/src/services/lifeCycleModels/util_calculate.ts`, the platform already treats resulting-process generation as a substantial workflow with:

- graph edge construction
- dependence assignment
- scaling propagation
- allocation
- final-product grouping
- primary/secondary projected process generation
- process-payload packaging

So a resulting process is not merely a thin handle and should not be routed through `process-automated-builder` by default.

### Responsibility split
- `process-automated-builder`
  - build a standalone process from flow/reference-flow evidence
- `lifecyclemodel-automated-builder`
  - assemble lifecycle model `json_ordered`
  - ensure the model carries `referenceToResultingProcess`
- `lifecyclemodel-resulting-process-builder`
  - compute and package projected resulting process datasets from lifecycle model topology and math
- `lifecyclemodel-recursive-orchestrator`
  - decide when to invoke each one and keep lineage consistent

## Current recommended orchestration chain

1. discover reusable models / resulting processes / processes
2. if needed, call `process-automated-builder` to fill missing process nodes
3. call `lifecyclemodel-automated-builder` to assemble submodel or root model `json_ordered`
4. call `lifecyclemodel-resulting-process-builder` to compute/package resulting process payloads
5. hand model payload plus projected process payloads/relation metadata to MCP / downstream publish path
6. let MCP derive graph presentation content from the canonical model/process JSON
