# Integration Notes

## Evidence from current codebase

### tiangong-lca-next
`src/services/lifeCycleModels/util_calculate.ts` already contains the core semantics for model-derived process projection:
- graph edge construction
- dependence assignment
- scaling factor propagation
- allocation
- final-product grouping
- generation of primary/secondary projected process payloads

This is strong evidence that resulting-process generation is its own domain workflow.

### tiangong-lca-next process service
`src/services/processes/api.ts` already persists `model_id` on process rows.

### tiangong-lca-mcp
`src/tools/life_cycle_model_file_tools.ts` already derives graph presentation (`json_tg`) from lifecycle model `jsonOrdered` plus referenced process rows.

## Recommended responsibility split

### lca-skills
- orchestration
- dry-run planning
- projection packaging contracts
- no direct DB mutation by default

### lifecyclemodel-resulting-process-builder
- projection computation and packaging
- resulting-process metadata stamping
- relation payload generation

### tiangong-lca-mcp
- validation and graph derivation for lifecycle model file intake
- CRUD bridge for approved insert/update operations

### tiangong-lca-next
- editing / preview / review UI
- graph and submodel presentation
- model/process relation display

## Key conclusion

A resulting process is best treated as a **computed projection artifact** of a lifecycle model. It is substantial enough to justify a dedicated skill, but distinct enough from process synthesis that it should not be merged into `process-automated-builder`.
