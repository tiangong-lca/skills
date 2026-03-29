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

### tiangong-lca-cli
`src/lib/lifecyclemodel-resulting-process.ts` now owns the local projection path, and `src/lib/lifecyclemodel-publish-resulting-process.ts` owns the local publish-handoff artifact generation.

## Recommended responsibility split

### lca-skills
- orchestration
- dry-run planning
- projection packaging contracts
- thin wrappers only
- no direct DB mutation by default

### lifecyclemodel-resulting-process-builder
- wrapper-level compatibility for callers like `lifecyclemodel-recursive-orchestrator`
- no business logic beyond invoking the CLI

### tiangong-lca-cli
- projection computation and packaging
- resulting-process metadata stamping
- relation payload generation
- publish handoff artifact generation

### tiangong-lca-next
- editing / preview / review UI
- graph and submodel presentation
- model/process relation display

## Key conclusion

A resulting process is best treated as a **computed projection artifact** of a lifecycle model. It is substantial enough to justify a dedicated skill, but distinct enough from process synthesis that it should not be merged into `process-automated-builder`.
