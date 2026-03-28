# Next / MCP Alignment Notes

## What current code already proves

### In `tiangong-lca-next`
- lifecycle model create/update already computes projected resulting processes
- projected processes are written with `model_id`
- `json_tg.submodels` already stores a projection summary list
- update path already distinguishes primary/secondary projected processes

### In `tiangong-lca-mcp`
- lifecycle model intake already derives `json_tg`
- graph presentation is derived from model `json_ordered` + referenced processes
- CRUD tool already special-cases lifecycle models

## What is still missing

A unified, explicit cross-project relation contract for:
- projected resulting process identity
- projection role
- version pinning
- projection signature / update matching
- graph asset references

## Recommended alignment

### Keep in Next for now
- lifecycle model editing UX
- graph preview UX
- submodel list UX
- `model_id` compatibility behavior

### Keep in MCP for now
- lifecycle model payload validation
- `json_tg` derivation
- controlled CRUD write path

### Move into skill contracts now
- recursive planning semantics
- resulting-process builder invocation contract
- explicit relation payload bundle
- lineage manifest schema

## Migration principle

Do not try to rewrite Next's current lifecycle-model calculation path immediately.

Instead:
1. document the projection contract in skills
2. make skill artifacts mirror the semantics already present in Next
3. later, if desired, extract projection logic from Next into a reusable library/service boundary

## Short-term safest move

Short-term, the safest architecture is:
- **skills define and test the contracts**
- **Next remains the richest reference implementation of projection semantics**
- **MCP remains the safest validated ingestion/writing layer**

That avoids premature rewrites while still converging the architecture.
