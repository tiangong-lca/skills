# Minimal Relation Fields Across lca-skills / tiangong-lca-next / downstream publish layers

## Why this file exists

Current code already exposes a partial relation model:

- `processes.model_id`
- `lifecyclemodels.json_tg.submodels[*]`
- `lifeCycleModelInformation.dataSetInformation.referenceToResultingProcess`

These are useful, but they are not yet a complete or explicit contract for recursive orchestration, projected resulting processes, and stable lineage.

This file defines the **minimum recommended relation fields** before deeper database/schema work.

## Design goals

1. Keep compatibility with current code paths.
2. Avoid immediate database migration requirements where possible.
3. Make recursion, projection, and version pinning explicit.
4. Separate canonical relation fields from derived presentation fields.

---

## Canonical relation fields

### A. On lifecycle model payload / record

These fields are canonical and should remain authoritative:

- `lifecyclemodel.id`
- `lifecyclemodel.version`
- `lifeCycleModelInformation.dataSetInformation.referenceToResultingProcess`

Recommended semantic strengthening:

- `referenceToResultingProcess` should represent the **primary projected resulting process** of that lifecycle model version.
- If the model also emits secondary/subproduct projected processes, they should not overload `referenceToResultingProcess`; they belong in a projection relation list.

### B. On process payload / record

Current field:

- `processes.model_id`

Recommended semantics:

- keep `model_id` as the **compatibility field** for "this process came from a lifecycle model"
- do not rely on it alone as the complete relation contract

Recommended additional logical fields (even if first stored in manifests / sidecar payloads before DB changes):

- `generated_from_lifecyclemodel_id`
- `generated_from_lifecyclemodel_version`
- `projection_role` = `primary` | `secondary`
- `projection_signature`
- `projection_source` = `lifecyclemodel_projection`

### C. On model projection relation objects

Recommended normalized relation object:

```json
{
  "lifecyclemodel_id": "lm_xxx",
  "lifecyclemodel_version": "00.00.001",
  "resulting_process_id": "proc_xxx",
  "resulting_process_version": "00.00.001",
  "projection_role": "primary",
  "projection_signature": "sha256:...",
  "is_primary": true
}
```

This relation object is the cleanest cross-project interchange format.

---

## Derived / presentation fields

These are useful but should not be treated as the canonical relation source:

- `json_tg.xflow`
- `json_tg.submodels`
- graph screenshots
- preview image URIs
- node labels / rendered graph layout

These can reference canonical relations, but should not replace them.

---

## Recommended minimum field set by project

### 1. lca-skills

Should emit these in manifests / payload bundles:

- `generated_from_lifecyclemodel_id`
- `generated_from_lifecyclemodel_version`
- `projection_role`
- `projection_signature`
- `reference_to_resulting_process` (primary only)

### 2. tiangong-lca-next

Should continue supporting:

- `processes.model_id`
- `json_tg.submodels`

But should conceptually map them toward:

- `model_id` -> compatibility alias for `generated_from_lifecyclemodel_id`
- `json_tg.submodels` -> derived projection summary list

### 3. downstream publish layer

Should accept payload bundles containing:

- lifecycle model `json_ordered`
- projected process `json_ordered`
- relation payload array

The publish layer may derive `json_tg` for lifecycle models, but it should not be the only keeper of model/process lineage.

---

## Compatibility plan

### Phase 0: now
No DB schema change required.

- Keep using `processes.model_id`
- Keep using `referenceToResultingProcess`
- Add richer relation objects to skill artifacts / manifests / local payload bundles

### Phase 1: soft standardization
When touching app/service types, add optional fields in TypeScript models and payload contracts:

- `generatedFromLifecycleModelId?`
- `generatedFromLifecycleModelVersion?`
- `projectionRole?`
- `projectionSignature?`

### Phase 2: explicit persistence
Only if needed later, add a dedicated relation table or explicit JSON metadata field for model/resulting-process projections.

---

## Recommended defaults

- `model_id` remains supported as a compatibility field.
- `referenceToResultingProcess` always points to the primary projected process.
- `projection_role=secondary` is stored only in relation metadata, not in `referenceToResultingProcess`.
- `projection_signature` should be stable for the same projection basis and useful for update-vs-create matching.

---

## Bottom line

If only one thing is standardized next, standardize this tuple first:

- `generated_from_lifecyclemodel_id`
- `generated_from_lifecyclemodel_version`
- `projection_role`
- `projection_signature`

That tuple is the minimum durable bridge between recursive orchestration, projected resulting processes, downstream publish intake, and Next UI behavior.
