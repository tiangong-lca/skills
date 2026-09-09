# Semantic work

Apply the current task's schema and template. The field names below explain their purpose; the supplied contract remains authoritative for the exact shape.

## Evidence and closure

For completed non-test work, provide a concrete basis and structured evidence identifying the source/context plus a quote, trace, file/field path or citation. List the context kinds actually reviewed. Full-context tasks require the supplied schema, methodology YAML, ruleset, classification/location schema, source row, entity payload and applicable profile/dependency context.

Preserve `authoring_package` and its hash, task-specific `authoring_context.context_bundle_sha256`, and each required `closes_action_items` association. A shared bundle saves repeated reading; it does not replace entity evidence or the task's own context digest. Do not mark an unreviewed context kind as used.

Use `decision_status: completed` or `patch_status: completed` only when the work is complete. Keep an unresolved item explicit instead of supplying a plausible placeholder.

## Identity decisions

Read completed current-row and relevant dependency preflight evidence. Compare names, dataset type, unit/property, geography, classifications, exchange context and source evidence; lexical similarity alone does not prove identity.

- `reuse_existing_reference` identifies the exact canonical table, ID and version supported by the selected candidate evidence.
- `create_new` requires evidence that candidates are not identity-equivalent and that the current workflow permits this decision. A historical profile never permits elementary-flow creation or account-local support writes. Prefer canonical reuse; retain a blocker when current permission or evidence is missing.
- `block_unresolved` records what was searched, the remaining conflict or absence of evidence, and the next evidence needed.

Preserve dataset type, ID/version, package/context hashes, evidence and action-item closure from the template. The runtime owns partitioning writes/reference-only/unresolved rows and rewriting dependent references.

## Classification and location decisions

Use valid codes from the supplied category/location schema, selecting the justified leaf where required. Preserve `category_type`, completion status, dataset identity and the decision task's context hash. Location decisions also retain the exact `target_path`.

Use all relevant source geography: operation/supply location, exchange locations, referenced flow, name mix/location information and provenance. Resolve conflicting evidence explicitly. Formal location fields use codes; natural-language restrictions belong in their supported description fields. Do not encode classification/location decisions as generic patches when a dedicated task is supplied.

## Field patches

Fill the supplied patch template and only its supported operations. Paths address the current row shape: a canonical row may put the domain payload under `json`, so preserve a supplied `/json/...` pointer instead of assuming the payload is the root object.

Each operation needs its basis, structured evidence, `resolution.mode`, reviewed context kinds and the exact action items it closes. Supporting cleanup should close the same item it is needed to resolve. Do not hand-edit rows or create deterministic apply/validation reports.

Populate formal fields when the source proves a value; do not hide provable information in a general comment. Separate source name fragments into the appropriate name fields and use evidence-backed descriptions rather than generated placeholders. True source rows describe traceable reports, publications or datasets; format/compliance metadata must not become a fabricated publication identity.

When evidence is insufficient, use only the work item's allowed resolution modes. `deferred_to_common_other` requires a structured unresolved trace with the blocked field, reason, evidence and next action; it cannot replace a mandatory schema value. Source-faithful exchange-completeness acceptance requires the explicit source evidence and permitted `source_trace_verified` mode. Missing annual supply is not a fabricated volume or arbitrary deferral: Foundry owns the deterministic `9999 missing-data-sentinel/year` policy where applicable.

After files are returned, Foundry owns validation, deterministic application, current-row preflight refresh, dependency evidence preservation, finalize and any later authorized write/readback. Do not shortcut those stages or infer completion from a filename.
