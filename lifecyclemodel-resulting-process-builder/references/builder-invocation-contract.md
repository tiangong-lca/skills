# Builder Invocation Contract

## Caller

Primary caller:
- `lifecyclemodel-recursive-orchestrator`

Secondary caller candidates:
- direct operator dry-run
- direct wrapper invocation from this skill

## Invocation purpose

Build a lifecycle model into one or more resulting process datasets whose exchange values are computed from model topology and allocation logic.

## Active execution path

- `scripts/run-lifecyclemodel-resulting-process-builder.sh build` delegates to `tiangong lifecyclemodel build-resulting-process`
- `scripts/run-lifecyclemodel-resulting-process-builder.sh publish` delegates to `tiangong lifecyclemodel publish-resulting-process`
- the wrapper still accepts `--request` and `--model-file` for compatibility, but the canonical build contract underneath is a CLI request file passed through `--input`

## Input contract

### Required
- `source_model.id`
- `source_model.version`
- `source_model.json_ordered` or `source_model.json_ordered_path`

### Optional
- `source_model.json_tg`
- `previous_projection_snapshot`
- `projection.mode` = `primary-only` | `all-subproducts`
- `projection.metadata_overrides`
- `projection.graph_snapshot_uri`
- `publish.intent`

## Example invocation payload

```json
{
  "source_model": {
    "id": "lm_xxx",
    "version": "00.00.001",
    "json_ordered_path": "/abs/path/lifecyclemodel.json"
  },
  "projection": {
    "mode": "all-subproducts",
    "metadata_overrides": {
      "type_of_data_set": "partly terminated system"
    },
    "graph_snapshot_uri": "file:///abs/path/model-preview.png"
  },
  "publish": {
    "intent": "prepare_only"
  }
}
```

## Output contract

### Required outputs
- `process-projection-bundle.json`
- `projection-report.json`
- `publish-bundle.json` and `publish-intent.json` after the publish-handoff step

### Bundle contents
- `source_model`
- `projected_processes[]`
- `relations[]`
- `report`

### Required relation fields per projected process
- `lifecyclemodel_id`
- `lifecyclemodel_version`
- `resulting_process_id`
- `resulting_process_version`
- `projection_role`
- `projection_signature`

## Decision rule: create vs update

Builder should not decide persistence on its own.

It may suggest:
- `create`
- `update`
- `reuse_existing_projection`

But final write behavior belongs to orchestrator / approved publish layer.

## Key separation rule

Builder may compute process payloads, but it should not replace:
- `process-automated-builder` for flow-to-process synthesis
- `lifecyclemodel-automated-builder` for model assembly

It only owns deterministic model-to-resulting-process construction.
