# Resulting Process Build Contract

## Canonical sources

### Source of truth
- lifecycle model `json_ordered`
- referenced process `json_ordered`

### Derived views
- `json_tg.xflow`
- screenshots / diagram assets
- preview summaries

## Output bundle shape

```json
{
  "source_model": {
    "id": "lm_xxx",
    "version": "00.00.001"
  },
  "projected_processes": [
    {
      "role": "primary",
      "id": "proc_xxx",
      "version": "00.00.001",
      "json_ordered": {},
      "metadata": {
        "generated_from_lifecyclemodel_id": "lm_xxx",
        "generated_from_lifecyclemodel_version": "00.00.001",
        "projection_signature": "...",
        "type_of_data_set": "partly terminated system"
      }
    }
  ],
  "relations": [
    {
      "lifecyclemodel_id": "lm_xxx",
      "lifecyclemodel_version": "00.00.001",
      "resulting_process_id": "proc_xxx",
      "resulting_process_version": "00.00.001",
      "projection_role": "primary"
    }
  ],
  "report": {
    "node_count": 0,
    "edge_count": 0,
    "scaling_summary": {},
    "allocation_summary": {}
  }
}
```

## Publish handoff artifacts

After the build step, `tiangong lifecyclemodel publish-resulting-process` derives:

- `publish-bundle.json`
- `publish-intent.json`

Those artifacts stay local until a later approved `tiangong publish run`.

## Required metadata semantics

Projected process payloads should preserve or derive:
- process UUID/version
- process name derived from model name and subproduct role
- quantitative reference exchange
- `typeOfDataSet`
- source-model linkage
- optional preview/screenshot references

## Rule of separation

- `process-automated-builder` creates a process from external flow evidence.
- `lifecyclemodel-resulting-process-builder` creates a resulting process from lifecycle model computation.

These are not interchangeable pipelines.
