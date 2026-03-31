# Invocation Contracts

## 1. Orchestrator -> process auto-build slice

Use only when a required node lacks a suitable process and must be synthesized from a reference flow or equivalent external evidence.

### Input contract
- target node context
- candidate root/reference flow payload
- reuse decision log
- requested run mode (`dry-run` | `execute`)

### Expected output contract
- prepared local run artifact path
- produced process dataset payload(s) when present under the run
- run artifact path
- unresolved issues / placeholders
- publish eligibility flag

## 2. Orchestrator -> lifecyclemodel auto-build slice

Use when a coherent set of process nodes should be assembled into a lifecycle model.

### Input contract
- selected process set
- root reference process selection
- connection plan
- model metadata shell
- run mode (`dry-run` | `execute`)

### Expected output contract
- lifecycle model `json_ordered`
- local assembly report
- referenceToResultingProcess contract presence
- local artifact path
- optional downstream `validate-build` / `publish-build` follow-up note when the orchestration plan stops at local model assembly

## 3. Orchestrator -> lifecyclemodel build-resulting-process slice

Use when a lifecycle model must emit one or more resulting process datasets whose exchanges come from model topology and allocation math.

### Input contract
- source lifecycle model `json_ordered`
- optional existing `json_tg`
- previous submodel/process snapshot for update matching
- projection mode (`primary-only` | `all-subproducts`)
- process metadata overrides
- publish intent
- optional graph/screenshot asset references

### Expected output contract
- projected process payload bundle
- relation payloads with at least:
  - `generated_from_lifecyclemodel_id`
  - `generated_from_lifecyclemodel_version`
  - `projection_role`
  - `projection_signature`
- projection report
- optional graph/screenshot asset references

## 4. Projector -> publish layer

For approved writes only.

### Input contract
- process payload bundle
- lifecycle model / resulting process relation bundle
- optional screenshot or attachment references

### Expected output contract
- inserted/updated process ids + versions
- inserted/updated relation metadata
- validation / rule verification outcome

## Key rule

Do not route ordinary model-derived resulting-process projection through `process-automated-builder`. That path is reserved for process synthesis, not model projection.

Also:

- do not reintroduce Python wrapper-specific config here
- the supported `process_builder` request surface is limited to `flow_file`, `flow_json`, and `run_id`
