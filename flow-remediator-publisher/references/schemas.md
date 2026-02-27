# Output Schemas (Initial Version)

These schemas are intentionally lightweight so they can evolve while the workflow is tested.

## `review/findings.jsonl`

One JSON object per finding.

Common fields:

- `flow_uuid`
- `base_version`
- `severity`: `error|warning|info`
- `rule_id`
- `message`
- `fixability`: `auto|manual|review-needed`
- `evidence` (optional object)
- `suggested_action` (optional)

Example:

```json
{
  "flow_uuid": "bdbb913b-620c-42a0-baf6-c5802a2b6c4b",
  "base_version": "01.01.000",
  "severity": "warning",
  "rule_id": "quantitative_reference_mismatch",
  "message": "Quantitative reference internal ID does not match the chosen reference flowProperty.",
  "fixability": "auto",
  "evidence": {
    "quant_ref_internal_id": "1",
    "expected_internal_id": "0"
  },
  "suggested_action": "Align quantitative reference with chosen flowProperty internal ID."
}
```

## `fix/fix_proposals.jsonl`

Two modes in initial version:

- `mode=applied`: deterministic patch ops already applied to `patched_flows`
- `mode=candidate`: non-deterministic/high-risk findings requiring manual review or regeneration

Example `applied`:

```json
{
  "flow_uuid": "bdbb913b-620c-42a0-baf6-c5802a2b6c4b",
  "base_version": "01.01.000",
  "mode": "applied",
  "rule_id": "safe_fix_batch",
  "patch_ops": [
    {
      "op": "set",
      "path": "/flowDataSet/flowInformation/quantitativeReference/referenceToReferenceFlowProperty",
      "value": "0",
      "rule_id": "quantitative_reference_alignment"
    }
  ]
}
```

Example `candidate`:

```json
{
  "flow_uuid": "bdbb913b-620c-42a0-baf6-c5802a2b6c4b",
  "base_version": "01.01.000",
  "mode": "candidate",
  "rule_id": "same_category_high_similarity",
  "severity": "warning",
  "message": "Another flow in the same classification/flowProperty/unitgroup group is highly similar.",
  "next_step": "manual-review-or-regenerate"
}
```

## `fix/patch_manifest.jsonl`

One row per patched/copied flow file, used by `publish`.

Fields:

- `flow_uuid`
- `base_version`
- `patched_version_before_publish`
- `source_file`
- `patched_file`
- `changed`
- `before_sha256`
- `after_sha256`

## `publish/publish_results.jsonl`

One row per publish attempt.

Status values:

- `dry-run`
- `inserted`
- `conflict`
- `error`

Important fields:

- `flow_uuid`
- `base_version`
- `latest_version_checked`
- `new_version`
- `mode`
- `status`
- `reason` (when conflict/error)
- `insert_result` (when `insert`)

## Future Alignment Target

When `lci-review --profile flow` is implemented, keep this skill compatible by:

- mapping external review findings into the same `findings.jsonl` shape
- preserving `fix_proposals.jsonl` and `patch_manifest.jsonl` contracts
- keeping publish append-only and versioned

