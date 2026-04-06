# Approved Decision Schema

## Canonical File Shape

`materialize-approved-decisions` accepts:

- JSON array of objects
- JSONL with one object per line

Recommended top-level fields:

- `cluster_id`: required stable cluster identifier
- `decision`: required, one of:
  - `merge_keep_one`
  - `keep_distinct`
  - `blocked_missing_db_flow`
- `canonical_flow`: required for `merge_keep_one`
- `flow_refs`: required for all three decision types
- `reason`: optional reviewer rationale or label

## Flow Ref Shape

Each flow ref may be expressed as either:

```json
{
  "id": "7a285e9a-a9f6-4b86-ab17-6ea17367400c",
  "version": "01.01.001"
}
```

or:

```json
"7a285e9a-a9f6-4b86-ab17-6ea17367400c@01.01.001"
```

## Decision Examples

### `merge_keep_one`

```json
{
  "cluster_id": "cluster-0001",
  "decision": "merge_keep_one",
  "canonical_flow": {
    "id": "7a285e9a-a9f6-4b86-ab17-6ea17367400c",
    "version": "01.01.001"
  },
  "flow_refs": [
    {
      "id": "7a285e9a-a9f6-4b86-ab17-6ea17367400c",
      "version": "01.01.001"
    },
    {
      "id": "017acdd0-7fd7-44cb-a410-1d559e59c506",
      "version": "01.01.001"
    }
  ],
  "reason": "same_property_semantic_review"
}
```

### `keep_distinct`

```json
{
  "cluster_id": "cluster-0015",
  "decision": "keep_distinct",
  "flow_refs": [
    "7a285e9a-a9f6-4b86-ab17-6ea17367400c@01.01.001",
    "017acdd0-7fd7-44cb-a410-1d559e59c506@01.01.001"
  ],
  "reason": "purity_conflict"
}
```

### `blocked_missing_db_flow`

```json
{
  "cluster_id": "cluster-0043",
  "decision": "blocked_missing_db_flow",
  "flow_refs": [
    "1c833e18-7cd2-4521-b649-62e5d6aa6935@01.01.001",
    "309c856b-67c2-48b2-a52f-e1974625bd3a@01.01.001"
  ],
  "reason": "db_rows_not_materialized"
}
```

## Wrapper Command

```bash
node scripts/run-flow-governance-review.mjs materialize-approved-decisions \
  --decision-file /abs/path/approved-decisions.json \
  --flow-rows-file /abs/path/materialized/review-input-rows.jsonl \
  --out-dir /abs/path/decision-artifacts
```

## Output Contract

The wrapper delegates to `tiangong flow materialize-decisions` and writes:

- `flow-dedup-canonical-map.json`
- `flow-dedup-rewrite-plan.json`
- `manual-semantic-merge-seed.current.json`
- `decision-summary.json`
- `blocked-clusters.json`

Interpretation:

- `flow-dedup-canonical-map.json` maps each successfully materialized cluster member to its canonical flow
- `flow-dedup-rewrite-plan.json` lists the concrete source-flow to canonical-flow rewrites
- `manual-semantic-merge-seed.current.json` is a versioned seed alias map for downstream alias materialization
- `blocked-clusters.json` captures `keep_distinct`, `blocked_missing_db_flow`, and merge rows that could not be materialized because the referenced flow rows were absent
