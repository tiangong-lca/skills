# Real-DB-First Runbook

## When This Runbook Applies

Use this path when the task explicitly says the conclusion must be based on real database flow rows, current DB JSON, or real `id/version` pairs.

Typical examples:

- cluster dedup review based on current DB rows
- re-review after `search flow` returned candidate refs
- approved decision materialization that must bind to exact DB rows before downstream rewrite or publish planning

## Non-Negotiable Rule

When the task requires real DB binding:

- do not use synthetic local rows to draw the conclusion
- do not invent UUIDs, versions, or row payloads
- do not treat `search flow` output as already-materialized review input

If the task needs a conclusion about real DB objects, first lock the exact flow refs, then materialize the corresponding DB rows.

## Canonical Sequence

1. Obtain candidate or exact refs.

   Source options:

   - `node ../flow-hybrid-search/scripts/run-flow-hybrid-search.mjs ...`
   - an existing refs file
   - a curated manual ref list

2. Prepare a refs file.

   Minimal shape:

   ```json
   {
     "id": "7a285e9a-a9f6-4b86-ab17-6ea17367400c",
     "version": "01.01.001",
     "state_code": 100,
     "cluster_id": "cluster-0001",
     "source": "search-flow"
   }
   ```

3. Materialize real DB rows through the wrapper:

   ```bash
   node scripts/run-flow-governance-review.mjs materialize-db-flows \
     --refs-file /abs/path/flow-refs.json \
     --out-dir /abs/path/materialized \
     --fail-on-missing
   ```

4. Review the materialized rows:

   ```bash
   node scripts/run-flow-governance-review.mjs review-flows \
     --rows-file /abs/path/materialized/review-input-rows.jsonl \
     --out-dir /abs/path/review
   ```

## Artifact Contract

`materialize-db-flows` writes:

- `resolved-flow-rows.jsonl`
- `review-input-rows.jsonl`
- `fetch-summary.json`
- `missing-flow-refs.jsonl`
- `ambiguous-flow-refs.jsonl`

Interpretation:

- `resolved-flow-rows.jsonl` keeps one row per successfully resolved input ref
- `review-input-rows.jsonl` collapses repeated refs by real `id@version` and records `_materialization.materialized_from_refs`
- `missing-flow-refs.jsonl` and `ambiguous-flow-refs.jsonl` are blocker artifacts, not advisory logs

## Blocked Cases

If `missing-flow-refs.jsonl` or `ambiguous-flow-refs.jsonl` is non-empty:

- do not replace the unresolved flows with synthetic rows
- do not issue a final cluster conclusion for the affected cluster
- mark the affected cluster as blocked on DB materialization

Partial continuation is allowed only for clusters whose members were fully materialized from the DB.
