# Process Write Routing

Use this routing rule before any remote `processes` write.

## 0. Complete-dataset gate comes before routing

Do not write a `process` row unless the outgoing payload is a complete dataset under the current schema contract.

Minimum gate:

- pass current `ProcessSchema.safeParse(...)` or an equivalent strict validator
- retain all schema-required nodes, even when some fields are semantically empty
- contain no unresolved placeholder / missing reference objects that the current task is expected to fix

If the official CLI / MCP / RPC accepts a payload that fails this gate:

- treat that as a CLI / command-endpoint gap
- block the write in the skill or case-local bridge
- record the exact blocker and the rejected row set in friction

Only after the complete-dataset gate passes should you choose the state-driven write path below.

## 1. Decide by `state_code`, then choose the tool

Do not start from a preferred endpoint such as RPC, MCP, or raw PostgREST update. Start from the dataset state and required lineage semantics.

## 2. `state_code=0`

Treat `state_code=0` as an unpublished current-user draft under active editing.

Required write semantics:

- keep the same `id`
- keep the same `version`
- update the current row in place
- save only after the complete-dataset gate passes

Reason:

- avoid creating many pre-publication versions for the same draft
- keep unpublished editing history simple and reduce review noise

Tooling implication:

- prefer a canonical `update` MCP / CLI adapter when available
- accept a save-draft path only if it preserves the same-row draft update semantics
- if the available write path creates a new version for every draft edit, treat that as a tooling mismatch and record CLI friction

## 3. `state_code=100`

Treat `state_code=100` as a public dataset revision task.

Required write semantics:

- keep the same `id`
- increment `version`
- publish the revised content as the next version in the same lineage
- save only after the complete-dataset gate passes

Reason:

- avoid producing multiple UUIDs that represent the same process with only editorial differences
- preserve a clean public version chain for downstream reuse and traceability

Tooling implication:

- prefer a version-bump adapter that writes the next version under the same UUID
- do not use clone/new-UUID behavior as the default update path
- only create a new UUID when the user explicitly wants a fork, a new lineage, or a materially different dataset identity

## 4. When the platform cannot satisfy the target semantics

If the available tool cannot satisfy the required lineage behavior:

- stop treating that tool as the default write path for this state
- record CLI or auth friction with the exact response and blocker
- keep the artifact-first plan and verification flow unchanged

## 5. Reporting

For every bulk write report, state explicitly:

- input `state_code`
- target lineage behavior (`in-place update` or `same UUID + version bump`)
- actual tool used
- whether the actual tool matched the intended lineage semantics

After every remote write, re-fetch the remote rows and run the canonical verifier. The skill wrapper delegates to the same CLI command:

```bash
tiangong process verify-rows \
  --rows-file /abs/path/process-list-report.json \
  --out-dir /abs/path/artifacts/<case_slug>/outputs/post-write-verification
```

The final gate is not “HTTP 200” or “row came back”. The final gate is:
- remote row re-fetched successfully
- `ProcessSchema.safeParse(...)` passes
- required process name fields still exist on the persisted payload
