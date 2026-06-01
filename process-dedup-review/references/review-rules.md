# Process Dedup Review Rules

## Evidence Hierarchy

Use evidence in this order:

1. Exact normalized exchange signature.
2. Remote process metadata (`state_code`, timestamps, exchange flow short descriptions).
3. Source-language process names.
4. Cross-object reference checks within the authenticated user's accessible scope, when available.
5. Wider visible/shared reference checks, when available.

If a higher-priority layer is missing, say so explicitly in the report instead of silently promoting a weaker heuristic.

## Exact Duplicate Rule

A duplicate group is an exact duplicate when every candidate has the same normalized multiset of:

- flow id
- direction
- mean amount
- resulting amount

Ignore:

- exchange row order
- exchange internal IDs
- source row order

## Keep/Delete Tie-Break Rules

Apply these in order:

1. Prefer the row whose name matches the exchange semantics.
2. Prefer the row with the more specific and better standardized source-language name.
3. If both still tie, prefer the row that is already referenced elsewhere.
4. If only session-scoped reference evidence is available, use it but state that wider visible/shared references were not checked.
5. If reference evidence is unavailable and the semantic/name score still ties, prefer the older row and record the unresolved tie.

## Name-Semantics Heuristics

These heuristics are only valid after exact duplicate structure is confirmed.

- If input and output keep the same product flow and there is no transformed output flow, penalize names such as `processing`, `production`, or `manufacturing`.
- If the auxiliary input is an explicit transport service, prefer `transport` over broader terms such as `logistics`.
- Prefer explicit and standardized wording such as `waste paper` over compressed variants such as `wastepaper`.
- Prefer broader-but-still-correct names only when the narrower alternative is semantically wrong. Do not keep a vague name just because it is shorter.

## Reporting Contract

For every recommended delete candidate, state:

- why the group is an exact duplicate
- why the kept row won the tie-break
- whether downstream-reference verification was completed
- whether the recommendation is `safe to delete now` or only `priority delete candidate`
