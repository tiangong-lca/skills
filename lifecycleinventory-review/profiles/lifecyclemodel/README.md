# lifecyclemodel profile (planned)

Status: **not implemented yet**.

Current wrapper behavior:

1. `scripts/run-lifecycleinventory-review.mjs --profile lifecyclemodel ...`
2. delegates to `tiangong review lifecyclemodel ...`
3. the CLI currently returns the planned contract help / not implemented status

Next step suggestion:
1. Define model-level review scope and required inputs.
2. Implement `tiangong review lifecyclemodel` in `tiangong-lca-cli`.
3. Keep this skill as a thin wrapper only; do not add a new Python review runtime here.
