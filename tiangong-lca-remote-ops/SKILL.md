---
name: tiangong-lca-remote-ops
description: Wrap TianGong CLI process maintenance commands for authenticated remote process refreshes and local post-write verification. Use when tasks need current-user process reference refreshes, resumable process maintenance artifacts, or strict local verification of fetched process rows.
---

# TianGong LCA Remote Ops

## Scope
- This skill is a thin wrapper around native `tiangong process ...` commands.
- Use it when the task is specifically about remote `processes` maintenance or post-write verification.
- Do not add business-specific runtime logic, Supabase auth code, or custom `.env` parsing into this skill. If capability is missing, add it to `tiangong-cli` first.

## Canonical Commands
- Refresh current-user process references:

```bash
node tiangong-lca-remote-ops/scripts/update-process-references.mjs \
  --out-dir /abs/path/process-refresh \
  --dry-run
```

- Commit the refresh after the local gate passes:

```bash
node tiangong-lca-remote-ops/scripts/update-process-references.mjs \
  --out-dir /abs/path/process-refresh \
  --apply
```

- Verify frozen rows after any remote write:

```bash
node tiangong-lca-remote-ops/scripts/verify-process-rows.mjs \
  --rows-file /abs/path/process-list-report.json \
  --out-dir /abs/path/post-write-verification
```

## Runtime Contract
- Remote refresh uses canonical CLI env only:
  - `TIANGONG_LCA_API_BASE_URL`
  - `TIANGONG_LCA_API_KEY`
  - `TIANGONG_LCA_SUPABASE_PUBLISHABLE_KEY`
- `TIANGONG_LCA_API_KEY` is sensitive and must come from env or the caller's secret store, never from skill files or committed scripts.
- `verify-process-rows` is local-only and does not require remote credentials by itself.
- Wrapper-local `--cli-dir` is the only supported override for choosing a local CLI checkout.

## Guardrails
- Never hardcode or print passwords, access tokens, decoded API-key payloads, or raw secret env values.
- Keep `--out-dir` explicit so manifests, progress logs, blockers, and verification artifacts are reproducible.
- Before any remote write, apply the state-driven routing rule in [references/process-write-routing.md](references/process-write-routing.md).
- Treat local `ProcessSchema` validation plus unresolved-reference checks as the hard gate, not HTTP success alone.
