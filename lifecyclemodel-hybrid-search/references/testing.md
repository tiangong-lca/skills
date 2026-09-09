# Testing

Run from the independently installed skill directory, without a CLI or Data Foundry checkout and without public environment variables.

## Dry run (no login or network request)

```bash
node scripts/run-lifecyclemodel-hybrid-search.mjs --published-cli --dry-run --json
```

Confirm the CLI's Production endpoint and region in the planned request. This must not create a session. A custom `--base-url` without its matching complete configuration must fail before network access.

## Human login and read-only smoke

```bash
pnpm dlx --package=@tiangong-lca/cli@0.1.13 tiangong-lca auth status --json
```

A fresh install reports `login-required` (exit 1). The human user then runs the pinned CLI's `auth login` in a trusted terminal and completes browser consent. Never ask an agent to collect a password, code, or token. After login:

```bash
pnpm dlx --package=@tiangong-lca/cli@0.1.13 tiangong-lca auth doctor-auth --json
node scripts/run-lifecyclemodel-hybrid-search.mjs --published-cli --json
```

## Direct CLI equivalent

```bash
pnpm dlx --package=@tiangong-lca/cli@0.1.13 tiangong-lca search lifecyclemodel --input ./assets/example-request.json --dry-run --json
```

Use `--cli-dir /path/to/tiangong-lca-cli` only for an explicitly selected matching local build. Custom environment setup is documented in `env.md`; it is not a Production prerequisite.

## Checklist

- A 200 response contains `data` (an array, possibly empty).
- A 400 response indicates a missing or invalid query.
- A 500 response indicates a provider/RPC failure; inspect service logs rather than changing the auth path.
- No credential or session content appears in output or test artifacts.
