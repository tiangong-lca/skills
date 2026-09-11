# Runtime and command contract

Use one explicitly selected TianGong CLI executable/argv prefix. Record its resolved location, `--version` output, capability response and capture/analysis paths in the run's local `runtime.json`. Do not put authentication material in that record.

The required protocol is `tiangong-lca.overview-capabilities.v1`, with `capture`, `catalog`, `analyze`, public all-owner visibility and read-only mode. Inspect it with:

```text
tiangong-lca --version
tiangong-lca dataset overview describe --json
tiangong-lca auth status --json
```

For a normally installed executable, use its resolved absolute path for subsequent invocations. If the capability is absent, explain the missing CLI dependency. Do not silently switch runtimes, install a floating version or recreate the missing algorithms in the skill.

## Feature rollout

The initial implementation depends on [CLI #284](https://github.com/tiangong-lca/tiangong-cli/issues/284). The Skills shared launcher is pinned to published CLI `0.1.8`; that release does **not** implement this protocol. The feature PR does not publish a new CLI package, and package version `0.1.11` alone is not evidence of the new capability.

During development/review, an **explicitly selected** CLI checkout containing the companion implementation may be used with this argv prefix:

```text
node <CLI_DIR>/bin/tiangong-lca.js
```

Prepare that checkout with its exact Node/pnpm toolchain and frozen lockfile, then build it before checking capabilities. This choice must come from the user or the active task's recorded CLI implementation path; never discover a sibling checkout. Invoke the public bin only, with arguments as separate values and without a shell-generated command string. Do not import private `dist` modules. This workflow-only skill does not bypass or change other skills' shared launcher pin. Once a compatible package is released, qualify the installed executable against the same protocol; selecting and advancing a shared release pin is separately reviewed work.

## Run sequence

Replace the executable below with the validated prefix. Paths are platform-neutral placeholders; use fresh output directories.

```text
tiangong-lca dataset overview capture --out-dir <run>/inventory --json
tiangong-lca dataset overview catalog --inventory <run>/inventory --scope <run>/scope.json --out-dir <run>/candidates-1 --json
tiangong-lca dataset overview analyze --inventory <run>/inventory --scope <run>/scope.json --out-dir <run>/analysis --json
```

Capture defaults to a requested page size of 250, a maximum of 250,000 rows per table and a 512 MiB total response budget. The CLI's current `capture --help` owns accepted flags and bounds. A server page cap is handled by exact-count pagination; raising page size is not a completeness workaround. A failed capture creates no completed marker. The finished capture proves pagination completeness under stable membership/order, not transaction-level snapshot isolation.

`catalog` and `analyze` are offline. They verify capture hashes, public row identity and counts; they do not accept arbitrary historical reports as a database inventory. Iterate into a new candidate/analysis directory after changing terms or membership. Reuse the same capture within one analysis so text and charts have the same observation basis.

The CLI owns OAuth and any configured headless runtime. If `auth status` requires login, the user runs `auth login` in a trusted terminal and the skill resumes afterward. No passwords, codes, tokens, session contents or alternate database client are needed by this skill.
