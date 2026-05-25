#!/usr/bin/env node
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import {
  normalizeCliRuntimeArgs,
  publishedCliCommand,
  runTiangongCommand,
} from '../../scripts/lib/cli-launcher.mjs';

class UsageError extends Error {}

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '..', '..');

function fail(message) {
  throw new UsageError(message);
}

function renderHelp() {
  return `Usage:
  node scripts/run-process-scope-statistics.mjs --out-dir <dir> [options]

Wrapper options:
  --cli-dir <dir>         Override the published CLI and use a local tiangong-lca-cli repository path
  -h, --help

Delegates to:
  tiangong process scope-statistics

Runtime:
  default                  ${publishedCliCommand}
  local override           --cli-dir /path/to/tiangong-cli or TIANGONG_LCA_CLI_DIR

Notes:
  - this wrapper is CLI-only; it does not own remote fetch logic or .env parsing
  - the TianGong CLI loads .env from the current working directory before reading TIANGONG_LCA_* env
  - prefer repeatable --state-code flags; --state-codes <csv> is still accepted as a compatibility alias

Examples:
  node scripts/run-process-scope-statistics.mjs --out-dir /abs/path/process-scope --state-code 0 --state-code 100 --json
  node scripts/run-process-scope-statistics.mjs --out-dir /abs/path/process-scope --scope current-user --reuse-snapshot
`.trim();
}

function main() {
  const { cliDir, args } = normalizeCliRuntimeArgs(process.argv.slice(2), { repoRoot });

  if (args.includes('-h') || args.includes('--help')) {
    console.log(renderHelp());
    process.exit(0);
  }

  if (args.length === 0) {
    fail('Missing required CLI arguments. Use --out-dir <dir>.');
  }

  const exitCode = runTiangongCommand(['process', 'scope-statistics', ...args], {
    cliDir,
    repoRoot,
  });
  process.exit(exitCode);
}

try {
  main();
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  process.exit(1);
}
