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

const actions = {
  validate: ['dataset', 'validate'],
  'rewrite-references': ['dataset', 'references', 'rewrite'],
  'save-lifecyclemodels': ['lifecyclemodel', 'save-draft'],
  'graph-lifecyclemodels': ['lifecyclemodel', 'graph'],
};

function fail(message) {
  throw new UsageError(message);
}

function renderHelp() {
  return `Usage:
  node current-account-dataset-review/scripts/run-current-account-dataset-review.mjs <action> [options]

Actions:
  validate               Delegate to tiangong dataset validate
  rewrite-references    Delegate to tiangong dataset references rewrite
  save-lifecyclemodels  Delegate to tiangong lifecyclemodel save-draft
  graph-lifecyclemodels Delegate to tiangong lifecyclemodel graph

Wrapper options:
  --cli-dir <dir>        Override the published CLI and use a local tiangong-cli repository path
  -h, --help

Runtime:
  default                ${publishedCliCommand}
  local override         --cli-dir /path/to/tiangong-cli or TIANGONG_LCA_CLI_DIR

Examples:
  node current-account-dataset-review/scripts/run-current-account-dataset-review.mjs validate --input /abs/path/rows.jsonl --type auto --out-dir /abs/path/dataset-validate
  node current-account-dataset-review/scripts/run-current-account-dataset-review.mjs rewrite-references --input /abs/path/rows.jsonl --from flow:<old>@01.00.000 --to flow:<new>@01.01.000 --type process --type lifecyclemodel --out-dir /abs/path/rewrite --dry-run
  node current-account-dataset-review/scripts/run-current-account-dataset-review.mjs save-lifecyclemodels --input /abs/path/lifecyclemodels.jsonl --out-dir /abs/path/save-draft --dry-run
  node current-account-dataset-review/scripts/run-current-account-dataset-review.mjs graph-lifecyclemodels --input /abs/path/lifecyclemodels.jsonl --out-dir /abs/path/graph --format all --check-connections

Notes:
  - this wrapper is CLI-only; it does not own Supabase auth, password parsing, schema validation, or direct DB queries
  - keep --out-dir explicit so validation, rewrite, save-draft, and graph artifacts stay reproducible
`.trim();
}

function main() {
  const { cliDir, args } = normalizeCliRuntimeArgs(process.argv.slice(2), { repoRoot });

  if (args.includes('-h') || args.includes('--help')) {
    console.log(renderHelp());
    process.exit(0);
  }

  const action = args[0];
  if (!action) {
    fail('Missing required action. Use --help for supported actions.');
  }

  const command = actions[action];
  if (!command) {
    fail(`Unsupported action: ${action}. Use --help for supported actions.`);
  }

  const exitCode = runTiangongCommand([...command, ...args.slice(1)], {
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
