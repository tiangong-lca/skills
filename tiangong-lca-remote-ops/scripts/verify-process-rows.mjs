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
  node tiangong-lca-remote-ops/scripts/verify-process-rows.mjs --rows-file <file> --out-dir <dir> [options]

Wrapper options:
  --cli-dir <dir>         Override the published CLI and use a local tiangong-cli repository path
  -h, --help

Delegates to:
  tiangong process verify-rows

Runtime:
  default                  ${publishedCliCommand}
  local override           --cli-dir /path/to/tiangong-cli or TIANGONG_LCA_CLI_DIR

Notes:
  - this wrapper is local-only; it validates a frozen rows file and does not require remote credentials by itself
  - accepted inputs include raw process rows JSON/JSONL and tiangong process list reports with rows[]
  - keep --out-dir explicit so verification artifacts are reproducible and easy to diff

Examples:
  node tiangong-lca-remote-ops/scripts/verify-process-rows.mjs --rows-file /abs/path/process-list-report.json --out-dir /abs/path/post-write-verification
  node tiangong-lca-remote-ops/scripts/verify-process-rows.mjs --rows-file /abs/path/processes.jsonl --out-dir /abs/path/process-verify
`.trim();
}

function main() {
  const { cliDir, args } = normalizeCliRuntimeArgs(process.argv.slice(2), { repoRoot });

  if (args.includes('-h') || args.includes('--help')) {
    console.log(renderHelp());
    process.exit(0);
  }

  if (args.length === 0) {
    fail('Missing required CLI arguments. Use --rows-file <file> --out-dir <dir>.');
  }

  const exitCode = runTiangongCommand(['process', 'verify-rows', ...args], {
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
