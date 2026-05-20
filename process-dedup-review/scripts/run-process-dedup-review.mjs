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
  node scripts/run-process-dedup-review.mjs --input <file> --out-dir <dir> [options]

Wrapper options:
  --cli-dir <dir>         Override the published CLI and use a local tiangong-lca-cli repository path
  -h, --help

Delegates to:
  tiangong process dedup-review

Runtime:
  default                  ${publishedCliCommand}
  local override           --cli-dir /path/to/tiangong-cli or TIANGONG_LCA_CLI_DIR

Notes:
  - canonical input is grouped JSON, not .xlsx
  - this wrapper is CLI-only; it does not embed workbook parsing or remote enrichment logic
  - if the source starts as a spreadsheet, convert it into grouped JSON before running this wrapper

Examples:
  node scripts/run-process-dedup-review.mjs --input /abs/path/duplicate-groups.json --out-dir /abs/path/process-dedup --json
  node scripts/run-process-dedup-review.mjs --input /abs/path/duplicate-groups.json --out-dir /abs/path/process-dedup --skip-remote
`.trim();
}

function main() {
  const { cliDir, args } = normalizeCliRuntimeArgs(process.argv.slice(2), { repoRoot });

  if (args.includes('-h') || args.includes('--help')) {
    console.log(renderHelp());
    process.exit(0);
  }

  if (args.length === 0) {
    fail('Missing required CLI arguments. Use --input <file> --out-dir <dir>.');
  }

  const exitCode = runTiangongCommand(['process', 'dedup-review', ...args], {
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
