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
  node tiangong-lca-remote-ops/scripts/update-process-references.mjs --out-dir <dir> [options]

Wrapper options:
  --cli-dir <dir>         Override the published CLI and use a local tiangong-cli repository path
  -h, --help

Delegates to:
  tiangong process refresh-references

Runtime:
  default                  ${publishedCliCommand}
  local override           --cli-dir /path/to/tiangong-cli or TIANGONG_LCA_CLI_DIR

Required env for remote refresh:
  TIANGONG_LCA_API_BASE_URL
  TIANGONG_LCA_API_KEY
  TIANGONG_LCA_SUPABASE_PUBLISHABLE_KEY

Notes:
  - this wrapper is CLI-only; it does not own Supabase auth, password parsing, or schema validation logic
  - do not pass SUPABASE_EMAIL, SUPABASE_PASSWORD, or SUPABASE_ACCESS_TOKEN to the skill layer
  - keep --out-dir explicit so manifests, progress, blockers, and reports stay reproducible

Examples:
  node tiangong-lca-remote-ops/scripts/update-process-references.mjs --out-dir /abs/path/process-refresh --dry-run
  node tiangong-lca-remote-ops/scripts/update-process-references.mjs --out-dir /abs/path/process-refresh --apply --limit 20
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

  const exitCode = runTiangongCommand(['process', 'refresh-references', ...args], {
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
