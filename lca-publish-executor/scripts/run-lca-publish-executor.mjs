#!/usr/bin/env node
import process from 'node:process';
import {
  normalizeCliRuntimeArgs,
  publishedCliCommand,
  runTiangongCommand,
} from '../../scripts/lib/cli-launcher.mjs';

class UsageError extends Error {}

function fail(message) {
  throw new UsageError(message);
}

function renderHelp() {
  return `Usage:
  node scripts/run-lca-publish-executor.mjs publish [options]

Wrapper options:
  --cli-dir <dir>           Override the published CLI and use a local tiangong-lca-cli repository path

Canonical CLI command:
  tiangong publish run --input <file> [options]

Compatibility aliases:
  --request <file>          Alias for the CLI's --input <file>

Examples:
  node scripts/run-lca-publish-executor.mjs publish --request assets/example-request.json --dry-run --json
  node scripts/run-lca-publish-executor.mjs publish --request assets/example-request.json --commit --json

Notes:
  - default runtime is ${publishedCliCommand}
  - this wrapper is CLI-only; there is no Python or MCP fallback path
  - publish execution is unified under tiangong publish run
`.trim();
}

function normalizeForwardArgs(args) {
  const forwardArgs = [];

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];

    if (arg === '--request') {
      if (index + 1 >= args.length) {
        fail('--request requires a value');
      }
      forwardArgs.push('--input', args[index + 1]);
      index += 1;
      continue;
    }

    if (arg.startsWith('--request=')) {
      forwardArgs.push(`--input=${arg.slice('--request='.length)}`);
      continue;
    }

    forwardArgs.push(arg);
  }

  return forwardArgs;
}

function main() {
  const { cliDir, args } = normalizeCliRuntimeArgs(process.argv.slice(2));
  const action = args[0];
  const forwardedArgs = normalizeForwardArgs(args.slice(1));

  if (!action || action === 'help' || action === '-h' || action === '--help') {
    console.log(renderHelp());
    process.exit(0);
  }

  if (action !== 'publish') {
    fail(`Unknown action: ${action}`);
  }

  process.exit(runTiangongCommand(['publish', 'run', ...forwardedArgs], { cliDir }));
}

try {
  main();
} catch (error) {
  if (error instanceof UsageError) {
    console.error(`Error: ${error.message}`);
    console.error('');
    console.error(renderHelp());
    process.exit(2);
  }

  const message = error instanceof Error ? error.message : String(error);
  console.error(`Error: ${message}`);
  process.exit(1);
}
