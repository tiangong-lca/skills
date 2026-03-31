#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

class UsageError extends Error {}

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const skillDir = path.resolve(scriptDir, '..');
const workspaceRoot = path.resolve(skillDir, '..', '..');
const defaultCliDir = path.join(workspaceRoot, 'tiangong-lca-cli');

function fail(message) {
  throw new UsageError(message);
}

function renderHelp() {
  return `Usage:
  node scripts/run-lifecyclemodel-recursive-orchestrator.mjs <plan|execute|publish> [options]

Wrapper options:
  --cli-dir <dir>           Override the tiangong-lca-cli repository path

Canonical CLI command:
  tiangong lifecyclemodel orchestrate <plan|execute|publish> [options]

Compatibility aliases:
  --request <file>          Alias for the CLI's --input <file>

Examples:
  node scripts/run-lifecyclemodel-recursive-orchestrator.mjs plan --request assets/example-request.json --out-dir /abs/path/run-001 --json
  node scripts/run-lifecyclemodel-recursive-orchestrator.mjs execute --request assets/example-request.json --out-dir /abs/path/run-001 --allow-process-build --allow-submodel-build --json
  node scripts/run-lifecyclemodel-recursive-orchestrator.mjs publish --run-dir /abs/path/run-001 --publish-lifecyclemodels --publish-resulting-process-relations --json

Notes:
  - this wrapper is CLI-only; there is no Python fallback path
  - recursive orchestration now lives in tiangong lifecyclemodel orchestrate
`.trim();
}

function resolveCliBin(cliDir) {
  const cliBin = path.join(cliDir, 'bin', 'tiangong.js');
  if (!existsSync(cliBin)) {
    fail(`Cannot find TianGong CLI at ${cliBin}. Set TIANGONG_LCA_CLI_DIR or pass --cli-dir.`);
  }
  return cliBin;
}

function runCommand(command, args) {
  const result = spawnSync(command, args, {
    stdio: 'inherit',
  });

  if (result.error) {
    throw new Error(`Failed to execute ${command}: ${result.error.message}`);
  }
  if (typeof result.status === 'number') {
    return result.status;
  }
  if (result.signal) {
    throw new Error(`${command} terminated with signal ${result.signal}.`);
  }
  return 1;
}

function normalizeTopLevelArgs(rawArgs) {
  let cliDir = process.env.TIANGONG_LCA_CLI_DIR?.trim() || defaultCliDir;
  const args = [];

  for (let index = 0; index < rawArgs.length; index += 1) {
    const arg = rawArgs[index];

    if (arg === '--cli-dir') {
      if (index + 1 >= rawArgs.length) {
        fail('--cli-dir requires a value');
      }
      cliDir = rawArgs[index + 1];
      index += 1;
      continue;
    }

    if (arg.startsWith('--cli-dir=')) {
      cliDir = arg.slice('--cli-dir='.length);
      continue;
    }

    args.push(arg);
  }

  return {
    cliDir,
    args,
  };
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
  const { cliDir, args } = normalizeTopLevelArgs(process.argv.slice(2));
  const action = args[0];
  const forwardedArgs = normalizeForwardArgs(args.slice(1));

  if (!action || action === 'help' || action === '-h' || action === '--help') {
    console.log(renderHelp());
    process.exit(0);
  }

  if (action !== 'plan' && action !== 'execute' && action !== 'publish') {
    fail(`Unknown action: ${action}`);
  }

  const cliBin = resolveCliBin(cliDir);
  process.exit(
    runCommand(process.execPath, [
      cliBin,
      'lifecyclemodel',
      'orchestrate',
      action,
      ...forwardedArgs,
    ]),
  );
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
