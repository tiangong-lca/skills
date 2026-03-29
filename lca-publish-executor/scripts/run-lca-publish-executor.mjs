#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const skillDir = path.resolve(scriptDir, '..');
const workspaceRoot = path.resolve(skillDir, '..', '..');
const defaultCliDir = path.join(workspaceRoot, 'tiangong-lca-cli');
const defaultInputFile = path.join(skillDir, 'assets', 'example-request.json');

function fail(message) {
  console.error(`Error: ${message}`);
  process.exit(2);
}

function printHelp() {
  console.log(`Usage:
  node scripts/run-lca-publish-executor.mjs publish [options]
  node scripts/run-lca-publish-executor.mjs [options]

Wrapper compatibility options:
  publish                 Optional compatibility subcommand
  --request <file>        Alias for the CLI's --input <file>
  --cli-dir <dir>         Override the tiangong-lca-cli repository path

Canonical CLI command:
  tiangong publish run --input <file>

Notes:
  - If --request/--input is omitted, the wrapper uses assets/example-request.json.
  - --commit forwards to tiangong publish run; without configured CLI executors,
    commit-time work will be reported as deferred in publish-report.json.`);
}

let cliDir = process.env.TIANGONG_LCA_CLI_DIR?.trim() || defaultCliDir;
const rawArgs = [];

for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];

  if (arg === '--cli-dir') {
    if (index + 1 >= process.argv.length) {
      fail('--cli-dir requires a value');
    }
    cliDir = process.argv[index + 1];
    index += 1;
    continue;
  }
  if (arg.startsWith('--cli-dir=')) {
    cliDir = arg.slice('--cli-dir='.length);
    continue;
  }
  rawArgs.push(arg);
}

const cliBin = path.join(cliDir, 'bin', 'tiangong.js');
if (!existsSync(cliBin)) {
  fail(`Cannot find TianGong CLI at ${cliBin}. Set TIANGONG_LCA_CLI_DIR or pass --cli-dir.`);
}

let argIndex = 0;
if (rawArgs[0] === 'publish') {
  argIndex = 1;
} else if (
  rawArgs[0] &&
  !rawArgs[0].startsWith('-') &&
  rawArgs[0] !== 'help' &&
  rawArgs[0] !== '-h' &&
  rawArgs[0] !== '--help'
) {
  fail(`Unknown subcommand: ${rawArgs[0]}`);
}

let hasInput = false;
let showHelp = false;
const forwardArgs = [];

for (; argIndex < rawArgs.length; argIndex += 1) {
  const arg = rawArgs[argIndex];

  switch (arg) {
    case '--request':
      if (argIndex + 1 >= rawArgs.length) {
        fail('--request requires a value');
      }
      hasInput = true;
      forwardArgs.push('--input', rawArgs[argIndex + 1]);
      argIndex += 1;
      break;
    case '--input':
      if (argIndex + 1 >= rawArgs.length) {
        fail('--input requires a value');
      }
      hasInput = true;
      forwardArgs.push(arg, rawArgs[argIndex + 1]);
      argIndex += 1;
      break;
    case 'help':
    case '-h':
    case '--help':
      showHelp = true;
      break;
    default:
      if (arg.startsWith('--request=')) {
        hasInput = true;
        forwardArgs.push(`--input=${arg.slice('--request='.length)}`);
        break;
      }
      if (arg.startsWith('--input=')) {
        hasInput = true;
      }
      forwardArgs.push(arg);
      break;
  }
}

if (showHelp) {
  printHelp();
  process.exit(0);
}

const commandArgs = [cliBin, 'publish', 'run'];
if (!hasInput) {
  commandArgs.push('--input', defaultInputFile);
}
commandArgs.push(...forwardArgs);

const result = spawnSync(process.execPath, commandArgs, {
  stdio: 'inherit',
});

if (result.error) {
  fail(`Failed to execute TianGong CLI: ${result.error.message}`);
}
if (typeof result.status === 'number') {
  process.exit(result.status);
}
if (result.signal) {
  fail(`TianGong CLI terminated with signal ${result.signal}.`);
}
process.exit(1);
