#!/usr/bin/env node
import path from 'node:path';
import {
  buildContext,
  fail,
  resolveCliBin,
  runCli,
} from '../../shared/run-tiangong-cli-wrapper.mjs';

const context = buildContext(import.meta.url);
const defaultInputFile = path.join(context.skillDir, 'assets', 'example-request.json');

function printHelp() {
  process.stdout.write(`Usage:
  scripts/run-lca-publish-executor.mjs publish [options]

Wrapper compatibility options:
  --request <file>   Alias for the CLI's --input <file>
  --cli-dir <dir>    Override the tiangong-lca-cli repository path

Canonical CLI command:
  tiangong publish run --input <file>
`);
}

const rawArgs = process.argv.slice(2);
let cliDir = process.env.TIANGONG_LCA_CLI_DIR ?? context.defaultCliDir;
let subcommand = '';
let hasInput = false;
let showHelp = false;
const forwardArgs = [];

for (let index = 0; index < rawArgs.length; index += 1) {
  const token = rawArgs[index];

  if (!subcommand && !token.startsWith('-')) {
    subcommand = token;
    continue;
  }

  if (token === '--cli-dir') {
    const value = rawArgs[index + 1];
    if (!value) {
      fail('--cli-dir requires a value');
    }
    cliDir = value;
    index += 1;
    continue;
  }

  if (token?.startsWith('--cli-dir=')) {
    cliDir = token.slice('--cli-dir='.length);
    continue;
  }

  if (token === '--request' || token === '--input') {
    const value = rawArgs[index + 1];
    if (!value) {
      fail(`${token} requires a value`);
    }
    hasInput = true;
    forwardArgs.push('--input', value);
    index += 1;
    continue;
  }

  if (token?.startsWith('--request=')) {
    hasInput = true;
    forwardArgs.push(`--input=${token.slice('--request='.length)}`);
    continue;
  }

  if (token?.startsWith('--input=')) {
    hasInput = true;
    forwardArgs.push(token);
    continue;
  }

  if (token === '-h' || token === '--help') {
    showHelp = true;
    continue;
  }

  forwardArgs.push(token);
}

if (showHelp) {
  printHelp();
  process.exit(0);
}

if (subcommand !== '' && subcommand !== 'publish') {
  fail(`Unsupported subcommand: ${subcommand}`);
}

const cliBin = resolveCliBin(cliDir);
const args = ['publish', 'run'];

if (!hasInput) {
  args.push('--input', defaultInputFile);
}

args.push(...forwardArgs);
runCli(cliBin, args);
