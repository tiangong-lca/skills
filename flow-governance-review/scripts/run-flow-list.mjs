#!/usr/bin/env node
import {
  buildContext,
  fail,
  resolveCliBin,
  runCli,
} from '../../shared/run-tiangong-cli-wrapper.mjs';

const context = buildContext(import.meta.url);

function printHelp() {
  process.stdout.write(`Usage:
  scripts/run-flow-list.mjs [options]

Compatibility options:
  --cli-dir <dir>          Override the tiangong-lca-cli repository path

Canonical CLI command:
  tiangong flow list [options]

Notes:
  - This wrapper is a thin compatibility layer over the unified CLI.
  - Remote reads require TIANGONG_LCA_API_BASE_URL and TIANGONG_LCA_API_KEY.
`);
}

const rawArgs = process.argv.slice(2);
let cliDir = process.env.TIANGONG_LCA_CLI_DIR ?? context.defaultCliDir;
let showHelp = false;
const forwardArgs = [];

for (let index = 0; index < rawArgs.length; index += 1) {
  const token = rawArgs[index];

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

const cliBin = resolveCliBin(cliDir);
runCli(cliBin, ['flow', 'list', ...forwardArgs]);
