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
  scripts/run-lifecycleinventory-review.mjs [options]

Wrapper compatibility options:
  --profile <name>  process | lifecyclemodel (default: process)
  --cli-dir <dir>   Override the tiangong-lca-cli repository path

Canonical CLI commands:
  tiangong review process --run-root <dir> --run-id <id> --out-dir <dir>
  tiangong review lifecyclemodel --input <file>
`);
}

const rawArgs = process.argv.slice(2);
let cliDir = process.env.TIANGONG_LCA_CLI_DIR ?? context.defaultCliDir;
let profile = 'process';
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

  if (token === '--profile') {
    const value = rawArgs[index + 1];
    if (!value) {
      fail('--profile requires a value');
    }
    profile = value;
    index += 1;
    continue;
  }

  if (token?.startsWith('--profile=')) {
    profile = token.slice('--profile='.length);
    continue;
  }

  if (token === '-h' || token === '--help') {
    showHelp = true;
    continue;
  }

  forwardArgs.push(token);
}

if (!['process', 'lifecyclemodel'].includes(profile)) {
  fail('--profile must be one of: process, lifecyclemodel');
}

if (showHelp) {
  printHelp();
  process.exit(0);
}

const cliBin = resolveCliBin(cliDir);
runCli(cliBin, ['review', profile, ...forwardArgs]);
