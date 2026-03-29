#!/usr/bin/env node
import path from 'node:path';
import {
  buildContext,
  fail,
  resolveCliBin,
  runCli,
} from '../../shared/run-tiangong-cli-wrapper.mjs';

const context = buildContext(import.meta.url);

function printHelp() {
  process.stdout.write(`Usage:
  scripts/run-review-flows.mjs [options]

Compatibility options:
  --cli-dir <dir>          Override the tiangong-lca-cli repository path
  --enable-llm             Forward semantic review to the CLI
  --disable-llm            Force rule-only review in the CLI wrapper
  --methodology-file <p>   Compatibility-only label source; forwarded as --methodology-id basename when needed
  --methodology-id <name>  Override the methodology label written by the CLI

Canonical CLI command:
  tiangong review flow (--rows-file <file> | --flows-dir <dir> | --run-root <dir>) --out-dir <dir>
`);
}

const rawArgs = process.argv.slice(2);
let cliDir = process.env.TIANGONG_LCA_CLI_DIR ?? context.defaultCliDir;
let showHelp = false;
let enableLlm = false;
let disableLlm = false;
let methodologyId = '';
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

  if (token === '--enable-llm') {
    enableLlm = true;
    continue;
  }

  if (token === '--disable-llm') {
    disableLlm = true;
    continue;
  }

  if (token === '--with-reference-context') {
    fail(
      '--with-reference-context is not implemented in the unified CLI yet. Remove the flag or finish the later reference-context migration slice first.',
    );
  }

  if (token === '--methodology-file') {
    const value = rawArgs[index + 1];
    if (!value) {
      fail('--methodology-file requires a value');
    }
    if (!methodologyId) {
      methodologyId = path.basename(value);
    }
    index += 1;
    continue;
  }

  if (token?.startsWith('--methodology-file=')) {
    if (!methodologyId) {
      methodologyId = path.basename(token.slice('--methodology-file='.length));
    }
    continue;
  }

  if (token === '--methodology-id') {
    const value = rawArgs[index + 1];
    if (!value) {
      fail('--methodology-id requires a value');
    }
    methodologyId = value;
    index += 1;
    continue;
  }

  if (token?.startsWith('--methodology-id=')) {
    methodologyId = token.slice('--methodology-id='.length);
    continue;
  }

  if (token === '-h' || token === '--help') {
    showHelp = true;
    continue;
  }

  forwardArgs.push(token);
}

if (enableLlm && disableLlm) {
  fail('Use at most one of --enable-llm or --disable-llm');
}

if (showHelp) {
  printHelp();
  process.exit(0);
}

const cliBin = resolveCliBin(cliDir);
const cliArgs = ['review', 'flow'];
if (enableLlm) {
  cliArgs.push('--enable-llm');
}
if (methodologyId) {
  cliArgs.push('--methodology-id', methodologyId);
}
cliArgs.push(...forwardArgs);
runCli(cliBin, cliArgs);
