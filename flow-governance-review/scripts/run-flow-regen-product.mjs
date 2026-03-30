#!/usr/bin/env node
import path from 'node:path';
import {
  buildContext,
  fail,
  resolveCliBin,
  runCli,
} from '../../shared/run-tiangong-cli-wrapper.mjs';

const context = buildContext(import.meta.url);
const DEFAULT_PROCESSES_FILE = path.join(
  context.skillDir,
  'assets',
  'artifacts',
  'flow-processing',
  'datasets',
  'process_pool.jsonl',
);
const DEFAULT_SCOPE_FLOW_FILE = path.join(
  context.skillDir,
  'assets',
  'artifacts',
  'flow-processing',
  'datasets',
  'flows_tidas_sdk_plus_classification_round2_sdk018_all_final_resolved.jsonl',
);
const DEFAULT_OUT_DIR = path.join(
  context.skillDir,
  'assets',
  'artifacts',
  'flow-processing',
  'remediation',
  'regen-product',
);

function printHelp() {
  process.stdout.write(`Usage:
  scripts/run-flow-regen-product.mjs [options]

Compatibility options:
  --cli-dir <dir>               Override the tiangong-lca-cli repository path
  --processes-file <file>       Local process rows JSON or JSONL input
  --scope-flow-file <file>      Repeatable scope flow rows JSON or JSONL input
  --scope-flow-files <files...> Compatibility alias for one or more scope flow files
  --catalog-flow-file <file>    Repeatable catalog flow rows JSON or JSONL input
  --catalog-flow-files <files...>
                                Compatibility alias for one or more catalog flow files
  --out-dir <dir>               Output directory for scan/repair/apply/validate artifacts

Canonical CLI command:
  tiangong flow regen-product --processes-file <file> --scope-flow-file <file> [options]

Defaults:
  --processes-file ${DEFAULT_PROCESSES_FILE}
  --scope-flow-file ${DEFAULT_SCOPE_FLOW_FILE}
  --out-dir ${DEFAULT_OUT_DIR}

Notes:
  - This wrapper is a thin compatibility layer over the unified CLI.
  - Pass --apply when you want patched-process and validation artifacts in addition to scan/repair planning.
  - The wrapper stays local-first and does not add any skill-local REST or MCP transport.
`);
}

function readFlagValue(argv, index, flag) {
  const value = argv[index + 1];
  if (!value) {
    fail(`${flag} requires a value`);
  }
  return value;
}

function readFlagValues(argv, startIndex, flag) {
  const values = [];
  let index = startIndex + 1;
  while (index < argv.length) {
    const token = argv[index];
    if (!token || token.startsWith('-')) {
      break;
    }
    values.push(token);
    index += 1;
  }
  if (values.length === 0) {
    fail(`${flag} requires at least one value`);
  }
  return { values, nextIndex: index - 1 };
}

const rawArgs = process.argv.slice(2);
let cliDir = process.env.TIANGONG_LCA_CLI_DIR ?? context.defaultCliDir;
let processesFile = null;
let outDir = null;
let showHelp = false;
const scopeFlowFiles = [];
const catalogFlowFiles = [];
const forwardArgs = [];

for (let index = 0; index < rawArgs.length; index += 1) {
  const token = rawArgs[index];

  if (token === '--cli-dir') {
    cliDir = readFlagValue(rawArgs, index, '--cli-dir');
    index += 1;
    continue;
  }

  if (token?.startsWith('--cli-dir=')) {
    cliDir = token.slice('--cli-dir='.length);
    continue;
  }

  if (token === '--processes-file') {
    processesFile = readFlagValue(rawArgs, index, '--processes-file');
    index += 1;
    continue;
  }

  if (token?.startsWith('--processes-file=')) {
    processesFile = token.slice('--processes-file='.length);
    continue;
  }

  if (token === '--scope-flow-file') {
    scopeFlowFiles.push(readFlagValue(rawArgs, index, '--scope-flow-file'));
    index += 1;
    continue;
  }

  if (token?.startsWith('--scope-flow-file=')) {
    scopeFlowFiles.push(token.slice('--scope-flow-file='.length));
    continue;
  }

  if (token === '--scope-flow-files') {
    const result = readFlagValues(rawArgs, index, '--scope-flow-files');
    scopeFlowFiles.push(...result.values);
    index = result.nextIndex;
    continue;
  }

  if (token === '--catalog-flow-file') {
    catalogFlowFiles.push(readFlagValue(rawArgs, index, '--catalog-flow-file'));
    index += 1;
    continue;
  }

  if (token?.startsWith('--catalog-flow-file=')) {
    catalogFlowFiles.push(token.slice('--catalog-flow-file='.length));
    continue;
  }

  if (token === '--catalog-flow-files') {
    const result = readFlagValues(rawArgs, index, '--catalog-flow-files');
    catalogFlowFiles.push(...result.values);
    index = result.nextIndex;
    continue;
  }

  if (token === '--out-dir') {
    outDir = readFlagValue(rawArgs, index, '--out-dir');
    index += 1;
    continue;
  }

  if (token?.startsWith('--out-dir=')) {
    outDir = token.slice('--out-dir='.length);
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
const cliArgs = [
  'flow',
  'regen-product',
  '--processes-file',
  processesFile ?? DEFAULT_PROCESSES_FILE,
];

for (const scopeFlowFile of scopeFlowFiles.length > 0 ? scopeFlowFiles : [DEFAULT_SCOPE_FLOW_FILE]) {
  cliArgs.push('--scope-flow-file', scopeFlowFile);
}

for (const catalogFlowFile of catalogFlowFiles) {
  cliArgs.push('--catalog-flow-file', catalogFlowFile);
}

cliArgs.push('--out-dir', outDir ?? DEFAULT_OUT_DIR, ...forwardArgs);
runCli(cliBin, cliArgs);
