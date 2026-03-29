#!/usr/bin/env node
import path from 'node:path';
import {
  buildContext,
  fail,
  resolveCliBin,
  runCli,
} from '../../shared/run-tiangong-cli-wrapper.mjs';

const context = buildContext(import.meta.url);
const DEFAULT_INPUT_FILE = path.join(
  context.skillDir,
  'assets',
  'artifacts',
  'flow-processing',
  'datasets',
  'flows_tidas_sdk_plus_classification_invalid.jsonl',
);
const DEFAULT_OUT_DIR = path.join(
  context.skillDir,
  'assets',
  'artifacts',
  'flow-processing',
  'remediation',
  'round1',
);
const LEGACY_OUTPUT_FLAGS = new Map([
  ['--out-all-file', 'flows_tidas_sdk_plus_classification_remediated_all.jsonl'],
  ['--out-valid-file', 'flows_tidas_sdk_plus_classification_remediated_ready_for_mcp.jsonl'],
  ['--out-manual-file', 'flows_tidas_sdk_plus_classification_residual_manual_queue.jsonl'],
  ['--out-report-file', 'flows_tidas_sdk_plus_classification_remediation_report.json'],
  ['--out-audit-file', 'flows_tidas_sdk_plus_classification_remediation_audit.jsonl'],
  ['--out-prompt-file', 'flows_tidas_sdk_plus_classification_residual_manual_queue_prompt.md'],
]);

function printHelp() {
  process.stdout.write(`Usage:
  scripts/run-remediate-flows.mjs [options]

Compatibility options:
  --cli-dir <dir>          Override the tiangong-lca-cli repository path
  --input-file <file>      Invalid flow rows JSON or JSONL input
  --out-dir <dir>          Output directory for the round1 remediation artifacts

Legacy compatibility:
  --out-all-file <file>
  --out-valid-file <file>
  --out-manual-file <file>
  --out-report-file <file>
  --out-audit-file <file>
  --out-prompt-file <file>

Canonical CLI command:
  tiangong flow remediate --input-file <file> --out-dir <dir>

Defaults:
  --input-file ${DEFAULT_INPUT_FILE}
  --out-dir ${DEFAULT_OUT_DIR}

Notes:
  - Legacy output-file flags are accepted only when they keep the canonical filenames above.
  - This wrapper implements deterministic local round1 remediation only.
`);
}

function readFlagValue(argv, index, flag) {
  const value = argv[index + 1];
  if (!value) {
    fail(`${flag} requires a value`);
  }
  return value;
}

const rawArgs = process.argv.slice(2);
let cliDir = process.env.TIANGONG_LCA_CLI_DIR ?? context.defaultCliDir;
let inputFile = null;
let outDir = null;
let showHelp = false;
const legacyOutputFiles = new Map();
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

  if (token === '--input-file') {
    inputFile = readFlagValue(rawArgs, index, '--input-file');
    index += 1;
    continue;
  }

  if (token?.startsWith('--input-file=')) {
    inputFile = token.slice('--input-file='.length);
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

  if (LEGACY_OUTPUT_FLAGS.has(token)) {
    legacyOutputFiles.set(token, readFlagValue(rawArgs, index, token));
    index += 1;
    continue;
  }

  const legacyEqualsEntry = Array.from(LEGACY_OUTPUT_FLAGS.keys()).find((flag) =>
    token?.startsWith(`${flag}=`),
  );
  if (legacyEqualsEntry) {
    legacyOutputFiles.set(legacyEqualsEntry, token.slice(legacyEqualsEntry.length + 1));
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

let effectiveOutDir = outDir ? path.resolve(outDir) : null;
for (const [flag, filePath] of legacyOutputFiles.entries()) {
  const resolvedPath = path.resolve(filePath);
  const expectedBaseName = LEGACY_OUTPUT_FLAGS.get(flag);
  if (path.basename(resolvedPath) !== expectedBaseName) {
    fail(
      `${flag} must keep the canonical file name ${expectedBaseName}. Use --out-dir when you only need a different directory.`,
    );
  }
  const legacyDir = path.dirname(resolvedPath);
  if (effectiveOutDir && effectiveOutDir !== legacyDir) {
    fail(`Legacy output flags must resolve to the same directory as --out-dir: ${effectiveOutDir}`);
  }
  effectiveOutDir = legacyDir;
}

const cliBin = resolveCliBin(cliDir);
const cliArgs = [
  'flow',
  'remediate',
  '--input-file',
  inputFile ?? DEFAULT_INPUT_FILE,
  '--out-dir',
  effectiveOutDir ?? DEFAULT_OUT_DIR,
  ...forwardArgs,
];
runCli(cliBin, cliArgs);
