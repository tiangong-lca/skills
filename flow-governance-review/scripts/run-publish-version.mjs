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
  'remediation',
  'round1',
  'flows_tidas_sdk_plus_classification_remediated_ready_for_mcp.jsonl',
);
const DEFAULT_OUT_DIR = path.join(
  context.skillDir,
  'assets',
  'artifacts',
  'flow-processing',
  'remediation',
  'mcp-sync',
);
const DEFAULT_TARGET_USER_ID = 'dbcf5d8a-60bb-4dfc-a2b3-e8b4ab9352c0';
const LEGACY_OUTPUT_FLAGS = new Map([
  ['--out-remote-failed-file', 'flows_tidas_sdk_plus_classification_remote_validation_failed.jsonl'],
  ['--out-success-list-file', 'flows_tidas_sdk_plus_classification_mcp_success_list.json'],
  ['--out-report-file', 'flows_tidas_sdk_plus_classification_mcp_sync_report.json'],
]);

function printHelp() {
  process.stdout.write(`Usage:
  scripts/run-publish-version.mjs [options]

Compatibility options:
  --cli-dir <dir>                  Override the tiangong-lca-cli repository path
  --input-file <file>              Ready-for-publish flow rows JSON or JSONL input
  --out-dir <dir>                  Output directory for publish-version artifacts
  --target-user-id <id>            Override the historical default owner fallback
  --commit                         Commit remote writes explicitly
  --dry-run                        Plan the publish-version stage without remote writes

Legacy compatibility:
  --out-remote-failed-file <file>
  --out-success-list-file <file>
  --out-report-file <file>

Canonical CLI command:
  tiangong flow publish-version --input-file <file> --out-dir <dir> --commit

Defaults:
  --input-file ${DEFAULT_INPUT_FILE}
  --out-dir ${DEFAULT_OUT_DIR}
  --target-user-id ${DEFAULT_TARGET_USER_ID}

Notes:
  - This wrapper preserves the historical publish-stage behavior by adding --commit unless --dry-run is passed.
  - Legacy output-file flags are accepted only when they keep the canonical filenames above.
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
let sawCommit = false;
let sawDryRun = false;
let sawTargetUserId = false;
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

  if (token === '--target-user-id') {
    sawTargetUserId = true;
    forwardArgs.push(token, readFlagValue(rawArgs, index, '--target-user-id'));
    index += 1;
    continue;
  }

  if (token?.startsWith('--target-user-id=')) {
    sawTargetUserId = true;
    forwardArgs.push(token);
    continue;
  }

  if (token === '--commit') {
    sawCommit = true;
    forwardArgs.push(token);
    continue;
  }

  if (token === '--dry-run') {
    sawDryRun = true;
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

if (sawCommit && sawDryRun) {
  fail('Use at most one of --commit or --dry-run');
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
  'publish-version',
  '--input-file',
  inputFile ?? DEFAULT_INPUT_FILE,
  '--out-dir',
  effectiveOutDir ?? DEFAULT_OUT_DIR,
];

if (!sawCommit && !sawDryRun) {
  cliArgs.push('--commit');
}
if (!sawTargetUserId) {
  cliArgs.push('--target-user-id', DEFAULT_TARGET_USER_ID);
}
cliArgs.push(...forwardArgs);
runCli(cliBin, cliArgs);
