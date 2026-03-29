#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const SKILL_DIR = path.resolve(SCRIPT_DIR, '..');
const WORKSPACE_ROOT = path.resolve(SKILL_DIR, '..', '..');

const DEFAULT_CLI_DIR = path.join(WORKSPACE_ROOT, 'tiangong-lca-cli');
const DEFAULT_INPUT_FILE = path.join(SKILL_DIR, 'assets', 'example-request.json');

let cliDir = process.env.TIANGONG_LCA_CLI_DIR || DEFAULT_CLI_DIR;
const tempPaths = [];

function cleanup() {
  for (const target of tempPaths) {
    if (target && existsSync(target)) {
      rmSync(target, { recursive: true, force: true });
    }
  }
}

function fail(message) {
  process.stderr.write(`Error: ${message}\n`);
  process.exit(2);
}

function printHelp() {
  process.stdout.write(`Usage:
  scripts/run-lifecyclemodel-resulting-process-builder.mjs build [options]
  scripts/run-lifecyclemodel-resulting-process-builder.mjs publish [options]

Build aliases:
  prepare
  project

Wrapper compatibility options for build:
  --request <file>          Alias for the CLI's --input <file>
  --model-file <file>       Synthesize a temporary CLI request from a lifecycle model file
  --projection-role <mode>  primary | all (maps to projection.mode)

Wrapper options:
  --cli-dir <dir>           Override the tiangong-lca-cli repository path

Canonical CLI commands:
  tiangong lifecyclemodel build-resulting-process --input <file>
  tiangong lifecyclemodel publish-resulting-process --run-dir <dir>
`);
}

function runCli(args) {
  const cliBin = path.join(cliDir, 'bin', 'tiangong.js');
  if (!existsSync(cliBin)) {
    fail(`Cannot find TianGong CLI at ${cliBin}. Set TIANGONG_LCA_CLI_DIR or pass --cli-dir.`);
  }

  const result = spawnSync(process.execPath, [cliBin, ...args], {
    stdio: 'inherit',
  });

  if (result.error) {
    fail(result.error.message);
  }

  process.exit(result.status ?? 1);
}

function writeModelRequest(modelFileRaw, projectionRole) {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), 'tg-lifecyclemodel-request-'));
  tempPaths.push(tempDir);
  const requestFile = path.join(tempDir, 'request.json');
  const mode = projectionRole === 'all' ? 'all-subproducts' : 'primary-only';
  const payload = {
    source_model: {
      json_ordered_path: path.resolve(modelFileRaw),
    },
    projection: {
      mode,
    },
    process_sources: {
      allow_remote_lookup: false,
    },
    publish: {
      intent: 'prepare_only',
      prepare_process_payloads: true,
      prepare_relation_payloads: true,
    },
  };

  writeFileSync(requestFile, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
  return requestFile;
}

function parseWrapperArgs(argv) {
  const passthrough = [];

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];

    if (token === '--cli-dir') {
      const value = argv[index + 1];
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

    passthrough.push(token);
  }

  return passthrough;
}

function runBuild(args) {
  let projectionRole = 'primary';
  let inputPath = '';
  let modelFile = '';
  let showHelp = false;
  const forwardArgs = [];

  for (let index = 0; index < args.length; index += 1) {
    const token = args[index];

    if (token === '--request' || token === '--input') {
      const value = args[index + 1];
      if (!value) {
        fail(`${token} requires a value`);
      }
      inputPath = value;
      index += 1;
      continue;
    }

    if (token?.startsWith('--request=')) {
      inputPath = token.slice('--request='.length);
      continue;
    }

    if (token?.startsWith('--input=')) {
      inputPath = token.slice('--input='.length);
      continue;
    }

    if (token === '--model-file') {
      const value = args[index + 1];
      if (!value) {
        fail('--model-file requires a value');
      }
      modelFile = value;
      index += 1;
      continue;
    }

    if (token?.startsWith('--model-file=')) {
      modelFile = token.slice('--model-file='.length);
      continue;
    }

    if (token === '--projection-role') {
      const value = args[index + 1];
      if (!value) {
        fail('--projection-role requires a value');
      }
      projectionRole = value;
      index += 1;
      continue;
    }

    if (token?.startsWith('--projection-role=')) {
      projectionRole = token.slice('--projection-role='.length);
      continue;
    }

    if (token === '-h' || token === '--help') {
      showHelp = true;
      continue;
    }

    forwardArgs.push(token);
  }

  if (!['primary', 'all'].includes(projectionRole)) {
    fail('--projection-role must be one of: primary, all');
  }

  if (showHelp) {
    runCli(['lifecyclemodel', 'build-resulting-process', '--help']);
  }

  if (inputPath && modelFile) {
    fail('Use either --request/--input or --model-file, not both.');
  }

  const resolvedInput =
    modelFile !== ''
      ? writeModelRequest(modelFile, projectionRole)
      : inputPath || DEFAULT_INPUT_FILE;

  runCli([
    'lifecyclemodel',
    'build-resulting-process',
    '--input',
    resolvedInput,
    ...forwardArgs,
  ]);
}

function runPublish(args) {
  if (args.includes('-h') || args.includes('--help')) {
    runCli(['lifecyclemodel', 'publish-resulting-process', '--help']);
  }

  runCli(['lifecyclemodel', 'publish-resulting-process', ...args]);
}

process.on('exit', cleanup);
process.on('SIGINT', () => {
  cleanup();
  process.exit(130);
});
process.on('SIGTERM', () => {
  cleanup();
  process.exit(143);
});

const args = parseWrapperArgs(process.argv.slice(2));
const subcommand = args[0];

if (!subcommand) {
  printHelp();
  process.exit(0);
}

switch (subcommand) {
  case '-h':
  case '--help':
  case 'help':
    printHelp();
    break;
  case 'build':
  case 'prepare':
  case 'project':
    runBuild(args.slice(1));
    break;
  case 'publish':
    runPublish(args.slice(1));
    break;
  default:
    fail(`Unknown subcommand: ${subcommand}`);
}
