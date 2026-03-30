#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const skillDir = path.resolve(scriptDir, '..');
const workspaceRoot = path.resolve(skillDir, '..', '..');
const defaultCliDir = path.join(workspaceRoot, 'tiangong-lca-cli');
const defaultInputFile = path.join(skillDir, 'assets', 'example-request.json');
const tempPaths = [];

function cleanup() {
  tempPaths.forEach((targetPath) => {
    if (existsSync(targetPath)) {
      rmSync(targetPath, { recursive: true, force: true });
    }
  });
}

process.on('exit', cleanup);

function fail(message) {
  console.error(`Error: ${message}`);
  process.exit(2);
}

function printHelp() {
  console.log(`Usage:
  node scripts/run-lifecyclemodel-resulting-process-builder.mjs build [options]
  node scripts/run-lifecyclemodel-resulting-process-builder.mjs publish [options]

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
  tiangong lifecyclemodel publish-resulting-process --run-dir <dir>`);
}

function runCli(cliBin, cliArgs) {
  const result = spawnSync(process.execPath, [cliBin, ...cliArgs], {
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
}

function writeModelRequest(modelFile, projectionRole) {
  const tempDir = mkdtempSync(path.join(tmpdir(), 'tg-lifecyclemodel-request-'));
  tempPaths.push(tempDir);

  const requestFile = path.join(tempDir, 'request.json');
  const payload = {
    source_model: {
      json_ordered_path: path.resolve(modelFile),
    },
    projection: {
      mode: projectionRole === 'all' ? 'all-subproducts' : 'primary-only',
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

function runBuild(cliBin, args) {
  let projectionRole = 'primary';
  let inputPath = '';
  let modelFile = '';
  let showHelp = false;
  const forwardArgs = [];

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];

    switch (arg) {
      case '--request':
      case '--input':
        if (index + 1 >= args.length) {
          fail(`${arg} requires a value`);
        }
        inputPath = args[index + 1];
        index += 1;
        break;
      case '--model-file':
        if (index + 1 >= args.length) {
          fail('--model-file requires a value');
        }
        modelFile = args[index + 1];
        index += 1;
        break;
      case '--projection-role':
        if (index + 1 >= args.length) {
          fail('--projection-role requires a value');
        }
        projectionRole = args[index + 1];
        index += 1;
        break;
      case '-h':
      case '--help':
        showHelp = true;
        break;
      default:
        if (arg.startsWith('--request=')) {
          inputPath = arg.slice('--request='.length);
        } else if (arg.startsWith('--input=')) {
          inputPath = arg.slice('--input='.length);
        } else if (arg.startsWith('--model-file=')) {
          modelFile = arg.slice('--model-file='.length);
        } else if (arg.startsWith('--projection-role=')) {
          projectionRole = arg.slice('--projection-role='.length);
        } else {
          forwardArgs.push(arg);
        }
        break;
    }
  }

  if (!['primary', 'all'].includes(projectionRole)) {
    fail('--projection-role must be one of: primary, all');
  }

  if (showHelp) {
    runCli(cliBin, ['lifecyclemodel', 'build-resulting-process', '--help']);
  }

  if (inputPath && modelFile) {
    fail('Use either --request/--input or --model-file, not both.');
  }

  if (modelFile) {
    inputPath = writeModelRequest(modelFile, projectionRole);
  } else if (!inputPath) {
    inputPath = defaultInputFile;
  }

  runCli(cliBin, [
    'lifecyclemodel',
    'build-resulting-process',
    '--input',
    inputPath,
    ...forwardArgs,
  ]);
}

function runPublish(cliBin, args) {
  let showHelp = false;
  const forwardArgs = [];

  args.forEach((arg) => {
    if (arg === '-h' || arg === '--help') {
      showHelp = true;
      return;
    }
    forwardArgs.push(arg);
  });

  if (showHelp) {
    runCli(cliBin, ['lifecyclemodel', 'publish-resulting-process', '--help']);
  }

  runCli(cliBin, ['lifecyclemodel', 'publish-resulting-process', ...forwardArgs]);
}

let cliDir = process.env.TIANGONG_LCA_CLI_DIR?.trim() || defaultCliDir;
const filteredArgs = [];

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
  filteredArgs.push(arg);
}

const cliBin = path.join(cliDir, 'bin', 'tiangong.js');
if (!existsSync(cliBin)) {
  fail(`Cannot find TianGong CLI at ${cliBin}. Set TIANGONG_LCA_CLI_DIR or pass --cli-dir.`);
}

const subcommand = filteredArgs[0];
if (!subcommand || subcommand === 'help' || subcommand === '-h' || subcommand === '--help') {
  printHelp();
  process.exit(0);
}

switch (subcommand) {
  case 'build':
  case 'prepare':
  case 'project':
    runBuild(cliBin, filteredArgs.slice(1));
    break;
  case 'publish':
    runPublish(cliBin, filteredArgs.slice(1));
    break;
  default:
    fail(`Unknown subcommand: ${subcommand}`);
}
