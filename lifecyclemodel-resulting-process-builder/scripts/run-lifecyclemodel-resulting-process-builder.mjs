#!/usr/bin/env node
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import {
  normalizeCliRuntimeArgs,
  publishedCliCommand,
  runTiangongCommand,
} from '../../scripts/lib/cli-launcher.mjs';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const skillDir = path.resolve(scriptDir, '..');
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
  --cli-dir <dir>           Override the published CLI and use a local tiangong-lca-cli repository path

Canonical CLI commands:
  tiangong lifecyclemodel build-resulting-process --input <file>
  tiangong lifecyclemodel publish-resulting-process --run-dir <dir>

Runtime:
  default                  ${publishedCliCommand}
  local override           --cli-dir /path/to/tiangong-lca-cli or TIANGONG_LCA_CLI_DIR`);
}

function runCli(cliDir, cliArgs) {
  process.exit(runTiangongCommand(cliArgs, { cliDir }));
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

function runBuild(cliDir, args) {
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
    runCli(cliDir, ['lifecyclemodel', 'build-resulting-process', '--help']);
  }

  if (inputPath && modelFile) {
    fail('Use either --request/--input or --model-file, not both.');
  }

  if (modelFile) {
    inputPath = writeModelRequest(modelFile, projectionRole);
  } else if (!inputPath) {
    inputPath = defaultInputFile;
  }

  runCli(cliDir, [
    'lifecyclemodel',
    'build-resulting-process',
    '--input',
    inputPath,
    ...forwardArgs,
  ]);
}

function runPublish(cliDir, args) {
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
    runCli(cliDir, ['lifecyclemodel', 'publish-resulting-process', '--help']);
  }

  runCli(cliDir, ['lifecyclemodel', 'publish-resulting-process', ...forwardArgs]);
}

const { cliDir, args: filteredArgs } = normalizeCliRuntimeArgs(process.argv.slice(2));

const subcommand = filteredArgs[0];
if (!subcommand || subcommand === 'help' || subcommand === '-h' || subcommand === '--help') {
  printHelp();
  process.exit(0);
}

switch (subcommand) {
  case 'build':
  case 'prepare':
  case 'project':
    runBuild(cliDir, filteredArgs.slice(1));
    break;
  case 'publish':
    runPublish(cliDir, filteredArgs.slice(1));
    break;
  default:
    fail(`Unknown subcommand: ${subcommand}`);
}
