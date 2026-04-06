#!/usr/bin/env node
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import {
  buildTiangongInvocation,
  normalizeCliRuntimeArgs,
  publishedCliCommand,
  renderShellCommand,
  runTiangongCommand,
} from '../../scripts/lib/cli-launcher.mjs';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const skillDir = path.resolve(scriptDir, '..');
const workspaceRoot = path.resolve(skillDir, '..', '..');
const defaultOutDir = path.join(workspaceRoot, 'artifacts', 'lifecyclemodel-automated-builder', 'default-run');

function fail(message) {
  console.error(`Error: ${message}`);
  process.exit(2);
}

function printHelp() {
  console.log(`Usage:
  node scripts/run-lifecyclemodel-automated-builder.mjs build [options]
  node scripts/run-lifecyclemodel-automated-builder.mjs validate [options]
  node scripts/run-lifecyclemodel-automated-builder.mjs publish [options]

Wrapper options:
  --cli-dir <dir>           Override the published CLI and use a local tiangong-lca-cli repository path

Build compatibility options:
  --manifest <file>         Alias for the CLI's --input <file>
  --request <file>          Alias for the CLI's --input <file>
  --dry-run                 Print the resolved CLI command and exit

Canonical CLI commands:
  tiangong lifecyclemodel auto-build --input <file> [--out-dir <dir>]
  tiangong lifecyclemodel validate-build --run-dir <dir>
  tiangong lifecyclemodel publish-build --run-dir <dir>

Runtime:
  default                  ${publishedCliCommand}
  local override           --cli-dir /path/to/tiangong-lca-cli or TIANGONG_LCA_CLI_DIR

Notes:
  - build is implemented and delegates to tiangong lifecyclemodel auto-build
  - validate delegates to tiangong lifecyclemodel validate-build and re-runs local validation on one existing build run
  - publish delegates to tiangong lifecyclemodel publish-build and prepares local publish handoff artifacts only`);
}

function runCli(cliDir, cliArgs) {
  process.exit(runTiangongCommand(cliArgs, { cliDir }));
}

function normalizeBuildArgs(args) {
  let inputPath = '';
  let outDir = '';
  let showHelp = false;
  let dryRun = false;
  const forwardArgs = [];

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];

    switch (arg) {
      case '--manifest':
      case '--request':
      case '--input':
        if (index + 1 >= args.length) {
          fail(`${arg} requires a value`);
        }
        inputPath = args[index + 1];
        index += 1;
        break;
      case '--out-dir':
        if (index + 1 >= args.length) {
          fail('--out-dir requires a value');
        }
        outDir = args[index + 1];
        forwardArgs.push('--out-dir', outDir);
        index += 1;
        break;
      case '--dry-run':
        dryRun = true;
        break;
      case '-h':
      case '--help':
        showHelp = true;
        break;
      default:
        if (arg.startsWith('--manifest=')) {
          inputPath = arg.slice('--manifest='.length);
        } else if (arg.startsWith('--request=')) {
          inputPath = arg.slice('--request='.length);
        } else if (arg.startsWith('--input=')) {
          inputPath = arg.slice('--input='.length);
        } else if (arg.startsWith('--out-dir=')) {
          outDir = arg.slice('--out-dir='.length);
          forwardArgs.push(arg);
        } else {
          forwardArgs.push(arg);
        }
        break;
    }
  }

  return {
    inputPath,
    outDir,
    showHelp,
    dryRun,
    forwardArgs,
  };
}

function runBuild(cliDir, args) {
  const normalized = normalizeBuildArgs(args);

  if (normalized.showHelp) {
    printHelp();
    process.exit(0);
  }

  if (!normalized.inputPath) {
    fail('build requires --input <file> (or --manifest / --request).');
  }

  const cliArgs = ['lifecyclemodel', 'auto-build', '--input', normalized.inputPath];
  const hasOutDir = normalized.forwardArgs.some(
    (arg) => arg === '--out-dir' || arg.startsWith('--out-dir='),
  );

  if (!hasOutDir) {
    cliArgs.push('--out-dir', defaultOutDir);
  }

  cliArgs.push(...normalized.forwardArgs);

  if (normalized.dryRun) {
    const invocation = buildTiangongInvocation(cliArgs, { cliDir });
    console.log(renderShellCommand(invocation.command, invocation.args));
    process.exit(0);
  }

  runCli(cliDir, cliArgs);
}

function runDelegatedLifecyclemodelCommand(cliDir, subcommand, args) {
  const showHelp = args.includes('-h') || args.includes('--help');
  if (showHelp) {
    runCli(cliDir, ['lifecyclemodel', subcommand, '--help']);
  }
  runCli(cliDir, ['lifecyclemodel', subcommand, ...args]);
}

const { cliDir, args: filteredArgs } = normalizeCliRuntimeArgs(process.argv.slice(2));

const subcommand = filteredArgs[0];
if (!subcommand || subcommand === 'help' || subcommand === '-h' || subcommand === '--help') {
  printHelp();
  process.exit(0);
}

switch (subcommand) {
  case 'build':
    runBuild(cliDir, filteredArgs.slice(1));
    break;
  case 'validate':
    runDelegatedLifecyclemodelCommand(cliDir, 'validate-build', filteredArgs.slice(1));
    break;
  case 'publish':
    runDelegatedLifecyclemodelCommand(cliDir, 'publish-build', filteredArgs.slice(1));
    break;
  default:
    fail(`Unknown subcommand: ${subcommand}`);
}
