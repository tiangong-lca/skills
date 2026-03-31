#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

class UsageError extends Error {}

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const skillDir = path.resolve(scriptDir, '..');
const workspaceRoot = path.resolve(skillDir, '..', '..');
const defaultCliDir = path.join(workspaceRoot, 'tiangong-lca-cli');

function fail(message) {
  throw new UsageError(message);
}

function renderHelp() {
  return `Usage:
  node scripts/run-review.mjs [--profile <process|lifecyclemodel>] [options]

Wrapper options:
  --profile <name>         process | lifecyclemodel (default: process)
  --cli-dir <dir>          Override the tiangong-lca-cli repository path

Profiles:
  process                  Delegate to tiangong review process
  lifecyclemodel           Reserved; not implemented yet

Examples:
  node scripts/run-review.mjs --profile process --run-root /path/to/artifacts/process_from_flow/<run_id> --run-id <run_id> --out-dir /abs/path/review
  node scripts/run-review.mjs --profile process --run-root /path/to/artifacts/process_from_flow/<run_id> --run-id <run_id> --out-dir /abs/path/review --enable-llm
`.trim();
}

function resolveCliBin(cliDir) {
  const cliBin = path.join(cliDir, 'bin', 'tiangong.js');
  if (!existsSync(cliBin)) {
    fail(`Cannot find TianGong CLI at ${cliBin}. Set TIANGONG_LCA_CLI_DIR or pass --cli-dir.`);
  }
  return cliBin;
}

function runCommand(command, args) {
  const result = spawnSync(command, args, {
    stdio: 'inherit',
  });

  if (result.error) {
    throw new Error(`Failed to execute ${command}: ${result.error.message}`);
  }
  if (typeof result.status === 'number') {
    return result.status;
  }
  if (result.signal) {
    throw new Error(`${command} terminated with signal ${result.signal}.`);
  }
  return 1;
}

function normalizeArgs(rawArgs) {
  let cliDir = process.env.TIANGONG_LCA_CLI_DIR?.trim() || defaultCliDir;
  let profile = 'process';
  const args = [];

  for (let index = 0; index < rawArgs.length; index += 1) {
    const arg = rawArgs[index];

    if (arg === '--cli-dir') {
      if (index + 1 >= rawArgs.length) {
        fail('--cli-dir requires a value');
      }
      cliDir = rawArgs[index + 1];
      index += 1;
      continue;
    }

    if (arg.startsWith('--cli-dir=')) {
      cliDir = arg.slice('--cli-dir='.length);
      continue;
    }

    if (arg === '--profile') {
      if (index + 1 >= rawArgs.length) {
        fail('--profile requires a value');
      }
      profile = rawArgs[index + 1];
      index += 1;
      continue;
    }

    if (arg.startsWith('--profile=')) {
      profile = arg.slice('--profile='.length);
      continue;
    }

    args.push(arg);
  }

  return {
    cliDir,
    profile,
    args,
  };
}

function main() {
  const { cliDir, profile, args } = normalizeArgs(process.argv.slice(2));

  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    if (profile === 'process' && args.length > 0) {
      const cliBin = resolveCliBin(cliDir);
      process.exit(runCommand(process.execPath, [cliBin, 'review', 'process', ...args]));
    }

    console.log(renderHelp());
    process.exit(0);
  }

  if (profile === 'process') {
    const cliBin = resolveCliBin(cliDir);
    process.exit(runCommand(process.execPath, [cliBin, 'review', 'process', ...args]));
  }

  if (profile === 'lifecyclemodel') {
    fail(
      "Profile 'lifecyclemodel' is not implemented yet. The canonical process path is `tiangong review process`.",
    );
  }

  fail(`Unknown profile: ${profile}`);
}

try {
  main();
} catch (error) {
  if (error instanceof UsageError) {
    console.error(`Error: ${error.message}`);
    console.error('');
    console.error(renderHelp());
    process.exit(2);
  }

  const message = error instanceof Error ? error.message : String(error);
  console.error(`Error: ${message}`);
  process.exit(1);
}
