#!/usr/bin/env node
import process from 'node:process';
import {
  normalizeCliRuntimeArgs,
  publishedCliCommand,
  runTiangongCommand,
} from '../../scripts/lib/cli-launcher.mjs';

class UsageError extends Error {}

function fail(message) {
  throw new UsageError(message);
}

function renderHelp() {
  return `Usage:
  node scripts/run-review.mjs [--profile <process|lifecyclemodel>] [options]

Wrapper options:
  --profile <name>         process | lifecyclemodel (default: process)
  --cli-dir <dir>          Override the published CLI and use a local tiangong-lca-cli repository path

Profiles:
  process                  Delegate to tiangong review process
  lifecyclemodel           Delegate to tiangong review lifecyclemodel

Runtime:
  default                  ${publishedCliCommand}
  local override           --cli-dir /path/to/tiangong-lca-cli or TIANGONG_LCA_CLI_DIR

Examples:
  node scripts/run-review.mjs --profile process --run-root /path/to/artifacts/process_from_flow/<run_id> --run-id <run_id> --out-dir /abs/path/review
  node scripts/run-review.mjs --profile process --run-root /path/to/artifacts/process_from_flow/<run_id> --run-id <run_id> --out-dir /abs/path/review --enable-llm
  node scripts/run-review.mjs --profile lifecyclemodel --run-dir /path/to/artifacts/lifecyclemodel_auto_build/<run_id> --out-dir /abs/path/lifecyclemodel-review
`.trim();
}

function normalizeArgs(rawArgs) {
  const { cliDir, args } = normalizeCliRuntimeArgs(rawArgs);
  let profile = 'process';
  const forwardedArgs = [];

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === '--profile') {
      if (index + 1 >= args.length) {
        fail('--profile requires a value');
      }
      profile = args[index + 1];
      index += 1;
      continue;
    }

    if (arg.startsWith('--profile=')) {
      profile = arg.slice('--profile='.length);
      continue;
    }

    forwardedArgs.push(arg);
  }

  return {
    cliDir,
    profile,
    args: forwardedArgs,
  };
}

function main() {
  const { cliDir, profile, args } = normalizeArgs(process.argv.slice(2));

  if (args.length === 0) {
    console.log(renderHelp());
    process.exit(0);
  }

  if (args.includes('-h') || args.includes('--help')) {
    if (profile === 'process') {
      process.exit(runTiangongCommand(['review', 'process', ...args], { cliDir }));
    }

    if (profile === 'lifecyclemodel') {
      process.exit(runTiangongCommand(['review', 'lifecyclemodel', ...args], { cliDir }));
    }

    console.log(renderHelp());
    process.exit(0);
  }

  if (profile === 'process') {
    process.exit(runTiangongCommand(['review', 'process', ...args], { cliDir }));
  }

  if (profile === 'lifecyclemodel') {
    process.exit(runTiangongCommand(['review', 'lifecyclemodel', ...args], { cliDir }));
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
