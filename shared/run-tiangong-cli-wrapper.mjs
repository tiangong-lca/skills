#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

function fail(message) {
  process.stderr.write(`Error: ${message}\n`);
  process.exit(2);
}

function buildContext(importMetaUrl) {
  const scriptDir = path.dirname(fileURLToPath(importMetaUrl));
  const skillDir = path.resolve(scriptDir, '..');
  const workspaceRoot = path.resolve(skillDir, '..', '..');

  return {
    scriptDir,
    skillDir,
    workspaceRoot,
    defaultCliDir: path.join(workspaceRoot, 'tiangong-lca-cli'),
  };
}

function resolveCliBin(cliDir) {
  const cliBin = path.join(cliDir, 'bin', 'tiangong.js');
  if (!existsSync(cliBin)) {
    fail(`Cannot find TianGong CLI at ${cliBin}. Set TIANGONG_LCA_CLI_DIR or pass --cli-dir.`);
  }
  return cliBin;
}

function runCli(cliBin, args) {
  const result = spawnSync(process.execPath, [cliBin, ...args], {
    stdio: 'inherit',
  });

  if (result.error) {
    fail(result.error.message);
  }

  process.exit(result.status ?? 1);
}

function parseSimpleWrapperArgs(argv) {
  let cliDir = null;
  let hasInput = false;
  let showHelp = false;
  const forwardArgs = [];

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

    if (token === '--input') {
      const value = argv[index + 1];
      if (!value) {
        fail('--input requires a value');
      }
      hasInput = true;
      forwardArgs.push(token, value);
      index += 1;
      continue;
    }

    if (token?.startsWith('--input=')) {
      hasInput = true;
      forwardArgs.push(token);
      continue;
    }

    if (token === '-h' || token === '--help') {
      showHelp = true;
      forwardArgs.push(token);
      continue;
    }

    forwardArgs.push(token);
  }

  return {
    cliDir,
    hasInput,
    showHelp,
    forwardArgs,
  };
}

export function runSimpleInputWrapper(options) {
  const context = buildContext(options.importMetaUrl);
  const parsed = parseSimpleWrapperArgs(options.argv);
  const cliDir = parsed.cliDir ?? process.env.TIANGONG_LCA_CLI_DIR ?? context.defaultCliDir;
  const cliBin = resolveCliBin(cliDir);
  const args = [...options.command];

  if (!parsed.showHelp && !parsed.hasInput && options.defaultInputFile) {
    args.push('--input', options.defaultInputFile);
  }

  args.push(...parsed.forwardArgs);
  runCli(cliBin, args);
}

export { buildContext, fail, parseSimpleWrapperArgs, resolveCliBin, runCli };
