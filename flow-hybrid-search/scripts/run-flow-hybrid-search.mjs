#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const skillDir = path.resolve(scriptDir, '..');
const workspaceRoot = path.resolve(skillDir, '..', '..');
const defaultCliDir = path.join(workspaceRoot, 'tiangong-lca-cli');
const defaultInputFile = path.join(skillDir, 'assets', 'example-request.json');

function fail(message) {
  console.error(`Error: ${message}`);
  process.exit(2);
}

let cliDir = process.env.TIANGONG_LCA_CLI_DIR?.trim() || defaultCliDir;
let hasInput = false;
let showHelp = false;
const forwardArgs = [];

for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];

  switch (arg) {
    case '--cli-dir':
      if (index + 1 >= process.argv.length) {
        fail('--cli-dir requires a value');
      }
      cliDir = process.argv[index + 1];
      index += 1;
      break;
    case '--input':
      if (index + 1 >= process.argv.length) {
        fail('--input requires a value');
      }
      hasInput = true;
      forwardArgs.push(arg, process.argv[index + 1]);
      index += 1;
      break;
    case '-h':
    case '--help':
      showHelp = true;
      forwardArgs.push(arg);
      break;
    default:
      if (arg.startsWith('--cli-dir=')) {
        cliDir = arg.slice('--cli-dir='.length);
        break;
      }
      if (arg.startsWith('--input=')) {
        hasInput = true;
      }
      forwardArgs.push(arg);
      break;
  }
}

const cliBin = path.join(cliDir, 'bin', 'tiangong.js');
if (!existsSync(cliBin)) {
  fail(`Cannot find TianGong CLI at ${cliBin}. Set TIANGONG_LCA_CLI_DIR or pass --cli-dir.`);
}

const commandArgs = [cliBin, 'search', 'flow'];
if (!showHelp && !hasInput) {
  commandArgs.push('--input', defaultInputFile);
}
commandArgs.push(...forwardArgs);

const result = spawnSync(process.execPath, commandArgs, {
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
