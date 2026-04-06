#!/usr/bin/env node
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import {
  normalizeCliRuntimeArgs,
  runTiangongCommand,
} from '../../scripts/lib/cli-launcher.mjs';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const skillDir = path.resolve(scriptDir, '..');
const defaultInputFile = path.join(skillDir, 'assets', 'example-request.json');

function fail(message) {
  console.error(`Error: ${message}`);
  process.exit(2);
}

let hasInput = false;
let showHelp = false;
const forwardArgs = [];
let cliDir = null;
let args = [];

try {
  ({ cliDir, args } = normalizeCliRuntimeArgs(process.argv.slice(2)));
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  fail(message);
}

for (let index = 0; index < args.length; index += 1) {
  const arg = args[index];

  switch (arg) {
    case '--input':
      if (index + 1 >= args.length) {
        fail('--input requires a value');
      }
      hasInput = true;
      forwardArgs.push(arg, args[index + 1]);
      index += 1;
      break;
    case '-h':
    case '--help':
      showHelp = true;
      forwardArgs.push(arg);
      break;
    default:
      if (arg.startsWith('--input=')) {
        hasInput = true;
      }
      forwardArgs.push(arg);
      break;
  }
}

const commandArgs = ['search', 'lifecyclemodel'];
if (!showHelp && !hasInput) {
  commandArgs.push('--input', defaultInputFile);
}
commandArgs.push(...forwardArgs);

try {
  process.exit(runTiangongCommand(commandArgs, { cliDir }));
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  fail(message);
}
