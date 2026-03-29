#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const skillDir = path.resolve(scriptDir, '..');
const workspaceRoot = path.resolve(skillDir, '..', '..');
const defaultCliDir = path.join(workspaceRoot, 'tiangong-lca-cli');
const legacyTargetScript = path.join(scriptDir, 'origin', 'process_from_flow_langgraph.py');
const canonicalSubcommands = new Set(['auto-build', 'resume-build', 'publish-build', 'batch-build']);

function fail(message) {
  console.error(`Error: ${message}`);
  process.exit(2);
}

function renderHelp() {
  return `Usage:
  node scripts/run-process-automated-builder.mjs <auto-build|resume-build|publish-build|batch-build> [options]
  node scripts/run-process-automated-builder.mjs legacy [legacy-wrapper-options] [-- python-args]

Wrapper options:
  --cli-dir <dir>           Override the tiangong-lca-cli repository path

Canonical commands:
  auto-build                Delegate to tiangong process auto-build
  resume-build              Delegate to tiangong process resume-build
  publish-build             Delegate to tiangong process publish-build
  batch-build               Delegate to tiangong process batch-build

auto-build compatibility options:
  --request <file>          Alias for the CLI's --input <file>
  --flow-file <path>        Build a temporary CLI request from a reference flow file
  --flow-json <json>        Build a temporary CLI request from inline flow JSON
  --flow-stdin              Build a temporary CLI request from stdin flow JSON
  --operation <mode>        produce | treat (default: produce)

Legacy path:
  - use the explicit 'legacy' subcommand for standalone Python/LangGraph modes
  - any invocation that does not start with a canonical subcommand is treated as legacy
  - scripts/run-process-automated-builder.sh is now a compatibility shim to this Node wrapper

Examples:
  node scripts/run-process-automated-builder.mjs auto-build --flow-file /abs/path/reference-flow.json --operation produce --json
  node scripts/run-process-automated-builder.mjs resume-build --run-id <run_id> --json
  node scripts/run-process-automated-builder.mjs publish-build --run-id <run_id> --json
  node scripts/run-process-automated-builder.mjs batch-build --input /abs/path/batch-request.json --json
  node scripts/run-process-automated-builder.mjs legacy --mode workflow --flow-file /abs/path/reference-flow.json -- --operation produce
`.trim();
}

function resolveCliBin(cliDir) {
  const cliBin = path.join(cliDir, 'bin', 'tiangong.js');
  if (!existsSync(cliBin)) {
    fail(`Cannot find TianGong CLI at ${cliBin}. Set TIANGONG_LCA_CLI_DIR or pass --cli-dir.`);
  }
  return cliBin;
}

function resolveDefaultPython() {
  const explicit = process.env.PAB_PYTHON_BIN?.trim();
  if (explicit) {
    return explicit;
  }

  const candidates = process.platform === 'win32'
    ? [path.join(skillDir, '.venv', 'Scripts', 'python.exe')]
    : [path.join(skillDir, '.venv', 'bin', 'python')];

  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }

  return process.platform === 'win32' ? 'python' : 'python3';
}

function runCommand(command, args, options = {}) {
  const result = spawnSync(command, args, {
    stdio: 'inherit',
    ...options,
  });

  if (result.error) {
    fail(`Failed to execute ${command}: ${result.error.message}`);
  }
  if (typeof result.status === 'number') {
    process.exit(result.status);
  }
  if (result.signal) {
    fail(`${command} terminated with signal ${result.signal}.`);
  }
  process.exit(1);
}

function writeTempJsonFile(prefix, value) {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), prefix));
  const filePath = path.join(tempDir, 'payload.json');
  writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  return {
    tempDir,
    filePath,
  };
}

function normalizeTopLevelArgs(rawArgs) {
  let cliDir = process.env.TIANGONG_LCA_CLI_DIR?.trim() || defaultCliDir;
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

    args.push(arg);
  }

  return {
    cliDir,
    args,
  };
}

function normalizeCliInputArgs(args) {
  let inputPath = null;
  let inputAliasSeen = false;
  const forwardArgs = [];

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];

    if (arg === '--request') {
      if (index + 1 >= args.length) {
        fail('--request requires a value');
      }
      inputPath = args[index + 1];
      inputAliasSeen = true;
      index += 1;
      continue;
    }

    if (arg.startsWith('--request=')) {
      inputPath = arg.slice('--request='.length);
      inputAliasSeen = true;
      continue;
    }

    if (arg === '--input') {
      if (index + 1 >= args.length) {
        fail('--input requires a value');
      }
      inputPath = args[index + 1];
      forwardArgs.push(arg, args[index + 1]);
      index += 1;
      continue;
    }

    if (arg.startsWith('--input=')) {
      inputPath = arg.slice('--input='.length);
      forwardArgs.push(arg);
      continue;
    }

    forwardArgs.push(arg);
  }

  if (inputAliasSeen && inputPath) {
    forwardArgs.unshift('--input', inputPath);
  }

  return {
    inputPath,
    forwardArgs,
  };
}

function runCanonicalAutoBuild(cliBin, args) {
  let inputPath = null;
  let flowFile = null;
  let flowJson = null;
  let flowFromStdin = false;
  let operation = 'produce';
  let showHelp = false;
  const forwardArgs = [];
  const tempDirs = [];

  try {
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
        case '--flow-file':
          if (index + 1 >= args.length) {
            fail('--flow-file requires a value');
          }
          flowFile = args[index + 1];
          index += 1;
          break;
        case '--flow-json':
          if (index + 1 >= args.length) {
            fail('--flow-json requires a value');
          }
          flowJson = args[index + 1];
          index += 1;
          break;
        case '--flow-stdin':
          flowFromStdin = true;
          break;
        case '--operation':
          if (index + 1 >= args.length) {
            fail('--operation requires a value');
          }
          operation = args[index + 1];
          index += 1;
          break;
        case '-h':
        case '--help':
          showHelp = true;
          forwardArgs.push(arg);
          break;
        default:
          if (arg.startsWith('--request=')) {
            inputPath = arg.slice('--request='.length);
            break;
          }
          if (arg.startsWith('--input=')) {
            inputPath = arg.slice('--input='.length);
            forwardArgs.push(arg);
            break;
          }
          if (arg.startsWith('--flow-file=')) {
            flowFile = arg.slice('--flow-file='.length);
            break;
          }
          if (arg.startsWith('--flow-json=')) {
            flowJson = arg.slice('--flow-json='.length);
            break;
          }
          if (arg.startsWith('--operation=')) {
            operation = arg.slice('--operation='.length);
            break;
          }
          forwardArgs.push(arg);
          break;
      }
    }

    if (showHelp) {
      runCommand(process.execPath, [cliBin, 'process', 'auto-build', '--help']);
    }

    const flowInputCount = [inputPath ? 1 : 0, flowFile ? 1 : 0, flowJson ? 1 : 0, flowFromStdin ? 1 : 0]
      .reduce((sum, value) => sum + value, 0);
    if (flowInputCount === 0) {
      fail('Missing input. Use --input/--request or one of --flow-file/--flow-json/--flow-stdin.');
    }
    if (inputPath && flowInputCount > 1) {
      fail('Use either --input/--request or flow wrapper options, not both.');
    }
    if (flowFile && flowJson) {
      fail('--flow-file and --flow-json are mutually exclusive.');
    }
    if (flowFile && flowFromStdin) {
      fail('--flow-file and --flow-stdin are mutually exclusive.');
    }
    if (flowJson && flowFromStdin) {
      fail('--flow-json and --flow-stdin are mutually exclusive.');
    }
    if (operation !== 'produce' && operation !== 'treat') {
      fail("--operation must be 'produce' or 'treat'.");
    }

    if (!inputPath) {
      let resolvedFlowPath = flowFile ? path.resolve(flowFile) : null;

      if (flowJson || flowFromStdin) {
        const flowPayload = flowJson ?? readFileSync(0, 'utf8');
        const tempFlow = writeTempJsonFile('tg-pab-flow-', JSON.parse(flowPayload));
        tempDirs.push(tempFlow.tempDir);
        resolvedFlowPath = tempFlow.filePath;
      }

      if (!resolvedFlowPath || !existsSync(resolvedFlowPath)) {
        fail(`Flow file not found: ${resolvedFlowPath ?? '(missing flow input)'}`);
      }

      const tempRequest = writeTempJsonFile('tg-pab-request-', {
        flow_file: resolvedFlowPath,
        operation,
      });
      tempDirs.push(tempRequest.tempDir);
      inputPath = tempRequest.filePath;
    }

    runCommand(process.execPath, [
      cliBin,
      'process',
      'auto-build',
      '--input',
      inputPath,
      ...forwardArgs,
    ]);
  } finally {
    for (const tempDir of tempDirs) {
      rmSync(tempDir, { recursive: true, force: true });
    }
  }
}

function runCanonicalInputCommand(cliBin, subcommand, args) {
  const { forwardArgs } = normalizeCliInputArgs(args);
  runCommand(process.execPath, [cliBin, 'process', subcommand, ...forwardArgs]);
}

function hasFlag(flag, values) {
  return values.includes(flag);
}

function runLegacyMode(args) {
  let mode = 'workflow';
  let flowFile = '';
  let flowJson = '';
  let flowFromStdin = false;
  let pythonBin = resolveDefaultPython();
  const forwardArgs = [];

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];

    switch (arg) {
      case '--mode':
        if (index + 1 >= args.length) {
          fail('--mode requires a value');
        }
        mode = args[index + 1];
        index += 1;
        break;
      case '--flow-file':
        if (index + 1 >= args.length) {
          fail('--flow-file requires a value');
        }
        flowFile = args[index + 1];
        index += 1;
        break;
      case '--flow-json':
        if (index + 1 >= args.length) {
          fail('--flow-json requires a value');
        }
        flowJson = args[index + 1];
        index += 1;
        break;
      case '--flow-stdin':
        flowFromStdin = true;
        break;
      case '--python-bin':
        if (index + 1 >= args.length) {
          fail('--python-bin requires a value');
        }
        pythonBin = args[index + 1];
        index += 1;
        break;
      case '-h':
      case '--help':
        console.error(renderHelp());
        process.exit(0);
      case '--':
        forwardArgs.push(...args.slice(index + 1));
        index = args.length;
        break;
      default:
        if (arg.startsWith('--mode=')) {
          mode = arg.slice('--mode='.length);
          break;
        }
        if (arg.startsWith('--flow-file=')) {
          flowFile = arg.slice('--flow-file='.length);
          break;
        }
        if (arg.startsWith('--flow-json=')) {
          flowJson = arg.slice('--flow-json='.length);
          break;
        }
        if (arg.startsWith('--python-bin=')) {
          pythonBin = arg.slice('--python-bin='.length);
          break;
        }
        forwardArgs.push(arg);
        break;
    }
  }

  if (mode !== 'workflow' && mode !== 'langgraph') {
    fail(`Invalid --mode: ${mode}`);
  }
  if (flowFile && flowJson) {
    fail('--flow-file and --flow-json are mutually exclusive.');
  }
  if (flowFile && flowFromStdin) {
    fail('--flow-file and --flow-stdin are mutually exclusive.');
  }
  if (flowJson && flowFromStdin) {
    fail('--flow-json and --flow-stdin are mutually exclusive.');
  }

  const tempDirs = [];

  try {
    if (flowJson || flowFromStdin) {
      const flowPayload = flowJson ?? readFileSync(0, 'utf8');
      const tempFlow = writeTempJsonFile('tg-pab-legacy-flow-', JSON.parse(flowPayload));
      tempDirs.push(tempFlow.tempDir);
      flowFile = tempFlow.filePath;
    }

    if (flowFile && !existsSync(flowFile)) {
      fail(`Flow file not found: ${flowFile}`);
    }

    if (hasFlag('--flow', forwardArgs) && flowFile) {
      fail('Do not pass --flow in forwarded args when using wrapper flow input options.');
    }

    let requireFlow = true;
    let langgraphSubcommand = '';
    const forwardedHasFlow = hasFlag('--flow', forwardArgs);

    if (mode === 'langgraph') {
      if (forwardArgs.length > 0) {
        const firstArg = forwardArgs[0];
        if (firstArg === 'flow-auto-build' || firstArg === 'process-update') {
          langgraphSubcommand = firstArg;
          requireFlow = false;
        }
      }
      if (
        hasFlag('--resume', forwardArgs) ||
        hasFlag('--publish-only', forwardArgs) ||
        hasFlag('--cleanup-only', forwardArgs)
      ) {
        requireFlow = false;
      }
    } else {
      langgraphSubcommand = 'workflow';
    }

    if (requireFlow && !flowFile && !forwardedHasFlow) {
      fail('Missing flow input. Use --flow-file/--flow-json/--flow-stdin (or forward --flow).');
    }

    if (langgraphSubcommand && langgraphSubcommand !== 'workflow' && flowFile) {
      fail(
        `Flow input is not used for langgraph subcommands (${langgraphSubcommand}); remove --flow-file/--flow-json/--flow-stdin.`,
      );
    }

    const flowArg = !langgraphSubcommand || langgraphSubcommand === 'workflow'
      ? (flowFile ? ['--flow', flowFile] : [])
      : [];

    const env = {
      ...process.env,
      PYTHONPATH: process.env.PYTHONPATH
        ? `${skillDir}${path.delimiter}${process.env.PYTHONPATH}`
        : skillDir,
    };

    if (langgraphSubcommand === 'workflow') {
      runCommand(pythonBin, [legacyTargetScript, langgraphSubcommand, ...flowArg, ...forwardArgs], {
        env,
      });
    }

    if (langgraphSubcommand) {
      runCommand(pythonBin, [legacyTargetScript, ...forwardArgs], {
        env,
      });
    }

    runCommand(pythonBin, [legacyTargetScript, ...flowArg, ...forwardArgs], {
      env,
    });
  } finally {
    for (const tempDir of tempDirs) {
      rmSync(tempDir, { recursive: true, force: true });
    }
  }
}

function main() {
  const { cliDir, args } = normalizeTopLevelArgs(process.argv.slice(2));

  if (args.length === 0) {
    console.error(renderHelp());
    process.exit(0);
  }

  const subcommand = args[0];

  if (subcommand === 'help' || subcommand === '-h' || subcommand === '--help') {
    console.error(renderHelp());
    process.exit(0);
  }

  if (subcommand === 'legacy') {
    runLegacyMode(args.slice(1));
  }

  const cliBin = resolveCliBin(cliDir);

  if (!canonicalSubcommands.has(subcommand)) {
    runLegacyMode(args);
  }

  const commandArgs = args.slice(1);

  switch (subcommand) {
    case 'auto-build':
      runCanonicalAutoBuild(cliBin, commandArgs);
      return;
    case 'resume-build':
      runCommand(process.execPath, [cliBin, 'process', 'resume-build', ...commandArgs]);
      return;
    case 'publish-build':
      runCommand(process.execPath, [cliBin, 'process', 'publish-build', ...commandArgs]);
      return;
    case 'batch-build':
      runCanonicalInputCommand(cliBin, 'batch-build', commandArgs);
      return;
    default:
      fail(`Unknown subcommand: ${subcommand}`);
  }
}

main();
