#!/usr/bin/env node
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import {
  normalizeCliRuntimeArgs,
  publishedCliCommand,
  runTiangongCommand,
} from '../../scripts/lib/cli-launcher.mjs';

class UsageError extends Error {}

const canonicalSubcommands = new Set(['auto-build', 'resume-build', 'publish-build', 'batch-build']);

function fail(message) {
  throw new UsageError(message);
}

function renderHelp() {
  return `Usage:
  node scripts/run-process-automated-builder.mjs <auto-build|resume-build|publish-build|batch-build> [options]

Wrapper options:
  --cli-dir <dir>           Override the published CLI and use a local tiangong-lca-cli repository path

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

Runtime:
  default                  ${publishedCliCommand}
  local override           --cli-dir /path/to/tiangong-lca-cli or TIANGONG_LCA_CLI_DIR

Notes:
  - the wrapper uses the CLI only; older alternate runtimes are not part of the supported path
  - there is no shell compatibility shim; call this .mjs entrypoint directly
  - auto-build and batch-build require --out-dir
  - resume-build and publish-build should use --run-dir so the output root stays explicit

Examples:
  node scripts/run-process-automated-builder.mjs auto-build --flow-file /abs/path/reference-flow.json --operation produce --out-dir /abs/path/artifacts/<case_slug>/process_from_flow/<run_id> --json
  node scripts/run-process-automated-builder.mjs resume-build --run-dir /abs/path/artifacts/<case_slug>/process_from_flow/<run_id> --run-id <run_id> --json
  node scripts/run-process-automated-builder.mjs publish-build --run-dir /abs/path/artifacts/<case_slug>/process_from_flow/<run_id> --run-id <run_id> --json
  node scripts/run-process-automated-builder.mjs batch-build --input /abs/path/batch-request.json --out-dir /abs/path/artifacts/<case_slug>/process_batch/<batch_id> --json
`.trim();
}

function parseJsonText(rawText, sourceLabel) {
  try {
    return JSON.parse(rawText);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    fail(`Invalid JSON from ${sourceLabel}: ${message}`);
  }
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

function hasFlag(flag, values) {
  return values.some((value) => value === flag || value.startsWith(`${flag}=`));
}

function requireFlag(flag, values, message) {
  if (!hasFlag(flag, values)) {
    fail(message);
  }
}

function normalizeCliInputArgs(args) {
  let inputPath = null;
  const forwardArgs = [];

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];

    if (arg === '--request') {
      if (index + 1 >= args.length) {
        fail('--request requires a value');
      }
      if (inputPath && inputPath !== args[index + 1]) {
        fail('Use only one of --request or --input.');
      }
      inputPath = args[index + 1];
      index += 1;
      continue;
    }

    if (arg.startsWith('--request=')) {
      const value = arg.slice('--request='.length);
      if (inputPath && inputPath !== value) {
        fail('Use only one of --request or --input.');
      }
      inputPath = value;
      continue;
    }

    if (arg === '--input') {
      if (index + 1 >= args.length) {
        fail('--input requires a value');
      }
      const value = args[index + 1];
      if (inputPath && inputPath !== value) {
        fail('Use only one of --request or --input.');
      }
      inputPath = value;
      forwardArgs.push('--input', value);
      index += 1;
      continue;
    }

    if (arg.startsWith('--input=')) {
      const value = arg.slice('--input='.length);
      if (inputPath && inputPath !== value) {
        fail('Use only one of --request or --input.');
      }
      inputPath = value;
      forwardArgs.push(arg);
      continue;
    }

    forwardArgs.push(arg);
  }

  if (inputPath && !hasFlag('--input', forwardArgs)) {
    forwardArgs.unshift('--input', inputPath);
  }

  return {
    forwardArgs,
  };
}

function runCanonicalAutoBuild(cliDir, args) {
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
          if (inputPath && inputPath !== args[index + 1]) {
            fail('Use only one of --request, --input, or flow wrapper options.');
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
          break;
        default:
          if (arg.startsWith('--request=')) {
            const value = arg.slice('--request='.length);
            if (inputPath && inputPath !== value) {
              fail('Use only one of --request, --input, or flow wrapper options.');
            }
            inputPath = value;
            break;
          }
          if (arg.startsWith('--input=')) {
            const value = arg.slice('--input='.length);
            if (inputPath && inputPath !== value) {
              fail('Use only one of --request, --input, or flow wrapper options.');
            }
            inputPath = value;
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
      return runTiangongCommand(['process', 'auto-build', '--help'], { cliDir });
    }

    const inputSourceCount = [inputPath ? 1 : 0, flowFile ? 1 : 0, flowJson ? 1 : 0, flowFromStdin ? 1 : 0].reduce(
      (sum, value) => sum + value,
      0,
    );

    if (inputSourceCount === 0) {
      fail('Missing input. Use --input/--request or one of --flow-file/--flow-json/--flow-stdin.');
    }
    if (inputPath && inputSourceCount > 1) {
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

    requireFlag(
      '--out-dir',
      forwardArgs,
      "auto-build requires --out-dir <dir>. Choose an explicit output path, for example /abs/path/artifacts/<case_slug>/.",
    );

    if (!inputPath) {
      let resolvedFlowPath = flowFile ? path.resolve(flowFile) : null;

      if (flowJson || flowFromStdin) {
        const flowPayload = flowJson ?? readFileSync(0, 'utf8');
        const tempFlow = writeTempJsonFile(
          'tg-pab-flow-',
          parseJsonText(flowPayload, flowJson ? '--flow-json' : 'stdin'),
        );
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

    return runTiangongCommand(['process', 'auto-build', '--input', inputPath, ...forwardArgs], {
      cliDir,
    });
  } finally {
    for (const tempDir of tempDirs) {
      rmSync(tempDir, { recursive: true, force: true });
    }
  }
}

function runCanonicalInputCommand(cliDir, subcommand, args) {
  const { forwardArgs } = normalizeCliInputArgs(args);
  requireFlag(
    '--out-dir',
    forwardArgs,
    `${subcommand} requires --out-dir <dir>. Choose an explicit output path, for example /abs/path/artifacts/<case_slug>/.`,
  );
  return runTiangongCommand(['process', subcommand, ...forwardArgs], { cliDir });
}

function main() {
  const { cliDir, args } = normalizeCliRuntimeArgs(process.argv.slice(2));

  if (args.length === 0) {
    console.error(renderHelp());
    return 0;
  }

  const subcommand = args[0];

  if (subcommand === 'help' || subcommand === '-h' || subcommand === '--help') {
    console.error(renderHelp());
    return 0;
  }
  if (!canonicalSubcommands.has(subcommand)) {
    fail(
      `Unknown subcommand: ${subcommand}. The legacy Python workflow was removed; use only the documented process CLI commands.`,
    );
  }

  const commandArgs = args.slice(1);

  switch (subcommand) {
    case 'auto-build':
      return runCanonicalAutoBuild(cliDir, commandArgs);
    case 'resume-build':
      requireFlag(
        '--run-dir',
        commandArgs,
        'resume-build requires --run-dir <dir>. Use an explicit run directory under /abs/path/artifacts/<case_slug>/... and pass --run-id only as an optional consistency check.',
      );
      return runTiangongCommand(['process', 'resume-build', ...commandArgs], { cliDir });
    case 'publish-build':
      requireFlag(
        '--run-dir',
        commandArgs,
        'publish-build requires --run-dir <dir>. Use an explicit run directory under /abs/path/artifacts/<case_slug>/... and pass --run-id only as an optional consistency check.',
      );
      return runTiangongCommand(['process', 'publish-build', ...commandArgs], { cliDir });
    case 'batch-build':
      return runCanonicalInputCommand(cliDir, 'batch-build', commandArgs);
    default:
      fail(`Unknown subcommand: ${subcommand}`);
  }
}

try {
  process.exitCode = main();
} catch (error) {
  if (error instanceof UsageError) {
    console.error(`Error: ${error.message}`);
    process.exitCode = 2;
  } else if (error instanceof Error) {
    console.error(`Error: ${error.message}`);
    process.exitCode = 1;
  } else {
    console.error(`Error: ${String(error)}`);
    process.exitCode = 1;
  }
}
