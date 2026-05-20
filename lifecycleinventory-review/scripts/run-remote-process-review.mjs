#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import {
  buildTiangongInvocation,
  normalizeCliRuntimeArgs,
  renderShellCommand,
  withCliRuntimeEnv,
} from '../../scripts/lib/cli-launcher.mjs';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '..', '..');
const runReviewScript = path.join(scriptDir, 'run-review.mjs');

class UsageError extends Error {}

function usage() {
  return `Usage:
  node scripts/run-remote-process-review.mjs --out-dir <dir> [wrapper-options] --list [process-list args] [--review [review args]]

Wrapper options:
  --out-dir <dir>         Artifact root for the frozen snapshot and review outputs
  --review-out-dir <dir>  Override the nested review output dir (default: <out-dir>/review)
  --report-file <file>    Reuse an existing tiangong process list --json report instead of fetching
  --json                  Print a compact wrapper summary JSON
  --cli-dir <dir>         Use a local TianGong CLI checkout instead of the published package
  -h, --help

Forwarding modes:
  --list                  All following args are forwarded to tiangong process list
  --review                All following args are forwarded to node scripts/run-review.mjs --profile process

Examples:
  node scripts/run-remote-process-review.mjs \\
    --out-dir /abs/path/artifacts/process-review \\
    --list --state-code 0 --state-code 100 --all

  node scripts/run-remote-process-review.mjs \\
    --out-dir /abs/path/artifacts/process-review \\
    --list --user-id <owner> --state-code 0 --all \\
    --review --logic-version 2026-04-14
`.trim();
}

function fail(message) {
  throw new UsageError(message);
}

function ensureDir(dirPath) {
  mkdirSync(dirPath, { recursive: true });
}

function writeText(filePath, text) {
  ensureDir(path.dirname(filePath));
  writeFileSync(filePath, text, 'utf8');
}

function writeJson(filePath, value) {
  writeText(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function writeJsonl(filePath, rows) {
  const lines = rows.map((row) => JSON.stringify(row)).join('\n');
  writeText(filePath, `${lines}${rows.length ? '\n' : ''}`);
}

function parseArgs(rawArgs) {
  const { cliDir, args } = normalizeCliRuntimeArgs(rawArgs, { repoRoot });
  const options = {
    cliDir,
    outDir: null,
    reviewOutDir: null,
    reportFile: null,
    json: false,
    listArgs: [],
    reviewArgs: [],
  };

  let mode = 'wrapper';
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];

    if (arg === '-h' || arg === '--help') {
      if (mode === 'wrapper') {
        console.log(usage());
        process.exit(0);
      }
      if (mode === 'list') {
        options.listArgs.push(arg);
        continue;
      }
      options.reviewArgs.push(arg);
      continue;
    }

    if (mode === 'wrapper') {
      if (arg === '--list') {
        mode = 'list';
        continue;
      }
      if (arg === '--review') {
        mode = 'review';
        continue;
      }
      if (arg === '--out-dir') {
        options.outDir = path.resolve(args[index + 1] ?? '');
        index += 1;
        continue;
      }
      if (arg.startsWith('--out-dir=')) {
        options.outDir = path.resolve(arg.slice('--out-dir='.length));
        continue;
      }
      if (arg === '--review-out-dir') {
        options.reviewOutDir = path.resolve(args[index + 1] ?? '');
        index += 1;
        continue;
      }
      if (arg.startsWith('--review-out-dir=')) {
        options.reviewOutDir = path.resolve(arg.slice('--review-out-dir='.length));
        continue;
      }
      if (arg === '--report-file') {
        options.reportFile = path.resolve(args[index + 1] ?? '');
        index += 1;
        continue;
      }
      if (arg.startsWith('--report-file=')) {
        options.reportFile = path.resolve(arg.slice('--report-file='.length));
        continue;
      }
      if (arg === '--json') {
        options.json = true;
        continue;
      }

      fail(`Unknown wrapper option: ${arg}`);
    }

    if (mode === 'list') {
      options.listArgs.push(arg);
      continue;
    }

    options.reviewArgs.push(arg);
  }

  if (!options.outDir) {
    fail('Missing required --out-dir.');
  }
  if (!options.reportFile && options.listArgs.length === 0) {
    fail('Provide either --report-file <file> or a --list forwarding block.');
  }

  return {
    ...options,
    reviewOutDir: options.reviewOutDir ?? path.join(options.outDir, 'review'),
  };
}

function runProcessList(cliDir, listArgs) {
  const invocation = buildTiangongInvocation(['process', 'list', ...listArgs, '--json'], {
    repoRoot,
    cliDir,
  });
  const result = spawnSync(invocation.command, invocation.args, {
    cwd: repoRoot,
    env: withCliRuntimeEnv(process.env, cliDir),
    stdio: 'pipe',
    encoding: 'utf8',
  });

  if (result.error) {
    throw result.error;
  }
  if (typeof result.status === 'number' && result.status !== 0) {
    const stderr = result.stderr?.trim() || result.stdout?.trim() || `exit code ${result.status}`;
    throw new Error(`Process list failed: ${stderr}`);
  }
  if (!result.stdout?.trim()) {
    throw new Error(
      `Process list returned no stdout for ${renderShellCommand(invocation.command, invocation.args)}`,
    );
  }

  return {
    command: renderShellCommand(invocation.command, invocation.args),
    stdout: result.stdout,
    stderr: result.stderr ?? '',
  };
}

function runReview(cliDir, rowsFile, reviewOutDir, reviewArgs) {
  const command = [
    runReviewScript,
    '--profile',
    'process',
    '--rows-file',
    rowsFile,
    '--out-dir',
    reviewOutDir,
    ...reviewArgs,
  ];
  const result = spawnSync(process.execPath, command, {
    cwd: repoRoot,
    env: withCliRuntimeEnv(process.env, cliDir),
    stdio: 'pipe',
    encoding: 'utf8',
  });

  if (result.error) {
    throw result.error;
  }
  if (typeof result.status === 'number' && result.status !== 0) {
    const stderr = result.stderr?.trim() || result.stdout?.trim() || `exit code ${result.status}`;
    throw new Error(`Process review failed: ${stderr}`);
  }

  return {
    command: renderShellCommand(process.execPath, command),
    stdout: result.stdout ?? '',
    stderr: result.stderr ?? '',
  };
}

function freezeReport(reportFile, snapshotReportFile) {
  if (!existsSync(reportFile)) {
    throw new Error(`Report file not found: ${reportFile}`);
  }
  ensureDir(path.dirname(snapshotReportFile));
  copyFileSync(reportFile, snapshotReportFile);
  return readFileSync(snapshotReportFile, 'utf8');
}

function parseRows(snapshotText, snapshotReportFile) {
  const parsed = JSON.parse(snapshotText);
  if (!parsed || typeof parsed !== 'object' || !Array.isArray(parsed.rows)) {
    throw new Error(`Expected a tiangong process list report with a rows array: ${snapshotReportFile}`);
  }
  return parsed.rows;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputsDir = path.join(args.outDir, 'inputs');
  const logsDir = path.join(args.outDir, 'logs');
  const outputsDir = path.join(args.outDir, 'outputs');
  ensureDir(inputsDir);
  ensureDir(logsDir);
  ensureDir(outputsDir);
  ensureDir(args.reviewOutDir);

  const snapshotReportFile = path.join(inputsDir, 'process-list-report.json');
  const snapshotRowsFile = path.join(inputsDir, 'processes.snapshot.rows.jsonl');

  let snapshotText = '';
  let listSummary = null;
  if (args.reportFile) {
    snapshotText = freezeReport(args.reportFile, snapshotReportFile);
    listSummary = {
      mode: 'reused_report',
      source_report: args.reportFile,
      frozen_report: snapshotReportFile,
    };
  } else {
    const listRun = runProcessList(args.cliDir, args.listArgs);
    snapshotText = listRun.stdout;
    writeText(snapshotReportFile, snapshotText.endsWith('\n') ? snapshotText : `${snapshotText}\n`);
    writeText(path.join(logsDir, 'process-list.stderr.log'), listRun.stderr);
    listSummary = {
      mode: 'fresh_fetch',
      command: listRun.command,
      frozen_report: snapshotReportFile,
    };
  }

  const rows = parseRows(snapshotText, snapshotReportFile);
  writeJsonl(snapshotRowsFile, rows);

  const reviewRun = runReview(args.cliDir, snapshotReportFile, args.reviewOutDir, args.reviewArgs);
  writeText(path.join(logsDir, 'review-process.stdout.log'), reviewRun.stdout);
  writeText(path.join(logsDir, 'review-process.stderr.log'), reviewRun.stderr);

  const summary = {
    generated_at_utc: new Date().toISOString(),
    status: 'completed_remote_process_review_wrapper',
    snapshot: {
      row_count: rows.length,
      report_file: snapshotReportFile,
      rows_file: snapshotRowsFile,
      ...listSummary,
    },
    review: {
      out_dir: args.reviewOutDir,
      command: reviewRun.command,
    },
  };
  writeJson(path.join(outputsDir, 'remote-process-review-summary.json'), summary);

  if (args.json) {
    process.stdout.write(`${JSON.stringify(summary)}\n`);
    return;
  }

  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
}

try {
  main();
} catch (error) {
  if (error instanceof UsageError) {
    console.error(`Error: ${error.message}`);
    console.error('');
    console.error(usage());
    process.exit(2);
  }

  const message = error instanceof Error ? error.message : String(error);
  console.error(`Error: ${message}`);
  process.exit(1);
}
