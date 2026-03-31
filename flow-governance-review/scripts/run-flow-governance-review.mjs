#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

class UsageError extends Error {}

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const skillDir = path.resolve(scriptDir, "..");
const workspaceRoot = path.resolve(skillDir, "..", "..");
const defaultCliDir = path.join(workspaceRoot, "tiangong-lca-cli");

const cliBackedCommands = new Map([
  ["review-flows", ["review", "flow"]],
  ["flow-get", ["flow", "get"]],
  ["flow-list", ["flow", "list"]],
  ["remediate-flows", ["flow", "remediate"]],
  ["publish-version", ["flow", "publish-version"]],
  ["publish-reviewed-data", ["flow", "publish-reviewed-data"]],
  ["build-flow-alias-map", ["flow", "build-alias-map"]],
  ["scan-process-flow-refs", ["flow", "scan-process-flow-refs"]],
  ["plan-process-flow-repairs", ["flow", "plan-process-flow-repairs"]],
  ["apply-process-flow-repairs", ["flow", "apply-process-flow-repairs"]],
  ["regen-product", ["flow", "regen-product"]],
  ["validate-processes", ["flow", "validate-processes"]],
]);

const removedCommands = new Set([
  "openclaw-entry",
  "openclaw-full-run",
  "run-governance",
  "flow-dedup-candidates",
  "export-openclaw-dedup-review-pack",
  "export-openclaw-ref-review-pack",
  "apply-openclaw-ref-decisions",
  "export-openclaw-text-review-pack",
  "export-openclaw-classification-review-pack",
  "apply-openclaw-text-decisions",
  "apply-openclaw-classification-decisions",
  "validate-openclaw-text-decisions",
  "validate-openclaw-classification-decisions",
]);

function fail(message) {
  throw new UsageError(message);
}

function renderHelp() {
  return `Usage:
  node scripts/run-flow-governance-review.mjs <command> [args...]

Wrapper options:
  --cli-dir <dir>           Override the tiangong-lca-cli repository path

CLI-backed commands:
  review-flows              Delegate to tiangong review flow
  flow-get                  Delegate to tiangong flow get
  flow-list                 Delegate to tiangong flow list
  remediate-flows           Delegate to tiangong flow remediate
  publish-version           Delegate to tiangong flow publish-version
  publish-reviewed-data     Delegate reviewed flow/process local publish preparation to tiangong flow publish-reviewed-data
  build-flow-alias-map      Delegate to tiangong flow build-alias-map
  scan-process-flow-refs    Delegate to tiangong flow scan-process-flow-refs
  plan-process-flow-repairs Delegate to tiangong flow plan-process-flow-repairs
  apply-process-flow-repairs Delegate to tiangong flow apply-process-flow-repairs
  regen-product             Delegate to tiangong flow regen-product
  validate-processes        Delegate to tiangong flow validate-processes

Notes:
  - no shell compatibility shim is kept; call this .mjs entrypoint directly
  - the wrapper is now CLI-only; it no longer exposes any Python fallback path
  - publish-reviewed-data now uses the CLI for both local preparation and commit-time process publish
  - removed OpenClaw / governance orchestration commands must be reintroduced as native tiangong subcommands before use

Examples:
  node scripts/run-flow-governance-review.mjs review-flows --rows-file /abs/path/flows.jsonl --out-dir /abs/path/review
  node scripts/run-flow-governance-review.mjs remediate-flows --input-file /abs/path/invalid-flows.jsonl --out-dir /abs/path/remediation
  node scripts/run-flow-governance-review.mjs publish-version --input-file /abs/path/ready-flows.jsonl --out-dir /abs/path/publish --dry-run
  node scripts/run-flow-governance-review.mjs publish-reviewed-data --flow-rows-file /abs/path/reviewed-flows.jsonl --out-dir /abs/path/publish-reviewed
  node scripts/run-flow-governance-review.mjs build-flow-alias-map --old-flow-file /abs/path/old-flows.jsonl --new-flow-file /abs/path/new-flows.jsonl --out-dir /abs/path/alias-map
  node scripts/run-flow-governance-review.mjs scan-process-flow-refs --processes-file /abs/path/processes.jsonl --scope-flow-file /abs/path/flows.jsonl --out-dir /abs/path/scan
  node scripts/run-flow-governance-review.mjs plan-process-flow-repairs --processes-file /abs/path/processes.jsonl --scope-flow-file /abs/path/flows.jsonl --out-dir /abs/path/repair-plan
  node scripts/run-flow-governance-review.mjs apply-process-flow-repairs --processes-file /abs/path/processes.jsonl --scope-flow-file /abs/path/flows.jsonl --out-dir /abs/path/repair-apply
  node scripts/run-flow-governance-review.mjs regen-product --processes-file /abs/path/processes.jsonl --scope-flow-file /abs/path/flows.jsonl --out-dir /abs/path/regen --apply
`.trim();
}

function resolveCliBin(cliDir) {
  const cliBin = path.join(cliDir, "bin", "tiangong.js");
  if (!existsSync(cliBin)) {
    fail(
      `Cannot find TianGong CLI at ${cliBin}. Set TIANGONG_LCA_CLI_DIR or pass --cli-dir.`,
    );
  }
  return cliBin;
}

function runCommand(command, args) {
  const result = spawnSync(command, args, {
    stdio: "inherit",
  });

  if (result.error) {
    throw new Error(`Failed to execute ${command}: ${result.error.message}`);
  }
  if (typeof result.status === "number") {
    return result.status;
  }
  if (result.signal) {
    throw new Error(`${command} terminated with signal ${result.signal}.`);
  }
  return 1;
}

function normalizeTopLevelArgs(rawArgs) {
  let cliDir = process.env.TIANGONG_LCA_CLI_DIR?.trim() || defaultCliDir;
  const args = [];

  for (let index = 0; index < rawArgs.length; index += 1) {
    const arg = rawArgs[index];

    if (arg === "--cli-dir") {
      if (index + 1 >= rawArgs.length) {
        fail("--cli-dir requires a value");
      }
      cliDir = rawArgs[index + 1];
      index += 1;
      continue;
    }

    if (arg.startsWith("--cli-dir=")) {
      cliDir = arg.slice("--cli-dir=".length);
      continue;
    }

    args.push(arg);
  }

  return {
    cliDir,
    args,
  };
}

function runCliBackedCommand(command, cliDir, forwardedArgs) {
  const cliSubcommand = cliBackedCommands.get(command);
  if (!cliSubcommand) {
    fail(`Unsupported CLI-backed command: ${command}`);
  }
  const cliBin = resolveCliBin(cliDir);
  return runCommand(process.execPath, [
    cliBin,
    ...cliSubcommand,
    ...forwardedArgs,
  ]);
}

function main() {
  const { cliDir, args } = normalizeTopLevelArgs(process.argv.slice(2));
  const command = args[0];
  const forwardedArgs = args.slice(1);

  if (
    !command ||
    command === "help" ||
    command === "-h" ||
    command === "--help"
  ) {
    console.log(renderHelp());
    process.exit(0);
  }

  if (cliBackedCommands.has(command)) {
    process.exit(runCliBackedCommand(command, cliDir, forwardedArgs));
  }

  if (removedCommands.has(command)) {
    fail(
      `Command '${command}' was removed with the legacy Python workflow. Reintroduce it as a native tiangong CLI command before use.`,
    );
  }

  fail(`Unknown command: ${command}`);
}

try {
  main();
} catch (error) {
  if (error instanceof UsageError) {
    console.error(`Error: ${error.message}`);
    console.error("");
    console.error(renderHelp());
    process.exit(2);
  }

  const message = error instanceof Error ? error.message : String(error);
  console.error(`Error: ${message}`);
  process.exit(1);
}
