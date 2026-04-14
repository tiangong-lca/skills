import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

export const publishedCliPackageSpec = '@tiangong-lca/cli@latest';
export const publishedCliCommand = `npm exec --yes --package=${publishedCliPackageSpec} -- tiangong`;

const launcherDir = path.dirname(fileURLToPath(import.meta.url));
const defaultSkillsRepoRoot = path.resolve(launcherDir, '..', '..');

function normalizeCliDir(cliDir) {
  const trimmed = cliDir?.trim();
  return trimmed ? trimmed : null;
}

function resolveNpmCommand() {
  return process.platform === 'win32' ? 'npm.cmd' : 'npm';
}

export function defaultLocalCliDirCandidates(repoRoot = defaultSkillsRepoRoot) {
  return ['tiangong-lca-cli', 'tiangong-cli'].map((dirName) =>
    path.join(path.dirname(repoRoot), dirName),
  );
}

export function resolveDefaultLocalCliDir(options = {}) {
  const repoRoot = options.repoRoot ?? defaultSkillsRepoRoot;
  const pathExists = options.pathExists ?? existsSync;
  return defaultLocalCliDirCandidates(repoRoot).find((candidate) => pathExists(candidate)) ?? null;
}

function isHelpInvocation(tiangongArgs) {
  return tiangongArgs.includes('--help') || tiangongArgs.includes('-h');
}

function buildPublishedFailureDiagnostic(invocation, tiangongArgs, result) {
  const status =
    typeof result.status === 'number' ? `exit code ${result.status}` : result.signal ?? 'unknown status';
  const summary = isHelpInvocation(tiangongArgs)
    ? `Published TianGong CLI invocation returned ${status} without any help output.`
    : `Published TianGong CLI invocation returned ${status} without any stdout/stderr output.`;

  return [
    summary,
    `Command: ${renderShellCommand(invocation.command, invocation.args)}`,
    `Local CLI auto-discovery checked: ${invocation.searchedCliDirs.join(', ')}`,
    'Use --cli-dir /path/to/tiangong-lca-cli or set TIANGONG_LCA_CLI_DIR to force a local working tree.',
  ].join('\n');
}

export function normalizeCliRuntimeArgs(rawArgs, options = {}) {
  const env = options.env ?? process.env;
  const defaultCliDir =
    normalizeCliDir(options.defaultCliDir) ??
    resolveDefaultLocalCliDir({
      repoRoot: options.repoRoot,
      pathExists: options.pathExists,
    });
  let cliDir = normalizeCliDir(env.TIANGONG_LCA_CLI_DIR) ?? defaultCliDir;
  const args = [];

  for (let index = 0; index < rawArgs.length; index += 1) {
    const arg = rawArgs[index];

    if (arg === '--cli-dir') {
      if (index + 1 >= rawArgs.length) {
        throw new Error('--cli-dir requires a value');
      }
      cliDir = normalizeCliDir(rawArgs[index + 1]);
      index += 1;
      continue;
    }

    if (arg.startsWith('--cli-dir=')) {
      cliDir = normalizeCliDir(arg.slice('--cli-dir='.length));
      continue;
    }

    args.push(arg);
  }

  return {
    cliDir,
    args,
  };
}

export function buildTiangongInvocation(tiangongArgs, options = {}) {
  const pathExists = options.pathExists ?? existsSync;
  const searchedCliDirs = defaultLocalCliDirCandidates(options.repoRoot);
  const cliDir =
    normalizeCliDir(options.cliDir) ??
    resolveDefaultLocalCliDir({
      repoRoot: options.repoRoot,
      pathExists,
    });

  if (cliDir) {
    const cliBin = path.join(cliDir, 'bin', 'tiangong.js');
    if (!pathExists(cliBin)) {
      throw new Error(
        `Cannot find TianGong CLI at ${cliBin}. Set TIANGONG_LCA_CLI_DIR or pass --cli-dir.`,
      );
    }

    return {
      mode: 'local',
      command: process.execPath,
      args: [cliBin, ...tiangongArgs],
      cliBin,
    };
  }

  return {
    mode: 'published',
    command: resolveNpmCommand(),
    args: ['exec', '--yes', `--package=${publishedCliPackageSpec}`, '--', 'tiangong', ...tiangongArgs],
    searchedCliDirs,
  };
}

export function runTiangongCommand(tiangongArgs, options = {}) {
  const spawnImpl = options.spawnImpl ?? spawnSync;
  const stdoutWrite = options.stdoutWrite ?? ((text) => process.stdout.write(text));
  const stderrWrite = options.stderrWrite ?? ((text) => process.stderr.write(text));
  const invocation = buildTiangongInvocation(tiangongArgs, options);
  const result = spawnImpl(invocation.command, invocation.args, {
    stdio: 'pipe',
    encoding: 'utf8',
    ...options.spawnOptions,
  });

  if (result.error) {
    throw new Error(`Failed to execute TianGong CLI: ${result.error.message}`);
  }

  if (result.stdout) {
    stdoutWrite(result.stdout);
  }
  if (result.stderr) {
    stderrWrite(result.stderr);
  }

  if (invocation.mode === 'published' && !result.stdout && !result.stderr) {
    stderrWrite(`${buildPublishedFailureDiagnostic(invocation, tiangongArgs, result)}\n`);
    return typeof result.status === 'number' && result.status !== 0 ? result.status : 1;
  }
  if (typeof result.status === 'number') {
    return result.status;
  }
  if (result.signal) {
    throw new Error(`TianGong CLI terminated with signal ${result.signal}.`);
  }
  return 1;
}

export function withCliRuntimeEnv(baseEnv, cliDir) {
  const env = { ...baseEnv };
  const normalizedCliDir = normalizeCliDir(cliDir);

  if (normalizedCliDir) {
    env.TIANGONG_LCA_CLI_DIR = normalizedCliDir;
  } else {
    delete env.TIANGONG_LCA_CLI_DIR;
  }

  return env;
}

export function renderShellCommand(command, args) {
  return [command, ...args]
    .map((value) =>
      /^[A-Za-z0-9_./:=+@-]+$/u.test(value) ? value : JSON.stringify(value),
    )
    .join(' ');
}
