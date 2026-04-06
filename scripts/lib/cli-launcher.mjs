import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';

export const publishedCliPackageSpec = '@tiangong-lca/cli@latest';
export const publishedCliCommand = `npx -y ${publishedCliPackageSpec}`;

function normalizeCliDir(cliDir) {
  const trimmed = cliDir?.trim();
  return trimmed ? trimmed : null;
}

function resolveNpxCommand() {
  return process.platform === 'win32' ? 'npx.cmd' : 'npx';
}

export function normalizeCliRuntimeArgs(rawArgs, options = {}) {
  const env = options.env ?? process.env;
  let cliDir = normalizeCliDir(env.TIANGONG_LCA_CLI_DIR) ?? normalizeCliDir(options.defaultCliDir);
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
  const cliDir = normalizeCliDir(options.cliDir);

  if (cliDir) {
    const cliBin = path.join(cliDir, 'bin', 'tiangong.js');
    if (!existsSync(cliBin)) {
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
    command: resolveNpxCommand(),
    args: ['-y', publishedCliPackageSpec, ...tiangongArgs],
  };
}

export function runTiangongCommand(tiangongArgs, options = {}) {
  const invocation = buildTiangongInvocation(tiangongArgs, options);
  const result = spawnSync(invocation.command, invocation.args, {
    stdio: 'inherit',
    ...options.spawnOptions,
  });

  if (result.error) {
    throw new Error(`Failed to execute TianGong CLI: ${result.error.message}`);
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
