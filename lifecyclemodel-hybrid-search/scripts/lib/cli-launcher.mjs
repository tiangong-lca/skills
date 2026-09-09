import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';

export const expectedNodeVersion = '24.19.0';
export const expectedPnpmVersion = '11.24.0';
export const publishedCliPackageSpec = '@tiangong-lca/cli@0.1.13';
export const publishedCliCommand = `pnpm dlx --package=${publishedCliPackageSpec} tiangong-lca`;

const expectedCliPackageName = '@tiangong-lca/cli';
const expectedCliPackageVersion = '0.1.13';
const expectedCliPackageManager = `pnpm@${expectedPnpmVersion}`;
const expectedCliNodeEngine = '>=24.19.0 <25';
const verifiedToolchainsBySpawn = new WeakMap();

function normalizeCliDir(cliDir) {
  const trimmed = cliDir?.trim();
  return trimmed ? path.resolve(trimmed) : null;
}

function resolvePnpmCommand(platform = process.platform) {
  return platform === 'win32' ? 'pnpm.exe' : 'pnpm';
}

export function normalizeCliRuntimeArgs(rawArgs, options = {}) {
  const env = options.env ?? process.env;
  let cliDir =
    env.TIANGONG_LCA_CLI_MODE === 'published'
      ? null
      : normalizeCliDir(env.TIANGONG_LCA_CLI_DIR);
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

    if (arg === '--published-cli') {
      cliDir = null;
      continue;
    }

    args.push(arg);
  }

  return {
    cliDir,
    args,
  };
}

function readLocalCliPackageEvidence(cliDir, options) {
  const pathExists = options.pathExists ?? existsSync;
  const readText = options.readText ?? ((filePath) => readFileSync(filePath, 'utf8'));
  const packageManifestPath = path.join(cliDir, 'package.json');
  const lockfilePath = path.join(cliDir, 'pnpm-lock.yaml');

  if (!pathExists(packageManifestPath)) {
    throw new Error(`Local TianGong CLI requires package.json: ${packageManifestPath}`);
  }
  if (!pathExists(lockfilePath)) {
    throw new Error(`Local TianGong CLI requires pnpm-lock.yaml: ${lockfilePath}`);
  }

  let manifest;
  try {
    manifest = JSON.parse(readText(packageManifestPath));
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`Cannot parse local TianGong CLI package.json: ${detail}`);
  }

  if (
    !manifest ||
    typeof manifest !== 'object' ||
    manifest.name !== expectedCliPackageName ||
    manifest.version !== expectedCliPackageVersion
  ) {
    throw new Error(
      `Local TianGong CLI package mismatch: expected ${publishedCliPackageSpec}.`,
    );
  }
  if (manifest.packageManager !== expectedCliPackageManager) {
    throw new Error(
      `Local TianGong CLI packageManager mismatch: expected ${expectedCliPackageManager}.`,
    );
  }
  if (
    !manifest.engines ||
    manifest.engines.node !== expectedCliNodeEngine ||
    manifest.engines.pnpm !== expectedPnpmVersion
  ) {
    throw new Error(
      `Local TianGong CLI engine mismatch: expected Node ${expectedCliNodeEngine} and pnpm ${expectedPnpmVersion}.`,
    );
  }

  const lockfile = readText(lockfilePath);
  if (
    !/^lockfileVersion:\s*['"]?9\.0['"]?\s*$/mu.test(lockfile) ||
    !/^importers:\s*$/mu.test(lockfile)
  ) {
    throw new Error(`Local TianGong CLI pnpm lockfile is not a supported frozen v9 lock: ${lockfilePath}`);
  }

  return {
    packageManifestPath,
    lockfilePath,
    packageVersion: manifest.version,
  };
}

export function buildTiangongInvocation(tiangongArgs, options = {}) {
  const pathExists = options.pathExists ?? existsSync;
  const cliDir = normalizeCliDir(options.cliDir);

  if (cliDir) {
    const cliBin = path.join(cliDir, 'bin', 'tiangong-lca.js');
    if (!pathExists(cliBin)) {
      throw new Error(
        `Cannot find TianGong CLI at ${cliBin}. Set TIANGONG_LCA_CLI_DIR or pass --cli-dir.`,
      );
    }
    const packageEvidence = readLocalCliPackageEvidence(cliDir, options);

    return {
      mode: 'local',
      command: process.execPath,
      args: [cliBin, ...tiangongArgs],
      cliDir,
      cliBin,
      ...packageEvidence,
    };
  }

  return {
    mode: 'published',
    command: resolvePnpmCommand(options.platform),
    args: ['dlx', `--package=${publishedCliPackageSpec}`, 'tiangong-lca', ...tiangongArgs],
    packageSpec: publishedCliPackageSpec,
  };
}

function newestMtimeMs(targetPath, options) {
  const pathExists = options.pathExists ?? existsSync;
  const readDir = options.readDir ?? readdirSync;
  const statPath = options.statPath ?? statSync;

  if (!pathExists(targetPath)) {
    return null;
  }

  const stat = statPath(targetPath);
  if (!stat.isDirectory()) {
    return stat.mtimeMs;
  }

  let newest = stat.mtimeMs;
  for (const child of readDir(targetPath)) {
    const childMtime = newestMtimeMs(path.join(targetPath, child), options);
    if (typeof childMtime === 'number' && childMtime > newest) {
      newest = childMtime;
    }
  }
  return newest;
}

function localCliNeedsBuild(invocation, options) {
  const pathExists = options.pathExists ?? existsSync;
  const statPath = options.statPath ?? statSync;
  const entryPath = path.join(invocation.cliDir, 'dist', 'src', 'main.js');
  const modulesStatePath = path.join(invocation.cliDir, 'node_modules', '.modules.yaml');
  if (!pathExists(entryPath) || !pathExists(modulesStatePath)) {
    return true;
  }

  const builtAt = statPath(entryPath).mtimeMs;
  const sourcePaths = [
    path.join(invocation.cliDir, 'src'),
    invocation.cliBin,
    invocation.packageManifestPath,
    invocation.lockfilePath,
    path.join(invocation.cliDir, 'tsconfig.build.json'),
  ];

  return sourcePaths.some((sourcePath) => {
    const sourceMtime = newestMtimeMs(sourcePath, options);
    return typeof sourceMtime === 'number' && sourceMtime > builtAt;
  });
}

export function assertSupportedToolchain(options = {}) {
  const nodeVersion = String(options.nodeVersion ?? process.versions.node).replace(/^v/u, '');
  if (nodeVersion !== expectedNodeVersion) {
    throw new Error(`Node ${expectedNodeVersion} is required; received ${nodeVersion}.`);
  }

  const pnpmCommand = resolvePnpmCommand(options.platform);
  const spawnImpl = options.toolchainSpawnImpl ?? spawnSync;
  const executionEnv = options.spawnOptions?.env ?? process.env;
  const envEntries = Object.entries(executionEnv);
  const readEnv = (name) =>
    envEntries.find(([key]) => key.toLowerCase() === name.toLowerCase())?.[1] ?? '';
  const verificationKey = [
    nodeVersion,
    options.platform ?? process.platform,
    pnpmCommand,
    readEnv('PATH'),
    readEnv('PATHEXT'),
  ].join('\0');
  const cacheEnabled = options.cacheToolchainVerification !== false;
  const cachedKeys = cacheEnabled ? verifiedToolchainsBySpawn.get(spawnImpl) : null;
  if (cachedKeys?.has(verificationKey)) {
    return pnpmCommand;
  }

  const result = spawnImpl(pnpmCommand, ['--version'], {
    env: executionEnv,
    stdio: 'pipe',
    encoding: 'utf8',
    shell: false,
  });
  if (result.error) {
    throw new Error(`Failed to verify pnpm ${expectedPnpmVersion}: ${result.error.message}`);
  }
  const actualPnpmVersion = result.stdout?.trim();
  if (result.status !== 0 || actualPnpmVersion !== expectedPnpmVersion) {
    throw new Error(
      `pnpm ${expectedPnpmVersion} is required; received ${actualPnpmVersion || 'unavailable'}.`,
    );
  }

  if (cacheEnabled) {
    const nextCachedKeys = cachedKeys ?? new Set();
    nextCachedKeys.add(verificationKey);
    verifiedToolchainsBySpawn.set(spawnImpl, nextCachedKeys);
  }

  return pnpmCommand;
}

function runLocalPreparationStep(invocation, args, label, options) {
  const spawnImpl = options.buildSpawnImpl ?? spawnSync;
  const result = spawnImpl(resolvePnpmCommand(options.platform), args, {
    cwd: invocation.cliDir,
    env: options.spawnOptions?.env ?? process.env,
    stdio: 'pipe',
    encoding: 'utf8',
    shell: false,
  });
  if (result.error) {
    throw new Error(`Failed to ${label} local TianGong CLI: ${result.error.message}`);
  }
  if (result.status !== 0) {
    const detail = [result.stdout, result.stderr].filter(Boolean).join('\n').trim();
    throw new Error(
      `Local TianGong CLI ${label} failed with exit code ${result.status}.${detail ? `\n${detail}` : ''}`,
    );
  }
}

function ensureLocalCliBuild(invocation, options) {
  if (invocation.mode !== 'local' || options.prepareLocalCli === false) {
    return;
  }
  if (!localCliNeedsBuild(invocation, options)) {
    return;
  }

  runLocalPreparationStep(
    invocation,
    ['install', '--frozen-lockfile'],
    'frozen install',
    options,
  );
  runLocalPreparationStep(invocation, ['run', 'build'], 'build', options);
}

export function executeTiangongCommand(tiangongArgs, options = {}) {
  const invocation = buildTiangongInvocation(tiangongArgs, options);
  assertSupportedToolchain(options);
  ensureLocalCliBuild(invocation, options);

  const spawnImpl = options.spawnImpl ?? spawnSync;
  const result = spawnImpl(invocation.command, invocation.args, {
    ...options.spawnOptions,
    stdio: 'pipe',
    encoding: 'utf8',
    shell: false,
  });

  if (result.error) {
    throw new Error(`Failed to execute TianGong CLI: ${result.error.message}`);
  }

  return {
    invocation,
    status: result.status,
    signal: result.signal,
    stdout: result.stdout ?? '',
    stderr: result.stderr ?? '',
  };
}

export function runTiangongCommand(tiangongArgs, options = {}) {
  const stdoutWrite = options.stdoutWrite ?? ((text) => process.stdout.write(text));
  const stderrWrite = options.stderrWrite ?? ((text) => process.stderr.write(text));
  const result = executeTiangongCommand(tiangongArgs, options);

  if (result.stdout) {
    stdoutWrite(result.stdout);
  }
  if (result.stderr) {
    stderrWrite(result.stderr);
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
    delete env.TIANGONG_LCA_CLI_MODE;
  } else {
    delete env.TIANGONG_LCA_CLI_DIR;
    env.TIANGONG_LCA_CLI_MODE = 'published';
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
