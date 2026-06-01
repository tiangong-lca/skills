import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildTiangongInvocation,
  normalizeCliRuntimeArgs,
  publishedCliCommand,
  runTiangongCommand,
} from '../scripts/lib/cli-launcher.mjs';

test('normalizeCliRuntimeArgs auto-discovers the alternate tiangong-cli sibling', () => {
  const { cliDir, args } = normalizeCliRuntimeArgs(['embedding-ft', '--help'], {
    repoRoot: '/workspace/tiangong-lca-skills',
    pathExists: (candidate) => candidate === '/workspace/tiangong-cli',
  });

  assert.equal(cliDir, '/workspace/tiangong-cli');
  assert.deepEqual(args, ['embedding-ft', '--help']);
});

test('normalizeCliRuntimeArgs keeps explicit cli-dir overrides above auto-discovery', () => {
  const { cliDir, args } = normalizeCliRuntimeArgs(
    ['--cli-dir', '/tmp/manual-cli', 'embedding-ft', '--help'],
    {
      repoRoot: '/workspace/tiangong-lca-skills',
      pathExists: () => true,
    },
  );

  assert.equal(cliDir, '/tmp/manual-cli');
  assert.deepEqual(args, ['embedding-ft', '--help']);
});

test('buildTiangongInvocation uses npm exec for the published CLI contract', () => {
  const invocation = buildTiangongInvocation(['qa', 'process', '--help'], {
    repoRoot: '/workspace/tiangong-lca-skills',
    pathExists: () => false,
  });

  assert.equal(invocation.mode, 'published');
  assert.equal(invocation.command, process.platform === 'win32' ? 'npm.cmd' : 'npm');
  assert.deepEqual(invocation.args, [
    'exec',
    '--yes',
    '--package=@tiangong-lca/cli@latest',
    '--',
    'tiangong-lca',
    'qa',
    'process',
    '--help',
  ]);
  assert.match(publishedCliCommand, /npm exec --yes --package=@tiangong-lca\/cli@latest -- tiangong-lca/u);
});

test('buildTiangongInvocation prefers an auto-discovered local CLI checkout', () => {
  const invocation = buildTiangongInvocation(['qa', 'process', '--help'], {
    repoRoot: '/workspace/tiangong-lca-skills',
    pathExists: (candidate) =>
      candidate === '/workspace/tiangong-cli' || candidate === '/workspace/tiangong-cli/bin/tiangong-lca.js',
  });

  assert.equal(invocation.mode, 'local');
  assert.equal(invocation.command, process.execPath);
  assert.deepEqual(invocation.args, ['/workspace/tiangong-cli/bin/tiangong-lca.js', 'qa', 'process', '--help']);
});

test('runTiangongCommand emits a clear diagnostic when the published help path returns no output', () => {
  let stderr = '';
  const exitCode = runTiangongCommand(['qa', 'process', '--help'], {
    repoRoot: '/workspace/tiangong-lca-skills',
    pathExists: () => false,
    spawnImpl: () => ({
      status: 0,
      stdout: '',
      stderr: '',
    }),
    stderrWrite: (text) => {
      stderr += text;
    },
  });

  assert.equal(exitCode, 1);
  assert.match(stderr, /returned exit code 0 without any help output/u);
  assert.match(stderr, /Local CLI auto-discovery checked:/u);
  assert.match(stderr, /Use --cli-dir/u);
});

test('runTiangongCommand rebuilds a stale local CLI before invocation', () => {
  const cliDir = '/workspace/tiangong-lca-cli';
  const paths = new Set([
    cliDir,
    `${cliDir}/bin`,
    `${cliDir}/bin/tiangong-lca.js`,
    `${cliDir}/dist/src/main.js`,
    `${cliDir}/src`,
    `${cliDir}/src/cli.ts`,
    `${cliDir}/package.json`,
    `${cliDir}/tsconfig.build.json`,
  ]);
  const directories = new Set([cliDir, `${cliDir}/bin`, `${cliDir}/src`]);
  const mtimes = new Map([
    [`${cliDir}/dist/src/main.js`, 10],
    [`${cliDir}/src`, 20],
    [`${cliDir}/src/cli.ts`, 20],
    [`${cliDir}/bin/tiangong-lca.js`, 5],
    [`${cliDir}/package.json`, 5],
    [`${cliDir}/tsconfig.build.json`, 5],
  ]);
  const calls = [];

  const exitCode = runTiangongCommand(['process', 'save-draft', '--help'], {
    repoRoot: '/workspace/tiangong-lca-skills',
    pathExists: (candidate) => paths.has(candidate),
    readDir: (candidate) => {
      if (candidate === `${cliDir}/src`) {
        return ['cli.ts'];
      }
      return [];
    },
    statPath: (candidate) => ({
      mtimeMs: mtimes.get(candidate) ?? 1,
      isDirectory: () => directories.has(candidate),
    }),
    buildSpawnImpl: (command, args, options) => {
      calls.push({ command, args, cwd: options.cwd, phase: 'build' });
      return { status: 0, stdout: '', stderr: '' };
    },
    spawnImpl: (command, args) => {
      calls.push({ command, args, phase: 'run' });
      return { status: 0, stdout: 'process save-draft help', stderr: '' };
    },
    stdoutWrite: () => {},
  });

  assert.equal(exitCode, 0);
  assert.equal(calls.length, 2);
  assert.deepEqual(calls[0], {
    command: process.platform === 'win32' ? 'npm.cmd' : 'npm',
    args: ['run', 'build'],
    cwd: cliDir,
    phase: 'build',
  });
  assert.equal(calls[1].command, process.execPath);
  assert.deepEqual(calls[1].args, [
    `${cliDir}/bin/tiangong-lca.js`,
    'process',
    'save-draft',
    '--help',
  ]);
});
