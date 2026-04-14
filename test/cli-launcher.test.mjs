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
  const invocation = buildTiangongInvocation(['review', 'process', '--help'], {
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
    'tiangong',
    'review',
    'process',
    '--help',
  ]);
  assert.match(publishedCliCommand, /npm exec --yes --package=@tiangong-lca\/cli@latest -- tiangong/u);
});

test('buildTiangongInvocation prefers an auto-discovered local CLI checkout', () => {
  const invocation = buildTiangongInvocation(['review', 'process', '--help'], {
    repoRoot: '/workspace/tiangong-lca-skills',
    pathExists: (candidate) =>
      candidate === '/workspace/tiangong-cli' || candidate === '/workspace/tiangong-cli/bin/tiangong.js',
  });

  assert.equal(invocation.mode, 'local');
  assert.equal(invocation.command, process.execPath);
  assert.deepEqual(invocation.args, ['/workspace/tiangong-cli/bin/tiangong.js', 'review', 'process', '--help']);
});

test('runTiangongCommand emits a clear diagnostic when the published help path returns no output', () => {
  let stderr = '';
  const exitCode = runTiangongCommand(['review', 'process', '--help'], {
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
