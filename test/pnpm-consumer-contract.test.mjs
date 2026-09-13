import test from 'node:test';
import assert from 'node:assert/strict';
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..');
const gitRepositoryLocationEnvNames = new Set([
  'GIT_DIR',
  'GIT_WORK_TREE',
  'GIT_INDEX_FILE',
  'GIT_OBJECT_DIRECTORY',
  'GIT_ALTERNATE_OBJECT_DIRECTORIES',
  'GIT_COMMON_DIR',
  'GIT_CEILING_DIRECTORIES',
  'GIT_PREFIX',
  'GIT_NAMESPACE',
  'GIT_QUARANTINE_PATH',
]);

function read(relativePath) {
  return readFileSync(path.join(repoRoot, relativePath), 'utf8');
}

function collectMarkdown(rootDir) {
  const result = runGit(rootDir, ['ls-files', '-z', '--', '*.md']);
  return result.stdout
    .split('\0')
    .filter(Boolean)
    .map((relativePath) => path.join(rootDir, relativePath));
}

function runGit(cwd, args) {
  const result = spawnSync('git', args, {
    cwd,
    env: sanitizedGitEnv(),
    encoding: 'utf8',
    stdio: 'pipe',
    shell: false,
  });
  assert.equal(result.error, undefined);
  assert.equal(result.status, 0, result.stderr || result.stdout);
  return result;
}

function sanitizedGitEnv(baseEnv = process.env) {
  return Object.fromEntries(
    Object.entries(baseEnv).filter(
      ([name]) => !gitRepositoryLocationEnvNames.has(name.toUpperCase()),
    ),
  );
}

function runBootstrapGit(cwd, args) {
  const result = spawnSync('git', args, {
    cwd,
    env: sanitizedGitEnv(),
    encoding: 'utf8',
    stdio: 'pipe',
    shell: false,
  });
  assert.equal(result.error, undefined);
  assert.equal(result.status, 0, result.stderr || result.stdout);
  return result;
}

test('hook-like Git environment cannot redirect fixture operations into the parent index', () => {
  const sandboxRoot = mkdtempSync(path.join(tmpdir(), 'skills-git-env-isolation-'));
  const parentRoot = path.join(sandboxRoot, 'parent');
  const fixtureRoot = path.join(sandboxRoot, 'fixture');
  const alternateObjects = path.join(sandboxRoot, 'alternate-objects');
  const inheritedValues = new Map();

  try {
    mkdirSync(parentRoot, { recursive: true });
    mkdirSync(fixtureRoot, { recursive: true });
    mkdirSync(alternateObjects, { recursive: true });
    runBootstrapGit(parentRoot, ['init', '--quiet']);
    writeFileSync(path.join(parentRoot, 'README.md'), '# parent original\n', 'utf8');
    runBootstrapGit(parentRoot, ['add', '--', 'README.md']);
    const parentTreeBefore = runBootstrapGit(parentRoot, ['write-tree']).stdout.trim();
    const parentWorktreeBefore = readFileSync(path.join(parentRoot, 'README.md'), 'utf8');

    runBootstrapGit(fixtureRoot, ['init', '--quiet']);
    writeFileSync(path.join(fixtureRoot, 'README.md'), '# isolated fixture\n', 'utf8');

    const inheritedGitEnvironment = {
      GIT_DIR: path.join(parentRoot, '.git'),
      GIT_WORK_TREE: fixtureRoot,
      GIT_INDEX_FILE: path.join(parentRoot, '.git', 'index'),
      GIT_OBJECT_DIRECTORY: path.join(parentRoot, '.git', 'objects'),
      GIT_ALTERNATE_OBJECT_DIRECTORIES: alternateObjects,
      GIT_COMMON_DIR: path.join(parentRoot, '.git'),
      GIT_CEILING_DIRECTORIES: sandboxRoot,
    };
    for (const [name, value] of Object.entries(inheritedGitEnvironment)) {
      inheritedValues.set(name, {
        existed: Object.prototype.hasOwnProperty.call(process.env, name),
        value: process.env[name],
      });
      process.env[name] = value;
    }

    runGit(fixtureRoot, ['add', '--', 'README.md']);

    for (const [name, previous] of inheritedValues) {
      if (previous.existed) {
        process.env[name] = previous.value;
      } else {
        delete process.env[name];
      }
    }
    inheritedValues.clear();

    assert.equal(
      runBootstrapGit(parentRoot, ['write-tree']).stdout.trim(),
      parentTreeBefore,
    );
    assert.equal(readFileSync(path.join(parentRoot, 'README.md'), 'utf8'), parentWorktreeBefore);
  } finally {
    for (const [name, previous] of inheritedValues) {
      if (previous.existed) {
        process.env[name] = previous.value;
      } else {
        delete process.env[name];
      }
    }
    rmSync(sandboxRoot, { recursive: true, force: true });
  }
});

test('markdown inventory excludes untracked nested repositories under .ci', () => {
  const fixtureRoot = mkdtempSync(path.join(tmpdir(), 'skills-markdown-inventory-'));
  const nestedRoot = path.join(fixtureRoot, '.ci', 'tiangong-lca-cli');

  try {
    runGit(fixtureRoot, ['init', '--quiet']);
    writeFileSync(path.join(fixtureRoot, 'README.md'), '# tracked root\n', 'utf8');
    runGit(fixtureRoot, ['add', '--', 'README.md']);

    mkdirSync(nestedRoot, { recursive: true });
    runGit(nestedRoot, ['init', '--quiet']);
    writeFileSync(
      path.join(nestedRoot, 'README.md'),
      'npx -y @tiangong-lca/cli@latest --help\n',
      'utf8',
    );
    runGit(nestedRoot, ['add', '--', 'README.md']);

    const inventory = collectMarkdown(fixtureRoot)
      .map((filePath) => path.relative(fixtureRoot, filePath))
      .sort();
    assert.deepEqual(inventory, ['README.md']);
  } finally {
    rmSync(fixtureRoot, { recursive: true, force: true });
  }
});

test('repository package contract pins the workspace Node and pnpm versions', () => {
  const manifest = JSON.parse(read('package.json'));
  assert.equal(manifest.packageManager, 'pnpm@11.24.0');
  assert.deepEqual(manifest.engines, {
    node: '24.19.0',
    pnpm: '11.24.0',
  });
  assert.match(read('pnpm-lock.yaml'), /^lockfileVersion: '9\.0'$/mu);
});

test('validation CI installs the exact CLI checkout through frozen pnpm only', () => {
  const workflow = read('.github/workflows/validate-skills.yml');

  assert.match(workflow, /repository: tiangong-lca\/cli/u);
  assert.match(workflow, /ref: a6c5815b06903b2b424c5ab892e4e9f3c99b3001/u);
  assert.match(
    workflow,
    /uses: pnpm\/setup@84cb39b217b10273981911c288cd62326dc7c6d2/u,
  );
  assert.match(workflow, /runtime: node@24\.19\.0/u);
  assert.match(workflow, /version: 11\.24\.0/u);
  assert.match(workflow, /install: false/u);
  assert.match(workflow, /cache: true/u);
  assert.match(workflow, /pnpm install --frozen-lockfile/u);
  assert.match(workflow, /pnpm run build/u);
  assert.match(workflow, /run-process-hybrid-search\.mjs --published-cli --help/u);
  assert.equal(
    workflow.match(
      /uses: actions\/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1/gu,
    )?.length,
    2,
  );
  assert.doesNotMatch(
    workflow,
    /actions\/checkout@v|pnpm\/action-setup|actions\/setup-node|package-lock\.json|\bnpm (?:ci|exec|run)\b/u,
  );
});

test('local push gate defaults published and validates explicit local evidence before mutation', () => {
  const hook = read('.githooks/pre-push');

  assert.match(hook, /pnpm install --frozen-lockfile/u);
  assert.match(hook, /pnpm run build/u);
  assert.match(hook, /pnpm prepush:gate/u);
  assert.match(hook, /TIANGONG_LCA_CLI_MODE=published/u);
  assert.doesNotMatch(hook, /\.\.\/tiangong-lca-cli|\.\.\/tiangong-cli/u);
  const evidenceIndex = hook.indexOf(
    'node "$repo_root/scripts/check-toolchain.mjs" --cli-dir "$cli_dir"',
  );
  const localInstallIndex = hook.indexOf(
    'cd "$cli_dir" && pnpm install --frozen-lockfile',
  );
  assert.ok(evidenceIndex >= 0);
  assert.ok(localInstallIndex > evidenceIndex);
  assert.doesNotMatch(hook, /package-lock\.json|\bnpm (?:ci|exec|run)\b/u);
});

test('checked-in docs pin the TianGong CLI while preserving external skills npx governance', () => {
  for (const filePath of collectMarkdown(repoRoot)) {
    const text = readFileSync(filePath, 'utf8');
    assert.doesNotMatch(
      text,
      /@tiangong-lca\/cli@latest|(?:npm exec|npx)[^\n]*@tiangong-lca\/cli/u,
      path.relative(repoRoot, filePath),
    );
  }

  assert.match(read('README.md'), /npx skills add https:\/\/github\.com\/tiangong-lca\/agent-skills/u);
  assert.match(read('README.zh-CN.md'), /npx skills add https:\/\/github\.com\/tiangong-lca\/agent-skills/u);
  assert.match(read('AGENTS.md'), /consumed through `npx skills`/u);
  assert.match(
    read('scripts/validate-skills.mjs'),
    /npx\[\^\\n\]\*@tiangong-lca\\\/cli/u,
  );
  assert.doesNotMatch(read('scripts/lib/cli-launcher.mjs'), /pnpm\.cmd/u);
});
