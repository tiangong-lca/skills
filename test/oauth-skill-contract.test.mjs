import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const remoteSkillFiles = [
  'lca-topic-overview/SKILL.md',
  'current-account-dataset-review/SKILL.md',
  'dataset-rls-maintenance/SKILL.md',
  'embedding-ft/SKILL.md',
  'external-dataset-curated-import/SKILL.md',
  'flow-governance-review/SKILL.md',
  'flow-hybrid-search/SKILL.md',
  'lca-publish-executor/SKILL.md',
  'lifecycleinventory-review/SKILL.md',
  'lifecyclemodel-hybrid-search/SKILL.md',
  'lifecyclemodel-resulting-process-builder/SKILL.md',
  'process-dedup-review/SKILL.md',
  'process-hybrid-search/SKILL.md',
  'process-scope-statistics/SKILL.md',
  'source-evidence-dataset-development/SKILL.md',
  'tiangong-lca-remote-ops/SKILL.md',
];

const activeRemoteDocs = [
  ...remoteSkillFiles,
  'embedding-ft/references/env.md',
  'embedding-ft/references/testing.md',
  'flow-governance-review/references/env.md',
  'flow-hybrid-search/references/env.md',
  'flow-hybrid-search/references/request-response.md',
  'flow-hybrid-search/references/testing.md',
  'lifecyclemodel-hybrid-search/references/env.md',
  'lifecyclemodel-hybrid-search/references/request-response.md',
  'lifecyclemodel-hybrid-search/references/testing.md',
  'process-hybrid-search/references/env.md',
  'process-hybrid-search/references/request-response.md',
  'process-hybrid-search/references/testing.md',
];

function read(relativePath) {
  return readFileSync(path.join(repoRoot, relativePath), 'utf8');
}

test('every active remote skill uses CLI OAuth with a human login handoff', () => {
  for (const relativePath of remoteSkillFiles) {
    const text = read(relativePath);
    assert.match(text, /tiangong-lca auth (?:status|doctor-auth)/u, relativePath);
    assert.match(text, /auth login/u, relativePath);
    assert.match(text, /human|人类/u, relativePath);
    assert.match(text, /password|密码/u, relativePath);
  }
});

test('active remote skill instructions contain no password-equivalent invocation', () => {
  const forbidden =
    /TIANGONG_LCA_API_KEY\s*=|--api-key(?:\s|`)|Authorization:\s*Bearer\s*<TIANGONG_LCA_API_KEY>|Auth variable:\s*`TIANGONG_LCA_API_KEY`/u;
  for (const relativePath of activeRemoteDocs) {
    assert.doesNotMatch(read(relativePath), forbidden, relativePath);
  }
});

test('headless and multi-account boundaries stay explicit', () => {
  for (const relativePath of [
    'current-account-dataset-review/SKILL.md',
    'dataset-rls-maintenance/SKILL.md',
    'flow-governance-review/SKILL.md',
    'tiangong-lca-remote-ops/SKILL.md',
  ]) {
    const text = read(relativePath);
    assert.match(text, /TIANGONG_LCA_SESSION_FILE|session file/u, relativePath);
  }
  for (const relativePath of [
    'embedding-ft/SKILL.md',
    'flow-hybrid-search/SKILL.md',
    'process-hybrid-search/SKILL.md',
    'lifecyclemodel-hybrid-search/SKILL.md',
  ]) {
    const text = read(relativePath);
    assert.match(text, /TIANGONG_LCA_ACCESS_TOKEN/u, relativePath);
    assert.match(text, /orchestrator/u, relativePath);
  }
});
