#!/usr/bin/env node
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  defaultLocalCliDirCandidates,
  normalizeCliRuntimeArgs,
  publishedCliCommand,
  withCliRuntimeEnv,
} from './lib/cli-launcher.mjs';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '..');
const localCliDirCandidates = defaultLocalCliDirCandidates(repoRoot);

const defaultSkillNames = [
  'process-hybrid-search',
  'flow-hybrid-search',
  'lifecyclemodel-hybrid-search',
  'embedding-ft',
  'process-automated-builder',
  'lifecyclemodel-automated-builder',
  'lifecyclemodel-resulting-process-builder',
  'lifecycleinventory-review',
  'flow-governance-review',
  'lifecyclemodel-recursive-orchestrator',
  'lca-publish-executor',
];

const removedQuickValidatePattern = new RegExp(String.raw`quick_validate` + String.raw`\.py`, 'u');
const removedLifecyclemodelReviewPattern = new RegExp(
  String.raw`run_lifecyclemodel_review` + String.raw`\.py`,
  'u',
);
const removedInitSkillPattern = new RegExp(String.raw`init_skill` + String.raw`\.py`, 'u');
const undocumentedInterfaceFlagPattern = new RegExp(String.raw`--` + String.raw`interface`, 'u');
const historicalValidatePyCurrentPathPattern = new RegExp(
  String.raw`validate` + String.raw`\.py` + String.raw` checks`,
  'iu',
);

const docGuards = [
  {
    file: 'process-hybrid-search/references/env.md',
    pattern: /shell wrapper/iu,
    message: 'Use Node `.mjs` wrapper wording in process-hybrid-search env docs.',
  },
  {
    file: 'flow-hybrid-search/references/env.md',
    pattern: /shell wrapper/iu,
    message: 'Use Node `.mjs` wrapper wording in flow-hybrid-search env docs.',
  },
  {
    file: 'lifecyclemodel-hybrid-search/references/env.md',
    pattern: /shell wrapper/iu,
    message: 'Use Node `.mjs` wrapper wording in lifecyclemodel-hybrid-search env docs.',
  },
  {
    file: 'embedding-ft/references/env.md',
    pattern: /shell wrapper/iu,
    message: 'Use Node `.mjs` wrapper wording in embedding-ft env docs.',
  },
  {
    file: 'lifecycleinventory-review/SKILL.md',
    pattern: /not implemented yet/iu,
    message: 'lifecycleinventory-review should not advertise lifecyclemodel as unimplemented.',
  },
  {
    file: 'lifecycleinventory-review/scripts/run-review.mjs',
    pattern: /not implemented yet/iu,
    message: 'run-review.mjs should delegate lifecyclemodel review to the CLI.',
  },
  {
    file: 'lifecycleinventory-review/profiles/lifecyclemodel/README.md',
    pattern: removedLifecyclemodelReviewPattern,
    message: 'lifecyclemodel profile docs should not reference a future Python review script filename.',
  },
  {
    file: 'lifecycleinventory-review/profiles/lifecyclemodel/README.md',
    pattern: /not implemented yet/iu,
    message: 'lifecyclemodel profile docs should describe the implemented CLI path.',
  },
  {
    file: 'AGENTS.md',
    pattern: removedQuickValidatePattern,
    message: 'AGENTS.md should point at node scripts/validate-skills.mjs instead of a removed Python validator.',
  },
  {
    file: 'AGENTS.md',
    pattern: removedInitSkillPattern,
    message: 'AGENTS.md should not require a missing Python bootstrap step.',
  },
  {
    file: 'AGENTS.md',
    pattern: undocumentedInterfaceFlagPattern,
    message: 'AGENTS.md should require a real agents/openai.yaml file, not an undocumented generator flag.',
  },
  {
    file: 'README.md',
    pattern: /~\/<agent>\/skills\//u,
    message: 'README.md should describe global install scope without assuming a Unix home-directory path.',
  },
  {
    file: 'README.zh-CN.md',
    pattern: /~\/<agent>\/skills\//u,
    message: 'README.zh-CN.md should describe global install scope without assuming a Unix home-directory path.',
  },
  {
    file: 'lifecyclemodel-automated-builder/references/source-analysis.md',
    pattern: historicalValidatePyCurrentPathPattern,
    message:
      'source-analysis.md should treat the old Python validator as historical context, not as the current execution path.',
  },
  {
    file: 'lca-publish-executor/assets/example-request.json',
    pattern: /"out_dir": "\/tmp\//u,
    message: 'lca-publish-executor example request should use a platform-neutral temp directory placeholder.',
  },
  {
    file: 'lifecyclemodel-resulting-process-builder/assets/example-request.json',
    pattern: /file:\/\/\/tmp\//u,
    message:
      'lifecyclemodel-resulting-process-builder example request should use a platform-neutral file URI placeholder.',
  },
];

const requiredDocPatterns = [
  {
    file: 'lifecycleinventory-review/SKILL.md',
    pattern: /--rows-file/u,
    message:
      'lifecycleinventory-review should document the native --rows-file process review path.',
  },
  {
    file: 'lifecycleinventory-review/scripts/run-review.mjs',
    pattern: /--rows-file/u,
    message: 'run-review.mjs help should include a rows-file process review example.',
  },
  {
    file: 'README.md',
    pattern: /process list --json/u,
    message: 'README.md should mention the native process list -> review process rows-file path.',
  },
  {
    file: 'README.zh-CN.md',
    pattern: /process list --json/u,
    message:
      'README.zh-CN.md should mention the native process list -> review process rows-file path.',
  },
];

const targetedSmokeChecks = [
  {
    skill: 'flow-governance-review',
    script: 'flow-governance-review/scripts/run-flow-governance-review.mjs',
    args: ['materialize-db-flows', '--help'],
    description: 'flow-governance-review materialize-db-flows help',
  },
  {
    skill: 'flow-governance-review',
    script: 'flow-governance-review/scripts/run-flow-governance-review.mjs',
    args: ['materialize-approved-decisions', '--help'],
    description: 'flow-governance-review materialize-approved-decisions help',
  },
  {
    skill: 'flow-governance-review',
    script: 'flow-governance-review/scripts/run-flow-governance-review-fixture.mjs',
    args: [],
    description: 'flow-governance-review end-to-end fixture',
  },
  {
    skill: 'lifecycleinventory-review',
    script: 'lifecycleinventory-review/scripts/run-review.mjs',
    args: ['--profile', 'process', '--help'],
    description: 'process review profile help',
  },
  {
    skill: 'lifecycleinventory-review',
    script: 'lifecycleinventory-review/scripts/run-review.mjs',
    args: ['--profile', 'lifecyclemodel', '--help'],
    description: 'lifecyclemodel review profile help',
  },
];

function fail(message) {
  throw new Error(message);
}

function parseArgs(rawArgs) {
  const { cliDir, args } = normalizeCliRuntimeArgs(rawArgs, { repoRoot });

  if (args.includes('-h') || args.includes('--help')) {
    printHelp();
    process.exit(0);
  }

  return {
    cliDir,
    targets: args,
  };
}

function printHelp() {
  console.log(`Usage:
  node scripts/validate-skills.mjs [--cli-dir <dir>] [skill-path ...]

Examples:
  node scripts/validate-skills.mjs
  node scripts/validate-skills.mjs lifecycleinventory-review process-hybrid-search
  node scripts/validate-skills.mjs --cli-dir ../tiangong-lca-cli lifecycleinventory-review
  node scripts/validate-skills.mjs --cli-dir ../tiangong-cli lifecycleinventory-review

What this validates:
  - SKILL.md frontmatter presence
  - agents/openai.yaml interface keys
  - Node syntax for skill wrapper .mjs files
  - wrapper --help smoke checks through the TianGong CLI
  - targeted doc guards that prevent stale shell/Python migration wording

CLI runtime:
  - default local repo validation uses the first sibling repo that exists:
    - ../tiangong-lca-cli
    - ../tiangong-cli
  - otherwise wrappers fall back to ${publishedCliCommand}
  - use --cli-dir or TIANGONG_LCA_CLI_DIR to force a local working tree
`.trim());
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    stdio: 'pipe',
    encoding: 'utf8',
    ...options,
  });

  if (result.error) {
    throw result.error;
  }

  if (typeof result.status === 'number' && result.status !== 0) {
    const stderr = result.stderr?.trim() || result.stdout?.trim() || `exit code ${result.status}`;
    fail(`${command} ${args.join(' ')} failed: ${stderr}`);
  }
}

function ensureCliBuild(cliDir, required) {
  if (!cliDir || !required) {
    return;
  }
  const cliBin = path.join(cliDir, 'bin', 'tiangong.js');
  const cliDist = path.join(cliDir, 'dist', 'src', 'main.js');
  if (!existsSync(cliBin)) {
    fail(`Cannot find TianGong CLI at ${cliBin}. Set TIANGONG_LCA_CLI_DIR or pass --cli-dir.`);
  }
  if (!existsSync(cliDist)) {
    fail(`TianGong CLI is missing built artifacts at ${cliDist}. Run npm run build in tiangong-lca-cli first.`);
  }
}

function normalizeSkillTarget(target) {
  const directPath = path.isAbsolute(target) ? target : path.join(repoRoot, target);
  if (existsSync(directPath) && statSync(directPath).isDirectory()) {
    return directPath;
  }

  const namedPath = path.join(repoRoot, target);
  if (existsSync(namedPath) && statSync(namedPath).isDirectory()) {
    return namedPath;
  }

  fail(`Skill path not found: ${target}`);
}

function collectWrapperScripts(skillDir) {
  const scriptsDir = path.join(skillDir, 'scripts');
  if (!existsSync(scriptsDir)) {
    return [];
  }

  return readdirSync(scriptsDir)
    .filter((entry) => entry.endsWith('.mjs'))
    .sort()
    .map((entry) => path.join(scriptsDir, entry));
}

function scriptUsesCliLauncher(scriptFile) {
  return readFileSync(scriptFile, 'utf8').includes('cli-launcher.mjs');
}

function assertSkillFrontmatter(skillDir) {
  const skillFile = path.join(skillDir, 'SKILL.md');
  if (!existsSync(skillFile)) {
    fail(`Missing SKILL.md in ${path.relative(repoRoot, skillDir)}`);
  }

  const text = readFileSync(skillFile, 'utf8');
  const frontmatterMatch = text.match(/^---\r?\n([\s\S]*?)\r?\n---/u);
  if (!frontmatterMatch) {
    fail(`SKILL.md in ${path.relative(repoRoot, skillDir)} must start with YAML frontmatter.`);
  }
  if (!/^\s*name:\s*.+$/mu.test(frontmatterMatch[1])) {
    fail(`SKILL.md in ${path.relative(repoRoot, skillDir)} is missing a frontmatter name.`);
  }
  if (!/^\s*description:\s*.+$/mu.test(frontmatterMatch[1])) {
    fail(`SKILL.md in ${path.relative(repoRoot, skillDir)} is missing a frontmatter description.`);
  }
}

function assertAgentMetadata(skillDir) {
  const agentFile = path.join(skillDir, 'agents', 'openai.yaml');
  if (!existsSync(agentFile)) {
    fail(`Missing agents/openai.yaml in ${path.relative(repoRoot, skillDir)}`);
  }

  const text = readFileSync(agentFile, 'utf8');
  for (const key of ['interface:', 'display_name:', 'short_description:', 'default_prompt:']) {
    if (!text.includes(key)) {
      fail(`${path.relative(repoRoot, agentFile)} is missing required key ${key}`);
    }
  }
}

function runNodeChecks(scriptFiles) {
  scriptFiles.forEach((scriptFile) => {
    run(process.execPath, ['--check', scriptFile], {
      cwd: repoRoot,
    });
  });
}

function runHelpSmoke(scriptFiles, cliDir) {
  scriptFiles.forEach((scriptFile) => {
    run(process.execPath, [scriptFile, '--help'], {
      cwd: repoRoot,
      env: withCliRuntimeEnv(process.env, cliDir),
    });
  });
}

function runTargetedSmokeChecks(skillDirs, cliDir) {
  let count = 0;
  const selectedSkills = new Set(skillDirs.map((skillDir) => path.basename(skillDir)));

  targetedSmokeChecks.forEach((check) => {
    if (!selectedSkills.has(check.skill)) {
      return;
    }

    const scriptFile = path.join(repoRoot, check.script);
    if (!existsSync(scriptFile)) {
      fail(`Targeted smoke script is missing for ${check.description}: ${check.script}`);
    }

    run(process.execPath, [scriptFile, ...check.args], {
      cwd: repoRoot,
      env: withCliRuntimeEnv(process.env, cliDir),
    });
    count += 1;
  });

  return count;
}

function runDocGuards() {
  docGuards.forEach((guard) => {
    const filePath = path.join(repoRoot, guard.file);
    if (!existsSync(filePath)) {
      fail(`Guarded file is missing: ${guard.file}`);
    }
    const text = readFileSync(filePath, 'utf8');
    if (guard.pattern.test(text)) {
      fail(`${guard.message} (${guard.file})`);
    }
  });
}

function runRequiredDocPatterns() {
  requiredDocPatterns.forEach((guard) => {
    const filePath = path.join(repoRoot, guard.file);
    if (!existsSync(filePath)) {
      fail(`Required-doc file is missing: ${guard.file}`);
    }
    const text = readFileSync(filePath, 'utf8');
    if (!guard.pattern.test(text)) {
      fail(`${guard.message} (${guard.file})`);
    }
  });
}

function main() {
  const { cliDir, targets } = parseArgs(process.argv.slice(2));
  runDocGuards();
  runRequiredDocPatterns();

  const skillDirs = (targets.length ? targets : defaultSkillNames)
    .map((target) => normalizeSkillTarget(target))
    .sort((left, right) => left.localeCompare(right));
  const skillPlans = skillDirs.map((skillDir) => ({
    skillDir,
    scriptFiles: collectWrapperScripts(skillDir),
  }));
  const needsCliRuntime =
    skillPlans.some(({ scriptFiles }) => scriptFiles.some((scriptFile) => scriptUsesCliLauncher(scriptFile))) ||
    targetedSmokeChecks.some((check) =>
      skillPlans.some(({ skillDir }) => path.basename(skillDir) === check.skill),
    );

  ensureCliBuild(cliDir, needsCliRuntime);

  let scriptCount = 0;
  skillPlans.forEach(({ skillDir, scriptFiles }) => {
    assertSkillFrontmatter(skillDir);
    assertAgentMetadata(skillDir);
    scriptCount += scriptFiles.length;
    runNodeChecks(scriptFiles);
    runHelpSmoke(scriptFiles, cliDir);
  });
  const targetedSmokeCount = runTargetedSmokeChecks(skillDirs, cliDir);

  console.log(
    `Validated ${skillDirs.length} skill directories, ${scriptCount} wrapper scripts, ${targetedSmokeCount} targeted smokes, ${docGuards.length} negative doc guards, and ${requiredDocPatterns.length} required doc patterns.`,
  );
}

try {
  main();
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Validation failed: ${message}`);
  process.exit(1);
}
