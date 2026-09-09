import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const entry = fileURLToPath(new URL("../foundry-tidas-import/", import.meta.url));
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const version = "0.1.7";
const source = "d0d2e7819e5ff573fb427063d13f83a2cb47ba70";

test("copied Foundry skill runs the public locked runtime and rejects changed installation inputs", {
  timeout: 1_200_000,
}, (t) => {
  const lockPath = path.join(entry, "scripts", "bootstrap-lock.json");
  assert.ok(fs.existsSync(lockPath), "The final independently verified F1 bootstrap lock must be shipped.");
  const lock = JSON.parse(fs.readFileSync(lockPath, "utf8"));
  assert.equal(lock.schema, "tiangong-lca.runtime-bootstrap-lock.v1");
  assert.equal(lock.manifest_url,
    `https://github.com/tiangong-lca/data-foundry/releases/download/foundry-runtime-v${version}/runtime-manifest.json`);
  const platform = `${process.platform}-${process.arch}`;
  assert.ok(["linux-x64", "linux-arm64", "darwin-arm64", "win32-x64"].includes(platform));
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "foundry-public-skill-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const copy = path.join(root, "copied skill");
  fs.cpSync(entry, copy, { recursive: true, dereference: false });
  const userRoot = path.join(root, "user");
  const temporary = path.join(root, "temp");
  const workspace = path.join(root, "项目 workspace");
  const localAppData = path.join(userRoot, "AppData", "Local");
  const appData = path.join(userRoot, "AppData", "Roaming");
  for (const directory of [userRoot, temporary, workspace, localAppData, appData])
    fs.mkdirSync(directory, { recursive: true });
  const windows = process.platform === "win32";
  const systemRoot = process.env.SystemRoot ?? process.env.SYSTEMROOT;
  if (windows) assert.ok(systemRoot);
  const systemPath = windows
    ? [path.join(systemRoot, "System32"), systemRoot].join(path.delimiter)
    : "/usr/bin:/bin:/usr/sbin:/sbin";
  const env = {
    HOME: userRoot, USERPROFILE: userRoot, LOCALAPPDATA: localAppData, APPDATA: appData,
    XDG_CACHE_HOME: path.join(userRoot, ".cache"), XDG_CONFIG_HOME: path.join(userRoot, ".config"),
    TEMP: temporary, TMP: temporary, TMPDIR: temporary,
    PATH: systemPath, Path: systemPath, LANG: "C", LC_ALL: "C", TZ: "UTC",
  };
  for (const key of ["SystemRoot", "SYSTEMROOT", "WINDIR", "windir", "ComSpec", "COMSPEC",
    "PATHEXT", "PROCESSOR_ARCHITECTURE", "PROCESSOR_ARCHITEW6432"])
    if (process.env[key] !== undefined) env[key] = process.env[key];
  let shell = "/bin/sh";
  if (windows) {
    const found = spawnSync(path.join(systemRoot, "System32", "where.exe"), ["pwsh"], {
      encoding: "utf8", timeout: 30_000, shell: false,
    });
    assert.equal(found.status, 0, "The native Windows qualification host requires PowerShell.");
    shell = found.stdout.trim().split(/\r?\n/u)[0];
    assert.ok(path.isAbsolute(shell) && fs.statSync(shell).isFile());
  }
  const script = path.join(copy, "scripts", `tiangong-runtime-bootstrap.${windows ? "ps1" : "sh"}`);
  const prefix = windows ? ["-NoProfile", "-NonInteractive", "-File", script] : [script];
  const cache = path.join(windows ? localAppData : process.platform === "darwin"
    ? path.join(userRoot, "Library", "Caches") : env.XDG_CACHE_HOME, "tiangong-lca", "runtimes", "v1");
  assert.equal(fs.existsSync(cache), false);
  const checks = [];
  const run = (phase, args, expectedExit = 0, action = null) => {
    const started = Date.now();
    const result = spawnSync(action?.executable ?? shell, action?.argv ?? [...prefix, ...args], {
      cwd: action?.cwd ?? workspace, env, shell: false, encoding: "utf8", timeout: 900_000,
      stdio: ["ignore", "pipe", "pipe"], maxBuffer: 8 * 1024 * 1024,
    });
    assert.ifError(result.error);
    assert.equal(result.signal, null);
    assert.equal(result.status, expectedExit, `${phase}: ${result.stderr.slice(0, 4096)}`);
    checks.push({ phase, exit: result.status, milliseconds: Date.now() - started });
    return result;
  };
  const operation = (phase, args, expectedExit = 0) => {
    const result = run(phase, args, expectedExit);
    const value = JSON.parse(result.stdout);
    assert.equal(value.schema, "tiangong-foundry.operation-result.v1");
    return value;
  };
  const initial = operation("cold-install", ["workspace", "init", "--workspace", workspace, "--json"]);
  assert.equal(initial.status, "ready");
  const manifestFile = path.join(cache, "manifests", `${lock.manifest_sha256}.json`);
  const manifestBytes = fs.readFileSync(manifestFile);
  assert.equal(manifestBytes.length, lock.manifest_bytes);
  assert.equal(hash(manifestBytes), lock.manifest_sha256);
  const manifest = JSON.parse(manifestBytes);
  assert.equal(manifest.product.version, version);
  const componentKeys = fs.readdirSync(path.join(cache, "components")).sort();
  const doctor = operation("warm-start", ["doctor", "--workspace", workspace, "--json"]);
  assert.equal(doctor.status, "ready");
  assert.equal(doctor.runtime_identity.foundry.package_version, version);
  const qualification = doctor.runtime_identity.qualification;
  assert.equal(qualification.status, "ready");
  assert.equal(qualification.identity.cli.package_version, "0.1.13");
  assert.equal(qualification.identity.cli.node_version, "24.19.0");
  assert.equal(qualification.identity.tidas.binary_version, "0.3.0");
  assert.deepEqual(fs.readdirSync(path.join(cache, "components")).sort(), componentKeys);
  const application = manifest.components.find((component) => component.id === "foundry" && component.platform === platform);
  assert.ok(application);
  // Receipts locate an installation; the independently shipped manifest owns
  // the content proof and the source expectation is reviewed separately.
  const matches = componentKeys.filter((key) => {
    const receipt = JSON.parse(fs.readFileSync(path.join(cache, "components", key, "receipt.json"), "utf8"));
    return receipt.archive_sha256 === application.archive.sha256 && receipt.content_sha256 === application.content_sha256;
  });
  assert.equal(matches.length, 1);
  const provenancePath = "metadata/runtime-provenance.json";
  const provenanceFact = application.files.find((file) => file.path === provenancePath);
  assert.ok(provenanceFact);
  const provenanceBytes = fs.readFileSync(path.join(cache, "components", matches[0], "root", provenancePath));
  assert.equal(provenanceBytes.length, provenanceFact.bytes);
  assert.equal(hash(provenanceBytes), provenanceFact.sha256);
  const provenance = JSON.parse(provenanceBytes);
  assert.equal(provenance.scope, "published-release");
  assert.equal(provenance.source.commit, source);
  assert.equal(provenance.package.version, version);
  assert.equal(provenance.published_package.source.gitCommit, source);
  assert.equal(provenance.cli.source.gitCommit, "b5e209259d3bb06205b9af131b1c0edc3fba6da2");

  // This is a credential-free local cleanup task, not the live RC01–RC06 account case.
  const selected = path.join(workspace, "source.jsonl");
  fs.writeFileSync(selected, '{"flowDataSet":{}}\n');
  const spec = path.join(workspace, "task.json");
  fs.writeFileSync(spec, JSON.stringify({
    schema: "tiangong-foundry.task-start.v1", request_id: "copied-skill-install",
    actor_id: "skill-qualifier", lane: "external-dataset-curated-import", profile_id: "generic",
    target_entities: ["flow"], sources: [{ path: selected }], seed: null, account_intent: null,
    preparation: { operation: "dataset-curation-cleanup", type: "flow", input: selected,
      source_input: null, output_directory: "outputs/cleanup" },
  }, null, 2));
  const task = operation("task-start", ["task", "start", "--workspace", workspace, "--spec", spec, "--json"]);
  assert.equal(task.status, "ready");
  assert.ok(task.task_id);
  const taskArgs = ["--workspace", workspace, "--task", task.task_id, "--actor", "skill-qualifier", "--json"];
  const status = operation("task-status", ["task", "status", ...taskArgs]);
  assert.equal(status.task_id, task.task_id);
  const action = status.next_actions.find((item) => item.kind === "command");
  assert.ok(action);
  assert.equal(action.code, "resume_local_preparation");
  assert.equal(fs.realpathSync(action.cwd), fs.realpathSync(workspace));
  const resumed = JSON.parse(run("returned-task-resume", [], 0, action).stdout);
  assert.equal(resumed.schema, "tiangong-foundry.operation-result.v1");
  assert.equal(resumed.task_id, task.task_id);
  assert.ok(["ready", "completed"].includes(resumed.status));
  assert.equal(resumed.runtime_identity.qualification.status, "ready");
  const manifestIndex = action.argv.indexOf("--manifest");
  assert.ok(manifestIndex > 0);
  const actionManifest = action.argv[manifestIndex + 1];
  assert.ok(path.isAbsolute(actionManifest));
  const manifestRelative = path.relative(fs.realpathSync(root), fs.realpathSync(actionManifest));
  assert.ok(manifestRelative && manifestRelative !== ".."
    && !manifestRelative.startsWith(`..${path.sep}`) && !path.isAbsolute(manifestRelative));
  assert.equal(fs.lstatSync(actionManifest).isSymbolicLink(), false);
  const actionManifestBytes = fs.readFileSync(actionManifest);
  assert.equal(hash(actionManifestBytes), lock.manifest_sha256);
  fs.appendFileSync(actionManifest, "\n");
  try {
    // The public CLI runtime-error contract returns EX_UNAVAILABLE (69).
    assert.match(run("returned-manifest-tamper-refused", [], 69, action).stderr,
      /RUNTIME_MANIFEST_INTEGRITY/u);
  } finally { fs.writeFileSync(actionManifest, actionManifestBytes); }
  assert.equal(operation("developer-command-refused", ["profiles-list", "--workspace", workspace, "--json"], 2).status, "needs_input");

  const mustNotExist = path.join(root, "must-not-exist");
  const originalScript = fs.readFileSync(script);
  fs.appendFileSync(script, "\n");
  try {
    assert.match(run("script-tamper-refused", ["workspace", "init", "--workspace", mustNotExist, "--json"], 1).stderr,
      /bootstrap_script_changed/u);
    assert.equal(fs.existsSync(mustNotExist), false);
  } finally { fs.writeFileSync(script, originalScript); }
  const lockCopy = path.join(copy, "scripts", "bootstrap-lock.json");
  const originalLock = fs.readFileSync(lockCopy);
  fs.unlinkSync(lockCopy);
  try {
    assert.match(run("missing-lock-refused", ["workspace", "init", "--workspace", mustNotExist, "--json"], 1).stderr,
      /missing_adjacent_lock/u);
    assert.equal(fs.existsSync(mustNotExist), false);
  } finally { fs.writeFileSync(lockCopy, originalLock); }
  fs.appendFileSync(manifestFile, "\n");
  try {
    assert.match(run("manifest-tamper-refused", ["doctor", "--workspace", workspace, "--json"], 1).stderr,
      /file_size_mismatch|file_sha256_mismatch/u);
  } finally { fs.writeFileSync(manifestFile, manifestBytes); }
  const keyPrefix = platform.replaceAll("-", "_");
  const integrity = path.join(cache, "components", lock[`${keyPrefix}_component_key`], "root", lock[`${keyPrefix}_integrity_path`]);
  const originalIntegrity = fs.readFileSync(integrity);
  fs.writeFileSync(integrity, "changed\n");
  try {
    assert.match(run("base-index-tamper-refused", ["doctor", "--workspace", workspace, "--json"], 1).stderr,
      /integrity_file_changed/u);
  } finally { fs.writeFileSync(integrity, originalIntegrity); }
  assert.equal(operation("restored-installation", ["doctor", "--workspace", workspace, "--json"]).status, "ready");
  const report = { schema: "tiangong-skills.foundry-public-install.v1", status: "passed", platform,
    version, source: provenance.source.commit, manifest_sha256: lock.manifest_sha256,
    initial_cache: "empty", credential_scope: "none", runtime_identity: doctor.runtime_identity, checks };
  const proofDirectory = process.env.FOUNDRY_INSTALL_PROOF_DIR;
  if (proofDirectory) {
    assert.ok(path.isAbsolute(proofDirectory));
    fs.mkdirSync(proofDirectory, { recursive: true });
    fs.writeFileSync(path.join(proofDirectory, `${platform}.json`), JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  }
  t.diagnostic(JSON.stringify({ platform, version, manifest_sha256: lock.manifest_sha256, checks }));
});
