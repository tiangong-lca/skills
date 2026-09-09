import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const scripts = fileURLToPath(new URL("../foundry-tidas-import/scripts/", import.meta.url));
// Immutable cli-v0.1.13 source: b5e209259d3bb06205b9af131b1c0edc3fba6da2.
// These original CLI scripts are distributed unchanged; F1 supplies the adjacent lock.
const expected = {
  // Independently verified final foundry-runtime-v0.1.7 public asset.
  "bootstrap-lock.json": "cd89d156f95c1af9e7a48342edf84c4d00c05e027eac99ff5d658a57f2fb4b01",
  "../assets/licenses/tiangong-cli-LICENSE": "5ca31e8840557caad889b275beef7c9d56e67efede031e71228c5ff8f0d4135e",
  "tiangong-runtime-bootstrap.sh": "7aa826448f7b0e1d25f59a19a6f8c906621a82f0ee956124d9f93484885b9fac",
  "tiangong-runtime-bootstrap.ps1": "a797bc5269386a0f45fbc5bebb7d634b37fc8a164500aaaa4905be5e61bf8f65",
};

test("Foundry bootstrap copies and final lock match independently verified public assets", () => {
  for (const [name, digest] of Object.entries(expected)) {
    const file = path.join(scripts, name);
    assert.ok(fs.lstatSync(file).isFile());
    assert.equal(createHash("sha256").update(fs.readFileSync(file)).digest("hex"), digest, name);
  }
});

test("an isolated bootstrap without its adjacent lock refuses before installing or launching", (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "foundry-skill-bootstrap-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const copy = path.join(root, "copied skill"), user = path.join(root, "user");
  const temporary = path.join(root, "temp");
  const localAppData = path.join(user, "AppData", "Local");
  const appData = path.join(user, "AppData", "Roaming");
  fs.mkdirSync(copy);
  for (const directory of [temporary, localAppData, appData])
    fs.mkdirSync(directory, { recursive: true });
  const windows = process.platform === "win32";
  const systemRoot = process.env.SystemRoot ?? process.env.SYSTEMROOT;
  if (windows) assert.ok(systemRoot);
  const shell = windows
    ? path.join(systemRoot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    : "/bin/sh";
  const script = path.join(copy, `tiangong-runtime-bootstrap.${windows ? "ps1" : "sh"}`);
  fs.copyFileSync(path.join(scripts, path.basename(script)), script);
  const env = windows ? {
    SystemRoot: systemRoot, USERPROFILE: user, LOCALAPPDATA: localAppData,
    PATH: path.join(systemRoot, "System32"),
  } : {
    HOME: user, USERPROFILE: user,
    XDG_CACHE_HOME: path.join(user, ".cache"),
    XDG_CONFIG_HOME: path.join(user, ".config"),
    LOCALAPPDATA: localAppData, APPDATA: appData,
    TEMP: temporary, TMP: temporary, TMPDIR: temporary,
    PATH: "/usr/bin:/bin:/usr/sbin:/sbin",
  };
  if (windows) {
    const started = Date.now();
    const startup = spawnSync(shell, [
      "-NoProfile", "-NonInteractive", "-Command", "Write-Output 'bootstrap-host-ready'",
    ], { cwd: copy, env, encoding: "utf8", shell: false, stdio: ["ignore", "pipe", "pipe"], timeout: 90_000 });
    if (startup.error) {
      const control = spawnSync(shell, [
        "-NoProfile", "-NonInteractive", "-Command", "Write-Output 'bootstrap-host-ready'",
      ], { cwd: copy, encoding: "utf8", shell: false, stdio: ["ignore", "pipe", "pipe"], timeout: 30_000 });
      assert.fail(JSON.stringify({ private: { status: startup.status, error: startup.error.code }, hostControl: { status: control.status, error: control.error?.code, ready: control.stdout?.trim() === "bootstrap-host-ready" } }));
    }
    assert.equal(startup.status, 0, startup.stderr);
    assert.equal(startup.stdout.trim(), "bootstrap-host-ready");
    t.diagnostic(`Private Windows PowerShell startup completed in ${Date.now() - started} ms.`);
  }
  const result = spawnSync(
    shell,
    windows ? ["-NoProfile", "-NonInteractive", "-File", script, "doctor", "--json"] : [script, "doctor", "--json"],
    { cwd: copy, env, encoding: "utf8", shell: false, stdio: ["ignore", "pipe", "pipe"], timeout: windows ? 90_000 : 30_000 },
  );
  assert.ifError(result.error);
  assert.notEqual(result.status, 0);
  assert.match(result.stdout + result.stderr, /bootstrap_error:missing_adjacent_lock/u);
  // The shell may initialize its own cache, but the rejected bootstrap must not
  // create the CLI-owned runtime namespace or alter the copied skill.
  for (const base of windows ? [localAppData] : [env.XDG_CACHE_HOME, path.join(user, "Library", "Caches")])
    assert.equal(fs.existsSync(path.join(base, "tiangong-lca", "runtimes")), false);
  assert.deepEqual(fs.readdirSync(copy), [path.basename(script)]);
});

test("the original C1 PowerShell download helper accepts a real public response", {
  skip: process.platform !== "win32",
}, async (t) => {
  const url = "https://nodejs.org/dist/v24.19.0/SHASUMS256.txt";
  const response = await fetch(url, { redirect: "error", signal: AbortSignal.timeout(30_000) });
  assert.equal(response.status, 200);
  const expectedBytes = Buffer.from(await response.arrayBuffer());
  assert.ok(expectedBytes.length > 0 && expectedBytes.length < 64 * 1024);
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "foundry-c1-download-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const original = path.join(scripts, "tiangong-runtime-bootstrap.ps1");
  const output = path.join(root, "checksums.txt"), probe = path.join(root, "probe.ps1");
  const quote = (value) => "'" + value.replaceAll("'", "''") + "'";
  fs.writeFileSync(probe, [
    "Set-StrictMode -Version Latest",
    "$ErrorActionPreference = 'Stop'",
    "$tokens = $null; $errors = $null",
    `$ast = [Management.Automation.Language.Parser]::ParseFile(${quote(original)}, [ref]$tokens, [ref]$errors)`,
    "if ($errors.Count) { throw 'Original bootstrap did not parse' }",
    "$functions = $ast.FindAll({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] }, $false)",
    "foreach ($function in $functions) { . ([scriptblock]::Create($function.Extent.Text)) }",
    `Download ${quote(url)} ${quote(output)} ${expectedBytes.length}`,
  ].join("\n") + "\n");
  // Exercise the unchanged download owner in the PowerShell host already used
  // by Foundry's Windows qualification, before any final F1 asset is published.
  const result = spawnSync("pwsh", ["-NoProfile", "-NonInteractive", "-File", probe], {
    cwd: root, encoding: "utf8", shell: false, stdio: ["ignore", "pipe", "pipe"],
    timeout: 60_000, maxBuffer: 1024 * 1024,
  });
  assert.ifError(result.error);
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(fs.readFileSync(output), expectedBytes);
});
