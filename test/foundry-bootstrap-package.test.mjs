import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const scripts = fileURLToPath(new URL("../foundry-tidas-import/scripts/", import.meta.url));
// Immutable cli-v0.1.11 source: 1f9f75fcae3c386e601b49a7da95df0d6a526f6f.
// These original CLI scripts are distributed unchanged; F1 supplies the adjacent lock.
const expected = {
  "../assets/licenses/tiangong-cli-LICENSE": "5ca31e8840557caad889b275beef7c9d56e67efede031e71228c5ff8f0d4135e",
  "tiangong-runtime-bootstrap.sh": "a7055855e89d6f0781b1d44ac4d05c71053a14855906fa5e0c7e7e3ebf5867f5",
  "tiangong-runtime-bootstrap.ps1": "8931fa991eb94cd1f801f71ca729d9c4b851d2039a914712b1ecb7895de38fa0",
};

test("Foundry bootstrap copies match the immutable C1 public scripts", () => {
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
  fs.mkdirSync(copy);
  fs.mkdirSync(user);
  const windows = process.platform === "win32";
  const script = path.join(copy, `tiangong-runtime-bootstrap.${windows ? "ps1" : "sh"}`);
  fs.copyFileSync(path.join(scripts, path.basename(script)), script);
  const env = { HOME: user, USERPROFILE: user, XDG_CACHE_HOME: user, LOCALAPPDATA: user };
  for (const name of ["PATH", "PATHEXT", "SystemRoot", "SYSTEMROOT", "WINDIR", "ComSpec", "COMSPEC"])
    if (process.env[name]) env[name] = process.env[name];
  const result = spawnSync(
    windows ? "powershell.exe" : "/bin/sh",
    windows ? ["-NoProfile", "-NonInteractive", "-File", script, "doctor", "--json"] : [script, "doctor", "--json"],
    { cwd: copy, env, encoding: "utf8", shell: false, timeout: 30_000 },
  );
  assert.ifError(result.error);
  assert.notEqual(result.status, 0);
  assert.match(result.stdout + result.stderr, /bootstrap_error:missing_adjacent_lock/u);
  assert.deepEqual(fs.readdirSync(user), []);
  assert.deepEqual(fs.readdirSync(copy), [path.basename(script)]);
});
