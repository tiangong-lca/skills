import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);
const cliDir = process.env.TIANGONG_OVERVIEW_TEST_CLI_DIR;
const outputDir = process.env.TIANGONG_OVERVIEW_TEST_OUTPUT_DIR;

test("topic overview installs independently with local reference navigation and paired metadata", () => {
  const root = mkdtempSync(path.join(tmpdir(), "installed-topic-skill-"));
  try {
    const installed = path.join(root, "lca-topic-overview");
    cpSync(path.join(repoRoot, "lca-topic-overview"), installed, {
      recursive: true,
    });
    assert.deepEqual(readdirSync(installed).sort(), [
      "SKILL.md",
      "agents",
      "references",
    ]);
    const skill = readFileSync(path.join(installed, "SKILL.md"), "utf8");
    assert.match(skill, /^---\nname: lca-topic-overview\ndescription: /u);
    const metadata = readFileSync(
      path.join(installed, "agents/openai.yaml"),
      "utf8",
    );
    assert.match(metadata, /\$lca-topic-overview/u);
    for (const match of skill.matchAll(/\[[^\]]+\]\(([^)]+)\)/gu)) {
      const reference = path.resolve(installed, match[1]);
      assert.ok(reference.startsWith(installed + path.sep));
      assert.ok(existsSync(reference));
    }
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test(
  "explicit companion CLI runs electricity and steel workflows from raw public fixture data",
  {
    skip:
      !cliDir &&
      "Set TIANGONG_OVERVIEW_TEST_CLI_DIR to qualify the built companion CLI public bin.",
  },
  () => {
    const root = outputDir
      ? path.resolve(outputDir)
      : mkdtempSync(path.join(tmpdir(), "topic-skill-contract-"));
    if (outputDir) mkdirSync(root); // an explicit retained qualification directory must be fresh
    try {
      const bin = path.join(path.resolve(cliDir), "bin", "tiangong-lca.js");
      const run = (args) => {
        const result = spawnSync(
          process.execPath,
          [bin, "dataset", "overview", ...args],
          { cwd: root, env: process.env, shell: false, encoding: "utf8" },
        );
        assert.equal(result.status, 0, result.stderr || result.stdout);
        return JSON.parse(result.stdout);
      };
      assert.equal(
        run(["describe", "--json"]).schema_version,
        "tiangong-lca.overview-capabilities.v1",
      );
      const uuid = (n) =>
        `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`;
      const version = "00.00.001";
      const name = (label) => ({
        baseName: [{ "@xml:lang": "en", "#text": label }],
        treatmentStandardsRoutes: { "#text": "recorded route" },
        mixAndLocationTypes: { "#text": "CN" },
        flowProperties: { "#text": "per unit" },
        functionalUnitFlowProperties: { "#text": "per unit" },
      });
      const exchange = (internal, flow, direction) => ({
        "@dataSetInternalID": String(internal),
        exchangeDirection: direction,
        referenceToFlowDataSet: {
          "@refObjectId": uuid(flow),
          "@version": version,
          "common:shortDescription": { "#text": "electricity input" },
        },
      });
      const processRow = (n, label, exchanges, revision = version) => ({
        id: uuid(n),
        version: revision,
        state_code: 100,
        json: {
          processDataSet: {
            processInformation: {
              dataSetInformation: { name: name(label) },
              quantitativeReference: { referenceToReferenceFlow: "0" },
              geography: {
                locationOfOperationSupplyOrProduction: { "@location": "CN" },
              },
              time: { "common:referenceYear": "2024" },
            },
            exchanges: { exchange: exchanges },
            modellingAndValidation: {
              LCIMethodAndAllocation: {
                typeOfDataSet: "Unit process, single operation",
              },
            },
          },
        },
      });
      const flow = (n, label) => ({
        id: uuid(n),
        version,
        state_code: 100,
        json: {
          flowDataSet: {
            flowInformation: { dataSetInformation: { name: name(label) } },
            modellingAndValidation: {
              LCIMethod: { typeOfDataSet: "Product flow" },
            },
          },
        },
      });
      const instance = (internal, n, connections = []) => ({
        "@dataSetInternalID": internal,
        referenceToProcess: { "@refObjectId": uuid(n), "@version": version },
        connections: { outputExchange: connections },
      });
      const tables = {
        processes: [
          processRow(1, "electricity generation", [exchange(0, 10, "Output")]),
          processRow(
            1,
            "electricity generation",
            [exchange(0, 10, "Output")],
            "00.00.002",
          ),
          processRow(2, "steel production", [
            exchange(0, 11, "Output"),
            exchange(1, 10, "Input"),
            exchange(2, 10, "Input"),
          ]),
        ],
        flows: [flow(10, "electricity product"), flow(11, "steel product")],
        lifecyclemodels: [
          {
            id: uuid(20),
            version,
            state_code: 100,
            json: {
              lifeCycleModelDataSet: {
                lifeCycleModelInformation: {
                  dataSetInformation: { name: name("electricity system") },
                  technology: {
                    processes: {
                      processInstance: [
                        instance("one", 1, [
                          {
                            "@flowUUID": uuid(10),
                            "@version": version,
                            downstreamProcess: {
                              "@id": "two",
                              "@flowUUID": uuid(10),
                              "@version": version,
                            },
                          },
                          {
                            "@flowUUID": uuid(10),
                            "@version": version,
                            downstreamProcess: {
                              "@id": "one",
                              "@flowUUID": uuid(10),
                              "@version": version,
                            },
                          },
                        ]),
                        instance("two", 2),
                      ],
                    },
                  },
                },
              },
            },
          },
        ],
      };
      const inventory = path.join(root, "inventory");
      mkdirSync(inventory);
      const hashes = {};
      for (const [table, rows] of Object.entries(tables)) {
        const bytes = rows.map((row) => JSON.stringify(row) + "\n").join("");
        writeFileSync(path.join(inventory, table + ".jsonl"), bytes);
        hashes[table] = createHash("sha256").update(bytes).digest("hex");
      }
      const completeness = {
        status: "complete",
        complete: true,
        strategy: "postgrest_exact_count_multi_request",
        requested_page_size: 250,
        page_count: 3,
        row_count: 6,
        entity_counts: Object.fromEntries(
          Object.entries(tables).map(([table, rows]) => [table, rows.length]),
        ),
        tables: Object.entries(tables).map(([table, rows]) => ({
          table,
          status: "complete",
          complete: true,
          strategy: "postgrest_exact_count",
          requested_page_size: 250,
          effective_page_size: rows.length,
          pages_fetched: 1,
          rows_fetched: rows.length,
          exact_total: rows.length,
          termination_reason: "content_range_total_reached",
          content_range_verified: true,
          ordering_verified: true,
          duplicate_count: 0,
        })),
      };
      writeFileSync(
        path.join(inventory, "capture.json"),
        JSON.stringify({
          schema_version: "tiangong-lca.overview-capture.v1",
          visibility: "public_state_100_all_owners",
          source: "https://fixture.invalid",
          started_at_utc: "2026-09-08T00:00:00Z",
          finished_at_utc: "2026-09-08T00:00:01Z",
          transactional_snapshot: false,
          completeness,
          sha256: hashes,
        }),
      );
      for (const [topic, processId] of [
        ["electricity", 1],
        ["steel", 2],
      ]) {
        const scope = {
          schema_version: 1,
          topic,
          boundary: "Fixture topic; a consumer is related, not core.",
          terms: [topic],
          core: [],
        };
        const scopeFile = path.join(root, topic + "-scope.json");
        writeFileSync(scopeFile, JSON.stringify(scope));
        run([
          "catalog",
          "--inventory",
          inventory,
          "--scope",
          scopeFile,
          "--out-dir",
          path.join(root, topic + "-candidates"),
        ]);
        const candidates = JSON.parse(
          readFileSync(
            path.join(root, topic + "-candidates", "catalog.json"),
            "utf8",
          ),
        ).candidates;
        assert.deepEqual(
          candidates
            .filter((record) => record.table === "processes")
            .map((record) => record.id),
          [uuid(processId)],
        );
        scope.core = candidates.map((record) => ({
          table: record.table,
          id: record.id,
          reason: "Direct topic in recorded name; fixture identity checked.",
        }));
        writeFileSync(scopeFile, JSON.stringify(scope));
        const analysis = path.join(root, topic + "-analysis");
        run([
          "analyze",
          "--inventory",
          inventory,
          "--scope",
          scopeFile,
          "--out-dir",
          analysis,
        ]);
        const report = JSON.parse(
          readFileSync(path.join(analysis, "overview.json"), "utf8"),
        );
        assert.equal(report.core_counts[0].objects, 1);
        assert.equal(
          report.core_counts[0].public_revisions,
          topic === "electricity" ? 2 : 1,
        );
        assert.equal(report.products.confirmed_flow_ids.length, 1);
        assert.equal(
          report.model_connections[0].status,
          "exact_public_references",
        );
        const power = report.flow_usage.find((use) => use.flow_id === uuid(10));
        assert.equal(power.process_count, 2);
        assert.equal(power.exchange_occurrences, 3);
        for (const file of [
          "overview.html",
          "overview.md",
          "overview.json",
          "scope.json",
          "records.csv",
          "statistics.csv",
          "flow-usage.csv",
          "model-connections.csv",
        ])
          assert.ok(existsSync(path.join(analysis, file)), file);
        assert.match(
          readFileSync(path.join(analysis, "overview.md"), "utf8"),
          /\| processes \| 1 \|/u,
        );
        const html = readFileSync(path.join(analysis, "overview.html"), "utf8");
        assert.deepEqual(
          JSON.parse(
            html.match(
              /<script id="overview-data" type="application\/json">([\s\S]+?)<\/script>/u,
            )[1],
          ),
          report,
        );
      }
    } finally {
      if (!outputDir) rmSync(root, { recursive: true, force: true });
    }
  },
);
