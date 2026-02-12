# Process From Flow Workflow Guide (LangGraph Core + Origin Orchestration)

This reference is a self-contained migration of the `process_from_flow` workflow guide. It is adapted for standalone skill runtime where the reference flow is provided directly by the agent (file path, inline JSON, or stdin via wrapper script), not from any fixed input directory.

## Overview
- Goal: Derive ILCD process datasets from a reference flow dataset (ILCD JSON). Exchange flow `uuid` and `shortDescription` must come from `search_flows` candidates; use placeholders only when no match exists.
- Scope: Covers the LangGraph core flow in `src/tiangong_lca_spec/process_from_flow/service.py` and the orchestration layer under `scripts/origin/`.
- Outputs: `process_datasets` and `source_datasets`, plus artifacts under `artifacts/process_from_flow/<run_id>/`.

## Cross References
- CLI entrypoints: `scripts/origin/process_from_flow_langgraph.py` and `scripts/origin/process_from_flow_workflow.py` (read module docstrings and `--help`).
- Wrapper entrypoint: `scripts/run-process-automated-builder.sh` (supports `--flow-file`, `--flow-json`, `--flow-stdin`).

## Architecture and Main Flow
### Layer Responsibilities
- LangGraph core layer: `ProcessFromFlowService` runs the main inference chain (references -> routes -> processes -> exchanges -> matching -> datasets).
- Origin orchestration layer: `scripts/origin` handles SI download/parse, usability tagging, run/resume, publishing, and cleanup.

### Main Flow Outline
- Step 0 load_flow: Parse reference flow and build summary.
- Step 1 references + tech routes: 1a search -> 1b fulltext -> 1c clustering -> technology routes.
- Step 2 split processes: Split unit processes into an ordered chain.
- Step 3 generate exchanges: Create per-process input/output exchanges.
- Step 3b enrich exchange amounts: Extract or estimate amounts/units from text and SI.
- Step 4 match flows: Search flows and select candidates to fill `uuid` and `shortDescription`.
- Step 4b align exchange units: Validate unit-group compatibility and convert to reference units.
- Step 4c density conversion (optional): Estimate density for mass<->volume mismatches on product/waste flows.
- Step 1f build sources: Generate ILCD source datasets and references.
- Step 5a intended applications: Write intended applications per process before dataset build.
- Step 5 build process datasets: Emit final ILCD process datasets.
- Step 6 resolve placeholders: Post-process unmatched exchanges with a second search pass.
- Step 7 balance review: Mass/energy balance check (reports only).
- Step 8 data cut-off and completeness: Summarize missing values/conversions, write data cut-off principles (LLM first, rule fallback), and rewrite process-level `dataTreatment` from final balance review.

## LangGraph Core Workflow (ProcessFromFlowService)
### Entry and Dependencies
- Entry: `ProcessFromFlowService.run(flow_path, operation="produce", initial_state=None, stop_after=None)`.
- Dependencies: LLM (routes/split/exchanges/selection), flow search `search_flows`, candidate selector (LLM selector recommended), optional Translator/MCP client.
- `stop_after`: `references`, `tech`, `processes`, `exchanges`, `matches`, `sources` (CLI orchestration also supports `datasets`).

### Node Details (from coarse to fine)
0) load_flow
- Read `flow_path`, build `flow_dataset` and `flow_summary` (multi-language names, classification, comments, UUID, version).

1) references + tech routes
- 1a reference_search: Search technical-route literature -> `scientific_references.step_1a_reference_search` (default `topK=10`).
- 1b reference_fulltext: Deduplicate DOIs and fetch full text (`filter: {"doi": [...]}` + `topK=1` + `extK`) -> `scientific_references.step_1b_reference_fulltext`.
- 1c reference_clusters: Cluster by system boundary/main chain/intermediate flows -> `scientific_references.step_1c_reference_clusters`.
- Step 1 route output: Produce `technology_routes` with route summary, key inputs/outputs, key unit processes, assumptions/scope, and attach `supported_dois` and `route_evidence`.
- If Step 1a/1b/1c has no usable references, Steps 1-3 fall back to common sense and must mark `expert_judgement` with reasons.

2) split_processes
- Split each route into ordered unit processes; chain intermediates must match, and the last process produces/treats `load_flow`.
- Required fields: `technology`, `inputs`, `outputs`, `boundary`, `assumptions`, and `exchange_keywords`.
- `name_parts` must include `base_name`, `treatment_and_route`, `mix_and_location`, `quantitative_reference`, where `quantitative_reference` is numeric.
- Provide a geography decision per process (ILCD location code) and document representativeness limits in `descriptionOfRestrictions` (for example, non-local input datasets).
- When evidence is aggregated, mark `aggregation_scope` and `allocation_strategy` in `assumptions`.
- If references are usable, extra split evidence can be retrieved and stored in `scientific_references.step2`.

3) generate_exchanges
- Use `EXCHANGES_PROMPT` to generate exchanges; `is_reference_flow` aligns with `reference_flow_name` (Output for production, Input for treatment).
- Exchange names must be searchable and not composite; fill unit/amount (placeholders if unknown).
- Emissions add media suffix (`to air`, `to water`, `to soil`), plus `flow_type` and `search_hints`.
- Append machine-readable tags to each exchange `generalComment`: `[tg_io_kind_tag=<flow_type>] [tg_io_uom_tag=<unit>]`.
- Do not use ambiguous tag keys such as `classification`, `category`, or `typeOfDataSet`.
- Assign `material_role` for each exchange (`raw_material|auxiliary|catalyst|energy|emission|product|waste|service|unknown`); `balance_exclude` is optional explicit metadata, while `balance_review` still excludes auxiliary/catalyst by role even when the flag is absent.
- Every exchange records `data_source` and `evidence`; inferred items must mark `source_type=expert_judgement`.
- If references are usable, extra exchange evidence can be retrieved and stored in `scientific_references.step3`.

3b) enrich_exchange_amounts
- Use `EXCHANGE_VALUE_PROMPT` with full text and SI to extract verifiable values, writing `value_citations` and `value_evidence`, then filling amount/unit.
- Missing values remain placeholders; if boundary plus quantitative reference exist, `INDUSTRY_AVERAGE_PROMPT` may estimate and store `scientific_references.industry_average`.
- Scalable exchanges use `basis_*` for conversion and add scaling notes.

4) match_flows
- Search flows for each exchange (top 10 candidates); use LLM selector when available, fallback to `SimilarityCandidateSelector` when no LLM.
- Record `flow_search.query/candidates/selected_uuid/selected_reason/selector/unmatched` and fill `uuid/shortDescription`.
- Only add matching info; do not overwrite `data_source` and `evidence`.

4b) align_exchange_units
- Validate unit-group compatibility using flow unit groups; convert same-dimension units to the flow reference unit and write back amount/unit.
- Cross-dimension mismatches (for example, `kg` vs `m3`) mark `flow_search.unit_check.status=mismatch`; unknown units mark `review`.

4c) density_conversion (optional)
- Enabled only when `allow_density_conversion=true` or CLI passes `--allow-density-conversion`.
- Triggered only when `unit_check` is mass<->volume mismatch and `flow_type` is `product` or `waste`.
- LLM returns `density_value`, `density_unit`, and `assumptions` (`source_type=expert_judgement`); conversion sets `unit_check.status=converted_by_density` and records `density_used` while preserving originals.
- Density assumptions/conversion evidence stay on exchange fields (`flow_search.unit_check` and `density_used`); process-level treatment text is rewritten in Step 8.

1f) build_sources
- Generate ILCD source datasets from references (`tidas_sdk.create_source`), writing `source_datasets` and `source_references`.
- Infer usage from `usage_tagging`, Step 1c summaries, Step 1b usability, and `industry_average`, then filter out `background_only`.

5a) intended_applications
- Before `build_process_datasets`, call LLM per process using `description`, `boundary`, and `assumptions` (fallback to global `technical_description`, `scope`, `assumptions`) and write intended applications (EN+ZH).
- Write to `administrativeInformation.common:commissionerAndGoal.common:intendedApplications`.

5) build_process_datasets
- Build ILCD process datasets (reference direction follows `operation`; optional Translator adds Chinese fields).
- `ProcessClassifier` falls back to Manufacturing on failure; missing flows use placeholders only.
- Try `DatabaseCrudClient.select_flow` to fill flow version, shortDescription, flowProperty, and unit group.
- Ensure reference-flow exchange; empty amounts fallback to `"1.0"`; validate via `tidas_sdk.create_process` (warnings only).
- Exchange `referencesToDataSource` prefer `value_citations/value_evidence`; remaining evidence is rolled up to process level.
- `build_process_datasets` no longer pre-writes process `dataTreatmentAndExtrapolationsPrinciples`; Step 8 rewrites that field after `balance_review` to align process text with final review output.

6) resolve_placeholders (post-processing)
- After `build_process_datasets`, scan exchanges with `referenceToFlowDataSet.unmatched:placeholder=true`.
- Rebuild a secondary `flow_search` query from `exchangeName/Direction/unit/flow_type/search_hints/generalComment`.
- Filter candidates by `flow_type`; for elementary flows also filter by medium (`air/water/soil`).
- Use selector (LLM/Rule) to choose; write `flow_search.secondary_query/resolution_*` and update process datasets.
- If still unmatched, keep placeholder and record `resolution_status/reason` for review.

7) balance_review (post-processing)
- Compute mass/energy balance from `matched_process_exchanges` (fallback `process_exchanges`), prefer flow unit groups, fallback to built-in unit mapping.
- Exclude `material_role=auxiliary|catalyst` or `balance_exclude=true` from balance stats.
- Output `balance_review` and `balance_review_summary`, mark `ok|check|insufficient`, and log warnings; no exchange values are rewritten.
- Record `unit_mismatches` and `density_estimates` for review.

8) dataCutOffAndCompletenessPrinciples
- After placeholder resolution and balance review, summarize missing amounts, placeholders, and unit/density conversions per process.
- Write `dataCutOffAndCompletenessPrinciples` per process to `modellingAndValidation.dataSourcesTreatmentAndRepresentativeness.dataCutOffAndCompletenessPrinciples`: use existing state/LLM output when available, otherwise deterministic fallback text and bilingualization.
- In the same step, rewrite process-level `dataTreatmentAndExtrapolationsPrinciples` from final `balance_review` and `balance_review_summary` (including counters such as `mass_core_check_processes`, `unit_mismatch_total`, `mapping_conflict_total`, `role_missing_total`).

## Origin Orchestration Workflow (scripts/origin)
### Goal and Order
- Goal: Write SI and usage tagging back before Steps 1-3 so prompts can read SI evidence.
- Orchestration order:
  `Step 0 -> Step 1a -> Step 1b -> 1b-usability -> Step 1c -> Step 1d -> Step 1e -> Step 1 -> Step 2 -> Step 3 -> Step 3b -> Step 4 -> Step 4b -> Step 4c -> Step 1f -> Step 5a -> Step 5 -> Step 6 -> Step 7 -> Step 8`

### Core Scripts (Main Chain)
- `process_from_flow_workflow.py`: Main orchestrator, runs 1b-usability/1d/1e before resuming the main flow.
- `process_from_flow_langgraph.py`: LangGraph CLI (run/resume/cleanup/publish), supports `--stop-after` and `--publish/--commit`.
- `process_from_flow_reference_usability.py`: Step 1b usability screening (LCIA vs LCI).
- `process_from_flow_download_si.py`: Download SI originals and write SI metadata (supports `--doi/--cluster/--recommendation` filters, `--dry-run`, `--no-update-state`).
- `mineru_for_process_si.py`: Parse PDF/image SI into JSON structure.
- `process_from_flow_reference_usage_tagging.py`: Tag reference usage (also writes tags to `step_1c_reference_clusters.reference_summaries`).

### Maintenance Utilities (Non-main-chain, Offline Backfill/Recompute)
- `process_from_flow_build_sources.py`: Backfill source datasets from cached state.
- `process_from_flow_placeholder_report.py`: Generate placeholder resolution report (writes `cache/placeholder_report.json`).
- `product_flow_sdk_insert.py`: Moved to `scripts/product_flow/product_flow_sdk_insert.py`; not part of the `process_from_flow` main chain.

### Run Notes
- New runs require explicit `--flow`; no default flow file path is assumed.
- `process_from_flow_workflow.py` does not expose `--no-llm` (Step 1b/1e require LLM); use `process_from_flow_langgraph.py --no-llm` for deterministic debugging.
- `--min-si-hint` controls SI download threshold (`none|possible|likely`), with `--si-max-links/--si-timeout`.
- Default run-id naming (when `--run-id` is omitted): `pfw_<flow_code>_<flow_uuid8>_<operation>_<UTC_TIMESTAMP>` (example: `pfw_01211_3a8d74d8_produce_20260211T105022Z`).
- `process_from_flow_langgraph.py --stop-after datasets` means run through dataset writeout; other values stop early and save state.
- `process_from_flow_workflow.py` writes fixed per-run logs to `artifacts/process_from_flow/<run_id>/cache/workflow_logs/*.log` and timing summary to `artifacts/process_from_flow/<run_id>/cache/workflow_timing_report.json`.
- `process_from_flow_workflow.py` prints stage progress in stderr (`stage x/y`, elapsed, ETA, log path); `match_flows` logs exchange-level progress with completed/total and ETA.
- `--allow-density-conversion` enables LLM density estimates for mass<->volume mismatches (product/waste flows only).
- Placeholder resolution runs once by default; to re-run, clear `placeholder_resolution_applied/placeholder_resolutions` then `--resume`.
- `process_from_flow_workflow.py` clears `stop_after` before resuming the main pipeline.
- `--stop-after matches` ends after Step 4 matching; it does not produce `process_datasets/source_datasets`, and therefore does not export `exports/flows`.

## State Fields (`state`)
- Input/context: `flow_path`, `flow_dataset`, `flow_summary`, `operation`, `scientific_references`.
- Routes/processes: `technology_routes`, `process_routes`, `selected_route_id`, `technical_description`, `assumptions`, `scope`, `processes`.
- Intended/completeness: `intended_applications`, `data_cut_off_and_completeness_principles`, `data_cutoff_principles_applied`, `data_cutoff_summary`, `data_treatment_and_extrapolations_principles`, `data_treatment_principles_applied`, `data_treatment_summary`.
- Exchanges/matching: `process_exchanges`, `exchange_value_candidates`, `exchange_values_applied`, `matched_process_exchanges`.
- Outputs: `process_datasets`, `source_datasets`, `source_references`.
- Evaluation/markers: `coverage_metrics`, `coverage_history`, `stop_rule_decision`, `step_markers`, `stop_after`.
- Unit alignment/density: `unit_alignment_applied`, `unit_alignment_summary`, `allow_density_conversion`, `density_conversion_applied`, `density_conversion_summary`.
- Placeholder resolution: `placeholder_report`, `placeholder_resolutions`, `placeholder_resolution_applied`.
- Balance review: `balance_review`, `balance_review_summary`.

## SI Injection Points (Actual Behavior)
- Step 1: `TECH_DESCRIPTION_PROMPT` reads `si_snippets`.
- Step 2: `PROCESS_SPLIT_PROMPT` reads `si_snippets`.
- Step 3: `EXCHANGES_PROMPT` reads `si_snippets`.
- Step 3b: `EXCHANGE_VALUE_PROMPT` reads `fulltext_references` and `si_snippets`.
- Step 4/Step 5 do not read SI directly.
- SI must be written back to `process_from_flow_state.json` before Step 1; otherwise rerun Step 1-3.
- `si_snippets` come from MinerU outputs plus direct text-style SI (`docx/xlsx/csv/tsv/txt/md`), prioritizing primary-cluster DOIs and ranked by `docx > tabular(xlsx/csv/tsv) > text > mineru`; max 3 DOIs, 1 snippet per DOI, 2000 chars each.

## Text Field Sources (Implementation Details)
- `processInformation.dataSetInformation.common:generalComment`: From Step 2 `process.description`, fallback to `technical_description`.
- `exchanges.exchange.generalComment`: From Step 3 `exchange.generalComment` plus evidence rollup, with enforced `tg_io_kind_tag` and `tg_io_uom_tag`.
- `processInformation.technology.technologyDescriptionAndIncludedProcesses`: Concatenates `technical_description` + `process.description` + global `assumptions` (not per-process `structure.assumptions`).
- `administrativeInformation.common:commissionerAndGoal.common:intendedApplications`: Step 5a per-process text from `description/boundary/assumptions`, fallback to global.
- `modellingAndValidation.dataSourcesTreatmentAndRepresentativeness.dataCutOffAndCompletenessPrinciples`: Step 8 per-process summary of placeholders/missing values/conversions.
- `modellingAndValidation.dataSourcesTreatmentAndRepresentativeness.dataTreatmentAndExtrapolationsPrinciples`: Step 8 post-balance rewrite aligned with `balance_review_summary` counters.

## Outputs and Debugging
- Output root: `artifacts/process_from_flow/<run_id>/` with `input/`, `cache/`, and `exports/`.
- State file: `cache/process_from_flow_state.json`.
- Workflow stage logs: `cache/workflow_logs/*.log`.
- Workflow timing report: `cache/workflow_timing_report.json`.
- MCP call snapshots: `cache/mcp_snapshots/*.jsonl`.
- Flow select cache: `cache/flow_select_cache.json` (first remote CRUD select per flow UUID/version, then local cache hits within and across resumed runs using the same run cache).
- `exports/flows` is generated from final process references (`referenceToFlowDataSet`) after datasets are built/published; it is not a dump of all search candidates.
- Placeholder report: `cache/placeholder_report.json` (from `resolve_placeholders` or `process_from_flow_placeholder_report.py`).
- Resume: `uv run python scripts/origin/process_from_flow_langgraph.py --resume --run-id <run_id>`.
- Maintenance backfill sources: `uv run python scripts/origin/process_from_flow_build_sources.py --run-id <run_id>`.
- Maintenance placeholder report: `uv run python scripts/origin/process_from_flow_placeholder_report.py --run-id <run_id>` (`--no-update-state` avoids writing to state).
- Publish existing run: `uv run python scripts/origin/process_from_flow_langgraph.py --publish-only --run-id <run_id> [--publish-flows] [--commit]`.
- Cleanup old runs: `uv run python scripts/origin/process_from_flow_langgraph.py --cleanup-only --retain-runs 3`.

## Publishing Flow (Flow/Source/Process)
Actual order in `process_from_flow_langgraph.py`:
- With `--publish-flows`: `flows -> sources -> processes`.
- Without `--publish-flows`: `sources -> processes`.

### Dependencies and Configuration
- Entrypoints: `FlowPublisher`, `ProcessPublisher`, `DatabaseCrudClient`.
- MCP service: configure `tiangong_lca_remote` in `.secrets/secrets.toml` (`Database_CRUD_Tool`).
- LLM optional: used for flow type/product category inference and bilingual field completion for new flows; failures fall back to deterministic defaults with logs.

### Step 0: Publish sources (optional but recommended)
- `--publish/--publish-only` publishes sources before processes.
- Only publish sources referenced by process/exchange `referenceToDataSource`.

### Step 1: Prepare alignment structure (for FlowPublisher)
- Structure: `[{ "process_name": "...", "origin_exchanges": { "<exchangeName>": [<exchange dict>, ...] } }]`.
- Each exchange dict must include: `exchangeName`, `exchangeDirection`, `unit`, `meanAmount|resultingAmount|amount`, `generalComment`, `referenceToFlowDataSet`.
- Optionally add `matchingDetail.selectedCandidate` mapped from `flow_search` for better classification/property selection.

### Step 2: Publish/update flows
- `FlowPublisher` builds flow datasets via shared `ProductFlowCreationService`, then validates through `tidas_sdk.create_flow` (validation fallback is allowed and logged).
- `FlowPublisher.prepare_from_alignment()` builds `FlowPublishPlan`:
  - Placeholder `referenceToFlowDataSet` -> insert.
  - Matched but missing flow property -> update (version +1).
  - Elementary flows are not created; product/waste flows generate ILCD flow datasets.
- `FlowPublisher.publish()` applies `FlowDedupService` at publish time and can switch final action among `insert/update/reuse` based on remote existence checks, so final action may differ from plan mode.
- Auto inference:
  - `FlowTypeClassifier`: LLM first, fallback rules.
  - `FlowProductCategorySelector`: Pick product category level by level.
  - `FlowPropertyRegistry`: Defaults to Mass (override per exchange if needed).
- After publish, use `FlowPublishPlan.exchange_ref` to replace placeholders in process datasets.

### Step 3: Publish processes
- `ProcessPublisher.publish(process_datasets)` defaults to dry-run; `--commit` writes.
- Always `close()` MCP clients after publishing.

## Literature Service Configuration and Operation
### Retrieval Strategy
- Build queries from flow name, operation, and technical description.
- Step 2/Step 3 can add retrievals, stored in `scientific_references.step2/step3`.
- Step 1b uses `filter: {"doi": [...]}` + `topK=1` + `extK` (default `extK=200`).

### Configuration
Configure `tiangong_kb_remote` in `.secrets/secrets.toml`:

```toml
[tiangong_kb_remote]
transport = "streamable_http"
service_name = "TianGong_KB_Remote"
url = "https://mcp.tiangong.earth/mcp"
api_key = "<YOUR_TG_KB_REMOTE_API_KEY>"
timeout = 180
```

If not configured or invalid, the workflow falls back to LLM common sense only.

### Logs
- `process_from_flow.mcp_client_created`: MCP client created.
- `process_from_flow.search_references`: Literature search succeeded (query + count).
- `process_from_flow.search_references_failed`: Literature search failed (non-blocking).
- `process_from_flow.mcp_client_closed`: MCP client closed.
- `process_from_flow.match_flows_started`: Flow matching starts with process/exchange totals.
- `process_from_flow.match_flows_progress`: Per-exchange matching progress (`completed/total`, elapsed, ETA).
- `process_from_flow.match_flows_completed`: Flow matching finished with total elapsed.
- `crud.select_flow_record_cache_hit`: Selected flow record served from local run cache.
- `crud.select_flow_cache_flushed`: Flow select cache persisted to disk (`flow_select_cache.json`).

### Performance
- Each literature search typically takes around 1-2 seconds (depends on network and service load).
- Step 1b full-text time depends on DOI count and `extK`.
- End-to-end runtime is usually dominated by remote calls (KB search, flow search, CRUD select, LLM) and can take minutes; Step 4 matching is often the longest stage.
- Use `cache/workflow_timing_report.json` as the source of truth for per-stage timing in each run.

### Testing
```bash
uv run python test/test_scientific_references.py
```

### Reference Usability Screening
- Optional step: check whether Step 1b full text supports route/process/exchange needs.
- Mark `unusable` if text only reports LCIA indicators (for example, ADP/AP/GWP/EP/PED/RI) or impact units like `kg CO2 eq`, `kg SO2 eq`, `kg Sb eq`, `kg PO4 eq`, with no LCI inventory rows (`kg`, `g`, `t`, `m2`, `m3`, `pcs`, `kWh`, `MJ`).
- If the paper hints supporting information/appendix for inventory tables, record `si_hint` (`likely|possible|none`) and `si_reason`; still keep `decision=unusable` if main text has no LCI tables.
- If `si_hint=none` with no `si_reason`, the script auto-scans keywords in text to backfill `likely/possible`.
- Prompt: `src/tiangong_lca_spec/process_from_flow/prompts.py` `REFERENCE_USABILITY_PROMPT`.
- Script: `uv run python scripts/origin/process_from_flow_reference_usability.py --run-id <run_id>`.
- Output: `scientific_references.usability` in `process_from_flow_state.json`.

## Usage Notes
- Ensure LLM is configured; `process_from_flow_workflow.py` does not expose `--no-llm`.
- Keep flow search/selector interfaces consistent (`FlowQuery` -> `(candidates, unmatched)`).
- CLI adds Chinese translations by default; disable with `--no-translate-zh`.

## Stop Rules
- Stop rules rely on coverage, not retrieval count; thresholds can evolve without changing node order.
- Coverage definitions:
  - `process_coverage` = processes with evidence / total planned processes.
  - `exchange_value_coverage` = key exchanges with evidence / total key exchanges.
- `stop_rule_decision` records `should_stop/action/reason/coverage_delta`; `coverage_history` stores each evaluation time.
- Default thresholds (adjustable):
  - Stop when `process_coverage >= 0.5` and `exchange_value_coverage >= 0.6`.
  - Stop when coverage delta vs previous evaluation is `< 0.1`.
- If below thresholds and usability shows `unusable` with `si_hint=none`, switch to `expert_judgement` and log reasons.
- Key exchanges: explicit `is_key_exchange/isKeyExchange`, `is_reference_flow`, `flow_type=elementary`, or input-side energy (`electricity/diesel/gasoline/heat`). If none, treat all exchanges as key exchanges.
