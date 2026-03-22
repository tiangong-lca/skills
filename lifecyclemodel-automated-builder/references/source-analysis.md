# Source Analysis

## TianGong App Behavior

Reference repo: `/home/huimin/projects/tiangong-lca-next`

- `src/services/lifeCycleModels/util.ts`
  - `genLifeCycleModelJsonOrdered` converts graph nodes and edges into native lifecycle model data.
  - Core native fields are:
    - `lifeCycleModelInformation.quantitativeReference.referenceToReferenceProcess`
    - `technology.processes.processInstance`
    - `connections.outputExchange.downstreamProcess`
- `src/services/lifeCycleModels/api.ts`
  - The application stores extra platform fields such as `json_tg` and `rule_verification`.
  - Those fields are outside this skill's scope after the current redesign.

## TIDAS SDK

Reference repo: `/home/huimin/projects/tidas-sdk`

- `createLifeCycleModel(data?, config?)` provides strict validation.
- The SDK is the native schema gate for the `json_ordered` artifact this skill emits.

## TIDAS Tools

Reference repo: `/home/huimin/projects/tidas-tools`

- `validate.py` checks lifecycle model classification hierarchy.
- Classification still has to pass even if the JSON passes strict `tidas-sdk` validation.

## TianGong MCP

Reference repo: `/home/huimin/projects/tiangong-lca-mcp`

- `src/tools/db_crud.ts`
  - `Database_CRUD_Tool` now accepts `jsonOrdered` for lifecyclemodels and delegates platform-specific preparation to `prepareLifecycleModelFile(...)`.
  - Lifecyclemodel `select` is intentionally sanitized to `id`, `version`, and `json_ordered`.
- `src/tools/life_cycle_model_file_tools.ts`
  - Provides the downstream native-model preparation logic used by MCP before write.
- Implication:
  - This skill should stop at native `json_ordered`.
  - If a remote write is approved later, the MCP layer is the correct place to derive `json_tg` and related platform fields.
