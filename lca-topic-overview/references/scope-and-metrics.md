# Scope and metrics

## Topic scope

```json
{
  "schema_version": 1,
  "topic": "电力行业",
  "boundary": "包括发电和输配电相关数据；仅使用电力的其他行业过程归为关联数据。",
  "terms": ["电力", "发电", "electricity", "power generation", "transmission"],
  "core": []
}
```

The empty list is the **discovery input**, not permission to skip selection. After reading candidates, insert entries of this shape:

```json
{
  "table": "processes",
  "id": "<UUID observed in the public capture>",
  "reason": "参考产品为电力，过程名称与分类均描述发电。"
}
```

Supported tables are `processes`, `flows`, `lifecyclemodels`. Use UUID membership, not names. The CLI selects that UUID's latest **public** version from this capture; a newer draft is irrelevant. Unknown or duplicate core identities are rejected. For another topic, change the boundary, terms and actual core identities; there is no electricity-specific runtime.

Discovery matches descriptive names, synonyms, classifications, dataset type and recorded technology description. It deliberately ignores names in exchanges, so merely consuming a product does not create core membership. Broaden topic terms using observed classification and terminology, review borderline candidates, and record why core records belong. Keep names in all available languages; missing name parts remain blank.

## Counts and evidence

| Observation | Basis |
| --- | --- |
| Core/public object counts | Distinct table + UUID, latest public `DD.DD.DDD` version |
| Public revisions | All public versions of the counted identities, separately stated |
| Reference products | Distinct Flow UUIDs reached by unambiguous reference exchanges whose exact Flow version resolves publicly; duplicated exchange IDs never confirm a product. Unresolved, ambiguous and absent-reference cases are separate |
| Classification | Recorded classification paths; distinct objects within each group, multiple paths can overlap |
| Geography, time, type | Recorded metadata; missing values are a visible bucket. Reference year is not an industry trend |
| Treatment/technology names | Recorded second name-part text; no semantic merging or claim of distinct technologies |
| Flow usage | Distinct Process counts by input/output role **and** exchange occurrence counts, based on latest public Process revisions |
| Models | Model objects, process instances and explicitly declared connections are different counts |

Each distribution includes `record_keys` in `overview.json` and `statistics.csv`; HTML bars drill into those same members. `records.csv` preserves individual name parts, table, UUID, version, classification and role. JSON preserves raw multilingual name parts too. Missing names never replace object identity or justify a merge.

## Relationships

1. A core Process's exact Flow references define the one-hop neighborhood; a core Flow includes public Process uses of that UUID. A shared exact public Flow can show possible suppliers and consumers. Usages with a missing or unavailable exact version remain in `unresolved_flow_usages` and reference context, without a resolved association edge or an increased related-Process count. These are structural supply/use observations, not a selected provider or a calculated quantity.
2. Core Models and public Models that refer to a core Process expose their recorded instances and explicit connections. Repeated Process UUIDs remain separate instances. An old exact Process revision remains reference context, without changing main counts.
3. Connection observations distinguish absent/ambiguous target instances, unresolved Process references, absent Flow identifiers/versions, unresolved public Flow references, unobserved endpoint exchanges and exact public references. The declared output and input may use different Flow UUIDs; preserve both UUID/version pairs and check each endpoint independently. Never infer missing versions/edges or replace a reference version.
4. Do not recursively expand the whole database. The initial command emits one-hop Flow associations plus relevant explicit Model structures. A graph displays bounded subsets and states shown/total counts; JSON/CSV retain the full analyzed relation set. Cycles are displayed as recorded, without recursive calculation or assumed allocation.

## Output review

Required files are `overview.md`, `overview.html`, `overview.json`, `scope.json`, `records.csv`, `statistics.csv`, `flow-usage.csv`, and `model-connections.csv`. Keep the verified inventory and runtime record with the run.

Read `overview.json` before adding prose. Explain the database's current composition, distribution, representative exact records and structural relationships. When inspecting a graph, compare its counts to the same JSON set and test a filter. Preserve output limits and unresolved-reference notes. Do not append recommendations, governance actions, quality rankings, future outlook, external benchmark percentages or new-data production tasks.
