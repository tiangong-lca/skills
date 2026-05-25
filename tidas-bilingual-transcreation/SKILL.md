---
name: tidas-bilingual-transcreation
description: Use when producing or reviewing high-quality Chinese/English bilingual TIDAS or ILCD fields for flow, process, or lifecyclemodel rows. This skill makes Codex do semantic transcreation from extracted translation units while the TianGong CLI performs deterministic extract, apply, validation, and evidence artifact generation.
---

# TIDAS Bilingual Transcreation

## Scope

- Produce reviewed bilingual translations for local TIDAS rows after `dataset bilingual extract`.
- Use TIDAS/ILCD context, process or flow review artifacts, source evidence, terminology, units, classification, geography, time, and technology context.
- Do not translate by mechanical term replacement. Write target-language text that is natural, professional, and faithful to the dataset meaning.
- Do not perform remote writes, publish, save-draft, or credential handling from this skill.

## Canonical CLI Flow

Run the deterministic CLI steps around the AI review:

```bash
tiangong-lca dataset bilingual extract \
  --input /abs/path/rows/processes.jsonl \
  --type process \
  --out-dir /abs/path/translation/process
```

Codex then reads `outputs/trans-units.jsonl` plus the relevant evidence and writes a reviewed JSONL file:

```jsonl
{"unit_id":"...","row_index":0,"field_path":"/json_ordered/processDataSet/.../name","source_lang":"en","target_lang":"zh","source_text":"Electricity mix, high voltage","translated_text":"高压电力组合","basis":"Translated from process name, geography, and quantitative-reference context.","review_status":"agent_reviewed","reviewer":"codex"}
```

Apply and validate:

```bash
tiangong-lca dataset bilingual apply \
  --input /abs/path/rows/processes.jsonl \
  --translations /abs/path/translation/process/trans-reviewed.jsonl \
  --out /abs/path/rows/processes.translated.jsonl \
  --out-dir /abs/path/translation/process/apply

tiangong-lca dataset bilingual validate \
  --input /abs/path/rows/processes.translated.jsonl \
  --type process \
  --out-dir /abs/path/translation/process/validate
```

Repeat for `--type flow` when reference flows are part of the target dataset.

## Translation Rules

- Preserve numeric values, units, CAS numbers, standards, UUID-like identifiers, geography codes, dates, and named methods unless the evidence explicitly supports a localized form.
- Translate the whole field in context. For example, avoid literal fragments such as `组件s`, `层压件s`, or `光伏 装置`.
- Keep technical nouns precise: use terms such as `过程`, `产品流`, `基本流`, `定量参考`, `高压电力组合`, or `光伏安装系统` only when they fit the actual field context.
- Keep source meaning narrower than style preference. Do not add process steps, environmental mechanisms, supplier claims, or assumptions not present in source evidence.
- If the English source is itself low quality, repair the target text only within supported meaning and record the limitation in `basis`.
- If the correct translation depends on missing evidence, output no translation for that unit and record a `manual_review` blocker outside the reviewed JSONL.

## Required Inputs

Use the smallest set needed for the requested rows:

- `outputs/trans-units.jsonl` from `dataset bilingual extract`.
- Source rows or selected field snippets for field-level context.
- Source evidence, if available.
- Process/flow review reports, if available.
- Flow governance decisions and process-flow reference rules, if the field names or comments mention linked flows.
- Project terminology or user-approved glossary, if provided.

## Output Contract

Each reviewed translation row should include:

- `unit_id`
- `row_index`
- `field_path`
- `source_lang`
- `target_lang`
- `source_text`
- `translated_text`
- `basis`
- `review_status`, normally `agent_reviewed`
- `reviewer`, normally `codex`

The CLI writes `translation-evidence.json` during `dataset bilingual apply`; do not hand-edit that evidence file except to inspect it.

## Stop Rules

Stop and report blockers instead of applying translations when:

- source evidence is missing for a high-impact field such as name, quantitative reference, geography, time, technology, or exchange meaning;
- translation would require inventing technical facts;
- a field contains mixed-language residue, placeholders, or schema-invalid text that needs content repair before translation;
- `dataset bilingual validate` reports blockers after apply.

The task is not target-quality ready until `dataset bilingual validate` passes and `translation-evidence.json` exists for all reviewed bilingual fields.

