# Public task workflow

The examples below show public Foundry arguments. Execute them through the qualified managed entry selected for this skill; a similarly named executable on ambient PATH is not runtime selection evidence. Keep returned executable/argv/cwd fields intact when continuing an action.

## Initialize or inspect

```text
tiangong-foundry workspace init --workspace <absolute-workspace> --json
tiangong-foundry doctor --workspace <absolute-workspace> --json
tiangong-foundry task status --workspace <absolute-workspace> --task <task-id> --actor <actor-id> --json
```

Initialization and doctor do not log in, grant writes or create business records. For an existing workspace that requires migration, use `workspace migrate --workspace <absolute-workspace> --dry-run --json`, review the resulting plan, and follow its separately explicit apply operation. Never delete the old workspace to bypass migration.

## Start with selected evidence

Write a JSON spec in the user workspace with exactly these fields:

| Field | Selection |
| --- | --- |
| `schema` | `tiangong-foundry.task-start.v1` |
| `request_id`, `actor_id` | Stable IDs for this user request and the independently selected actor. |
| `lane` | `external-dataset-curated-import` or `source-evidence-dataset-development`. |
| `profile_id` | The applicable profile from the selected runtime/task contract. |
| `target_entities` | Ordered unique dataset types: `contact`, `source`, `support`, `flow`, `flowproperty`, `unitgroup`, `process`, `lifecyclemodel`. |
| `sources` | Selected original evidence as objects containing only `path`. |
| `seed` | `null` for native package conversion, or a selected candidate JSON seed as `{ "path": "..." }`. Source-evidence tasks require the seed among `sources`. |
| `account_intent` | `null` for local preparation without an account, or the intended `project_ref`, `user_id`, `session_reference` (path or `null`) and optional `account_mode` (`ordinary` or `production-test`). No credential contents. |
| `preparation` | Normally `null`, so Foundry drives the full workflow. Use a selected cleanup preparation only when that is the intended local operation. |

Select account intent before starting work that needs remote identity checks or writes; do not guess it from an unrelated logged-in session. Paths in the spec resolve against the explicit workspace. Foundry captures input bytes and binds revisions; editing files in a registered task is not how to select changed inputs.

```text
tiangong-foundry task start --workspace <absolute-workspace> --spec <spec-file> --json
tiangong-foundry task resume --workspace <absolute-workspace> --task <returned-task-id> --actor <actor-id> --json
```

One resume advances one registered stage. Conversion, context, rows, assessment, identity preflight, finalization, authorization and execution may require successive current actions. Preserve the returned task ID and status; unchanged blockers need the stated input, not repeated unchanged resumes. Changed selected input creates a retained revision, and earlier consumed attempts still require their original recovery.

## Submit current semantic work

The descriptor selected by `--semantic-input` contains exactly `schema`, `task_id`, `actor_id`, `assessment_sha256` and `submissions`. Use schema `tiangong-foundry.semantic-input.v1`; take the assessment digest from the current registered artifact. Each submission contains exactly:

- `kind`: `patch`, `classification`, `location` or `identity`;
- `authoring_task_sha256`: the current work-item digest;
- `file`: the selected decision/patch file, resolved against the workspace;
- `sha256`: the SHA-256 of that file's actual bytes.

Use each owner's generated template and required full-context evidence. Select one owner per row type, then reassess before another owner uses the changed rows. Do not insert illustrative or historical hashes. A rejected proposal leaves the current rows unchanged and retains diagnostics.

```text
tiangong-foundry task resume --workspace <absolute-workspace> --task <task-id> --actor <actor-id> --semantic-input <descriptor-file> --json
```

Authorization is a separate selection through `--authorization-input`; it cannot be combined with `--semantic-input` or `--reference-input`. Use the runtime's current approval action and artifact contract, including its task/account/input/action/expiry bindings. A prior profile waiver or copied historical approval does not grant the current operation.

## Select explicit reference evidence

Use this selection only when supported by the verified descriptor schema shipped with the qualified runtime. It is a separate input stage, not an approval or a way to bypass a blocked dependency. An unsupported runtime needs the qualified successor distribution; do not invent a new lock or invoke a developer command.

The `tiangong-foundry.reference-input.v1` descriptor contains exactly:

| Field | Selection |
| --- | --- |
| `schema` | `tiangong-foundry.reference-input.v1` |
| `task_id`, `actor_id` | The current registered task and independently selected actor. |
| `rows_manifest_sha256` | Digest of the current indexed `foundry-rows.json` artifact. |
| `dataset_type` | The selected concrete row type: `process`, `flow`, `source`, `contact`, `lifecyclemodel`, `unitgroup` or `flowproperty`. |
| `qa_reference_rows` | Explicit `{ "file": "...", "sha256": "..." }` selections for Process QA, or `[]`. Other types cannot select QA reference rows. |
| `intent` | One selected `{ "file": "...", "sha256": "..." }` exact-reference intent, or `null`. |
| `review_files` | Every review file used by that intent, independently selected as file/SHA-256 pairs, or `[]` when no intent is selected. |

Select QA evidence and/or an intent. Each list allows at most 128 files; files must be regular, readable and at most 8 MiB each, within a 64 MiB aggregate selection. Descriptor file paths resolve against the explicit workspace; review locators inside the CLI intent follow that protocol and must resolve to the independently selected reviews. File digests refer to actual bytes, while consumer/selected-reference payload digests follow the qualified CLI's canonical protocol; do not substitute one for the other or hash a seed wrapper as a final consumer payload.

An exact-reference intent and its reviews use the CLI-owned `dataset-exact-reference-intent.v1` and `dataset-exact-reference-review.v1` contracts. Use the actual finalizer-selected consumer rows and reviewed reference observations for the intended account/project. Foundry verifies selection and transport; CLI decides reference eligibility. An unavailable, unrelated or foreign private reference cannot become usable merely by writing a review file.

```text
tiangong-foundry task resume --workspace <absolute-workspace> --task <task-id> --actor <actor-id> --reference-input <descriptor-file> --json
```

Do not combine this option with semantic input, authorization input or an explicit cleanup preparation. Empty paths are invalid, including through the JavaScript facade. Foundry snapshots selected QA/review bytes, retains the original intent, and derives only its review-file locators toward those snapshots. A changed selection invalidates prior finalization; changed rows require current evidence. Prepared, consumed and completed scopes retain their original selection. Let Foundry re-finalize and verify; do not modify indexed snapshots or proof reports.

## Select native execution intent with approval

For `input_kind=final_rows` and a Flow, Process or Source scope, the current `tiangong-foundry.authorization-input.v1` descriptor may include `execution_contract: { "file": "...", "sha256": "..." }`. The selected file must be the qualified CLI's `dataset-save-draft-execution-contract.v1` insert-only contract for the exact ordered final rows, desired payload digests, intended project/account and owner draft state `0`. Other types and prepared-row approval cannot use this field. Keep the raw file hash distinct from the CLI's canonical contract digest.

The contract accompanies a separately valid grant and its current evidence; it does not authorize a write itself. Foundry binds a task snapshot and returns a sealed execution action. Invalid selection cannot fall back to a different writer. After dispatch, use only the original task's readback/recovery action. A missing response or missing native receipt cannot justify another contract, task revision or mutation. Completion still requires the matching execution evidence and independent root/owner/state/payload verification, plus the same reference intent/reviews when selected.

## Interpret the result

The result schema is `tiangong-foundry.operation-result.v1`. `ready` and `running` describe progress; `needs_input` and `needs_auth` identify required intervention; `blocked` and `failed` retain reasons; `completed` requires current registered completion evidence. Inspect permission state separately (`not_required`, `required`, `granted`, `invalid`).

For missing qualification, repair/select the trusted runtime through its manager. For identity mismatch, preserve the registered destination and obtain the correct session. For stale context, return to the current assessment/work-item producer. For `mutation_readback_required` or predecessor-attempt blockers, keep the original scope and perform only its returned recovery. Do not downgrade these conditions into success or direct write retries.
