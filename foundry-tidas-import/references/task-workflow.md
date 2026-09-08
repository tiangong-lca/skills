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

Authorization is a separate selection through `--authorization-input`; it cannot be combined with `--semantic-input`. Use the runtime's current approval action and artifact contract, including its task/account/input/action/expiry bindings. A prior profile waiver or copied historical approval does not grant the current operation.

## Interpret the result

The result schema is `tiangong-foundry.operation-result.v1`. `ready` and `running` describe progress; `needs_input` and `needs_auth` identify required intervention; `blocked` and `failed` retain reasons; `completed` requires current registered completion evidence. Inspect permission state separately (`not_required`, `required`, `granted`, `invalid`).

For missing qualification, repair/select the trusted runtime through its manager. For identity mismatch, preserve the registered destination and obtain the correct session. For stale context, return to the current assessment/work-item producer. For `mutation_readback_required` or predecessor-attempt blockers, keep the original scope and perform only its returned recovery. Do not downgrade these conditions into success or direct write retries.
