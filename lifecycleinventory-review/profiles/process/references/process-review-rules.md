# Process Review Rules

适用场景：
- review `process_from_flow` 等本地 process 构建产物；
- review 冻结后的远端 `processes` snapshot；
- 目标是先固定输入，再按 ILCD + skill 内置 rubric 做本地 review，最后输出可执行的修改清单；
- 如果任务只是 review，不直接远端写回。

## Execution Contract

1. 先建立 review 输出目录：
   - `inputs/`: 用户请求、run manifest 或远端 snapshot manifest、原始 rows JSONL
   - `outputs/`: rulebook JSON、findings JSONL、process-level patch plan、summary JSON
   - `logs/`: CLI 调用、分页抓取、重试、脚本 stdout/stderr
   - `reports/`: operations log、process change list、final summary、verification
   - `friction/`: 可选的工具限制说明
2. review 必须基于冻结后的输入进行；远端任务先 freeze snapshot，本地 run 也不要边读边改原始产物。
3. `node scripts/run-review.mjs --profile process` 是本地 process review 的 canonical CLI 入口。
4. 当前远端 snapshot review 的 canonical skill 入口是 `node scripts/run-remote-process-review.mjs`：先冻结 `tiangong process list --json` 输出，再调用 `node scripts/run-review.mjs --profile process --rows-file ...`。只有在现有 `process list` 过滤维度不足以表达任务时，才使用补充 bridge，并在 review 备注或 `friction/` 目录中记录限制。
5. LLM 语义审核层默认关闭；只有显式启用时才可作为补充建议层，且不能替代硬规则。

## Required Inputs

- 本地 run review：至少提供 `--run-root`、`--run-id` 或等价可复核的 process artifact bundle。
- 远端 snapshot review：冻结后的 `processes` 行至少包含：
  - `id`
  - `version`
  - `state_code`
  - `model_id`
  - `modified_at`
  - `json`
- 若需要抓取远端 snapshot，认证信息来自 `TIANGONG_LCA_*` 环境变量或等价认证。
- 若 `model_id` 非空，还应补充所属 lifecyclemodel 的名称或上下文，供前景系统语境检查。

## Review Axes

说明：
- 默认 process rubric 不检查 “ILCD 分类是否映射”。
- 如果用户后续明确要求 taxonomy 治理，可在单独 case 中追加 classification review；当前默认只检查 schema-required 结构完整性、命名、定量参考、单位与平衡、代表性、前景系统语境和工具生成语言问题。
- 这里的 “不检查 ILCD 分类映射” 仅指不默认判定 classification 语义是否正确；若 classification 节点本身属于 schema-required，则其存在性仍属于结构完整性检查范围。

### 1. Schema-required Structural Completeness
来源：
- 当前 TIDAS `ProcessSchema`
- 当前前端 `src/pages/Processes/requiredFields.ts`

检查点：
- review 默认应检查最终 `jsonOrdered` 是否满足当前 process schema-required 字段存在性要求，而不只检查若干经验规则。
- 至少应覆盖：
  - `processDataSet` 命名空间与版本头；
  - `processInformation.dataSetInformation` 下的 required 节点；
  - `quantitativeReference` required 节点；
  - `time` required 节点；
  - `geography.locationOfOperationSupplyOrProduction` required 节点；
  - `modellingAndValidation.LCIMethodAndAllocation` required 节点；
  - 当前前端 `requiredFields.ts` 中声明为必填的 process 字段。
- schema-required 但当前语义可空的字段，不得直接省略；应保留为空多语言数组或等价 schema-safe 空结构。
- 最终 review/repair 复验不应只看“写回成功”，还应至少对最终 payload 执行一次 `ProcessSchema.safeParse(...)` 或等价严格校验。

处理原则：
- 结构 requiredness 与语义正确性分开记录：
  - required 节点缺失：记为结构缺陷，默认必须修；
  - 节点存在但语义一般：按对应轴继续评估，不混为“结构已通过”。
- classification 默认不做 taxonomy 语义裁定，但 schema-required 的 classification 容器缺失时，仍记为结构缺陷。

### 2. Process / Reference Flow Naming
来源：
- Rule 12-17
- Rule 20

检查点：
- 单一 reference flow 的 process，名称应与 reference flow 一致。
- 名称应使用技术性、结构化表达，不要保留 customs/catalog/legal enumeration 风格。
- `processDataSet.processInformation.dataSetInformation.name` 默认按以下四个字段审查与填写，不得把整串 short description 全塞进 `baseName`：
  - `baseName`：只写核心对象/产品/服务本体，不混入路线、mix、规格、电压、交付地点。
  - `treatmentStandardsRoutes`：写工艺路线、处理方式、技术路径、标准/子类型等“做法”信息。
  - `mixAndLocationTypes`：写 `production mix` / `technology mix` / `at plant` / `at farm gate` 等混合或交付/地点语义。
  - `functionalUnitFlowProperties`：写电压、等级、状态、品质、库容、粒径等功能单位或流属性限定。
- `baseName`、`treatmentStandardsRoutes`、`mixAndLocationTypes` 在当前 TIDAS process schema 和前端 `requiredFields` 中都按 required 处理；review 时不得省略这三个键。
- 若只有 2-3 段语义适用，只在适用字段填写文本；不适用但 schema-required 的字段也要保留为空多语言数组或等价 schema-safe 空结构。不要因为缺 1 个字段就把全部内容回退到 `baseName`，也不要直接删键。
- `flowProperties` 不是默认兜底垃圾桶；没有明确流属性语义时，不要为了凑字段把 route/location 文案塞进去。
- 地理信息不应混进过程/流名称本体，应放在专门字段。

典型问题：
- 名称过长；
- 含 `whether or not`、HS/税则式长串枚举；
- process 名称和 reference flow short description 不一致。
- `baseName` 里直接写成 `Alternating current; hydropower; technology mix; 35-330kV`，而 `treatmentStandardsRoutes` / `mixAndLocationTypes` / `functionalUnitFlowProperties` 全空。
- `baseName` 里直接写成 `Alfalfa for forage and silage; Fresh, unprocessed produce; Production mix, at farm gate`，没有把“鲜品，未加工”和“生产混合，在农场”拆到对应字段。

修改动作：
- 用 reference flow 名称回写 process 名称；
- 按四字段拆开写：
  - `Alternating current; hydropower; technology mix; 35-330kV`
    应拆为 `baseName=Alternating current`、`treatmentStandardsRoutes=hydropower`、`mixAndLocationTypes=technology mix`、`functionalUnitFlowProperties=35-330kV`。
  - `Alfalfa for forage and silage; Fresh, unprocessed produce; Production mix, at farm gate`
    应至少拆为 `baseName=Alfalfa for forage and silage`、`treatmentStandardsRoutes=[]`、`functionalUnitFlowProperties=Fresh, unprocessed produce`、`mixAndLocationTypes=Production mix, at farm gate`。
- 把不必要的条目式修饰词移到合适字段或删除。

### 3. Dataset Type and Linked Flow Integrity
来源：
- TianGong process structural review rule
- ILCD process dataset minimum structural completeness expectations

检查点：
- `modellingAndValidation.LCIMethodAndAllocation.typeOfDataSet` 不能为空。
- 每条 exchange 的 `referenceToFlowDataSet` 必须指向真实存在的 flow；不能保留占位符、失效 UUID 或只剩 short description 的假引用。
- flow 引用应更新到该 flow id 当前可访问的最新版本。
- 当 flow 版本更新时，`common:shortDescription` 也应同步刷新，避免版本和名称上下文脱节。

典型问题：
- type of dataset 未选择；
- input/output 里仍引用不存在的 flow；
- 旧 flow version 没有刷新到最新版本；
- 占位符 flow 被当成正式流保留下来。

修改动作：
- 补齐合法的 `typeOfDataSet`；
- 将失效 flow 引用替换为真实 flow，或在无法映射时保留为待建流问题，不得伪装成已完成数据；
- 将 flow 引用升级到最新版本，并同步 short description。

### 4. Quantitative Reference Integrity
来源：
- `docs/ILCD_rules/1-...General-guide-for-LCA...pdf`
- Section 6.4 `Function, functional unit, and reference flow`

检查点：
- `referenceToReferenceFlow` 能定位到真实 exchange。
- 该 exchange 必须是 `Output`。
- 定量数值必须为正。
- `functionalUnitOrOther` 必须存在。
- 中英文功能单位文本必须和 reference flow / exchange unit 一致。

典型问题：
- 功能单位缺失；
- `of of of` 之类重复词；
- 中文功能单位写成“1 单位”而 exchange 实际是 `kg`；
- reference exchange 指向 Input。

修改动作：
- 修正 reference exchange；
- 以 reference flow + 数量 + 单位重写中英文功能单位。

### 5. Unit Plausibility and Direct-Evidence Rule
来源：
- process review v2.1 historical rules

检查点：
- 当且仅当存在直接证据时，才记录“单位疑似错误”。
- 直接证据应来自名称、功能单位、exchange 语义、注释或可追溯上下文中的明确矛盾，而不是“看起来不合理”。
- 单位问题记录必须包含：
  - flow UUID
  - 当前单位
  - 建议正确单位
  - 依据
  - 置信度

处理原则：
- 仅因经验感觉不合理，不记录为正式 finding。
- 无法确认时，明确写“证据不足，不下结论”。

### 6. Geography / Time / Technology Representativeness
来源：
- Section 6.8.2 `Technological representativeness`
- Section 6.8.3 `Geographical representativeness`
- 时间代表性相关条款

检查点：
- geography 不能只停留在 `GLO` 占位，除非确实无法细化且限制说明是 process-specific 的。
- `referenceYear` 要有，且应补时间代表性说明。
- `technologyDescriptionAndIncludedProcesses` 不能为空。
- `technologicalApplicability` 在会影响复用时不应留空。

典型问题：
- geography 用统一 fallback 文案；
- time description 为空；
- technology description 完全缺失。

修改动作：
- 补真实市场/地区范围，或写明过程特定的限制说明；
- 补时间代表性；
- 补工艺范围、包含步骤、主要排除项。

### 7. Completeness / Cut-off / Balance Notes
来源：
- Section 6.6 `Deriving system boundaries and cut-off criteria`
- process review v2.1 historical balance scope

远端或本地产物中常见可直接解析的字段：
- `dataCutOffAndCompletenessPrinciples`
- `dataTreatmentAndExtrapolationsPrinciples`

检查点：
- 是否还存在 `placeholder remains unresolved`
- 是否还存在 `unit mismatch remains`
- 是否出现 `energy balance is insufficient`
- 物料平衡核查默认口径：
  - 左侧：`raw material input`
  - 右侧：`product output + by-product output + waste output`
  - `energy input`（如电、热、燃料）默认不计入质量平衡，只单列记录

处理原则：
- unresolved placeholder / unit mismatch：默认视为需要修改，不能当作已可发布版本。
- energy balance insufficient：若过程本来只做质量衡算，可保留并明确说明；否则应补足能量输出或说明排除原因。
- 平衡审查先做“存在性与可计算性”核查，不强行做缺乏证据的行业语义推断。

### 8. Foreground-System Context
这是 skill 内置的 review rule，不是 ILCD 原文条款，但对 TianGong 当前 process 很关键。

仅当 `model_id` 或等价前景系统上下文存在时启用。

检查点：
- 过程说明是否回到所属产品系统中的角色；
- 是否仍只是“通用路线模板 + proxy 解释”；
- 是否和所属 lifecyclemodel / 产品系统的功能语境脱节。

修改动作：
- 把描述改写成“这个过程在该产品系统中做什么”，而不是只描述抽象工艺路线。

### 9. Tool-authored Language Cleanup
这是本批 process review 的高频补充规则。

检查点：
- 文案里是否残留：
  - `Evidence basis`
  - `applied here as a proxy`
  - `interpreted as`
  - `users should ...`
  - `未提供...因此...`
  - `证据基础`
  - `作为...代理`
- 是否保留明显 prompt/cluster/代理路线口吻。

处理原则：
- 这些内容可保留在 case evidence 或 source note 中，不应直接以这种口吻留在 process dataset 的正式说明里。
- 正式数据集文本应简短、确定、可复核。

## Output Contract

所有 process review 至少应产出：
- 一份机器可读 rubric / findings / plan / summary
- 一份 `operations-log`
- 一份 `final-summary`
- 一份 `verification`
- 限制说明（`cli-friction` / `skill-friction`）

若任务是 snapshot/account review，至少产出：
- `outputs/process-review-rubric.json`
- `outputs/process-review-findings.jsonl`
- `outputs/process-review-plan.json`
- `outputs/process-review-summary.json`
- `reports/processes-needing-modification.md`
- `reports/processes-needing-modification.zh-CN.md`

若任务走 `node scripts/run-review.mjs --profile process`，当前 canonical CLI 输出为：
- `one_flow_rerun_timing.md`
- `one_flow_rerun_review_v2_1_zh.md`
- `one_flow_rerun_review_v2_1_en.md`
- `flow_unit_issue_log.md`
- `review_summary_v2_1.json`
- `process-review-report.json`

每个 review 文件都应包含：
- 证据充足结论
- 证据不足结论/限制

## Current Limitations

当前原生支持尚不包括：
- 按 review-report presence、反向引用关系等复杂条件直接筛选远端 process 的原生命令；
- 把 snapshot review 和下游治理/修复编排成一个 CLI 单命令的 native flow。

因此遇到更复杂的远端全量 process review 任务时：
- 优先使用 `run-remote-process-review.mjs` + `tiangong process list` 的现有 canonical 组合；
- 只有在组合仍不足以表达任务时，才允许使用补充 bridge；
- 补充 bridge 应视为临时方案，而不是 skill 的默认用法。
