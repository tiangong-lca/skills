# Process-Automated-Builder 拆分方案

## 1. 文档目的

本文档给出一个面向 OpenClaw 的重构方案：把当前 `/home/huimin/projects/lca-skills/process-automated-builder` 从“大一统 skill + 大一统运行链”拆成更细的 skill，并明确哪些部分应由 OpenClaw 的 orchestration / workspace / subagent 能力承担，哪些部分仍应保留为共享 runtime library。

本文档重点解决两个问题：

1. 当前 `process-automated-builder` 的能力边界过宽，单个 skill 同时承载了资料检索、资料清洗、SI 下载、过程拆分、exchange 抽取、flow 匹配、占位修复、平衡审查、发布修复等职责。
2. 当前流程对“用户直接指定参考来源”的支持不够一等公民。现有主链默认以 KB 检索和 SI 下载为主，用户自带的文档、表格、人工整理文本、标准条文、企业资料、专家笔记，缺乏统一的规范化注入接口。

本文档目标不是做一个纯概念图，而是给出一套可以直接进入实施的：

- skill 拆分建议
- 共享 artifact / contract 设计
- OpenClaw 协作编排方式
- subagent 并发边界
- 迁移路径
- 你提出的“先把参考资料拆解成规范化 json/text，再作为 process 拆分和 exchange 取值依据”的落地方案


## 2. 现状分析

### 2.1 当前 `process-automated-builder` 实际上已经包含两层流程

当前 skill 并不是一个纯单阶段脚本，而是两层叠加：

1. 外层编排层
   - 入口：`scripts/run-process-automated-builder.sh`
   - 主编排：`scripts/origin/process_from_flow_langgraph.py workflow`
   - 该层当前已经显式拆出 7 个阶段：
     - `01_references`
     - `02_usability`
     - `03_si_download`
     - `04_mineru`
     - `05_usage_tagging`
     - `06_clear_stop_after`
     - `07_main_pipeline`

2. 内层主推理链
   - 入口：`tiangong_lca_spec/process_from_flow/service.py`
   - 当前主链节点已经细分为：
     - `load_flow`
     - `describe_technology`
     - `split_processes`
     - `generate_exchanges`
     - `enrich_exchange_amounts`
     - `preflight_chain_continuity`
     - `match_flows`
     - `sync_chain_link_uuids`
     - `align_exchange_units`
     - `density_conversion`
     - `build_sources`
     - `generate_intended_applications`
     - `build_process_datasets`
     - `resolve_placeholders`
     - `verify_chain_link_uuids`
     - `balance_review`
     - `generate_data_cutoff_principles`

结论：当前实现已经天然暴露出可拆分边界，只是这些边界还没有被升格为独立 skill / 独立 artifact contract。


### 2.2 当前 repo 中已经存在一批可独立升格的脚本

当前 repo 已经有多条适合直接升格为 skill 的 CLI，而不是必须从零重写：

- `scripts/origin/process_from_flow_reference_usability.py`
- `scripts/origin/process_from_flow_download_si.py`
- `scripts/origin/mineru_for_process_si.py`
- `scripts/origin/process_from_flow_reference_usage_tagging.py`
- `scripts/origin/process_from_flow_build_sources.py`
- `scripts/origin/process_from_flow_placeholder_report.py`
- `scripts/origin/process_from_flow_langgraph.py flow-auto-build`
- `scripts/origin/process_from_flow_langgraph.py process-update`

这意味着拆分不应该以“重写 Python 逻辑”为第一优先，而应该优先做：

1. 重新定义 skill 边界；
2. 抽出稳定的输入输出 contract；
3. 让 orchestrator 调用现有脚本或共享 runtime；
4. 再逐步把共享逻辑从单 repo 中抽成公共库。


### 2.3 当前流程的核心问题

#### 问题 A：单个 skill 的职责过宽

当前 `process-automated-builder` 同时做了：

- 文献检索
- 文献 fulltext 汇总
- 文献可用性判断
- SI 下载
- SI 解析
- 参考资料 usage tagging
- 路线推理
- process 拆分
- exchange 生成
- 数值抽取
- flow 匹配
- 单位对齐
- 密度换算
- source dataset 构建
- process dataset 构建
- placeholder 修复
- 平衡校验
- 发布修复和 publish

这会带来三个直接后果：

1. 任何一处需求变化都容易把整个 skill prompt / reference / runtime 改乱。
2. 无法自然利用 OpenClaw 的 subagent / multi-skill 协作能力。
3. 用户指定的外部来源只能被“硬塞”进现有 state，而不是成为一等公民。


#### 问题 B：当前 state 写入模型不适合多个 agent 直接协作

当前明确存在单写者约束：

- `cache/process_from_flow_state.json`
- `cache/process_from_flow_state.json.lock`

这说明在一个 `run_id` 下，多个 agent 不能同时写同一个 canonical state 文件。

如果不改变 state 组织方式，而直接让多个 subagent 一起“各写一点到 state”，会导致：

- 写冲突
- 状态污染
- 合并不可审计
- resume 语义失真

因此后续架构必须采用：

- specialist skill 输出自己的独立 artifact；
- orchestrator 作为唯一 canonical writer；
- final state 由 orchestrator 统一归并。


#### 问题 C：用户外部来源没有稳定注入层

你提出的核心需求是正确的：

> 先把参考资料拆解整理成规范化 json 或 text 文件，再作为 process 拆分、exchange 数值提取的参考来源。

这本质上要求把“来源准备”从主链里独立出来，并让主链改为消费一个统一的 `reference_bundle` / `evidence_bundle`，而不是默认假设来源主要来自：

- KB 搜索
- DOI fulltext
- SI 下载

如果不做这一步，未来你无论接标准、报告、企业台账、Excel、Word、人工摘要、网页摘录，都会继续卡在“如何塞进当前 state”的低效集成方式。


## 3. OpenClaw 现状对本方案的约束与机会

### 3.1 当前 OpenClaw 已经具备的能力

基于当前 `.openclaw` 配置，可直接利用的机制包括：

- skill 仓库加载
  - `openclaw.json -> skills.load.extraDirs` 已包含：
    - `/home/huimin/projects/custom-skill`
    - `/home/huimin/projects/lca-skills`
- 多 agent / 多 workspace 隔离
  - 当前实例已采用 `agent + workspace` 作为真实隔离边界
- subagent 并发
  - `agents.defaults.subagents.maxConcurrent = 8`
- orchestrator 风格的本地 manifest 产物
  - 现有 lifecyclemodel orchestration 已形成：
    - `assembly-plan.json`
    - `graph-manifest.json`
    - `lineage-manifest.json`
    - `invocations/*.json`
    - `publish-bundle.json`

这些能力说明：

1. 你不需要为了这次拆分重新设计 OpenClaw 平台层。
2. 你需要做的是让新的 process skill 群和现有 OpenClaw orchestration 风格对齐。


### 3.2 当前 OpenClaw 的硬约束

#### 约束 1：保密和上下文隔离边界是 `agent + workspace`

这意味着：

- 客户资料
- 企业内部表格
- 标准草稿
- 用户指定来源

不应混放在共享 workspace 里，更不应把“来源 bundle”直接沉淀到公共 skill 仓库中当作长期默认上下文。

正确做法是：

- 新的来源规范化 artifact 放在项目 agent 的 workspace 或单独 run 目录下；
- skill 只定义“如何处理来源”，不保存具体来源内容。


#### 约束 2：当前会话中 skill 可见性仍有“镜像枚举”问题

OpenClaw 运行时可从 `/home/huimin/projects/lca-skills` 直接加载 skill；
但当前 Codex / OpenClaw 会话的 skill 枚举仍依赖 `~/.agents/skills` 镜像。

因此 rollout 时要区分两件事：

1. skill 代码真实位置
   - 仍然是 `/home/huimin/projects/lca-skills/<skill-name>`
2. 当前会话可见性
   - 需要时用：
   - `/home/huimin/.openclaw/scripts/sync_project_skill.sh`


### 3.3 当前 OpenClaw 对新方案最有价值的地方

最重要的不是“能开多少个 agent”，而是这三点：

1. 可把大流程拆成多个 skill，由 orchestrator 调用。
2. 可把语义密集、证据密集的子任务交给 subagent 并行完成。
3. 可把每次 run 固化为本地 manifest，使结果可追溯、可恢复、可复查。


## 4. 设计原则

### 4.1 Artifact-First，而不是 State-First

每个 specialist skill 输出独立 artifact，orchestrator 再归并。

推荐原则：

- specialist skill 不直接写 canonical `orchestration_state.json`
- specialist skill 输出：
  - `request.normalized.json`
  - `result.json`
  - `report.json`
  - `evidence.jsonl`
  - `handoff-summary.json`
- orchestrator 是唯一 canonical writer


### 4.2 Evidence-First，而不是 KB-First

主链应优先消费统一的 evidence bundle。
KB、SI、用户自带文档、手工整理 text、表格、标准附件都只是来源类型，不应在架构上分高低。

推荐优先级由 `source_policy` 控制，而不是写死在代码里。


### 4.3 技能薄、运行库厚

不建议把 `tiangong_lca_spec` 中的所有 Python 模块各复制一份到每个 skill。

正确做法：

- skill：边界、wrapper、reference、request/response contract
- shared runtime：模型、validator、mapper、artifact builder、flow search client、publish helper


### 4.4 并发只做“可独立、可归并、可追溯”的部分

适合 subagent 并发的部分：

- 来源拆解
- 来源规范化
- 可用性判断
- SI 解析
- 数值证据抽取
- route / process split 候选方案生成
- 审核 / review

不适合并发直接写 canonical state 的部分：

- final route selection
- final process chain selection
- placeholder 最终改写
- publish


### 4.5 兼容式迁移，而不是一次性推倒重来

短期内应允许：

- 新的 evidence bundle 注入现有 `scientific_references`
- 现有 `process_from_flow` 主链继续工作
- 老的 `process-automated-builder` 继续作为 umbrella skill / compatibility shim


## 5. 目标分层架构

建议分成 4 层。

### 5.1 第 0 层：共享 runtime library

建议保留或抽出的共享模块：

- `tiangong_lca_spec/core/*`
- `tiangong_lca_spec/process_from_flow/prompts.py`
- `tiangong_lca_spec/process_extraction/*`
- `tiangong_lca_spec/flow_search/*`
- `tiangong_lca_spec/flow_alignment/*`
- `tiangong_lca_spec/product_flow_creation/*`
- `tiangong_lca_spec/publishing/*`
- `tiangong_lca_spec/workflow/artifacts.py`
- `tiangong_lca_spec/state_lock.py`

这层不建议直接做成对用户暴露的 skill。
这层的职责是“算法与结构复用”，不是“协作边界”。


### 5.2 第 1 层：细粒度 specialist skills

以下是建议的目标 skill 列表。

| Skill 名称 | 主要职责 | 主要输入 | 主要输出 | 当前可复用实现 |
| --- | --- | --- | --- | --- |
| `lca-reference-bundle-normalizer` | 把用户指定来源规范化为统一 bundle | 本地文件、URL、纯文本、表格、手工摘要 | `reference_bundle.json`、`reference_index.jsonl`、`evidence_items.jsonl` | 新增为主，复用 OpenClaw 通用文档 skill |
| `lca-reference-kb-retriever` | KB/DOI 检索、fulltext 汇总、聚类 | flow + query policy | `kb_reference_bundle.json` | `service.py` 中 reference search/fulltext/cluster 逻辑 |
| `lca-reference-usability-review` | 判断参考资料是否可用于 step1/2/3 | `reference_bundle` 或 `kb_reference_bundle` | `usability_report.json` | `process_from_flow_reference_usability.py` |
| `lca-si-bundle-builder` | 下载、解析、抽取 SI，并写回 bundle | DOI / 来源 bundle | `si_bundle.json`、`si_snippets.jsonl` | `process_from_flow_download_si.py`、`mineru_for_process_si.py` |
| `lca-reference-usage-tagger` | 给来源打 `tech_route/process_split/exchange_values/background_only` 标签 | `reference_bundle`、`si_bundle` | `usage_tagging.json` | `process_from_flow_reference_usage_tagging.py` |
| `process-route-planner` | 生成技术路线候选 | flow + evidence bundle | `route_plan.json` | `describe_technology` |
| `process-chain-splitter` | 生成 unit process 链条 | flow + route + evidence bundle | `process_chain.json` | `split_processes` |
| `process-exchange-generator` | 生成每个 process 的 input/output exchanges | `process_chain.json` + evidence | `exchange_plan.json` | `generate_exchanges` |
| `process-exchange-value-extractor` | 从来源中提取 exchange amount/unit/basis | `exchange_plan.json` + evidence + SI | `exchange_values.json` | `enrich_exchange_amounts` |
| `process-chain-review` | 做 chain continuity、uuid sync、结构审查 | `process_chain.json`、`exchange_values.json` | `chain_review.json` | `preflight_chain_continuity`、`sync_chain_link_uuids`、`verify_chain_link_uuids` |
| `flow-match-resolver` | flow search、candidate selection、单位对齐、密度修复 | `exchange_values.json` | `flow_match_bundle.json` | `match_flows`、`align_exchange_units`、`density_conversion` |
| `source-dataset-builder` | 构建 source datasets | `reference_bundle`、usage tagging | `source_bundle.json` | `build_sources`、`process_from_flow_build_sources.py` |
| `process-dataset-builder` | intended applications + process dataset 生成 | `process_chain.json`、`flow_match_bundle.json`、`source_bundle.json` | `process_bundle.json` | `generate_intended_applications`、`build_process_datasets` |
| `process-placeholder-balance-review` | placeholder 修复、placeholder report、balance review、cutoff summary | `process_bundle.json` | `qa_bundle.json`、修订后的 `process_bundle.json` | `resolve_placeholders`、`process_from_flow_placeholder_report.py`、`balance_review`、`generate_data_cutoff_principles` |
| `lca-publish-executor` | 统一 publish contract 层 | `publish_bundle.json` 或 run_id | `publish_report.json` | 现有 skill，可直接复用 |


### 5.3 第 2 层：orchestrator skill

建议把现有 `process-automated-builder` 逐步收缩为 orchestrator 层。

推荐职责：

- intake request
- 选择来源策略
- 调度 specialist skills
- 管理 run 目录
- 记录 invocation manifest
- 控制 resume / retry / fail-fast
- 统一生成 `assembly-plan.json` / `graph-manifest.json` / `lineage-manifest.json`
- 统一写 `agent_handoff_summary.json`
- 与 publish skill 对接

推荐命名：

- 兼容期：仍叫 `process-automated-builder`
- 目标名：`process-from-flow-orchestrator`


### 5.4 第 3 层：workspace / agent 层

OpenClaw 层负责：

- 项目隔离
- 来源文件存放
- 长期 memory
- 对话上下文
- run artifact 浏览
- subagent 调度

skill 不负责：

- 客户资料长期存放
- 项目级别 memory
- 跨会话路由


## 6. 哪些东西不建议拆成 skill

以下内容不建议各自拆成独立 skill：

1. `tidas_mapping`
   - 这是 schema 映射与规范化逻辑，适合作为 runtime library。
2. `core/config/models/logging`
   - 这些是基础设施，不是协作边界。
3. `state_lock`
   - 这是底层并发保护，不是用户可调用能力。
4. `workflow/artifacts.py`
   - 这是 artifact builder 工具层，适合作为 builder skill 的依赖，不适合作为独立 skill。
5. `flow_search` 纯 client 层
   - 若已有 `flow-hybrid-search` / `process-hybrid-search`，则保持搜索能力与“process-from-flow 主链决策”解耦，不建议再复制一层“纯搜索 skill”。


## 7. 核心 contract 设计

### 7.1 顶层 request contract

建议新增统一 request 文件，例如 `pff-request.json`。

最小形态如下：

```json
{
  "request_id": "pff-aluminum-foil-001",
  "flow_file": "/abs/path/reference-flow.json",
  "operation": "produce",
  "workspace_run_root": "/abs/path/run-001",
  "source_inputs": [
    {
      "source_id": "src-paper-001",
      "type": "local_file",
      "path": "/abs/path/paper.pdf",
      "intended_roles": ["tech_route", "process_split"]
    },
    {
      "source_id": "src-table-001",
      "type": "local_file",
      "path": "/abs/path/mass-balance.xlsx",
      "intended_roles": ["exchange_values"]
    },
    {
      "source_id": "src-note-001",
      "type": "local_text",
      "path": "/abs/path/expert-notes.md",
      "intended_roles": ["assumptions", "process_split"]
    }
  ],
  "source_policy": {
    "step1_route": {
      "preferred": ["user_bundle", "kb_bundle"],
      "fallback": "expert_judgement"
    },
    "step2_process_split": {
      "preferred": ["user_bundle.process_split", "si_bundle", "kb_bundle.process_split"],
      "fallback": "expert_judgement"
    },
    "step3b_exchange_values": {
      "preferred": ["user_bundle.exchange_values", "si_bundle", "kb_bundle.exchange_values"],
      "require_numeric_evidence": true,
      "allow_estimation": true
    }
  },
  "execution_policy": {
    "mode": "orchestrated",
    "allow_subagents": true,
    "max_subagents": 6,
    "fail_fast": true,
    "publish": false
  }
}
```


### 7.2 `reference_bundle.json` 设计

这是整套方案的核心。

建议 bundle 至少包含 4 层信息：

1. `sources[]`
   - 每个原始来源对象
2. `documents[]`
   - 每个来源拆解后的文档结构
3. `evidence_items[]`
   - 每条可被引用的证据
4. `coverage`
   - 哪些 step 已经被哪些来源覆盖

建议形态：

```json
{
  "bundle_id": "bundle-001",
  "generated_at": "2026-03-22T12:00:00Z",
  "reference_flow": {
    "name": "Aluminum foil",
    "uuid": "..."
  },
  "sources": [
    {
      "source_id": "src-paper-001",
      "origin_type": "user_local_file",
      "path": "/abs/path/paper.pdf",
      "mime_type": "application/pdf",
      "language": "en",
      "citation": {
        "title": "Example paper",
        "doi": "10.xxxx/xxxx",
        "url": ""
      },
      "normalization_status": "completed"
    }
  ],
  "documents": [
    {
      "document_id": "doc-paper-001",
      "source_id": "src-paper-001",
      "doc_type": "pdf_fulltext",
      "normalized_text_path": "evidence/text/doc-paper-001.md",
      "structured_json_path": "evidence/structured/doc-paper-001.json",
      "supported_roles": ["tech_route", "process_split", "exchange_values"]
    }
  ],
  "evidence_items": [
    {
      "evidence_id": "evi-001",
      "source_id": "src-paper-001",
      "document_id": "doc-paper-001",
      "role": "process_split",
      "claim_type": "unit_process",
      "subject": {
        "process_name": "Melting",
        "exchange_name": null
      },
      "claim_text": "The process includes melting, casting, and rolling.",
      "normalized_value": null,
      "quote": "melting, casting, and rolling",
      "location_hint": "CN",
      "year_hint": "2024",
      "confidence": 0.92
    },
    {
      "evidence_id": "evi-002",
      "source_id": "src-table-001",
      "document_id": "doc-table-001",
      "role": "exchange_values",
      "claim_type": "exchange_amount",
      "subject": {
        "process_name": "Melting",
        "exchange_name": "Electricity"
      },
      "claim_text": "Electricity demand is 540 kWh per tonne input.",
      "normalized_value": {
        "amount": 540,
        "unit": "kWh",
        "basis_amount": 1,
        "basis_unit": "t"
      },
      "quote": "540 kWh/t",
      "table_locator": {
        "sheet": "Energy",
        "row": 8,
        "column": "B"
      },
      "confidence": 0.98
    }
  ],
  "coverage": {
    "tech_route": ["src-paper-001"],
    "process_split": ["src-paper-001", "src-note-001"],
    "exchange_values": ["src-table-001"]
  }
}
```


### 7.3 为什么 `reference_bundle` 必须独立存在

因为它解决了三件以前没有统一解决的事：

1. 把来源准备和 process 生成解耦；
2. 把“用户自带来源”和“KB 来源”放到同一消费接口；
3. 让 step2 与 step3b 可以显式知道“证据到底来自哪里”。


### 7.4 其他中间 artifact contract

建议每个 specialist skill 只收上一步产物，不直接读全局 state。

推荐中间产物：

- `route_plan.json`
- `process_chain.json`
- `exchange_plan.json`
- `exchange_values.json`
- `chain_review.json`
- `flow_match_bundle.json`
- `source_bundle.json`
- `process_bundle.json`
- `qa_bundle.json`
- `publish_bundle.json`

其中：

- `process_chain.json` 负责结构
- `exchange_values.json` 负责量化
- `flow_match_bundle.json` 负责 flow reference 和单位修复
- `process_bundle.json` 负责最终 ILCD process/source dataset
- `qa_bundle.json` 负责 placeholder/balance/cutoff 结果


### 7.5 与旧 state 的兼容映射

为避免一次性重写主链，建议先引入一个 compatibility adapter，把 `reference_bundle` 转成旧 `scientific_references` 结构。

建议映射：

- `reference_bundle.sources + documents` -> `scientific_references.step_1a_reference_search`
- `reference_bundle.documents.fulltext` -> `scientific_references.step_1b_reference_fulltext`
- `reference_bundle.coverage / route clustering` -> `scientific_references.step_1c_reference_clusters`
- `reference_bundle.usability_report` -> `scientific_references.usability`
- `reference_bundle.si_snippets` -> `scientific_references.si_snippets`
- `reference_bundle.usage_tagging` -> `scientific_references.usage_tagging`

这样可以做到：

1. 旧主链先继续跑；
2. 新来源先能注入；
3. 之后再把节点逐步改成直接消费新 contract。


## 8. 用户指定来源的落地方式

### 8.1 推荐的来源注入策略

建议把来源分成三类：

1. `user_bundle`
   - 用户明确指定、项目特定、优先级最高
2. `si_bundle`
   - 附件或 supporting information
3. `kb_bundle`
   - 平台 KB / DOI 检索结果

再由 `source_policy` 按 step 控制优先级。

示例：

- step1 `process-route-planner`
  - `user_bundle` > `kb_bundle` > `expert_judgement`
- step2 `process-chain-splitter`
  - `user_bundle.process_split` > `si_bundle` > `kb_bundle.process_split`
- step3b `process-exchange-value-extractor`
  - `user_bundle.exchange_values` > `si_bundle` > `kb_bundle.exchange_values` > `industry_average`


### 8.2 这样设计后，你的例子会如何落地

你举的例子是：

1. 你先把参考资料拆解整理成规范化 json 或 text；
2. 这些文件作为 process 拆分和 exchange 数值提取的参考来源；
3. 来源不再局限于 KB。

落地后流程会变成：

1. `lca-reference-bundle-normalizer`
   - 读取你提供的 pdf/docx/xlsx/md/txt/url
   - 输出统一的 `reference_bundle.json`
2. `process-route-planner`
   - 直接消费 `reference_bundle`
   - route summary 的主证据来自你指定的资料
3. `process-chain-splitter`
   - 优先消费 `evidence_items.role=process_split`
4. `process-exchange-value-extractor`
   - 优先消费 `evidence_items.role=exchange_values`
   - 对表格类来源保留 `sheet/row/column` 定位
5. 当 bundle 里没有足够证据时，才回退到：
   - SI
   - KB
   - 行业平均
   - expert judgement


### 8.3 对“直接指定来源”的额外建议

为了后续审计和修复，建议每条 evidence 最少保留：

- `source_id`
- `document_id`
- `role`
- `claim_type`
- `quote`
- `normalized_value`
- `locator`
- `confidence`

这样后面如果某个 exchange 数值有问题，可以逆向查到：

- 它来自哪份文件
- 文件哪一页/哪张表
- 是原始值还是推导值


## 9. OpenClaw 协作和 subagent 方案

### 9.1 建议的协作角色

推荐由一个 orchestrator agent 驱动多个 specialist subagent。

#### orchestrator agent

职责：

- 读取 `pff-request.json`
- 生成 `assembly-plan.json`
- 启动 specialist skills / subagents
- 归并 artifact
- 写 canonical state / summary

#### evidence workers

职责：

- 按来源文件并行做拆解、清洗、结构化、摘录

写入边界：

- 只写 `run_root/evidence/*`

#### modeling workers

职责：

- route 候选生成
- process split 候选生成
- exchange value 候选抽取

写入边界：

- 只写 `run_root/stage_outputs/<stage>/candidates/*`

#### review workers

职责：

- chain review
- placeholder review
- balance diagnosis

写入边界：

- 只写 `run_root/reviews/*`


### 9.2 并发矩阵

| 阶段 | 适合并发的单位 | 是否适合 subagent | 备注 |
| --- | --- | --- | --- |
| 来源规范化 | 每个 source / document | 是 | 最适合并发，天然独立 |
| usability review | 每个 DOI / source | 是 | 输出独立 report |
| SI 解析 | 每个 SI 文件 | 是 | 但统一命名和归档要由 orchestrator 完成 |
| route 方案生成 | 每个 route prompt variant | 是 | 只生成候选，不直接定稿 |
| process split | 每个 route / 每个 evidence slice | 是 | 最终链条只能单点归并 |
| exchange value 抽取 | 每个 process / 每个 source chunk | 是 | 输出候选值及证据 |
| flow search | 每个 exchange | 不建议用 subagent | 当前代码已支持线程并行，更适合留在 skill 内 |
| 单位对齐 / 密度修复 | 每个 exchange | 不建议用 subagent | 规则密集，宜在本地 deterministic 逻辑中完成 |
| build datasets | 整个 process bundle | 否 | 需要统一 schema 和一致性 |
| publish | 整个 run | 否 | 远端写操作必须单点控制 |


### 9.3 单写者规则

新的 orchestrator 仍然必须保留“单写者”原则，但写入对象应从旧 `process_from_flow_state.json` 升级为更通用的：

- `run_root/state/orchestration_state.json`

原则：

- subagent 不写 canonical state
- subagent 只写 stage-local artifact
- orchestrator 统一 merge


## 10. 推荐的 run 目录结构

建议向现有 lifecyclemodel orchestrator 风格对齐。

```text
run_root/
  request/
    pff-request.json
    request.normalized.json
    source-policy.json
  evidence/
    incoming/
    normalized/
      reference_bundle.json
      reference_index.jsonl
      evidence_items.jsonl
      usability_report.json
      si_bundle.json
      usage_tagging.json
    text/
    structured/
  stage_outputs/
    01_route/
      route_plan.json
      candidates/
    02_process_split/
      process_chain.json
      candidates/
    03_exchange_plan/
      exchange_plan.json
    04_exchange_values/
      exchange_values.json
      candidates/
    05_chain_review/
      chain_review.json
    06_flow_match/
      flow_match_bundle.json
    07_source_build/
      source_bundle.json
    08_process_build/
      process_bundle.json
    09_qa/
      qa_bundle.json
    10_publish/
      publish_bundle.json
  manifests/
    assembly-plan.json
    graph-manifest.json
    lineage-manifest.json
    invocation-index.json
  invocations/
    *.json
  reviews/
    placeholder_report.json
    balance_review.json
    method_policy_report.json
  state/
    orchestration_state.json
  logs/
    *.log
  handoff/
    agent_handoff_summary.json
```

这套目录有几个好处：

1. specialist skill 的输出边界清楚；
2. subagent 可安全并行写不同目录；
3. resume/retry 可按 stage 重进；
4. 最终 handoff 给其他 skill 或人工时，结构稳定。


## 11. 现有代码到新 skill 的映射建议

### 11.1 可直接提取为 skill 的部分

#### `lca-reference-usability-review`

直接来源：

- `scripts/origin/process_from_flow_reference_usability.py`

建议保留：

- 现有 prompt 与 run-id 写回逻辑

建议改造：

- 支持读取 `reference_bundle.json` 而不是只读旧 state


#### `lca-si-bundle-builder`

直接来源：

- `scripts/origin/process_from_flow_download_si.py`
- `scripts/origin/mineru_for_process_si.py`

建议保留：

- DOI 过滤、cluster 过滤、`--min-si-hint`

建议改造：

- 输出统一 `si_bundle.json`
- 不再默认只依赖旧 `scientific_references`


#### `lca-reference-usage-tagger`

直接来源：

- `scripts/origin/process_from_flow_reference_usage_tagging.py`

建议改造：

- 输入改为 `reference_bundle + si_bundle`
- 输出独立 tagging artifact


#### `process-placeholder-balance-review`

直接来源：

- `resolve_placeholders`
- `process_from_flow_placeholder_report.py`
- `balance_review`
- `generate_data_cutoff_principles`

建议改造：

- placeholder 修复和 balance review 共享同一 QA skill，避免切得过碎


### 11.2 适合作为共享 runtime 再由 wrapper skill 调用的部分

#### `process-route-planner`

直接来源：

- `describe_technology`

建议不要复制 Python 主逻辑，而是：

- 把 route 生成相关逻辑抽成 service
- 新 skill 仅包一层 request/response wrapper


#### `process-chain-splitter`

直接来源：

- `split_processes`

建议特殊处理：

- `reference_output_unit_policy`
- `chain_contract` 生成


#### `process-exchange-generator`

直接来源：

- `generate_exchanges`

建议保留为纯生成 skill，不负责数值抽取和 flow 匹配。


#### `process-exchange-value-extractor`

直接来源：

- `enrich_exchange_amounts`

这里最值得接入你说的“外部规范化来源 bundle”。
这是新方案的高优先级核心。


#### `flow-match-resolver`

直接来源：

- `match_flows`
- `align_exchange_units`
- `density_conversion`
- `tiangong_lca_spec/flow_alignment/*`
- `tiangong_lca_spec/flow_search/*`

建议保持为一个 skill，不再继续细切，因为：

- flow 匹配
- 单位对齐
- 密度修复

三者强耦合，共享同一 exchange 上下文。


#### `process-dataset-builder`

直接来源：

- `generate_intended_applications`
- `build_process_datasets`
- `workflow/artifacts.py`

建议让它只负责：

- process/source dataset 构建
- 本地 schema validation

不负责 publish。


## 12. 推荐的实施阶段

### Phase 0：先加 contract，不拆逻辑

先新增：

- `pff-request.json`
- `reference_bundle.json`
- `source_policy`
- `orchestration_state.json`

目标：

- 让外部来源先能进入系统
- 不急着拆代码目录


### Phase 1：先拆最成熟、最独立的 4 个 skill

优先拆：

1. `lca-reference-usability-review`
2. `lca-si-bundle-builder`
3. `lca-reference-usage-tagger`
4. `process-placeholder-balance-review`

原因：

- 它们已经有独立脚本
- 风险低
- 对主链侵入小


### Phase 2：新增 `lca-reference-bundle-normalizer`

这是整个方案里最关键的一步。

因为从这一阶段开始：

- 用户自带来源成为一等公民
- step2 / step3b 开始真正脱离“只能依赖 KB”

建议内部复用 OpenClaw 通用能力：

- `pdf`
- `docx`
- `xlsx`
- `extract`
- `document-granular-decompose`

LCA skill 只做“LCA 证据规范化”，不做通用格式解析重复造轮子。


### Phase 3：拆 foreground 生成链

拆出：

1. `process-route-planner`
2. `process-chain-splitter`
3. `process-exchange-generator`
4. `process-exchange-value-extractor`
5. `flow-match-resolver`
6. `process-dataset-builder`

到这一步后，原 `process-automated-builder` 基本只剩 orchestrator 价值。


### Phase 4：将 `process-automated-builder` 收缩为 orchestrator

这一阶段建议：

- 保留旧名字做兼容
- `SKILL.md` 改为强调 orchestration / plan / invoke / resume / publish handoff
- specialist 技能成为主入口能力


## 13. 推荐的 repo 结构

推荐在 `/home/huimin/projects/lca-skills` 下新增：

```text
lca-reference-bundle-normalizer/
lca-reference-kb-retriever/
lca-reference-usability-review/
lca-si-bundle-builder/
lca-reference-usage-tagger/
process-route-planner/
process-chain-splitter/
process-exchange-generator/
process-exchange-value-extractor/
process-chain-review/
flow-match-resolver/
source-dataset-builder/
process-dataset-builder/
process-placeholder-balance-review/
process-from-flow-orchestrator/    # 或暂时保留 process-automated-builder
```

如果短期不想改 skill 名：

- 先保留 `process-automated-builder`
- 后续新建 `process-from-flow-orchestrator`
- 最后让旧 skill 变成 thin compatibility wrapper


## 14. 风险与控制

### 风险 1：skill 过细导致 contract 漂移

控制方式：

- 所有中间 artifact 都版本化：
  - `schema_version`
- 每个 skill 自带 request/response schema
- orchestrator 在 merge 前做 schema 校验


### 风险 2：subagent 并发把同一 run 写乱

控制方式：

- subagent 只写 stage-local artifact
- orchestrator 单写 canonical state
- 保留 lock 文件机制


### 风险 3：用户来源质量不齐，影响链条稳定性

控制方式：

- `reference_bundle` 中增加：
  - `confidence`
  - `origin_type`
  - `supported_roles`
  - `quality flags`
- `source_policy` 决定是否允许低质量来源主导 step2 / step3b


### 风险 4：共享 runtime 被复制后版本漂移

控制方式：

- runtime 抽出成单独 package 或统一依赖路径
- skill 只放 wrapper，不复制核心 Python 模块


### 风险 5：项目来源污染共享 workspace

控制方式：

- 具体来源文件只留在项目 agent workspace / run 目录
- skill repo 不保存项目具体资料
- 外部敏感项目使用独立 agent/workspace


## 15. 最小可执行版本建议

如果只做一个最小可执行版本，我建议不是一次性拆 10 个 skill，而是先做下面 4 件事：

1. 新增 `lca-reference-bundle-normalizer`
   - 先解决用户指定来源的问题
2. 给现有 `process-automated-builder` 增加 `reference_bundle` 输入
   - 先通过 compatibility adapter 注入旧 state
3. 把已独立的脚本升格为 4 个 specialist skills
   - usability
   - si
   - usage tagging
   - placeholder/balance review
4. 给 orchestrator 加上统一 manifest
   - `assembly-plan.json`
   - `graph-manifest.json`
   - `lineage-manifest.json`
   - `agent_handoff_summary.json`

这样做的收益最大：

- 你能最快获得“外部来源优先”的能力；
- 你能最快把单体 skill 变成可协作体系；
- 同时不会把现有成功率已经验证过的主链完全打散。


## 16. 最终建议

最终推荐方案不是“把当前 skill 机械切成十几个 prompt 文件”，而是：

1. 把 `process-automated-builder` 重构成一个 orchestrator；
2. 把来源准备、过程推理、匹配修复、QA、publish handoff 拆成 specialist skills；
3. 把 `reference_bundle` 作为新的核心输入层；
4. 把 `agent + workspace` 当作项目来源与 memory 的真实隔离边界；
5. 把 subagent 用在来源拆解、证据抽取、候选方案生成和 review 上，而不是让多个 agent 直接写同一个 state；
6. 把 `tiangong_lca_spec` 保持为共享 runtime，而不是复制到每个 skill。

如果按这个方案推进，你要的能力会自然出现：

- 参考来源不再局限于 KB；
- 你可直接指定文档、表格、文本摘要作为建模证据；
- OpenClaw 能够以 orchestrator + skill + subagent 的方式合作完成整条 process-from-flow 链；
- 整个流程的 artifact、证据、修复、发布都能被清楚追溯。
