"""Prompt templates for the process-from-flow LangGraph workflow."""

TECH_DESCRIPTION_PROMPT = (
    "You are an expert LCA practitioner and process engineer.\n"
    "Given a single ILCD flow definition (the 'reference flow'), list the plausible technology/process routes "
    "for producing (or treating/disposal of) the flow.\n"
    "\n"
    "Rules:\n"
    "- Base your answer strictly on the provided flow context (name, classification, general comment, treatment/mix fields).\n"
    "- If scientific references are provided, use them as primary evidence for the route descriptions and assumptions.\n"
    "- If step_1c_reference_clusters are provided, prioritize the primary cluster; do not mix incompatible clusters.\n"
    "- If SI snippets are provided, use them to confirm route steps and cite them in route_evidence.\n"
    "- Do NOT invent numeric quantities.\n"
    "- Output 1..4 routes; if multiple routes are plausible (e.g., different production technologies), include them as separate routes.\n"
    "- Keep each route concise but specific enough to derive unit processes and exchanges later.\n"
    "- For each route, include route_evidence with source_type and citations; keep supported_dois aligned to the evidence used.\n"
    "\n"
    "Return strict JSON:\n"
    "{\n"
    '  "routes": [\n'
    "    {\n"
    '      "route_id": "R1",\n'
    '      "route_name": "...",\n'
    '      "route_summary": "...",\n'
    '      "key_unit_processes": ["..."],\n'
    '      "key_inputs": ["..."],\n'
    '      "key_outputs": ["..."],\n'
    '      "assumptions": ["..."],\n'
    '      "scope": "...",\n'
    '      "supported_dois": ["..."],\n'
    '      "route_evidence": {\n'
    '        "source_type": "literature|si|expert_judgement",\n'
    '        "citations": ["..."],\n'
    '        "notes": "..."\n'
    "      }\n"
    "    }\n"
    "  ]\n"
    "}\n"
)

INTENDED_APPLICATIONS_PROMPT = (
    "You are writing the intended applications field for an ILCD process dataset.\n"
    "Use the provided technical description, scope, and assumptions to summarize the intended application(s) "
    "of the data collection and modelling.\n"
    "\n"
    "Rules:\n"
    "- Base your answer strictly on the provided inputs; do NOT invent claims.\n"
    "- Provide both English and Chinese versions.\n"
    "- Keep 1..3 concise sentences per language.\n"
    "- Avoid marketing language and avoid repeating the input verbatim.\n"
    "\n"
    "Return strict JSON:\n"
    "{\n"
    '  "intended_applications": {\n'
    '    "en": "...",\n'
    '    "zh": "..."\n'
    "  }\n"
    "}\n"
)

DATA_CUTOFF_COMPLETENESS_PROMPT = (
    "You are writing dataCutOffAndCompletenessPrinciples for an ILCD process dataset.\n"
    "Use the provided summary of exchange completeness, placeholders, unit conversions, density conversions, "
    "and balance review checks.\n"
    "\n"
    "Rules:\n"
    "- Describe cut-off/completeness based only on the provided summary.\n"
    "- Provide both English and Chinese versions.\n"
    "- Mention missing amounts or unresolved placeholders when present.\n"
    "- Mention unit/density conversions and remaining unit mismatches when present.\n"
    "- Keep 1..3 concise sentences per language.\n"
    "\n"
    "Return strict JSON:\n"
    "{\n"
    '  "data_cut_off_and_completeness_principles": {\n'
    '    "en": "...",\n'
    '    "zh": "..."\n'
    "  }\n"
    "}\n"
)

PROCESS_SPLIT_PROMPT = (
    "You are selecting/using the route options and decomposing each route into unit processes (single operations).\n"
    "Input context includes the reference flow summary, the route options from Step 1, and any technical description.\n"
    "\n"
    "Rules:\n"
    "- If scientific references are provided, use them to identify and split unit processes; avoid adding steps without evidence and capture gaps in assumptions.\n"
    "- If SI snippets are provided, use them to confirm unit operations and add citations to assumptions where possible.\n"
    "- Output 1..4 routes, each with 1..6 processes.\n"
    "- For each route, processes must be ordered from upstream to downstream (P1 -> P2 -> ...).\n"
    "- If multiple processes in a route, the reference flow of process i must be an input exchange for process i+1; "
    "the last process directly produces (or treats/disposes) the reference flow.\n"
    "- Exactly one process per route MUST be marked as `is_reference_flow_process=true` (the last process when multiple).\n"
    "- Use clear, short process names.\n"
    "- Provide `process_id` values like P1, P2, ...\n"
    "- Each process must include structured fields split by: technology/process, inputs, outputs, boundary, assumptions.\n"
    "- Each process MUST define reference_flow_name (the main output flow of the process).\n"
    "- Process name must include four modules: base_name, treatment_and_route, mix_and_location, quantitative_reference.\n"
    "- quantitative_reference must be a numeric expression like '1 kg of <reference_flow_name>' or '1 unit of <reference_flow_name>'. If unit is unknown, use 'unit'.\n"
    "- For each process, provide a geography decision describing where the process occurs.\n"
    "- Use ILCD/TIDAS location codes (see input_data/location). Choose the most specific code supported by evidence.\n"
    "- If the process is located in China but some inputs use non-China datasets, keep location_code=CN and explain the substitution in description_of_restrictions.\n"
    "- If the geography is mixed or unclear, use GLO and explain why.\n"
    "- Ensure chain consistency: the reference_flow_name of process i must appear verbatim in process i+1 inputs.\n"
    "- Provide inputs/outputs as clean flow names (no f1/f2 labels); labels are added in post-processing.\n"
    "- If step_1c_reference_clusters are provided in the context, prioritize the primary cluster and only use supplementary clusters "
    "when they do not change the main process chain or system boundary.\n"
    "- Do NOT mix clusters with incompatible system boundaries or granularity.\n"
    "\n"
    "Return strict JSON:\n"
    "{\n"
    '  "selected_route_id": "R1",\n'
    '  "routes": [\n'
    "    {\n"
    '      "route_id": "R1",\n'
    '      "route_name": "...",\n'
    '      "processes": [\n'
    "        {\n"
    '          "process_id": "P1",\n'
    '          "reference_flow_name": "...",\n'
    '          "name_parts": {\n'
    '            "base_name": "...",\n'
    '            "treatment_and_route": "...",\n'
    '            "mix_and_location": "...",\n'
    '            "quantitative_reference": "..."\n'
    "          },\n"
    '          "name": "...",\n'
    '          "description": "...",\n'
    '          "structure": {\n'
    '            "technology": "...",\n'
    '            "inputs": ["..."],\n'
    '            "outputs": ["..."],\n'
    '            "boundary": "...",\n'
    '            "assumptions": ["..."]\n'
    "          },\n"
    '          "geography": {\n'
    '            "location_code": "CN|GLO|...",\n'
    '            "location_name": "...",\n'
    '            "description_of_restrictions_en": "...",\n'
    '            "description_of_restrictions_zh": "..."\n'
    "          },\n"
    '          "is_reference_flow_process": true|false\n'
    "        }\n"
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}\n"
)

EXCHANGES_PROMPT = (
    "You are defining the inventory exchanges (inputs/outputs) for each process.\n"
    "Input context includes the reference flow summary, a technical description, and a list of processes.\n"
    "\n"
    "Rules:\n"
    "- Provide plausible exchange names that can be searched in a flow catalogue (prefer English names).\n"
    "- If process provides structured fields (structure/inputs/outputs), use them as primary candidates.\n"
    "- Use scientific references (if provided) to confirm each exchange flow name and amount; only use numeric amounts explicitly supported by references.\n"
    "- Use SI snippets (if provided) as evidence for exchange values and cite them in evidence.\n"
    "- If step_1c_reference_clusters are provided in the context, only use exchange evidence from the primary cluster; "
    "use supplementary clusters only when consistent with the main chain and boundary.\n"
    "- If an exchange amount is not supported by references, set amount to null and note the assumption in generalComment.\n"
    "- Preserve chain naming: when process i outputs intermediate flow name X, process i+1 must include X as an input with the exact same string.\n"
    "- Inputs/outputs may be labeled like 'f1: <name>'; strip the label and use only the flow name for exchangeName.\n"
    "- Do NOT use composite exchange names (e.g., 'energy and machinery', 'air emissions', 'auxiliary materials'). Split into specific flows.\n"
    "- For energy, split into carriers such as electricity, diesel, gasoline, natural gas, or heat as applicable.\n"
    "- For emissions, split into elementary flows (e.g., methane, nitrous oxide, ammonia, CO2, NOx, particulates) "
    "or waterborne pollutants (e.g., nitrate, phosphate, pesticides) when relevant.\n"
    "- For labor, split by activity if multiple (e.g., 'Labor, harvesting' and 'Labor, post-harvest handling').\n"
    "- Add flow_type for each exchange: product | elementary | waste | service.\n"
    "- In generalComment, append machine-readable tags using EXACT keys: [tg_io_kind_tag=<flow_type>] [tg_io_uom_tag=<unit>].\n"
    "- Do NOT use ambiguous tag keys such as classification/category/typeOfDataSet.\n"
    "- Add material_role for each exchange: raw_material | auxiliary | catalyst | energy | emission | product | waste | service | unknown.\n"
    "- Use auxiliary/catalyst for inputs that are not embodied in the main product; set balance_exclude=true for those.\n"
    "- Provide role_reason to justify the material_role choice when it is not obvious.\n"
    "- For emissions, include 'to air' / 'to water' / 'to soil' in exchangeName when applicable.\n"
    "- Provide unit for each exchange (e.g., kg, kWh, MJ, m3, unit). If unsure, use 'unit'.\n"
    "- Provide amount as a numeric string; use null when unknown (placeholders are filled later).\n"
    "- For every exchange, provide data_source and evidence: data_source.source_type must be literature|si|expert_judgement.\n"
    "- data_source.citations must include DOI or URL when available (e.g., 'DOI 10.xxx' or 'https://doi.org/...').\n"
    "- evidence must be a list of non-DOI supporting notes (e.g., 'Doe 2021 Table 2', 'SI Table S3', or inference notes).\n"
    "- For every process, output 1..12 exchanges.\n"
    "- For each process, include exactly one exchange matching reference_flow_name and set is_reference_flow=true.\n"
    "- For the final process (is_reference_flow_process=true), the reference_flow_name must correspond to the load_flow.\n"
    "- Use exchangeDirection='Output' when operation is produce; use exchangeDirection='Input' when operation is treat/dispose.\n"
    "- Use exchangeDirection exactly 'Input' or 'Output'.\n"
    "\n"
    "Return strict JSON with keys:\n"
    "{\n"
    '  "processes": [\n'
    "    {\n"
    '      "process_id": "P1",\n'
    '      "exchanges": [\n'
    "        {\n"
    '          "exchangeDirection": "Input|Output",\n'
    '          "exchangeName": "...",\n'
    '          "generalComment": "...",\n'
    '          "unit": "...",\n'
    '          "amount": null,\n'
    '          "is_reference_flow": true|false,\n'
    '          "flow_type": "product|elementary|waste|service",\n'
    '          "material_role": "raw_material|auxiliary|catalyst|energy|emission|product|waste|service|unknown",\n'
    '          "balance_exclude": true|false,\n'
    '          "role_reason": "...",\n'
    '          "data_source": {"source_type": "literature|si|expert_judgement", "citations": ["DOI ..."]},\n'
    '          "evidence": ["..."]\n'
    "        }\n"
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}\n"
)

EXCHANGE_VALUE_PROMPT = (
    "You are extracting quantitative exchange values from evidence.\n"
    "Input context includes process_exchanges (with exchangeName/unit placeholders), fulltext references, and SI snippets.\n"
    "\n"
    "Rules:\n"
    "- Only use numeric amounts explicitly stated in the provided fulltext references or SI snippets.\n"
    "- Do NOT infer or estimate values. If no explicit value exists for an exchange, omit it from the output.\n"
    "- Match exchangeName exactly to the names in process_exchanges (case-insensitive matching is ok).\n"
    "- Provide unit and amount as a numeric string (e.g., '0.45', '12.3').\n"
    "- Provide evidence citing table/figure or SI location.\n"
    "- If a DOI or URL is available, include it explicitly in evidence.\n"
    "- If the reported value is given per a specific functional unit, include basis_amount, basis_unit, and basis_flow.\n"
    "- source_type must be literature|si.\n"
    "\n"
    "Return strict JSON:\n"
    "{\n"
    '  "processes": [\n'
    "    {\n"
    '      "process_id": "P1",\n'
    '      "exchanges": [\n'
    "        {\n"
    '          "exchangeName": "...",\n'
    '          "amount": "0.0",\n'
    '          "unit": "kg|MJ|kWh|m3|unit",\n'
    '          "basis_amount": "1.0",\n'
    '          "basis_unit": "kg|MJ|kWh|m3|unit",\n'
    '          "basis_flow": "<reference flow name>",\n'
    '          "source_type": "literature|si",\n'
    '          "evidence": ["DOI ... Table X", "SI ..."]\n'
    "        }\n"
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}\n"
)

INDUSTRY_AVERAGE_PROMPT = (
    "You are filling missing exchange amounts using industry-average values.\n"
    "Input context includes the process metadata, exchange metadata, and optional scientific references.\n"
    "\n"
    "Rules:\n"
    "- If references are provided, use them as primary evidence for an industry-average amount per functional unit.\n"
    "- If references are insufficient, you may estimate a reasonable industry-average value, but mark evidence as "
    "'Industry average estimate (expert judgement)'.\n"
    "- If allow_estimate_without_references is false and references are empty, return null for amount.\n"
    "- Only return a value when the evidence (or estimate) matches the SAME system boundary and functional unit "
    "as the process; if boundary or functional unit cannot be matched, return null.\n"
    "- Keep units consistent with the exchange unit when possible; otherwise choose a standard unit.\n"
    "- Return a single numeric value (no ranges). If you cannot estimate, return null for amount.\n"
    "\n"
    "Return strict JSON:\n"
    "{\n"
    '  "amount": "0.0" | null,\n'
    '  "unit": "kg|MJ|kWh|m3|unit" | null,\n'
    '  "evidence": ["..."],\n'
    '  "notes": "short rationale"\n'
    "}\n"
)

DENSITY_ESTIMATE_PROMPT = (
    "You are estimating density for unit conversion between mass and volume.\n"
    "Use common-sense values grounded in the process context and technical description.\n"
    "\n"
    "Rules:\n"
    "- Only estimate density for product/waste flows when mass<->volume conversion is required.\n"
    "- If density cannot be inferred from the context, return null for density_value.\n"
    "- Provide a single numeric density_value (no ranges).\n"
    "- density_unit must be one of: kg/m3, g/cm3, kg/L, g/L, g/mL.\n"
    "- assumptions must describe temperature/pressure/concentration/phase if relevant.\n"
    "- source_type must be expert_judgement.\n"
    "\n"
    "Return strict JSON:\n"
    "{\n"
    '  "density_value": "0.0" | null,\n'
    '  "density_unit": "kg/m3|g/cm3|kg/L|g/L|g/mL" | null,\n'
    '  "assumptions": "...",\n'
    '  "source_type": "expert_judgement",\n'
    '  "notes": "short rationale"\n'
    "}\n"
)

REFERENCE_CLUSTER_PROMPT = (
    "You are clustering scientific references into consistent process systems for process_from_flow.\n"
    "\n"
    "Goal:\n"
    "- Group DOIs that share the same system boundary, main process chain, and key intermediate flow names.\n"
    "- Prefer a self-contained chain that covers Step1+Step2+Step3 when available.\n"
    "- If chains conflict in boundary or granularity, keep them separate (do NOT merge).\n"
    "\n"
    "Return strict JSON:\n"
    "{\n"
    '  "clusters": [\n'
    "    {\n"
    '      "cluster_id": "C1",\n'
    '      "dois": ["..."],\n'
    '      "system_boundary": "cradle-to-gate|gate-to-gate|gate-to-grave|unspecified",\n'
    '      "granularity": "coarse|medium|fine|unknown",\n'
    '      "key_process_chain": ["..."],\n'
    '      "key_intermediate_flows": ["..."],\n'
    '      "supported_steps": ["step1", "step2", "step3"],\n'
    '      "recommendation": "primary|supplement|exclude",\n'
    '      "reason": "..."\n'
    "    }\n"
    "  ],\n"
    '  "primary_cluster_id": "C1",\n'
    '  "selection_guidance": "..."'
    "}\n"
)

PLACEHOLDER_QUERY_BUILDER_PROMPT = (
    "You are building a single structured flow-search query for one unmatched LCA exchange.\n"
    "\n"
    "Goal:\n"
    "- Generate ONE precise query payload that preserves the exchange semantics.\n"
    "- Do not broaden the exchange scope.\n"
    "- Keep direction, flow_type, unit, and compartment constraints consistent with the input.\n"
    "\n"
    "Rules:\n"
    "- Use exchange_name + general_comment as primary context.\n"
    "- If CAS exists, return one CAS number in standard format (e.g., 64-17-5).\n"
    "- classification_hints should be short canonical nouns/phrases (max 6 items).\n"
    "- flow_type must be one of: product | elementary | waste | service | null.\n"
    "- direction must be Input | Output | null.\n"
    "- compartment must be air | water | soil | null.\n"
    "- If information is missing, return null instead of guessing.\n"
    "\n"
    "Return strict JSON:\n"
    "{\n"
    '  "exchange_name": "...",\n'
    '  "description": "...",\n'
    '  "cas": "xx-xx-x" | null,\n'
    '  "classification_hints": ["...", "..."],\n'
    '  "flow_type": "product|elementary|waste|service" | null,\n'
    '  "direction": "Input|Output" | null,\n'
    '  "unit": "kg|m3|MJ|kWh|unit|..." | null,\n'
    '  "compartment": "air|water|soil" | null\n'
    "}\n"
)

PLACEHOLDER_UUID_SELECTOR_PROMPT = (
    "You are selecting the best flow UUID from retrieved flow candidates for one unmatched exchange.\n"
    "\n"
    "Goal:\n"
    "- Choose at most one UUID from candidates, or null if none is appropriate.\n"
    "\n"
    "Rules:\n"
    "- The selected UUID MUST be from the provided candidates list.\n"
    "- Prefer semantic consistency with exchange_name/description and constraints: flow_type, direction, unit, compartment.\n"
    "- If CAS is provided in the query and candidate CAS exists, prefer exact CAS match.\n"
    "- If no candidate is clearly valid, return null.\n"
    "- Keep reason concise (1 sentence).\n"
    "\n"
    "Return strict JSON:\n"
    "{\n"
    '  "selected_uuid": "<uuid>" | null,\n'
    '  "reason": "...",\n'
    '  "confidence": 0.0\n'
    "}\n"
)

REFERENCE_USABILITY_PROMPT = (
    "You are screening a scientific article for usefulness in the process_from_flow workflow.\n"
    "\n"
    "The workflow needs evidence for:\n"
    "- Step 1 (technology routes): concrete descriptions of production/treatment routes or major process stages.\n"
    "- Step 2 (unit process split): explicit unit operations, process sequences, or intermediate products.\n"
    "- Step 3 (exchanges): inventory inputs/outputs, emissions, resources, or quantified exchanges.\n"
    "\n"
    "Decision rules:\n"
    "- Mark 'usable' only if the article provides process-level or inventory details that can support at least one step above.\n"
    "- Mark 'unusable' if it only reports LCIA impact indicators (ADP/AP/GWP/EP/PED/RI) or impact units like 'kg CO2 eq' "
    "without any LCI inventory tables/rows showing physical flows (kg, g, t, m2, m3, pcs, kWh, MJ as inventory).\n"
    "- Mark 'unusable' if it is background-only (policy, market, nutrition/health, generic LCA discussion) without process/inventory detail.\n"
    "- Record si_hint as likely/possible/none when the text points to supporting information, supplementary material, or appendices "
    "that may contain inventory tables; keep decision=unusable unless the main text itself includes LCI tables.\n"
    "- If evidence is weak or indirect, choose 'unusable' and explain the gap.\n"
    "\n"
    "Return strict JSON:\n"
    "{\n"
    '  "decision": "usable|unusable",\n'
    '  "supported_steps": ["step1", "step2", "step3"],\n'
    '  "reason": "...",\n'
    '  "evidence": ["..."],\n'
    '  "si_hint": "none|possible|likely",\n'
    '  "si_reason": "..."\n'
    "}\n"
)

REFERENCE_USAGE_TAGGING_PROMPT = (
    "You are tagging how a scientific reference should be used in the process_from_flow workflow.\n"
    "\n"
    "Tagging categories:\n"
    "- tech_route: supports Step 1 (technology routes or process stages).\n"
    "- process_split: supports Step 2 (unit process split or ordered operations).\n"
    "- exchange_values: supports Step 3/3b (inventory exchanges, inputs/outputs, emissions, or quantified values).\n"
    "- background_only: background context only; no direct support for Steps 1-3.\n"
    "\n"
    "Rules:\n"
    "- Pick all that apply, but never include background_only with other tags.\n"
    "- Use fulltext and SI snippets if available.\n"
    "- If evidence is weak or absent, return background_only.\n"
    "\n"
    "Return strict JSON:\n"
    "{\n"
    '  "usage_tags": ["tech_route", "process_split", "exchange_values", "background_only"],\n'
    '  "reason": "..."'
    "}\n"
)
