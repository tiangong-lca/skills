"""LLM-backed extractors used in the process extraction stage."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any, Protocol

from tiangong_lca_spec.core.exceptions import ProcessExtractionError
from tiangong_lca_spec.core.json_utils import parse_json_response
from tiangong_lca_spec.core.logging import get_logger
from tiangong_lca_spec.tidas import FieldSummary, get_schema_repository
from tiangong_lca_spec.tidas.level_hierarchy import (
    HierarchyEntry,
    get_process_category_navigator,
    get_product_flow_category_navigator,
)

LOGGER = get_logger(__name__)


class LanguageModelProtocol(Protocol):
    """Minimal protocol required from language models used in the pipeline."""

    def invoke(self, input_data: dict[str, Any]) -> Any: ...


def _truncate(text: str, limit: int = 160) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _schema_type_hint(schema: dict[str, Any] | None) -> str:
    if not isinstance(schema, dict):
        return ""
    type_value = schema.get("type")
    if isinstance(type_value, list):
        # Preserve order but avoid duplicate fragments
        seen: list[str] = []
        for item in type_value:
            if item not in seen:
                seen.append(str(item))
        type_hint = " | ".join(seen)
    elif isinstance(type_value, str):
        type_hint = type_value
    elif "enum" in schema:
        type_hint = "enum"
    else:
        return ""
    return f" [{type_hint}]"


def _schema_is_array(schema: dict[str, Any] | None) -> bool:
    if not isinstance(schema, dict):
        return False
    type_value = schema.get("type")
    if isinstance(type_value, list):
        return "array" in type_value
    return type_value == "array"


def _render_enum_values(values: list[Any], indent: int) -> list[str]:
    prefix = "  " * indent
    formatted = ", ".join(str(value) for value in values)
    return [f"{prefix}- allowed values: {formatted}"]


def _is_multilang_field(field: FieldSummary, schema: dict[str, Any] | None) -> bool:
    if field.reference and "MultiLang" in field.reference:
        return True
    if not isinstance(schema, dict):
        return False
    options = schema.get("anyOf") or schema.get("oneOf") or schema.get("allOf")
    if not isinstance(options, list):
        return False
    for option in options:
        if not isinstance(option, dict):
            continue
        properties = option.get("properties")
        if isinstance(properties, dict) and "@xml:lang" in properties and "#text" in properties:
            return True
    return False


def _render_schema_details(schema: dict[str, Any] | None, indent: int, seen: set[int]) -> list[str]:
    if not isinstance(schema, dict):
        return []
    schema_id = id(schema)
    if schema_id in seen:
        prefix = "  " * indent
        return [f"{prefix}- ... (recursive reference)"]
    seen.add(schema_id)

    lines: list[str] = []
    prefix = "  " * indent

    for key in ("allOf", "anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list) and options:
            for index, option in enumerate(options, start=1):
                option_hint = _schema_type_hint(option)
                lines.append(f"{prefix}- {key} option {index}{option_hint}")
                lines.extend(_render_schema_details(option, indent + 1, seen))
            seen.remove(schema_id)
            return lines

    if _schema_is_array(schema) and isinstance(schema.get("items"), dict):
        extras: list[str] = []
        if schema.get("uniqueItems"):
            extras.append("uniqueItems")
        if "minItems" in schema:
            extras.append(f"minItems={schema['minItems']}")
        if "maxItems" in schema:
            extras.append(f"maxItems={schema['maxItems']}")
        extras_text = f" ({', '.join(extras)})" if extras else ""
        item_schema = schema["items"]
        lines.append(f"{prefix}- items{_schema_type_hint(item_schema)}{extras_text}")
        lines.extend(_render_schema_details(item_schema, indent + 1, seen))
        seen.remove(schema_id)
        return lines

    properties = schema.get("properties")
    if isinstance(properties, dict) and properties:
        required = set(schema.get("required", []))
        for name, child_schema in properties.items():
            type_hint = _schema_type_hint(child_schema)
            flags: list[str] = []
            if name in required:
                flags.append("required")
            if "maxLength" in child_schema:
                flags.append(f"maxLength={child_schema['maxLength']}")
            if "minLength" in child_schema:
                flags.append(f"minLength={child_schema['minLength']}")
            if "pattern" in child_schema:
                flags.append(f"pattern={child_schema['pattern']}")
            suffix = f" ({'; '.join(flags)})" if flags else ""
            lines.append(f"{prefix}- {name}{type_hint}{suffix}")
            enum_values = child_schema.get("enum")
            if isinstance(enum_values, list) and enum_values:
                lines.extend(_render_enum_values(enum_values, indent + 1))
            lines.extend(_render_schema_details(child_schema, indent + 1, seen))

    seen.remove(schema_id)
    return lines


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("#text", "text", "value"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
        parts = [_normalize_text(item) for item in value.values() if isinstance(item, (dict, list, str))]
        return "; ".join(part for part in parts if part)
    if isinstance(value, list):
        parts = [_normalize_text(item) for item in value]
        return "; ".join(part for part in parts if part)
    return str(value).strip()


def _build_process_summary(process_info: dict[str, Any]) -> dict[str, str]:
    data_info = process_info.get("dataSetInformation", {})
    name_block = data_info.get("name", {}) if isinstance(data_info.get("name"), dict) else {}
    summary: dict[str, str] = {}
    summary["baseName"] = _normalize_text(name_block.get("baseName"))
    summary["treatmentStandardsRoutes"] = _normalize_text(name_block.get("treatmentStandardsRoutes"))
    summary["mixAndLocationTypes"] = _normalize_text(name_block.get("mixAndLocationTypes"))
    summary["functionalUnit"] = _normalize_text(data_info.get("quantitativeReference", {}).get("functionalUnitOrOther"))
    general_comment = _normalize_text(data_info.get("common:generalComment")) or _normalize_text(process_info.get("common:generalComment"))
    if general_comment:
        summary["generalComment"] = _truncate(general_comment, limit=800)
    summary = {key: value for key, value in summary.items() if value}
    if not summary:
        uuid = data_info.get("common:UUID")
        if isinstance(uuid, str) and uuid.strip():
            summary["processUUID"] = uuid.strip()
    return summary or {"note": "Insufficient named attributes; rely on LLM reasoning."}


def _build_flow_summary(
    flow_dataset: dict[str, Any],
    source_payload: dict[str, Any] | None = None,
) -> dict[str, str]:
    flow_info = flow_dataset.get("flowInformation", {})
    if not isinstance(flow_info, dict):
        flow_info = {}
    data_info = flow_info.get("dataSetInformation", {})
    if not isinstance(data_info, dict):
        data_info = {}
    name_block = data_info.get("name", {}) if isinstance(data_info.get("name"), dict) else {}
    summary: dict[str, str] = {}
    summary["baseName"] = _normalize_text(name_block.get("baseName"))
    summary["treatmentStandardsRoutes"] = _normalize_text(name_block.get("treatmentStandardsRoutes"))
    summary["mixAndLocationTypes"] = _normalize_text(name_block.get("mixAndLocationTypes"))
    summary["flowProperties"] = _normalize_text(name_block.get("flowProperties"))
    summary["synonyms"] = _normalize_text(data_info.get("common:synonyms"))
    summary["generalComment"] = _normalize_text(data_info.get("common:generalComment"))
    technology = flow_info.get("technology", {})
    if isinstance(technology, dict):
        summary["technology"] = _normalize_text(technology.get("technologicalApplicability"))
    if source_payload:
        summary["jsonldCategory"] = _normalize_text(source_payload.get("category"))
        summary["jsonldDescription"] = _normalize_text(source_payload.get("description"))
        summary["jsonldSynonyms"] = _normalize_text(source_payload.get("synonyms"))
        summary["jsonldCAS"] = _normalize_text(source_payload.get("cas"))
        summary["jsonldFormula"] = _normalize_text(source_payload.get("formula"))
    summary = {key: value for key, value in summary.items() if value}
    if not summary:
        uuid = data_info.get("common:UUID")
        if isinstance(uuid, str) and uuid.strip():
            summary["flowUUID"] = uuid.strip()
    return summary or {"note": "Insufficient attributes; rely on LLM reasoning."}


def _format_fields(
    fields: list[FieldSummary],
    schema_node: dict[str, Any] | None,
    *,
    indent: int = 0,
    depth: int = 2,
) -> list[str]:
    lines: list[str] = []
    prefix = "  " * indent
    properties = schema_node.get("properties") if isinstance(schema_node, dict) else {}
    for field in fields:
        field_schema = properties.get(field.name, {}) if isinstance(properties, dict) else {}
        type_hint = f" [{field.type}]" if field.type else ""
        required = " (required)" if field.required else ""
        description = f": {_truncate(field.description)}" if field.description else ""
        lines.append(f"{prefix}- {field.name}{type_hint}{required}{description}")

        enum_values = field_schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            lines.extend(_render_enum_values(enum_values, indent + 1))

        if _is_multilang_field(field, field_schema):
            lines.extend(_render_schema_details(field_schema, indent + 1, seen=set()))

        if field.children and depth > 1:
            next_schema = field_schema
            if _schema_is_array(field_schema) and isinstance(field_schema.get("items"), dict):
                next_schema = field_schema["items"]
            lines.extend(
                _format_fields(
                    field.children,
                    next_schema,
                    indent=indent + 1,
                    depth=depth - 1,
                )
            )
    return lines


def _render_summary(title: str, pointer: str, depth: int = 2) -> str:
    repo = get_schema_repository()
    fields = repo.summarize_properties("tidas_processes.json", pointer)
    schema_node = repo.resolve_with_references("tidas_processes.json", pointer)
    lines = [title]
    lines.extend(_format_fields(fields, schema_node, indent=1, depth=depth))
    return "\n".join(lines)


process_guidelines = (
    "Process extraction guidelines:\n"
    "When extracting processes from the document, you must first list every activity or unit"
    " operation that could possibly be a process—without filtering—based on headings, tables,"
    " process descriptions, and any described operations. Only after listing all candidates do"
    " you apply the provided process_guidelines and module_guidelines to classify each candidate"
    " as a valid process, rejected process, or uncertain process. Never skip or silently exclude"
    " any potential process; if unsure, include it as a candidate and mark it uncertain later. The"
    " number of final + rejected + uncertain processes must always equal the number of initial"
    " candidates you listed. Always follow this workflow: (1) enumerate all candidates; (2) apply"
    " the rules; (3) output all results. This prevents omissions.\n"
    "1. Process Definition: a concrete activity that produces a product or service, "
    "consumes resources (energy, materials, land, transport, services), and has "
    "quantified LCI exchanges.\n"
    "2. Only create a process when quantifiable LCI data is provided; descriptive text "
    "without amounts is ignored.\n"
    "3. Only introduce a new process when the document explicitly labels a unit "
    "operation (in tables, section headings, or prose) and associates it with its own "
    "inventory or functional output.\n"
    "4. Do not promote a single exchange from another dataset's inventory into its own "
    "process unless the literature provides an independent LCI for that activity.\n"
    "5. When both parent (aggregated system) and subprocess data are present, create "
    "entries for each subprocess. Treat the parent dataset as the aggregation of its "
    "subprocesses and document that relation in `common:generalComment` instead of "
    "creating an additional record.\n"
    "6. If the literature mentions subprocesses in the text but provides only a total, "
    "system-boundary inventory in the data tables (black-box), create only one process "
    "representing the entire system, and do not create entries for the subprocesses "
    "lacking independent LCI data.\n"
    "7. Treat shared preparation steps, raw material staging, or unallocated "
    '"common" flows that lack their own functional unit as supplemental information. '
    "Do not create a separate Process for them. Write such details, or their total "
    "values, into the relevant subprocess `common:generalComment`.\n"
    "8. Every Process created must define one, and only one, primary product or service "
    "output directly related to its function, which serves as the **Reference Flow**.\n"
    "9. When identifying the Reference Flow, do not blindly assume the overall table "
    "header is the functional unit. You **must** look for the unique functional output "
    "(name and amount) explicitly associated with **this specific unit process** within "
    "the prose, table structure, or dedicated captions. The exchange amount **must be "
    "the exact numerical value specified in the literature**.\n"
    "10. The **Reference Flow** must not be an environmental emission or resource "
    "consumption (Elementary Flow); it **must** be a Product/Service flow "
    "(Technosphere Flow).\n"
    "11. If a Process yields multiple valuable products, you **must** clearly document "
    'the allocation method and basis described in the literature (e.g., "allocation by '
    'economic value," "mass allocation") inside the `common:generalComment`.\n'
    "12. If the same activity has distinct LCI variants by geography, year, or technology "
    "route, create separate records; otherwise merge them.\n"
    "13. Always capture which subprocesses are bundled together, the functional unit, "
    "and key allocation or shared-resource notes inside the process-level "
    "`common:generalComment`.\n"
    "14. Stage 3 flow alignment performs serial MCP lookups; therefore you **must** "
    "reproduce each table row or inventory line as its own `exchange` entry. Never "
    "merge, drop, or average distinct rows—even if values are similar. Preserve the "
    "original units, qualifiers, scenario labels, and footnotes inside `generalComment` "
    "so downstream alignment can trace every source datum.\n"
    "15. Normalize exchange names to Tiangong/ILCD canonical wording (e.g., "
    '"Electricity, medium voltage", "Carbon dioxide, fossil") and ensure every '
    "`generalComment` begins with the exact prefix `FlowSearch hints:` followed by "
    "the pipe-delimited template "
    "`basename=... | treatment=... | mix_location=... | flow_properties=... | "
    "en_synonyms=... | zh_synonyms=... | abbreviation=... | formula_or_CAS=... | "
    "state_purity=... | source_or_pathway=... | usage_context=...`\n"
    "`basename`: English technical name mirroring authoritative datasets. State "
    "the substance/product/waste as industry uses it, separate descriptors with commas "
    "(never semicolons), avoid geography/quantities, spell out acronyms on first use, "
    "and include state/grade cues (e.g., “Polypropylene, PP, granulate”; “Sulfur "
    "dioxide, gaseous”; “Waste glass cullet, mixed colors”).\n"
    "`treatment`: comma-separated qualitative qualifiers ordered from intrinsic "
    "modifications to contextual uses—surface/material treatments (Hot rolled, "
    "Sterilised), referenced standards/grades (EN 10025 S355, ASTM D4806), key "
    "performance attributes (UV-resistant, food-grade), intended uses (for wafer "
    "production, medical packaging), and production/recycling routes (primary "
    "production route, secondary feedstock, steam cracking route). Keep each entry "
    "concise and technical.\n"
    "`mix_location`: comma-separated mix and delivery descriptors (“Production "
    "mix, at plant”; “Consumption mix, to consumer”; “Technology-specific, to "
    "wholesale”; “Production mix, to waste incineration plant”). Use “at” for handover "
    "nodes and “to” when transport up to that node is included.\n"
    "`flow_properties`: comma-separated quantitative properties with explicit "
    "bases (“45 % Fe mass/mass”; “9.6 MJ/kg net calorific value”; “90.5 % methane by "
    "volume”; “750 g/L total dissolved solids”), using SI or accepted industry units "
    "and clarifying the measurement basis when it deviates from mass fraction.\n"
    "`en_synonyms`: semicolon-separated English alternative names/trade names/"
    "abbreviations（例如 “Electric power supply; Grid electricity; Utility electricity”）；输出到 FlowSearch hints "
    "字符串时严禁带方括号或引用整个 Python 列表。\n"
    "`zh_synonyms`: semicolon-separated Chinese equivalents（如“电力；电网供电”），与 `en_synonyms` 含义对应且不重复 `basename`，同样不得输出 "
    "`['...']` 形式。\n"
    "`abbreviation`: canonical abbreviations or short codes ( “MV electricity”, "
    "“NCM622-SiGr”).\n"
    "`state_purity`: physical state, purity or grade, and key operating conditions(“AC 10–30 kV, 50 Hz”, "
    "“Liquid, battery grade, 31–37 wt% HCl”).\n"
    "`source_or_pathway`: English description of supply routes, origin, or geography(“Regional grid, CN; "
    "Secondary aluminium route; Steam reforming feed”).\n"
    "`usage_context`: scenario references (table IDs, process step, functional role)such as “Input to cathode "
    "coating line, Table 3”.\n"
    "`formula_or_CAS`: chemical identifiers or CAS numbers.\n"
    "this ordered template (all fields except `zh_synonyms` must first be written in "
    "English; if you need to add a Chinese gloss, append it in parentheses rather "
    "than interleaving tokens):\n"
    "Populate every required field with substantive bilingual descriptors—never use "
    "`NA`/`N/A` placeholders, and never relocate essential content into notes while "
    "leaving structured slots empty. Append table references, conversion assumptions, "
    "or calculation details **after** the template so downstream alignment retains "
    "provenance.\n"
    "16. Set `exchangeName` to the exact wording printed in the literature tables "
    "(e.g., “Liquid nitrogen”), never a bare acronym. The first entry in `basename` must "
    "repeat that wording before listing abbreviations in `en_synonyms`.\n"
    "17. Fill `treatment`, `mix_location`, and `source_or_pathway` with descriptive, "
    "technical phrases (for example, “High purity, cryogenic liquid” or “Production mix, "
    "at plant (CN)”). Placeholders such as `GLO`, `CN`, `--`, `NA`, `N/A`, `Unspecified`, "
    "or acronyms with ≤3 uppercase letters are forbidden."
)

module_guidelines = (
    "Populate these required fields whenever evidence exists:\n"
    "- processInformation.dataSetInformation:\n"
    "  * `baseName`: concise description of the main activity or function "
    '(e.g., "Coal mining", "Electricity generation").\n'
    "  * `treatmentStandardsRoutes`: provide technical descriptors of the good, "
    "service, or process. Include treatment steps (e.g., pyrolysis, catalytic "
    "reforming), referenced standards (list the full identifier when available), "
    "product quality or grade, intended use, recognised production-route names, "
    "principal feedstocks (educts), and whether the product is primary or "
    "secondary. Separate descriptors with commas and use established industrial "
    "terminology. Exclude end-of-pipe environmental controls.\n"
    "  * `mixAndLocationTypes`: state whether the record is a production or "
    'consumption mix and note the availability location type (e.g., "to consumer", '
    '"at plant"). Separate descriptors with commas.\n'
    "  * `functionalUnitFlowProperties`: list technical qualifiers such as "
    "constituent content or energy content per unit. Separate values with commas. "
    "Non-qualifying metadata (CAS numbers, synonyms, formulas) belongs in the "
    "associated flow dataset, not here.\n"
    "  * `time.referenceYear`: calendar year when the dataset becomes valid. For "
    "multi-year data, use the first year of coverage. If the source omits this, "
    "infer it from the publication year or other documented coverage periods.\n"
    '  * `geography.@location`: explicit ISO/ILCD location; if absent, use "GLO" '
    "to match normalisation defaults.\n"
    "  * `technology`: short description of included technology and system "
    "boundary.\n"
    "- administrativeInformation:\n"
    "  * `common:generalComment`: two to four sentences covering scope, "
    "boundary, and any cross-process notes. This is a required ILCD field—never "
    "leave it empty.\n"
    "  * `common:commissionerAndGoal.common:intendedApplications`: summarise the "
    "stated intended applications of the dataset. Required even if you must infer "
    "from the paper’s study goal (note assumptions).\n"
    "- modellingAndValidation:\n"
    "  * `LCIMethodAndAllocation.typeOfDataSet`, `LCIMethodAndAllocation."
    "LCIMethodPrinciple`, and allocation details. These subfields are mandatory—"
    "write explicit strings such as “Attributional, cradle-to-gate” or “Consequential, "
    "system expansion”.\n"
    "  * `validation.review`: always include a review block. Use "
    '`{"review": {"@type": "Not reviewed"}}` when no external review exists.\n'
    "  * `complianceDeclarations`: fill in the standard ILCD compliance block "
    "(nomenclature/methodological/documentation/quality). Never omit it.\n"
    "  * `dataSourcesTreatmentAndRepresentativeness."
    "dataCutOffAndCompletenessPrinciples`: document the rules applied when "
    "deciding which product, waste, and elementary flows to include (e.g., "
    "cutting infrastructure, cooling water, or auxiliaries below 1 % of impacts).\n"
    "  * `dataSourcesTreatmentAndRepresentativeness."
    "dataSelectionAndCombinationPrinciples`, "
    "`dataTreatmentAndExtrapolationsPrinciples`, and "
    "`referenceToDataHandlingPrinciples`: describe how foreground data were "
    "chosen/combined, how extrapolations were handled, and cite relevant "
    "guidelines or databases.\n"
    "  * `dataSourcesTreatmentAndRepresentativeness.referenceToDataSource`: cite "
    "literature references, reports, database records, or on-site "
    "documentation with author(s), publication year, and a persistent identifier "
    "(DOI, URL, accession). Do not use table titles or captions as standalone "
    "sources. Include original sources for converted datasets and note "
    "parameterisation references when mathematical models are used. For LCI "
    "results or partly terminated systems, list influential background data "
    "sources as well.\n"
    "- exchanges.exchange (for each flow):\n"
    '  * `exchangeDirection`: "Input" or "Output".\n'
    "  * `meanAmount`, `unit`, and `resultingAmount`.\n"
    "  * `exchangeName` / `flowName`: copy the exact wording from the source tables' "
    '"Name of substance" (or equivalent) column—never leave it blank or replace it with '
    "generic codes. This value must match the first entry in FlowSearch `basename`.\n"
    "  * `generalComment`: output a single string in the format described above "
    "(`FlowSearch hints: en_synonyms=Electricity; Electric power; Utility electricity | ... | usage_context=...`). "
    "Flatten every list field into a semicolon-delimited value; never emit Python-style brackets or quoted arrays "
    "inside this comment. Include bilingual synonyms, abbreviations, chemical identifiers, state/purity, source "
    "or supply pathway, and explicit usage context. Every field must contain "
    "substantive information—if the literature omits details, infer them responsibly "
    "or explain the best available descriptor instead of using placeholders such as "
    "`CN`, `GLO`, or `--`. After "
    "the structured segments, append concise notes on data provenance, allocation, "
    "conversions, or table references.\n"
    "  * Omit `referenceToFlowDataSet` and other `referenceTo...` placeholders; Stage 3 "
    "will populate flow references after alignment.\n"
    '  * `@dataSetInternalID`: sequential identifiers as strings starting from "0".'
)

exchange_requirements = (
    "Exchange JSON requirements:\n"
    "Each `exchanges.exchange` entry MUST include `exchangeName`, `exchangeDirection`, "
    "`meanAmount`, `unit`, `resultingAmount`, and a `flowHints` object. The `flowHints` "
    "object must provide: `basename`, `treatment`, `mix_location`, `source_or_pathway`, "
    "`en_synonyms` (array), `zh_synonyms` (array), `abbreviation`, `state_purity`, "
    "`flow_properties`, `usage_context`, and optionally `formula_or_CAS`. Every value "
    'must be a descriptive technical phrase—placeholders such as "", `-`, `NA`, '
    "`N/A`, `TBD`, `Unspecified`, `GLO`, `GLOBAL`, `CN`, or acronyms of ≤3 uppercase "
    "letters are forbidden.\n"
    'Example (geography=CN, table row "Liquid nitrogen"):\n'
    "```\n"
    "{\n"
    '  "exchangeName": "Liquid nitrogen",\n'
    '  "exchangeDirection": "Input",\n'
    '  "meanAmount": "3.80E+01",\n'
    '  "unit": "kg",\n'
    '  "flowHints": {\n'
    '    "basename": "Liquid nitrogen",\n'
    '    "treatment": "High purity, cryogenic liquid",\n'
    '    "mix_location": "Production mix, at plant (CN)",\n'
    '    "source_or_pathway": "Air separation unit, regional supply (CN)",\n'
    '    "en_synonyms": ["Liquid nitrogen", "Nitrogen, liquid"],\n'
    '    "zh_synonyms": ["液氮"],\n'
    '    "abbreviation": "LN2",\n'
    '    "state_purity": "Liquid, 99.999% (5N)",\n'
    '    "flow_properties": "Mass flow, kg",\n'
    '    "usage_context": "Input to all-component physical recovery line",\n'
    '    "formula_or_CAS": "N2; 7727-37-9"\n'
    "  }\n"
    "}\n"
    "```\n"
    "Output strictly valid JSON—no comments, trailing commas, or prose outside the JSON object."
)


@cache
def _build_section_prompt() -> str:
    repo = get_schema_repository()
    metadata_schema = repo.resolve_with_references("tidas_processes.json", "/properties/processDataSet")
    metadata_fields = [field for field in repo.summarize_properties("tidas_processes.json", "/properties/processDataSet") if field.name.startswith("@")]
    metadata_lines = ["processDataSet metadata (auto-populated if omitted):"]
    metadata_lines.extend(_format_fields(metadata_fields, metadata_schema, indent=1, depth=1))
    metadata = "\n".join(metadata_lines)
    process_info = _render_summary(
        "processInformation fields:",
        "/properties/processDataSet/properties/processInformation",
        depth=2,
    )
    modelling = _render_summary(
        "modellingAndValidation fields:",
        "/properties/processDataSet/properties/modellingAndValidation",
        depth=2,
    )
    administrative = _render_summary(
        "administrativeInformation fields:",
        "/properties/processDataSet/properties/administrativeInformation",
        depth=2,
    )
    exchanges = _render_summary(
        "exchanges.exchange fields:",
        "/properties/processDataSet/properties/exchanges",
        depth=2,
    )
    return (
        "You are an expert LCA analyst. Extract structured content that conforms to the "
        "TIDAS ILCD `processDataSet` schema. Before filling fields, follow the guidelines "
        "below.\n\n"
        f"{process_guidelines}\n\n"
        f"{module_guidelines}\n\n"
        f"{exchange_requirements}\n\n"
        "Return JSON with a top-level key `processDataSets` whose value is an array of one or more "
        "objects matching the schema excerpts below. If only a single process is identified, still "
        "return it as a single-element array. Only include fields supported by the schema and omit "
        "entries that are not supported by evidence in the paper. Ensure the JSON is valid and do "
        "not wrap the result in Markdown or a code block.\n\n"
        f"{metadata}\n\n"
        f"{process_info}\n\n"
        f"{modelling}\n\n"
        f"{administrative}\n\n"
        f"{exchanges}"
    )


SECTION_PROMPT = _build_section_prompt()
PROCESS_LIST_PROMPT = (
    "You are enumerating every process or unit operation described in the document. "
    'Read the entire clean text and return JSON of the form {"processes": [...]}. Inclusion rules: (1) keep only '
    "processes whose evidence shows at least one physical LCI row (mass/volume/area/energy/count with explicit units "
    'such as kg, g, t, m2, m3, pcs, kWh, MJ). Treat phrases like "life cycle", "scenario", "system" or '
    '"parent system" as risk hints—not automatic exclusions. If the supporting table still lists ordinary LCI units '
    'keep it; if the table contains only LCIA indicators (ADP/AP/GWP/EP/PED/RI, units such as "kg CO2 eq.", '
    '"kg SO2 eq.", "kg Sb eq.", "kg PO4 eq.", "kg PM2.5 eq.", or MJ explicitly labelled as impacts/depletion) '
    'then exclude it as LCIA-only. Likewise, if the text merely says "inventory shown in Table 4" but the numeric '
    "rows are absent from the provided clean_text, treat it as lacking quantitative LCI. (2) Merge duplicates across "
    "sections/tables into a single entry and aggregate evidence. (3) Exclude narrative mentions, literature-only "
    "references without in-text quantification, and shared utilities or preparation steps lacking an independent "
    "functional unit or LCI. Do not split one table row into multiple processes. Each item must include processId, "
    "name, optional aliases, description, and evidence citing table numbers/section headings/quotes."
)

AGGREGATE_SYSTEM_PROMPT = (
    "You are analysing a life cycle assessment document. Identify every top-level or parent "
    "process system described (for example, production routes, technology options, or supply "
    "chains that contain multiple subprocesses with their own LCIs). Return JSON with the key "
    "`parentProcesses`, whose value is an array. Each item must include `name` (string), optional "
    "`aliases` (array of alternative names), optional `keywords` (array of distinguishing terms), "
    "and optional `subprocessHints` (array summarising important subprocesses mentioned). Only "
    "include parents that have at least one quantified subprocess in the text, and skip shared "
    "preparation steps or utilities that lack independent LCIs or a functional unit. If the "
    "document only provides a single black-box inventory with no decomposed subprocesses, return "
    "an empty array. Ensure every qualifying parent mentioned in the document appears exactly once."
)

CLASSIFICATION_LEVEL_PROMPT = (
    "You are selecting level {level} of the Tiangong (ISIC-based) process classification. "
    "Use `context.process` for the process summary and `context.candidates` for the allowed "
    "options. Choose exactly one candidate and return JSON with '@level', '@classId', and "
    "'#text'. The '@classId' MUST be one of the provided candidate codes, and '#text' must "
    "copy the candidate description verbatim. Do not include explanations or extra fields."
)

CLASSIFICATION_RETRY_ATTEMPTS = 2

FLOW_CLASSIFICATION_LEVEL_PROMPT = (
    "You are selecting level {level} of the Tiangong product flow classification. Use "
    "`context.flow` for the flow summary and `context.candidates` for the allowed options. "
    "Choose exactly one candidate and return JSON with '@level', '@classId', and '#text'. "
    "The '@classId' MUST match one of the provided candidate codes, and '#text' must copy "
    "the candidate description verbatim. Do not include reasoning or extra fields."
)

FLOW_CLASSIFICATION_RETRY_ATTEMPTS = 2

LOCATION_PROMPT = (
    "Normalize the process geography for the schema field "
    "`processInformation.geography.locationOfOperationSupplyOrProduction`. The JSON context "
    "contains `processInformation` or `flowInformation`, plus optional `rawLocationHint`, "
    "`initialCodeSuggestion`, and `candidateLocations` (each entry lists an ILCD code and "
    "description from the official hierarchy). Choose the most specific ILCD code supported "
    "by the evidence—prefer one of the candidate codes when appropriate, otherwise fall back "
    "to another valid ILCD code. Return JSON with `code` and optional `description` or "
    "`subLocation` notes. Do not fabricate new codes."
)


@dataclass
class ProcessListExtractor:
    llm: LanguageModelProtocol

    def run(self, clean_text: str, *, retry_feedback: str | None = None) -> list[dict[str, Any]]:
        LOGGER.info("process_extraction.process_list")
        prompt = PROCESS_LIST_PROMPT
        if retry_feedback:
            prompt = f"{prompt}\n\nPrevious attempt issues (do NOT remove or merge processes; only add/fix entries):\n" f"{retry_feedback}"
        payload = {
            "prompt": prompt,
            "context": clean_text,
            "response_format": {"type": "json_object"},
        }
        response = self.llm.invoke(payload)
        data = _ensure_dict(response)
        processes = data.get("processes")
        if not isinstance(processes, list):
            raise ProcessExtractionError("Process list extractor must return a 'processes' array.")
        return processes


@dataclass
class SectionExtractor:
    llm: LanguageModelProtocol

    def run(
        self,
        clean_text: str,
        *,
        focus_process: dict[str, Any] | None = None,
        retry_feedback: str | None = None,
    ) -> dict[str, Any]:
        LOGGER.info("process_extraction.section_extraction")
        prompt = SECTION_PROMPT
        if focus_process:
            aliases = focus_process.get("aliases") or []
            alias_text = f" (aliases: {', '.join(aliases)})" if aliases else ""
            description = focus_process.get("description") or ""
            evidence = focus_process.get("evidence") or []
            evidence_text = "\n".join(f"- {item}" for item in evidence) if evidence else "- (no citations provided)"
            focus_directive = (
                "You must extract exactly ONE `processDataSet` matching this target. "
                "Never drop, merge, or rename the process; keep the same `processId`.\n"
                f"processId: {focus_process.get('processId')}\n"
                f"name: {focus_process.get('name')}{alias_text}\n"
                f"summary: {description}\n"
                f"evidence:\n{evidence_text}\n"
                'Return JSON shaped as {"processDataSet": {...}} containing only this process.'
            )
            prompt = f"{prompt}\n\n{focus_directive}"
        if retry_feedback:
            prompt = (
                f"{prompt}\n\nPrevious attempt issues:\n"
                f"{retry_feedback}\n"
                "Regenerate the ENTIRE `processDataSet` for this process, fixing every issue above while preserving the same process identity."
            )
        payload = {
            "prompt": prompt,
            "context": clean_text,
            "response_format": {"type": "json_object"},
        }
        response = self.llm.invoke(payload)
        data = _ensure_dict(response)
        return data


@dataclass
class ProcessClassifier:
    llm: LanguageModelProtocol

    def __post_init__(self) -> None:
        self._navigator = get_process_category_navigator()
        self._max_level = self._navigator.max_level

    def run(self, process_info: dict[str, Any]) -> list[dict[str, Any]]:
        LOGGER.info("process_extraction.classification")
        summary = _build_process_summary(process_info)
        selections: list[dict[str, Any]] = []
        parent_code: str | None = None

        for level in range(0, self._max_level + 1):
            candidates = self._navigator.children(parent_code)
            if not candidates:
                break
            selection = self._select_level(level, candidates, summary)
            if selection is None:
                if level == 0:
                    raise ProcessExtractionError("Unable to determine level-0 classification for process.")
                LOGGER.warning(
                    "process_extraction.classification_level_fallback",
                    level=level,
                    parent=parent_code,
                )
                break
            selections.append(selection)
            parent_code = selection["@classId"]
        return selections

    def _select_level(
        self,
        level: int,
        candidates: list[HierarchyEntry],
        summary: dict[str, str],
    ) -> dict[str, Any] | None:
        candidate_codes = {entry.code for entry in candidates}
        code_to_description = {entry.code: entry.description for entry in candidates}
        context = {
            "process": summary,
            "candidates": [
                {
                    "code": entry.code,
                    "description": entry.description,
                    "level": entry.level,
                }
                for entry in candidates
            ],
        }
        payload = {
            "prompt": CLASSIFICATION_LEVEL_PROMPT.format(level=level),
            "context": context,
            "response_format": {"type": "json_object"},
        }
        for attempt in range(CLASSIFICATION_RETRY_ATTEMPTS):
            response = self.llm.invoke(payload)
            selection = self._parse_selection(response, level, candidate_codes, code_to_description)
            if selection:
                return selection
            LOGGER.warning(
                "process_extraction.classification_retry",
                level=level,
                attempt=attempt + 1,
            )
        return None

    def _parse_selection(
        self,
        response: Any,
        level: int,
        allowed_codes: set[str],
        descriptions: dict[str, str],
    ) -> dict[str, Any] | None:
        try:
            data = _ensure_dict(response)
        except ValueError:
            return None
        raw_code = data.get("@classId") or data.get("classId") or data.get("code")
        class_id = str(raw_code).strip() if raw_code is not None else ""
        if class_id not in allowed_codes:
            return None
        description = data.get("#text") or data.get("description") or descriptions.get(class_id) or ""
        description = description.strip() or descriptions.get(class_id, "")
        return {
            "@level": str(level),
            "@classId": class_id,
            "#text": description,
        }


@dataclass
class ProductFlowClassifier:
    llm: LanguageModelProtocol

    def __post_init__(self) -> None:
        self._navigator = get_product_flow_category_navigator()
        self._max_level = self._navigator.max_level

    def run(
        self,
        flow_dataset: dict[str, Any],
        *,
        source_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        LOGGER.info("flow_extraction.classification")
        summary = _build_flow_summary(flow_dataset, source_payload)
        selections: list[dict[str, Any]] = []
        parent_code: str | None = None

        for level in range(0, self._max_level + 1):
            candidates = self._navigator.children(parent_code)
            if not candidates:
                break
            selection = self._select_level(level, candidates, summary)
            if selection is None:
                if level == 0:
                    raise ProcessExtractionError("Unable to determine level-0 classification for flow.")
                LOGGER.warning(
                    "flow_extraction.classification_level_fallback",
                    level=level,
                    parent=parent_code,
                )
                break
            selections.append(selection)
            parent_code = selection["@classId"]
        return selections

    def _select_level(
        self,
        level: int,
        candidates: list[HierarchyEntry],
        summary: dict[str, str],
    ) -> dict[str, Any] | None:
        candidate_codes = {entry.code for entry in candidates}
        descriptions = {entry.code: entry.description for entry in candidates}
        context = {
            "flow": summary,
            "candidates": [
                {
                    "code": entry.code,
                    "description": entry.description,
                    "level": entry.level,
                }
                for entry in candidates
            ],
        }
        payload = {
            "prompt": FLOW_CLASSIFICATION_LEVEL_PROMPT.format(level=level),
            "context": context,
            "response_format": {"type": "json_object"},
        }
        for attempt in range(FLOW_CLASSIFICATION_RETRY_ATTEMPTS):
            response = self.llm.invoke(payload)
            selection = self._parse_selection(response, level, candidate_codes, descriptions)
            if selection:
                return selection
            LOGGER.warning(
                "flow_extraction.classification_retry",
                level=level,
                attempt=attempt + 1,
            )
        return None

    def _parse_selection(
        self,
        response: Any,
        level: int,
        allowed_codes: set[str],
        descriptions: dict[str, str],
    ) -> dict[str, Any] | None:
        try:
            data = _ensure_dict(response)
        except ValueError:
            return None
        raw_code = data.get("@classId") or data.get("classId") or data.get("code")
        class_id = str(raw_code).strip() if raw_code is not None else ""
        if class_id not in allowed_codes:
            return None
        description = data.get("#text") or data.get("description") or descriptions.get(class_id) or ""
        description = description.strip() or descriptions.get(class_id, "")
        return {
            "@level": str(level),
            "@classId": class_id,
            "#text": description,
        }


@dataclass
class LocationNormalizer:
    llm: LanguageModelProtocol

    def run(
        self,
        process_info: dict[str, Any],
        *,
        hint: str | None = None,
        candidates: list[dict[str, str]] | None = None,
        initial_code: str | None = None,
    ) -> dict[str, Any]:
        LOGGER.info("process_extraction.location_normalization")
        if isinstance(process_info, dict) and any(key in process_info for key in ("processInformation", "flowInformation")):
            context_payload: dict[str, Any] = dict(process_info)
        else:
            context_payload = {"processInformation": process_info}
        if hint:
            context_payload["rawLocationHint"] = hint
        if candidates:
            context_payload["candidateLocations"] = candidates
        if initial_code:
            context_payload["initialCodeSuggestion"] = initial_code
        response = self.llm.invoke(
            {
                "prompt": LOCATION_PROMPT,
                "context": context_payload,
                "response_format": {"type": "json_object"},
            }
        )
        return _ensure_dict(response)


@dataclass
class AggregateSystemExtractor:
    llm: LanguageModelProtocol

    def run(self, clean_text: str) -> dict[str, Any]:
        LOGGER.info("process_extraction.aggregate_system_identification")
        response = self.llm.invoke(
            {
                "prompt": AGGREGATE_SYSTEM_PROMPT,
                "context": clean_text,
                "response_format": {"type": "json_object"},
            }
        )
        return _ensure_dict(response)


def _ensure(response: Any) -> Any:
    if hasattr(response, "content"):
        response = getattr(response, "content")
    if isinstance(response, str):
        return parse_json_response(response)
    return response


def _ensure_dict(response: Any) -> dict[str, Any]:
    data = _ensure(response)
    if not isinstance(data, dict):
        raise ValueError("Expected dictionary output from language model")
    return data
