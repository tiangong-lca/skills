from __future__ import annotations

import json
import re
import tomllib
from typing import Any, Protocol

from openai import OpenAI

from tiangong_lca_spec.core.json_utils import parse_json_response

TRANSLATION_PROMPT = (
    "You are a professional translator for LCA datasets. Translate the provided text into the requested "
    "target language. Use Simplified Chinese when target_lang is 'zh'. Preserve technical terms, units, and "
    'proper nouns. Return strict JSON: {"translation": "..."}.'
)

CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


class LanguageModelProtocol(Protocol):
    def invoke(self, input_data: dict[str, Any]) -> Any: ...


class Translator:
    """Lightweight translation helper supporting both LLMs and OpenAI chat."""

    def __init__(
        self,
        *,
        model: str | None = None,
        secrets_path: str = ".secrets/secrets.toml",
        llm: LanguageModelProtocol | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._model = model
        self._llm = llm
        self._max_tokens = max_tokens
        self._client = None if llm is not None else self._make_client(secrets_path)
        self._cache: dict[tuple[str, str], str] = {}

    def _make_client(self, secrets_path: str) -> OpenAI:
        client_kwargs: dict[str, Any] = {}
        try:
            with open(secrets_path, "rb") as fh:
                secrets = tomllib.load(fh)
            openai_conf = secrets.get("openai", {})
            api_key = openai_conf.get("api_key")
            if api_key:
                client_kwargs["api_key"] = api_key
            if not self._model and openai_conf.get("model"):
                self._model = openai_conf.get("model")
        except Exception:
            pass
        if not self._model:
            self._model = "gpt-4o-mini"
        return OpenAI(**client_kwargs)

    def translate(self, text: str, target_lang: str) -> str | None:
        """Translate to target_lang ('en' or 'zh'), returning None on failure."""
        if not text or target_lang not in {"en", "zh"}:
            return None
        cleaned = text.strip()
        if not cleaned:
            return None
        if target_lang == "zh" and CJK_PATTERN.search(cleaned):
            return cleaned
        if target_lang == "en" and not CJK_PATTERN.search(cleaned):
            return cleaned

        cache_key = (cleaned, target_lang)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._llm is not None:
            translated = self._translate_with_llm(cleaned, target_lang)
        else:
            translated = self._translate_with_client(cleaned, target_lang)
        if translated:
            self._cache[cache_key] = translated
        return translated

    def _translate_with_llm(self, text: str, target_lang: str) -> str | None:
        payload = {
            "prompt": TRANSLATION_PROMPT,
            "context": {"target_lang": target_lang, "text": text},
            "response_format": {"type": "json_object"},
        }
        try:
            raw = self._llm.invoke(payload)
        except Exception:
            return None
        parsed = self._parse_translation(raw)
        if parsed:
            return parsed
        return self._coerce_string(raw)

    def _translate_with_client(self, text: str, target_lang: str) -> str | None:
        prompt = f"Target language: {target_lang}\nText:\n{text}"
        system = "Translate text concisely, preserve technical terms. Return only the translation."
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_completion_tokens=self._estimate_max_tokens(text),
            )
            content = resp.choices[0].message.content if resp.choices else None
            if not content:
                return None
            return content.strip()
        except Exception:
            return None

    def _estimate_max_tokens(self, text: str) -> int:
        if self._max_tokens is not None:
            return self._max_tokens
        return min(2000, max(256, len(text) // 2))

    @staticmethod
    def _parse_translation(raw: Any) -> str | None:
        if isinstance(raw, dict):
            value = raw.get("translation")
            return value.strip() if isinstance(value, str) and value.strip() else None
        if isinstance(raw, str):
            try:
                parsed = parse_json_response(raw)
            except Exception:
                return None
            if isinstance(parsed, dict):
                value = parsed.get("translation")
                return value.strip() if isinstance(value, str) and value.strip() else None
        return None

    @staticmethod
    def _coerce_string(raw: Any) -> str | None:
        if raw is None:
            return None
        text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        text = text.strip()
        return text or None


__all__ = ["Translator", "LanguageModelProtocol"]
