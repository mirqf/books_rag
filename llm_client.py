from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence
from urllib import error, request

from settings import get_secret, get_setting


@dataclass
class LLMContext:
    citation_id: int
    book_title: str
    location: str
    text: str


class OpenRouterClient:
    def __init__(self, api_key: str, model: str, app_name: str = "hackathon-rag-bot") -> None:
        self.api_key = api_key
        self.model = model
        self.app_name = app_name

    @classmethod
    def from_env(cls) -> Optional["OpenRouterClient"]:
        api_key = get_secret("OPENROUTER_API_KEY", "")
        model = get_setting("OPENROUTER_MODEL", "openrouter/free") or "openrouter/free"
        if not api_key:
            return None
        return cls(api_key=api_key, model=model)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model)

    def validate_configuration(self) -> None:
        if not self.enabled:
            raise RuntimeError("OPENROUTER_API_KEY or OPENROUTER_MODEL is missing")
        result = self._chat(
            messages=[
                {"role": "system", "content": "Return exactly OK."},
                {"role": "user", "content": "Check configuration."},
            ],
            temperature=0.0,
            max_tokens=4,
        )
        if not result.strip():
            raise RuntimeError("LLM validation failed: empty response")

    def synthesize_answer(self, question: str, contexts: Sequence[LLMContext]) -> dict:
        context_blocks = []
        for context in contexts:
            context_blocks.append(
                f"[{context.citation_id}] {context.book_title} | {context.location}\n{context.text}"
            )
        prompt = "\n\n".join(context_blocks) if context_blocks else "Контекст не найден."

        raw = self._chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты corrective-RAG ассистент по книгам. "
                        "Используй только факты из контекста. "
                        "Если контекста недостаточно, не выдумывай. "
                        "Верни только JSON-объект формата: "
                        "{\"sufficient\": true|false, "
                        "\"answer\": \"...\", "
                        "\"citation_ids\": [1,2], "
                        "\"needs_retry\": true|false, "
                        "\"retry_query\": \"...\"}. "
                        "Если данных недостаточно, answer должен быть ровно "
                        "\"Недостаточно данных в загруженных книгах.\" ЕСЛИ ТЫ ВЕРНЕШЬ ЧТО-то КРОМЕ JSON Я СБРОШУСЬ с 18 ЭТАЖА"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Вопрос: {question}\n\n"
                        f"Контекст:\n{prompt}\n\n"
                        "Выбери только реально полезные цитаты. "
                        "Если текущий retrieval промахнулся, поставь needs_retry=true "
                        "и предложи один более точный retry_query."
                    ),
                },
            ],
            temperature=0.05,
            max_tokens=900,
        )
        data = self._parse_json(raw)
        if not isinstance(data, dict):
            raise RuntimeError(f"Expected JSON object from answer synthesis, got: {data}")
        return data

    def _chat(
        self,
        messages: Sequence[dict],
        temperature: float = 0.1,
        max_tokens: int = 700,
    ) -> str:
        if not self.enabled:
            raise RuntimeError("OpenRouter is not configured")

        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": list(messages),
        }
        data = self._request_chat(payload)
        return self._extract_text_response(data)

    def _request_chat(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://openrouter.ai",
                "X-Title": self.app_name,
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenRouter connection error: {exc.reason}") from exc

    def _parse_json(self, raw: str) -> Any:
        raw = raw.strip()
        candidates = [raw]
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.S)
        candidates.extend(fenced)
        object_match = re.findall(r"(\{.*\})", raw, flags=re.S)
        array_match = re.findall(r"(\[.*\])", raw, flags=re.S)
        candidates.extend(object_match)
        candidates.extend(array_match)

        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        raise RuntimeError(f"Failed to parse JSON from LLM response: {raw}")

    def _extract_text_response(self, data: dict) -> str:
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenRouter response: {data}") from exc

        message = choice.get("message") if isinstance(choice, dict) else None
        content = None
        if isinstance(message, dict):
            content = message.get("content")

        text = self._content_to_text(content)
        if text:
            return text

        fallback_candidates = [
            choice.get("text") if isinstance(choice, dict) else None,
            message.get("reasoning") if isinstance(message, dict) else None,
            message.get("refusal") if isinstance(message, dict) else None,
            data.get("output_text") if isinstance(data, dict) else None,
        ]
        for candidate in fallback_candidates:
            text = self._content_to_text(candidate)
            if text:
                return text

        raise RuntimeError(f"OpenRouter returned empty content: {data}")

    def _content_to_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    if item.strip():
                        parts.append(item.strip())
                    continue
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type in {"text", "output_text"}:
                    value = item.get("text")
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
                elif "text" in item and isinstance(item["text"], str) and item["text"].strip():
                    parts.append(item["text"].strip())
            return "\n".join(parts).strip()
        if isinstance(content, dict):
            for key in ("text", "content", "value"):
                value = content.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""
