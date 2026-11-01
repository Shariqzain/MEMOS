"""
MEMOS — LLM Provider Abstraction
Supports Ollama, OpenAI, Gemini, and Anthropic via a unified interface.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMResponse:
    def __init__(self, content: str, model: str = "", usage: Optional[Dict] = None):
        self.content = content
        self.model = model
        self.usage = usage or {}


class LLMProvider:
    """Base class for LLM providers."""

    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        raise NotImplementedError

    async def health_check(self) -> bool:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    def __init__(self):
        import httpx
        self.client = httpx.AsyncClient(base_url=settings.llm_base_url, timeout=120)
        self.model = settings.llm_model

    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"

        resp = await self.client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        return LLMResponse(content=content, model=self.model, usage=data.get("eval_count", {}))

    async def health_check(self) -> bool:
        try:
            resp = await self.client.get("/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


class OpenAIProvider(LLMProvider):
    def __init__(self):
        import httpx
        base_url = settings.llm_base_url or "https://api.openai.com/v1"
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=120,
        )
        self.model = settings.llm_model

    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = await self.client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return LLMResponse(content=content, model=self.model, usage=data.get("usage", {}))

    async def health_check(self) -> bool:
        try:
            resp = await self.client.get("/models", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


_provider: Optional[LLMProvider] = None


def get_llm_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        prov = settings.llm_provider
        if prov == "ollama":
            _provider = OllamaProvider()
        elif prov in ("openai", "gemini", "anthropic"):
            _provider = OpenAIProvider()
        else:
            raise ValueError(f"Unknown LLM provider: {prov}")
        logger.info("LLM provider: %s / model: %s", prov, settings.llm_model)
    return _provider


async def llm_complete(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 2048,
    json_mode: bool = False,
) -> str:
    """Convenience wrapper: returns content string."""
    provider = get_llm_provider()
    response = await provider.complete(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
    )
    return response.content


async def llm_json(
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """Call LLM and parse JSON response. Returns empty dict on failure."""
    provider = get_llm_provider()
    try:
        response = await provider.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        content = response.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning("LLM returned invalid JSON: %s", e)
        return {}
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        raise
