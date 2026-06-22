"""OpenAI-SDK client construction shared by every LLM call path.

Every provider speaks the OpenAI wire protocol, so there is one transport; only
base_url / api_key / per-role model differ. Each provider is a row in
``PROVIDERS`` — add a provider by adding a row, not by editing the resolvers.

  * openai     → OPENAI_BASE_URL (ends in /v1), Bearer OPENAI_API_KEY
  * ollama     → {OLLAMA_BASE_URL}/v1, placeholder key (Ollama ignores auth, but
                 the SDK requires a non-empty api_key)
  * openrouter → OPENROUTER_BASE_URL (ends in /v1), Bearer OPENROUTER_API_KEY;
                 no embedding support (embed_model is None)

``resolve_chat`` / ``resolve_chat_async`` / ``resolve_vision`` / ``resolve_embed``
return ``(client, model)`` for the configured provider + role. They raise
``ClientConfigError`` on an unknown provider, a missing base URL / API key, or a
role the provider can't serve; callers translate that into their own domain error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from openai import AsyncOpenAI, OpenAI

from config import AppConfig

# Ollama ignores auth, but openai-python rejects an empty api_key at construction.
_OLLAMA_PLACEHOLDER_KEY = "ollama"


class ClientConfigError(RuntimeError):
    """Provider misconfigured: unknown name, missing base URL / API key, or
    unsupported role."""


def _require(value: str, name: str) -> str:
    if not value:
        raise ClientConfigError(f"{name} is not set")
    return value


def _ollama_base(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise ClientConfigError("OLLAMA_BASE_URL is not set")
    return f"{base}/v1"


@dataclass(frozen=True)
class ProviderSpec:
    """How to reach one OpenAI-wire provider. Every value is resolved from cfg at
    call time. ``embed_model`` is None when the provider can't embed."""

    base_url: Callable[[AppConfig], str]
    api_key: Callable[[AppConfig], str]
    chat_model: Callable[[AppConfig], str]
    vision_model: Callable[[AppConfig], str]
    embed_model: Callable[[AppConfig], str] | None


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        base_url=lambda c: c.OPENAI_BASE_URL,
        api_key=lambda c: _require(c.OPENAI_API_KEY, "OPENAI_API_KEY"),
        chat_model=lambda c: c.OPENAI_MODEL,
        vision_model=lambda c: c.OPENAI_VISION_MODEL,
        embed_model=lambda c: c.OPENAI_EMBED_MODEL,
    ),
    "ollama": ProviderSpec(
        base_url=lambda c: _ollama_base(c.OLLAMA_BASE_URL),
        api_key=lambda c: _OLLAMA_PLACEHOLDER_KEY,
        chat_model=lambda c: c.OLLAMA_LLM_MODEL,
        vision_model=lambda c: c.OLLAMA_VISION_MODEL,
        embed_model=lambda c: c.OLLAMA_EMBED_MODEL,
    ),
    "openrouter": ProviderSpec(
        base_url=lambda c: c.OPENROUTER_BASE_URL,
        api_key=lambda c: _require(c.OPENROUTER_API_KEY, "OPENROUTER_API_KEY"),
        chat_model=lambda c: c.OPENROUTER_MODEL,
        vision_model=lambda c: c.OPENROUTER_VISION_MODEL,
        embed_model=None,
    ),
}


def _spec(provider: str) -> ProviderSpec:
    spec = PROVIDERS.get(provider.strip().lower())
    if spec is None:
        raise ClientConfigError(
            f"unknown provider {provider!r} (expected one of {sorted(PROVIDERS)})"
        )
    return spec


def _build(*, base_url: str, api_key: str, timeout: float) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def resolve_chat(cfg: AppConfig) -> tuple[OpenAI, str]:
    """(client, model) for text/JSON chat — keyed on LLM_PROVIDER (sync)."""
    s = _spec(cfg.LLM_PROVIDER)
    return _build(base_url=s.base_url(cfg), api_key=s.api_key(cfg), timeout=cfg.LLM_TIMEOUT), s.chat_model(cfg)


def resolve_chat_async(
    cfg: AppConfig, *, provider: str | None = None, model: str | None = None
) -> tuple[AsyncOpenAI, str]:
    """(async client, model) for the streamed retrieval agent loop.

    ``provider``/``model`` override the env defaults per request (chat screen);
    credentials / base_url / timeout always come from cfg for the resolved
    provider — the request never carries secrets.
    """
    s = _spec(provider or cfg.LLM_PROVIDER)
    client = AsyncOpenAI(base_url=s.base_url(cfg), api_key=s.api_key(cfg), timeout=cfg.LLM_TIMEOUT)
    return client, model or s.chat_model(cfg)


def resolve_vision(cfg: AppConfig) -> tuple[OpenAI, str]:
    """(client, model) for vision — keyed on LLM_PROVIDER."""
    s = _spec(cfg.LLM_PROVIDER)
    return _build(base_url=s.base_url(cfg), api_key=s.api_key(cfg), timeout=cfg.LLM_TIMEOUT), s.vision_model(cfg)


def resolve_embed(cfg: AppConfig) -> tuple[OpenAI, str]:
    """(client, model) for embeddings — keyed on EMBED_PROVIDER (independent of LLM)."""
    s = _spec(cfg.EMBED_PROVIDER)
    if s.embed_model is None:
        raise ClientConfigError(f"provider {cfg.EMBED_PROVIDER!r} does not support embeddings")
    return _build(base_url=s.base_url(cfg), api_key=s.api_key(cfg), timeout=cfg.LLM_TIMEOUT), s.embed_model(cfg)
