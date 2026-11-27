"""One small LLM client for every provider.

Groq, Gemini, Ollama, OpenAI and Anthropic all expose an OpenAI-compatible
endpoint, so the `openai` SDK talks to all of them. Swapping models is a
config change, not a code change.
"""
import json
import logging
import os
import re

from openai import AsyncOpenAI

from . import config

log = logging.getLogger(__name__)

# provider -> (base_url, api-key env var, default model)
PROVIDERS: dict[str, tuple[str, str | None, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-flash-latest",
    ),
    "ollama": ("http://localhost:11434/v1", None, "llama3.1"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY", "claude-sonnet-5-5"),
}


def _client(provider: str) -> tuple[AsyncOpenAI, str]:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider '{provider}'. Choose from {list(PROVIDERS)}")
    base_url, key_env, _ = PROVIDERS[provider]
    api_key = os.getenv(key_env, "") if key_env else "ollama"
    if not api_key:
        raise RuntimeError(f"{key_env} is not set (needed for provider '{provider}')")
    return AsyncOpenAI(base_url=base_url, api_key=api_key), base_url


async def _call(provider: str, model: str, system: str, user: str, temperature: float) -> str:
    client, _ = _client(provider)
    model = model or PROVIDERS[provider][2]
    resp = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content or ""


async def chat(system: str, user: str, temperature: float = 0.2) -> str:
    """Ask the main model; if it fails (rate limit, outage), try the fallback."""
    if config.MOCK_MODE:
        from .mock import mock_chat

        return mock_chat(system, user)
    try:
        return await _call(config.LLM_PROVIDER, config.LLM_MODEL, system, user, temperature)
    except Exception as exc:  # noqa: BLE001
        if not config.FALLBACK_PROVIDER:
            raise
        log.warning("Primary LLM failed (%s); using fallback %s", exc, config.FALLBACK_PROVIDER)
        return await _call(config.FALLBACK_PROVIDER, config.FALLBACK_MODEL, system, user, temperature)


def parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (handles ```json fences)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in model output: {text[:200]}")
    return json.loads(match.group(0))
