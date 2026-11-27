"""Web search tools. DuckDuckGo needs no key; Tavily has a free monthly quota.

DuckDuckGo blocks bursts of parallel requests, so searches go out at most two at
a time, failed searches are retried with a pause and a shorter query, and
Tavily is used as a backup when a TAVILY_API_KEY is set.
"""
import asyncio
import logging
import random
import re

import httpx

from . import config

log = logging.getLogger(__name__)

_DDG_SLOTS = asyncio.Semaphore(2)  # max parallel DuckDuckGo calls
_STOPWORDS = {
    "what", "how", "why", "which", "who", "when", "where", "are", "is", "the", "a", "an",
    "and", "or", "of", "in", "for", "to", "with", "being", "used", "does", "do", "their",
    "associated", "ways", "use",
}


def shorten(query: str) -> str:
    """'In what ways are AI agents applied to DevOps in 2026?' -> 'AI agents applied DevOps 2026'"""
    words = re.findall(r"[\w\-\.]+", query)
    kept = [w for w in words if w.lower() not in _STOPWORDS]
    return " ".join(kept[:8]) or query


async def _duckduckgo_once(query: str, k: int) -> list[dict]:
    from ddgs import DDGS

    def run() -> list[dict]:
        try:
            return list(DDGS().text(query, max_results=k))
        except Exception as exc:  # ddgs raises when it gets nothing back
            if "no results" in str(exc).lower():
                return []
            raise

    async with _DDG_SLOTS:
        rows = await asyncio.to_thread(run)
        await asyncio.sleep(1)  # be polite between calls
    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in rows
        if r.get("href")
    ]


async def _duckduckgo(query: str, k: int) -> list[dict]:
    """Try the full query, then a shorter one, pausing longer after each miss."""
    attempts = [query, shorten(query), shorten(query)]
    last_error = None
    for i, q in enumerate(attempts):
        try:
            results = await _duckduckgo_once(q, k)
            if results:
                return results
        except Exception as exc:  # noqa: BLE001  (rate limit, timeout...)
            last_error = exc
            log.warning("DuckDuckGo attempt %d failed for %r: %s", i + 1, q, exc)
        await asyncio.sleep(2 * (i + 1) + random.random())
    if last_error:
        raise last_error
    return []


async def _tavily(query: str, k: int) -> list[dict]:
    if not config.TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set")
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}"},
            json={"query": query, "max_results": k, "search_depth": "basic"},
        )
        resp.raise_for_status()
    return [
        {"title": r.get("title", ""), "url": r["url"], "snippet": r.get("content", "")}
        for r in resp.json().get("results", [])
    ]


async def web_search(query: str, k: int | None = None) -> list[dict]:
    k = k or config.RESULTS_PER_QUERY
    if config.MOCK_MODE:
        from .mock import mock_search

        return mock_search(query, k)
    if config.SEARCH_PROVIDER == "tavily":
        return await _tavily(query, k)

    try:
        results = await _duckduckgo(query, k)
    except Exception:  # noqa: BLE001
        if not config.TAVILY_API_KEY:
            raise
        results = []
    if not results and config.TAVILY_API_KEY:
        log.info("DuckDuckGo found nothing; trying Tavily for %r", query)
        results = await _tavily(query, k)
    return results
