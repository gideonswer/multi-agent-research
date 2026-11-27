"""Fake LLM + search so the app and tests run offline with zero API keys."""
import json
import re


def mock_search(query: str, k: int) -> list[dict]:
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:40]
    return [
        {
            "title": f"Article {i + 1} about {query}",
            "url": f"https://example.com/{slug}/{i + 1}",
            "snippet": f"Key fact #{i + 1} about {query}.",
        }
        for i in range(min(k, 3))
    ]


def mock_chat(system: str, user: str) -> str:
    if "PLANNER" in system:
        q = user.split("Question:", 1)[-1].strip()
        return json.dumps(
            {"sub_questions": [f"What is {q}?", f"Why does {q} matter?", f"What is next for {q}?"]}
        )
    if "RESEARCHER" in system:
        return "- Main finding from the first source [1].\n- Supporting detail from another source [2]."
    return (
        "# Research Report\n\n## Summary\nThis is a mock report built from the findings [1][2].\n\n"
        "## Details\nEach section cites its sources [3]."
    )
