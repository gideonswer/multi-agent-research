"""Runs the whole pipeline offline in MOCK_MODE (no API keys needed)."""
import json
import os

os.environ["MOCK_MODE"] = "1"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.graph import clean_citations, graph, renumber  # noqa: E402
from app.main import app  # noqa: E402


@pytest.mark.asyncio
async def test_full_pipeline_produces_cited_report():
    events = []
    final = {}
    async for mode, chunk in graph.astream(
        {"question": "solid-state batteries", "findings": []}, stream_mode=["custom", "values"]
    ):
        events.append(chunk) if mode == "custom" else final.update(chunk)

    agents = {e["agent"] for e in events}
    assert {"planner", "researcher-1", "researcher-2", "researcher-3", "writer"} <= agents
    assert len(final["findings"]) == 3
    assert "## Sources" in final["report"]
    assert final["sources"] and all(s["url"].startswith("https://") for s in final["sources"])


def test_renumber_dedupes_urls_and_rewrites_citations():
    a = {"title": "A", "url": "https://a"}
    b = {"title": "B", "url": "https://b"}
    findings = [
        {"index": 0, "sub_question": "q1", "notes": "x [1] y [2]", "sources": [a, b]},
        {"index": 1, "sub_question": "q2", "notes": "z [1] w [9]", "sources": [b]},
    ]
    text, sources = renumber(findings)
    assert sources == [a, b]
    assert "x [1] y [2]" in text
    assert "z [2] w " in text  # b is global #2; made-up [9] removed


def test_api_streams_steps_and_report():
    client = TestClient(app)
    with client.stream("GET", "/api/research", params={"q": "quantum computing"}) as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    types = [e["type"] for e in events]
    assert types[0] == "step" and types[-1] == "end"
    report = next(e for e in events if e["type"] == "report")
    assert report["report"].startswith("# ")


def test_clean_citations_handles_gpt_oss_style():
    text = "fact 【1†L1-L4】 more 【12】 and [3†src] ok [4]"
    assert clean_citations(text) == "fact [1] more [12] and [3] ok [4]"
