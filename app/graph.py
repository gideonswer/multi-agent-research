"""The multi-agent workflow, built with LangGraph.

    START -> planner -> (researcher x N, in parallel) -> writer -> END

* planner    splits the question into focused sub-questions
* researcher searches the web for one sub-question and writes cited notes
* writer     merges every note into one report with numbered citations

Each agent reports its progress through LangGraph's stream writer, so the
frontend can show every step live.
"""
import operator
import re
from typing import Annotated, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import config
from .llm import chat, parse_json
from .search import web_search


# ---------- state ----------
class Source(TypedDict):
    title: str
    url: str


class Finding(TypedDict):
    index: int
    sub_question: str
    notes: str
    sources: list[Source]


class ResearchState(TypedDict, total=False):
    question: str
    sub_questions: list[str]
    findings: Annotated[list[Finding], operator.add]  # researchers append in parallel
    report: str
    sources: list[Source]


class ResearcherInput(TypedDict):
    question: str
    sub_question: str
    index: int


_CITE = re.compile(r"[\[【]\s*(\d+)\s*(?:†[^\]】]*)?[\]】]")


def clean_citations(text: str) -> str:
    """Turn model-specific styles like 【3†L1-L4】 or [3†source] into plain [3]."""
    return _CITE.sub(lambda m: f"[{m.group(1)}]", text)


def emit(agent: str, status: str, message: str, **data) -> None:
    """Send a live progress event to whoever is streaming the graph."""
    get_stream_writer()({"agent": agent, "status": status, "message": message, **data})


# ---------- prompts ----------
PLANNER_PROMPT = """You are the PLANNER agent in a research team.
Break the user's question into {n} or fewer focused sub-questions that can each be
answered with a web search. Together they must fully cover the question; avoid overlap.
Reply with JSON only: {{"sub_questions": ["...", "..."]}}"""

RESEARCHER_PROMPT = """You are a RESEARCHER agent. Using ONLY the numbered search results,
write 3-6 concise bullet points that answer the sub-question.
End every bullet with the result number(s) it came from, like [1] or [2][3].
Use plain square brackets only, like [1]. Never use 【】 brackets or line ranges.
If the results do not answer it, say so. Never invent facts or sources."""

WRITER_PROMPT = """You are the WRITER agent. Write a clear, well-structured research report
in Markdown that answers the main question using ONLY the research notes provided.
- Start with a title (#) and a 2-3 sentence "## Summary".
- Then use ## sections organised by theme (not by sub-question).
- Keep every citation number exactly as given, e.g. [3]. Do not add new numbers.
- Use plain square brackets only, like [3]. Never use 【】 brackets or line ranges.
- Do NOT write a sources/references list; it is added automatically.
- Note any gaps or conflicting information honestly."""


# ---------- agents ----------
async def planner(state: ResearchState) -> dict:
    emit("planner", "start", "Breaking the question into sub-questions…")
    reply = await chat(
        PLANNER_PROMPT.format(n=config.MAX_SUB_QUESTIONS), f"Question: {state['question']}"
    )
    try:
        subs = [s.strip() for s in parse_json(reply).get("sub_questions", []) if s.strip()]
    except ValueError:
        subs = []
    subs = subs[: config.MAX_SUB_QUESTIONS] or [state["question"]]
    emit("planner", "done", f"Planned {len(subs)} sub-questions", sub_questions=subs)
    return {"sub_questions": subs}


def fan_out(state: ResearchState) -> list[Send]:
    """One researcher per sub-question, all running at the same time."""
    return [
        Send("researcher", {"question": state["question"], "sub_question": sq, "index": i})
        for i, sq in enumerate(state["sub_questions"])
    ]


async def researcher(task: ResearcherInput) -> dict:
    name, sq = f"researcher-{task['index'] + 1}", task["sub_question"]
    emit(name, "start", f"Searching: {sq}", sub_question=sq)
    try:
        results = await web_search(sq)
    except Exception as exc:  # noqa: BLE001
        emit(name, "error", f"Search failed: {exc}")
        results = []

    if not results:
        notes = "- No search results were found for this sub-question."
        emit(name, "done", "No results found")
        return {"findings": [{"index": task["index"], "sub_question": sq, "notes": notes, "sources": []}]}

    emit(name, "progress", f"Found {len(results)} sources, reading…",
         sources=[{"title": r["title"], "url": r["url"]} for r in results])
    numbered = "\n\n".join(
        f"[{i}] {r['title']}\nURL: {r['url']}\n{r['snippet']}" for i, r in enumerate(results, 1)
    )
    notes = await chat(
        RESEARCHER_PROMPT,
        f"Main question: {task['question']}\nSub-question: {sq}\n\nSearch results:\n{numbered}",
    )
    notes = clean_citations(notes)
    emit(name, "done", "Notes written", notes=notes)
    return {
        "findings": [{
            "index": task["index"],
            "sub_question": sq,
            "notes": notes.strip(),
            "sources": [{"title": r["title"], "url": r["url"]} for r in results],
        }]
    }


def renumber(findings: list[Finding]) -> tuple[str, list[Source]]:
    """Give every unique URL one global number and rewrite each note's [n] to match."""
    sources: list[Source] = []
    by_url: dict[str, int] = {}
    blocks = []
    for f in sorted(findings, key=lambda f: f["index"]):
        local_to_global = {}
        for local, src in enumerate(f["sources"], 1):
            if src["url"] not in by_url:
                sources.append(src)
                by_url[src["url"]] = len(sources)
            local_to_global[local] = by_url[src["url"]]

        def swap(m: re.Match) -> str:
            n = int(m.group(1))
            return f"[{local_to_global[n]}]" if n in local_to_global else ""

        notes = re.sub(r"\[(\d+)\]", swap, f["notes"])
        blocks.append(f"### {f['sub_question']}\n{notes}")
    return "\n\n".join(blocks), sources


async def writer(state: ResearchState) -> dict:
    emit("writer", "start", f"Combining {len(state['findings'])} research notes into a report…")
    notes, sources = renumber(state["findings"])
    report = await chat(
        WRITER_PROMPT, f"Main question: {state['question']}\n\nResearch notes:\n{notes}", 0.3
    )
    report = clean_citations(report)
    # Drop any citation the model made up, then add the real sources list.
    report = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(sources) else "",
                    report).strip()
    if sources:
        report += "\n\n## Sources\n" + "\n".join(
            f"{i}. [{s['title'] or s['url']}]({s['url']})" for i, s in enumerate(sources, 1)
        )
    emit("writer", "done", "Report ready")
    return {"report": report, "sources": sources}


# ---------- graph ----------
def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("planner", planner)
    g.add_node("researcher", researcher)
    g.add_node("writer", writer)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fan_out, ["researcher"])
    g.add_edge("researcher", "writer")  # waits for every parallel researcher
    g.add_edge("writer", END)
    return g.compile()


graph = build_graph()
