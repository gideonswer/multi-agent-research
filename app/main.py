"""FastAPI server: streams every agent step to the browser with Server-Sent Events."""
import json
import logging
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from . import config
from .graph import graph

logging.basicConfig(level=logging.INFO)
STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Multi-Agent Research Assistant", version="1.0.0")


_runs: dict[str, deque] = defaultdict(deque)


def allowed(ip: str) -> bool:
    """Simple in-memory limit so a public demo can't burn the free API quotas."""
    if config.RATE_LIMIT_PER_HOUR <= 0:
        return True
    now, window = time.time(), _runs[ip]
    while window and now - window[0] > 3600:
        window.popleft()
    if len(window) >= config.RATE_LIMIT_PER_HOUR:
        return False
    window.append(now)
    return True


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm": config.LLM_PROVIDER,
        "fallback": config.FALLBACK_PROVIDER or None,
        "search": config.SEARCH_PROVIDER,
        "mock_mode": config.MOCK_MODE,
    }


@app.get("/api/research")
async def research(
    request: Request, q: str = Query(..., min_length=3, max_length=500)
) -> StreamingResponse:
    ok = allowed(client_ip(request))

    async def stream():
        if not ok:
            yield sse({"type": "error", "message":
                       f"Demo limit reached ({config.RATE_LIMIT_PER_HOUR} questions per hour). "
                       "Please try again later, or run it locally from GitHub."})
            yield sse({"type": "end"})
            return
        try:
            final = {}
            async for mode, chunk in graph.astream(
                {"question": q, "findings": []}, stream_mode=["custom", "values"]
            ):
                if mode == "custom":
                    yield sse({"type": "step", **chunk})
                else:
                    final = chunk
            yield sse({"type": "report", "report": final.get("report", ""),
                       "sources": final.get("sources", [])})
        except Exception as exc:  # noqa: BLE001
            logging.exception("Research run failed")
            yield sse({"type": "error", "message": str(exc)})
        yield sse({"type": "end"})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
