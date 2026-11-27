"""All settings come from environment variables (or a .env file)."""
import os

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# LLM: "groq" | "gemini" | "ollama" | "anthropic" | "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "")  # empty -> provider default
FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "gemini")  # "" to disable
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "")

# Web search: "duckduckgo" (free, no key) | "tavily" (free 1,000 credits/month)
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "duckduckgo")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
RESULTS_PER_QUERY = int(os.getenv("RESULTS_PER_QUERY", "5"))

# Planner limits
MAX_SUB_QUESTIONS = int(os.getenv("MAX_SUB_QUESTIONS", "4"))

# Mock mode: fake LLM + fake search. Lets you run the demo/tests with no keys.
MOCK_MODE = _bool("MOCK_MODE", False)
