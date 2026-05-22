import os
from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class _UsageTracker(BaseCallbackHandler):
    """Accumulates token usage across all LLM calls in the pipeline."""

    def __init__(self):
        super().__init__()
        self.records: list[dict] = []

    def on_llm_end(self, response: LLMResult, **kwargs):
        try:
            tu = (response.llm_output or {}).get("token_usage", {})
            if tu:
                self.records.append({
                    "input":  tu.get("prompt_tokens", 0),
                    "output": tu.get("completion_tokens", 0),
                    "total":  tu.get("total_tokens", 0),
                })
        except Exception:
            pass

    def reset(self):
        self.records.clear()


_tracker = _UsageTracker()


def get_token_usage() -> dict:
    """Return cumulative token usage across all LLM calls since process start."""
    total_in  = sum(r["input"]  for r in _tracker.records)
    total_out = sum(r["output"] for r in _tracker.records)
    return {
        "calls":         len(_tracker.records),
        "input_tokens":  total_in,
        "output_tokens": total_out,
        "total_tokens":  total_in + total_out,
        "per_call":      list(_tracker.records),
    }


def get_llm(agent_cfg: dict) -> BaseChatModel:
    provider    = agent_cfg["provider"].lower()
    model       = agent_cfg["model"]
    temperature = agent_cfg.get("temperature", 0.7)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=os.environ["ANTHROPIC_API_KEY"],
            callbacks=[_tracker],
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=os.environ["GOOGLE_API_KEY"],
            callbacks=[_tracker],
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key=os.environ["GROQ_API_KEY"],
            callbacks=[_tracker],
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
