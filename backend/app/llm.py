from __future__ import annotations

import os
from typing import Any

from langchain_community.llms import FakeListLLM

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None  # type: ignore


DEFAULT_FAKE_RESPONSE = (
    "Root Cause: Payment service connection pool exhaustion after v3.2.1 deployment\n"
    "Confidence: 0.88\n"
    "Evidence: payment timeouts, deployment at 09:58, latency spike at 10:03"
)


def get_llm() -> Any:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and ChatOpenAI is not None:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    return FakeListLLM(responses=[DEFAULT_FAKE_RESPONSE])


def extract_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if content:
        return content
    return str(response)
