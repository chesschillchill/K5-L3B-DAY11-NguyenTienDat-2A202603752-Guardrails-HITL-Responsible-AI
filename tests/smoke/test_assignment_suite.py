import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from openai import APIStatusError

from assignment import pipeline
from core.openai_runtime import OpenAIRunner


def test_assignment_suite_retries_current_query_after_gateway_timeout(monkeypatch, tmp_path):
    calls = []

    async def complete(self, agent, text, *, max_tokens):
        calls.append(text)
        if len(calls) == 2:
            response = httpx.Response(
                504, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
            )
            raise APIStatusError("Gateway timeout", response=response, body=None)
        return "Banking assistance."

    sleep = AsyncMock()
    monkeypatch.setattr(OpenAIRunner, "complete", complete)
    monkeypatch.setattr(pipeline, "asyncio", SimpleNamespace(sleep=sleep))
    monkeypatch.setattr(pipeline, "__file__", str(tmp_path / "src/assignment/pipeline.py"))
    audit, monitor = pipeline.build_observability()

    results = asyncio.run(pipeline.run_assignment_suite({
        "plugins": pipeline.build_production_plugins(),
        "audit": audit,
        "monitor": monitor,
    }))

    sleep.assert_awaited_once_with(5)
    assert calls[:3] == [
        results["safe_queries"][0]["input"],
        results["safe_queries"][1]["input"],
        results["safe_queries"][1]["input"],
    ]
    assert len(calls) == 17
    assert len(audit.logs) == monitor.total_requests == 30
    assert (tmp_path / "outputs/results.json").is_file()
