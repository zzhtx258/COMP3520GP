import json
import os
from unittest.mock import MagicMock

import pytest

RAG_STORAGE = os.environ.get("RAG_STORAGE_DIR", "")
OUTPUT_DIR = os.environ.get("RAG_OUTPUT_DIR", "")
RAG_CONFIG_PATH = os.environ.get("RAG_CONFIG_PATH", "")


def _make_mock_loop() -> MagicMock:
    from nanobot.providers.anthropic_provider import AnthropicProvider

    loop = MagicMock()
    loop.model = os.environ["RAG_LLM_MODEL"]
    loop.provider.api_key = os.environ.get("RAG_LLM_API_KEY", "")
    loop.provider.api_base = os.environ.get("RAG_LLM_API_BASE") or None
    provider_name = os.environ.get("RAG_LLM_PROVIDER", "openai").lower()
    loop.provider.__class__ = AnthropicProvider if provider_name == "anthropic" else object
    return loop


@pytest.mark.skipif(
    not (RAG_STORAGE and RAG_CONFIG_PATH and os.environ.get("RAG_LLM_MODEL")),
    reason="RAG env not set",
)
async def test_rag_query_returns_string():
    from nanobot.agent.tools.rag import (
        RAGAddonConfig,
        RAGQueryTool,
        _build_embedding_func,
        _build_llm_func,
    )

    with open(RAG_CONFIG_PATH, encoding="utf-8") as f:
        config = RAGAddonConfig.model_validate(json.load(f))

    loop = _make_mock_loop()
    tool = RAGQueryTool(
        storage_dir=RAG_STORAGE,
        llm_model_func=_build_llm_func(loop),
        embedding_func=_build_embedding_func(config),
    )
    result = await tool.execute("What are the core required courses?", mode="hybrid")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.skipif(not OUTPUT_DIR, reason="RAG_OUTPUT_DIR not set")
async def test_rag_grep_finds_markdown():
    from nanobot.agent.tools.rag_grep import RAGMarkdownGrepTool

    tool = RAGMarkdownGrepTool(output_dir=OUTPUT_DIR)
    result = await tool.execute("COMP", output_mode="files_with_matches")
    assert isinstance(result, str)
    assert ".md" in result


@pytest.mark.skipif(not OUTPUT_DIR, reason="RAG_OUTPUT_DIR not set")
async def test_rag_grep_is_recursive():
    from nanobot.agent.tools.rag_grep import RAGMarkdownGrepTool

    tool = RAGMarkdownGrepTool(output_dir=OUTPUT_DIR)
    result = await tool.execute(".", output_mode="files_with_matches")
    md_files = [line for line in result.splitlines() if line.endswith((".md", ".mdx"))]
    assert len(md_files) > 1, "Expected files from multiple nested directories"
