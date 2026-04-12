import json
import os
from pathlib import Path
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


async def test_rag_query_initializes_with_config_signature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Regression test for RAGAnything constructor compatibility.

    Ensures `RAGQueryTool._get_rag()` uses the official
    `RAGAnything(config=RAGAnythingConfig(working_dir=...))` pattern rather
    than passing `working_dir` directly.
    """
    from nanobot.agent.tools.rag import RAGQueryTool
    import raganything

    captured: dict[str, object] = {}

    class FakeConfig:
        def __init__(self, *, working_dir: str) -> None:
            self.working_dir = working_dir

    class FakeRAG:
        def __init__(
            self,
            *,
            config: FakeConfig,
            llm_model_func,
            vision_model_func,
            embedding_func,
        ) -> None:
            captured["config"] = config
            captured["llm_model_func"] = llm_model_func
            captured["vision_model_func"] = vision_model_func
            captured["embedding_func"] = embedding_func

        async def _ensure_lightrag_initialized(self) -> dict:
            return {"success": True}

    monkeypatch.setattr(raganything, "RAGAnything", FakeRAG)
    monkeypatch.setattr(raganything, "RAGAnythingConfig", FakeConfig)

    tool = RAGQueryTool(
        storage_dir=tmp_path,
        llm_model_func=lambda *_a, **_k: "ok",
        embedding_func=object(),
    )

    rag = await tool._get_rag()

    assert isinstance(rag, FakeRAG)
    assert isinstance(captured["config"], FakeConfig)
    assert captured["config"].working_dir == str(tmp_path.resolve())
    assert captured["vision_model_func"] is None


def test_rag_workspace_path_resolution_uses_workspace_relative_paths(tmp_path: Path) -> None:
    from nanobot.agent.tools.rag import _resolve_workspace_data_path

    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()

    assert _resolve_workspace_data_path("data/index", workspace) == (
        workspace / "data" / "index"
    ).resolve()


async def test_rag_grep_forces_output_root_and_md_only(tmp_path: Path) -> None:
    from nanobot.agent.tools.rag_grep import RAGMarkdownGrepTool

    output_dir = tmp_path / "raw"
    output_dir.mkdir()
    (output_dir / "inside.md").write_text("needle\n", encoding="utf-8")
    (output_dir / "inside.mdx").write_text("needle\n", encoding="utf-8")

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "outside.md").write_text("needle\n", encoding="utf-8")

    tool = RAGMarkdownGrepTool(output_dir=output_dir, workspace=tmp_path)
    result = await tool.execute(
        "needle",
        path=str(outside_dir),
        output_mode="files_with_matches",
    )

    assert "inside.md" in result
    assert "inside.mdx" not in result
    assert "outside.md" not in result


async def test_rag_grep_function_context_uses_markdown_headings(tmp_path: Path) -> None:
    from nanobot.agent.tools.rag_grep import RAGMarkdownGrepTool

    output_dir = tmp_path / "raw"
    target_dir = output_dir / "2017_BEng(CompSc)_6a13fc8d"
    target_dir.mkdir(parents=True)
    (target_dir / "plan.md").write_text(
        "# Programme\n## Core Courses\nCOMP3516 is required\n",
        encoding="utf-8",
    )

    tool = RAGMarkdownGrepTool(output_dir=output_dir, workspace=tmp_path)
    result = await tool.execute(
        "COMP3516",
        subpath="2017_BEng(CompSc)_6a13fc8d",
        function_context=True,
    )

    assert "plan.md:3" in result
    assert "section: Programme > Core Courses" in result


async def test_rag_grep_subpath_suggests_similar_directory(tmp_path: Path) -> None:
    from nanobot.agent.tools.rag_grep import RAGMarkdownGrepTool

    output_dir = tmp_path / "raw"
    (output_dir / "2017_BEng(CompSc)_6a13fc8d").mkdir(parents=True)
    (output_dir / "2018_BEng(CompSc)_f4648e1c").mkdir(parents=True)

    tool = RAGMarkdownGrepTool(output_dir=output_dir, workspace=tmp_path)
    result = await tool.execute(
        "COMP3516",
        subpath="CompSc",
    )

    assert result.startswith("Error: subpath not found: CompSc")
    assert "Did you mean:" in result
    assert "2017_BEng(CompSc)_6a13fc8d" in result


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
    md_files = [line for line in result.splitlines() if line.endswith(".md")]
    assert len(md_files) > 1, "Expected files from multiple nested directories"
