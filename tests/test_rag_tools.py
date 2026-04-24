import json
import os
from pathlib import Path
from types import SimpleNamespace
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


def test_build_embedding_func_uses_provider_default_api_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.agent.tools.rag import EmbeddingConfig, RAGAddonConfig, _build_embedding_func
    from nanobot.config.schema import Config, ProviderConfig, ProvidersConfig

    def _fake_openai_embed_func(*_args, **_kwargs):
        return []

    monkeypatch.setattr(
        "lightrag.llm.openai.openai_embed",
        SimpleNamespace(func=_fake_openai_embed_func),
        raising=False,
    )
    monkeypatch.setattr(
        "nanobot.config.loader.load_config",
        lambda: Config(
            providers=ProvidersConfig(
                dashscope=ProviderConfig(api_key="sk-test", api_base=None)
            )
        ),
    )

    embedding_func = _build_embedding_func(
        RAGAddonConfig(
            embedding=EmbeddingConfig(
                provider="dashscope",
                model="text-embedding-v3",
                dim=1024,
            )
        )
    )

    assert embedding_func.func.keywords.get("api_key") == "sk-test"
    assert (
        embedding_func.func.keywords.get("base_url")
        == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def test_build_embedding_func_prefers_explicit_rag_api_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.agent.tools.rag import EmbeddingConfig, RAGAddonConfig, _build_embedding_func
    from nanobot.config.schema import Config

    def _fake_openai_embed_func(*_args, **_kwargs):
        return []

    monkeypatch.setattr(
        "lightrag.llm.openai.openai_embed",
        SimpleNamespace(func=_fake_openai_embed_func),
        raising=False,
    )
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: Config())

    embedding_func = _build_embedding_func(
        RAGAddonConfig(
            embedding=EmbeddingConfig(
                provider="dashscope",
                model="text-embedding-v3",
                dim=1024,
                api_base="https://custom-embedding-endpoint.example/v1",
            )
        )
    )

    assert (
        embedding_func.func.keywords.get("base_url")
        == "https://custom-embedding-endpoint.example/v1"
    )


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
    raganything = pytest.importorskip("raganything")

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


def test_register_rag_addon_only_registers_rag_query(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from nanobot.agent.tools.rag import (
        EmbeddingConfig,
        RAGAddonConfig,
        RAGQueryTool,
        _register_rag_addon,
    )

    registered: list[object] = []

    class _DummyRegistry:
        def register(self, tool: object) -> None:
            registered.append(tool)

    loop = MagicMock()
    loop.workspace = tmp_path / "workspace"
    loop.tools = _DummyRegistry()

    monkeypatch.setattr(
        "nanobot.agent.tools.rag._load_addon_config",
        lambda: RAGAddonConfig(
            enable=True,
            storage_dir="data/indexes",
            output_dir="data/raw",
            embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-large", dim=1024),
        ),
    )
    monkeypatch.setattr("nanobot.agent.tools.rag._build_llm_func", lambda _loop: object())
    monkeypatch.setattr("nanobot.agent.tools.rag._build_embedding_func", lambda _config: object())

    _register_rag_addon(loop)

    assert len(registered) == 1
    assert isinstance(registered[0], RAGQueryTool)
    assert registered[0].name == "rag_query"


def test_prompt_templates_prefer_grep_in_data_content() -> None:
    identity = Path("nanobot/templates/agent/identity.md").read_text(encoding="utf-8")
    tools = Path("nanobot/templates/TOOLS.md").read_text(encoding="utf-8")
    skill = Path("nanobot/skills/rag/SKILL.md").read_text(encoding="utf-8")

    assert "data/content" in identity
    assert "prefer `rag_query`" in identity
    assert "corpus-facing retrieval terms" in identity
    assert "start discovery from `data/content` itself" in identity
    assert "writing and running code" in identity
    assert "consider using a subagent" in identity
    assert 'path="data/content"' in tools
    assert "relevant directories under `data/content` first" in tools
    assert "rag_grep" not in skill
    assert "data/content" in skill
    assert "identify the relevant programme/year directories first" in skill
    assert "literal text search" in skill
    assert "semantic + knowledge-graph retrieval" in skill
    assert "How to write `rag_query` queries" in skill
    assert "Treat the `rag_query` string as a retrieval query" in skill
    assert "Bad:" in skill
    assert "consider writing and running code" in skill
    assert "consider using a subagent" in skill


def test_rag_query_tool_description_teaches_query_rewriting() -> None:
    from nanobot.agent.tools.rag import RAGQueryTool

    tool = RAGQueryTool(
        storage_dir=".",
        llm_model_func=lambda *_a, **_k: "ok",
        embedding_func=object(),
    )

    query_schema = tool.parameters["properties"]["query"]
    assert "short corpus-facing terms" in query_schema["description"]
    assert "avoid workflow phrases" in query_schema["description"]
    assert "short retrieval terms" in tool.description


def test_summarize_rag_output_truncates_long_logs() -> None:
    from nanobot.agent.tools.rag import _summarize_rag_output

    text = "x" * 3000
    summarized = _summarize_rag_output(text, max_chars=50)

    assert summarized.startswith("x" * 50)
    assert summarized.endswith("... (truncated)")


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


async def test_rag_query_analyzes_large_context_for_agent() -> None:
    from nanobot.agent.tools.rag import RAGQueryTool
    from nanobot.providers.base import LLMResponse

    large_context = "entity\n" * 2000

    class FakeRAG:
        async def aquery(self, *_args, **_kwargs):
            return large_context

    class FakeProvider:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def chat_with_retry(self, **kwargs):
            self.calls.append(kwargs)
            return LLMResponse(
                content="RAG Analyst Summary\n\nRelevant Findings:\n- salary field appears in 2022 table",
                finish_reason="stop",
            )

    provider = FakeProvider()
    tool = RAGQueryTool(
        storage_dir=".",
        llm_model_func=lambda *_a, **_k: "ok",
        embedding_func=object(),
        provider=provider,
        model="test-model",
    )
    tool._rag = FakeRAG()

    result = await tool.execute("Which programme has the highest average salary?", mode="mix")

    assert result.startswith("RAG Analyst Summary")
    assert len(provider.calls) == 1
    assert provider.calls[0]["model"] == "test-model"
    assert provider.calls[0]["reasoning_effort"] == "minimal"
    assert "Which programme has the highest average salary?" in provider.calls[0]["messages"][1]["content"]


async def test_rag_query_falls_back_to_raw_context_when_analysis_fails() -> None:
    from nanobot.agent.tools.rag import RAGQueryTool
    from nanobot.providers.base import LLMResponse

    large_context = "relation\n" * 2000

    class FakeRAG:
        async def aquery(self, *_args, **_kwargs):
            return large_context

    class FakeProvider:
        async def chat_with_retry(self, **_kwargs):
            return LLMResponse(content="Error calling LLM", finish_reason="error")

    tool = RAGQueryTool(
        storage_dir=".",
        llm_model_func=lambda *_a, **_k: "ok",
        embedding_func=object(),
        provider=FakeProvider(),
        model="test-model",
    )
    tool._rag = FakeRAG()

    result = await tool.execute("What matters?", mode="mix")

    assert result == large_context


async def test_rag_query_skips_analysis_for_short_context() -> None:
    from nanobot.agent.tools.rag import RAGQueryTool

    short_context = "short context"

    class FakeRAG:
        async def aquery(self, *_args, **_kwargs):
            return short_context

    class FakeProvider:
        def __init__(self) -> None:
            self.called = False

        async def chat_with_retry(self, **_kwargs):
            self.called = True
            raise AssertionError("should not be called")

    provider = FakeProvider()
    tool = RAGQueryTool(
        storage_dir=".",
        llm_model_func=lambda *_a, **_k: "ok",
        embedding_func=object(),
        provider=provider,
        model="test-model",
    )
    tool._rag = FakeRAG()

    result = await tool.execute("What matters?", mode="mix")

    assert result == short_context
    assert provider.called is False


async def test_rag_query_suppresses_its_own_logs_in_quiet_context(monkeypatch) -> None:
    from nanobot.agent.tools import rag as rag_module
    from nanobot.agent.tools.rag import RAGQueryTool, suppress_rag_runtime_logs

    class FakeRAG:
        async def aquery(self, *_args, **_kwargs):
            return "quiet context"

    tool = RAGQueryTool(
        storage_dir=".",
        llm_model_func=lambda *_a, **_k: "ok",
        embedding_func=object(),
    )
    tool._rag = FakeRAG()

    recorded: list[tuple[str, tuple, dict]] = []
    monkeypatch.setattr(
        rag_module.logger,
        "info",
        lambda *args, **kwargs: recorded.append(("info", args, kwargs)),
    )
    monkeypatch.setattr(
        rag_module.logger,
        "debug",
        lambda *args, **kwargs: recorded.append(("debug", args, kwargs)),
    )

    with suppress_rag_runtime_logs(True):
        result = await tool.execute("quiet query", mode="mix")

    assert result == "quiet context"
    assert recorded == []


async def test_warmup_rag_addon_uses_quiet_context(monkeypatch) -> None:
    from nanobot.agent.tools import rag as rag_module

    class FakeTool:
        def __init__(self) -> None:
            self.quiet_flags: list[bool] = []

        async def _get_rag(self):
            self.quiet_flags.append(rag_module._RAG_RUNTIME_LOGS_SUPPRESSED.get())
            return object()

    tool = FakeTool()
    loop = SimpleNamespace(tools=SimpleNamespace(get=lambda name: tool if name == "rag_query" else None))
    monkeypatch.setattr(rag_module, "RAGQueryTool", FakeTool)

    await rag_module.warmup_rag_addon(loop)

    assert tool.quiet_flags == [True]


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
