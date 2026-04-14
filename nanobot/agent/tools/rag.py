"""RAG graph retrieval tool backed by RAGAnything/LightRAG."""

import asyncio
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from nanobot.agent.tools.base import Tool, tool_parameters


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    provider: str = "openai"
    model: str
    dim: int = Field(default=1536, gt=0)
    api_key: str | None = None
    api_base: str | None = None


class RAGAddonConfig(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    enable: bool = True
    storage_dir: str = "data/indexes"
    output_dir: str = "data/raw"
    embedding: EmbeddingConfig


def _load_addon_config() -> RAGAddonConfig | None:
    from nanobot.config.loader import get_config_path

    path = get_config_path().parent / "rag.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return RAGAddonConfig.model_validate(data)


def _resolve_workspace_data_path(raw_path: str, workspace: Path) -> Path:
    workspace_root = workspace.expanduser().resolve()
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (workspace_root / candidate).resolve()


async def warmup_rag_addon(loop: Any) -> None:
    """Pre-initialize the LightRAG instance in the background at loop startup."""
    tool = loop.tools.get("rag_query")
    if tool is None or not isinstance(tool, RAGQueryTool):
        return
    try:
        await tool._get_rag()
    except Exception as exc:
        from lightrag.utils import logger as rag_logger
        rag_logger.warning(f"RAG warmup failed: {exc}")


def _register_rag_addon(loop: Any) -> None:
    config = _load_addon_config()
    if config is None or not config.enable:
        return

    llm_func = _build_llm_func(loop)
    embedding_func = _build_embedding_func(config)
    workspace = Path(loop.workspace).expanduser().resolve()
    storage_dir = _resolve_workspace_data_path(config.storage_dir, workspace)

    loop.tools.register(
        RAGQueryTool(
            storage_dir=storage_dir,
            llm_model_func=llm_func,
            embedding_func=embedding_func,
        )
    )


def _build_llm_func(loop: Any) -> Any:
    from nanobot.providers.anthropic_provider import AnthropicProvider

    provider = loop.provider
    model = loop.model
    api_key = getattr(provider, "api_key", None) or ""
    # Prefer _effective_base (resolves spec.default_api_base) over the raw api_base attr.
    api_base = getattr(provider, "_effective_base", None) or getattr(provider, "api_base", None) or None

    if isinstance(provider, AnthropicProvider):
        from lightrag.llm.anthropic import anthropic_complete_if_cache

        async def anthropic_llm(prompt, system_prompt=None, history_messages=[], **kwargs):
            kwargs.pop("model", None)
            return await anthropic_complete_if_cache(
                model, prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                **kwargs,
            )

        return anthropic_llm

    from lightrag.llm.openai import openai_complete_if_cache

    extra = {"base_url": api_base} if api_base else {}

    async def openai_llm(prompt, system_prompt=None, history_messages=[], **kwargs):
        kwargs.pop("model", None)
        return await openai_complete_if_cache(
            model, prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            **extra,
            **kwargs,
        )

    return openai_llm


def _build_embedding_func(config: RAGAddonConfig) -> Any:
    import functools

    from lightrag.utils import EmbeddingFunc

    from nanobot.config.loader import load_config

    emb = config.embedding
    provider_name = emb.provider.lower()

    nanobot_cfg = load_config()
    p = getattr(nanobot_cfg.providers, provider_name.replace("-", "_"), None)
    api_key = emb.api_key or (p.api_key if p else None) or ""
    api_base = emb.api_base or (p.api_base if p else None) or None

    if not api_base:
        from nanobot.providers.registry import find_by_name

        spec = find_by_name(provider_name)
        if spec and getattr(spec, "default_api_base", None):
            api_base = spec.default_api_base

    if provider_name == "ollama":
        from lightrag.llm.ollama import ollama_embed

        raw_func = functools.partial(
            ollama_embed,
            model=emb.model,
            host=api_base or "http://localhost:11434",
        )
    else:
        from lightrag.llm.openai import openai_embed

        extra: dict[str, Any] = {}
        if api_base:
            extra["base_url"] = api_base
        # Use .func to bypass openai_embed's built-in EmbeddingFunc(1536) wrapper and let our EmbeddingFunc(emb.dim) handle validation.
        raw_func = functools.partial(
            openai_embed.func,
            model=emb.model,
            api_key=api_key,
            **extra,
        )

    return EmbeddingFunc(
        embedding_dim=emb.dim,
        max_token_size=8192,
        func=raw_func,
    )


@tool_parameters(
    {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language question to answer from the knowledge graph.",
                "minLength": 1,
            },
            "mode": {
                "type": "string",
                "enum": ["local", "global", "hybrid", "naive", "mix", "bypass"],
                "description": "Retrieval mode. mix (default): knowledge graph + vector. local: context-dependent. global: broad knowledge. naive: basic vector search.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top chunks/entities to retrieve (default 60).",
                "minimum": 1,
                "maximum": 200,
            },
        },
        "required": ["query"],
    }
)
class RAGQueryTool(Tool):
    """Query the RAGAnything knowledge graph built from ingested degree plan PDFs."""

    def __init__(
        self,
        storage_dir: str | Path,
        llm_model_func: Any,
        embedding_func: Any,
    ) -> None:
        self._storage_dir = str(Path(storage_dir).resolve())
        self._llm_model_func = llm_model_func
        self._embedding_func = embedding_func
        self._rag: Any = None
        self._lock = asyncio.Lock()

    async def _get_rag(self) -> Any:
        async with self._lock:
            if self._rag is None:
                from raganything import RAGAnything, RAGAnythingConfig

                rag = RAGAnything(
                    config=RAGAnythingConfig(working_dir=self._storage_dir),
                    llm_model_func=self._llm_model_func,
                    vision_model_func=None,
                    embedding_func=self._embedding_func,
                )
                init_result = await rag._ensure_lightrag_initialized()
                if not init_result.get("success"):
                    raise RuntimeError(
                        f"Failed to initialize LightRAG: {init_result.get('error')}"
                    )
                self._rag = rag
            return self._rag

    @property
    def name(self) -> str:
        return "rag_query"

    @property
    def description(self) -> str:
        return (
            "Query the local knowledge graph built from HKU degree programme documents. "
            "Use for questions about curriculum, course requirements, credit rules, prerequisites, "
            "programme regulations, graduate employment statistics, and salary data. "
            "Always try this before searching online. Default mode='mix'."
        )

    @property
    def read_only(self) -> bool:
        return True

    async def execute(
        self,
        query: str,
        mode: str = "mix",
        top_k: int = 60,
        **kwargs: Any,
    ) -> str:
        try:
            rag = await self._get_rag()
            result = await rag.aquery(
                query,
                mode=mode,
                top_k=top_k,
                only_need_context=True,
            )
            if result is None:
                return "No results found."
            return str(result)
        except Exception as exc:
            return f"Error querying RAG: {exc}"
