"""RAG graph retrieval tool backed by RAGAnything/LightRAG."""

from __future__ import annotations

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
    storage_dir: str = "/data/index"
    output_dir: str = "/data/raw"
    embedding: EmbeddingConfig


def _load_addon_config() -> RAGAddonConfig | None:
    from nanobot.config.loader import get_config_path

    path = get_config_path().parent / "rag.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return RAGAddonConfig.model_validate(data)


def _register_rag_addon(loop: Any) -> None:
    from nanobot.agent.tools.rag_grep import RAGMarkdownGrepTool

    config = _load_addon_config()
    if config is None or not config.enable:
        return

    llm_func = _build_llm_func(loop)
    embedding_func = _build_embedding_func(config)

    loop.tools.register(
        RAGQueryTool(
            storage_dir=config.storage_dir,
            llm_model_func=llm_func,
            embedding_func=embedding_func,
        )
    )
    loop.tools.register(
        RAGMarkdownGrepTool(
            output_dir=config.output_dir,
            workspace=loop.workspace,
        )
    )


def _build_llm_func(loop: Any) -> Any:
    import functools

    from nanobot.providers.anthropic_provider import AnthropicProvider

    provider = loop.provider
    model = loop.model
    api_key = getattr(provider, "api_key", None) or ""
    api_base = getattr(provider, "api_base", None) or None

    if isinstance(provider, AnthropicProvider):
        from lightrag.llm.anthropic import anthropic_complete_if_cache

        return functools.partial(
            anthropic_complete_if_cache,
            model=model,
            api_key=api_key,
        )

    from lightrag.llm.openai import openai_complete_if_cache

    return functools.partial(
        openai_complete_if_cache,
        model=model,
        api_key=api_key,
        **({"base_url": api_base} if api_base else {}),
    )


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

    if provider_name == "ollama":
        from lightrag.llm.ollama import ollama_embedding

        raw_func = functools.partial(
            ollama_embedding,
            model=emb.model,
            host=api_base or "http://localhost:11434",
        )
    else:
        from lightrag.llm.openai import openai_embedding

        extra: dict[str, Any] = {}
        if api_base:
            extra["base_url"] = api_base
        extra["dimensions"] = emb.dim
        raw_func = functools.partial(
            openai_embedding,
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
                "description": (
                    "Retrieval mode. "
                    "local: focuses on context-dependent information. "
                    "global: utilizes global knowledge. "
                    "hybrid: combines local and global retrieval methods. "
                    "naive: performs a basic search without advanced techniques. "
                    "mix: integrates knowledge graph and vector retrieval (default). "
                    "bypass: skips retrieval and queries the LLM directly."
                ),
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
                from raganything import RAGAnything

                self._rag = RAGAnything(
                    working_dir=self._storage_dir,
                    llm_model_func=self._llm_model_func,
                    vision_model_func=None,
                    embedding_func=self._embedding_func,
                )
            return self._rag

    @property
    def name(self) -> str:
        return "rag_query"

    @property
    def description(self) -> str:
        return (
            "Query the knowledge graph built from ingested degree plan PDFs. "
            "Use this for questions about curriculum structure, course requirements, "
            "credit rules, prerequisite chains, and programme regulations across all "
            "available degree programmes. "
            "Prefer mode='mix' unless you have a specific reason to use another mode."
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
