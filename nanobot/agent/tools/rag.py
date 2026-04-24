"""RAG graph retrieval tool backed by RAGAnything/LightRAG."""

import asyncio
import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.providers.base import LLMResponse
from nanobot.utils.helpers import truncate_text

_RAG_LOG_OUTPUT_MAX_CHARS = 2000
_RAG_RUNTIME_LOGS_SUPPRESSED: ContextVar[bool] = ContextVar(
    "rag_runtime_logs_suppressed",
    default=True,
)
_RAG_NOISY_LOGGERS = ("lightrag", "nano-vectordb")

_RAG_CONTEXT_ANALYST_SYSTEM_PROMPT = """You are preparing retrieved knowledge-graph context for another agent.

Transform raw RAG context into a compact, agent-friendly briefing that is easier to act on than the original dump.

Rules:
- Do not invent facts or answer beyond the supplied context.
- Preserve exact numbers, years, programme names, employer names, job titles, salary labels, and file/path hints when present.
- Remove obvious duplication, repetitive graph boilerplate, and low-signal metadata.
- Keep details that may matter for follow-up grep/code analysis, even if they seem only possibly relevant.
- Be explicit about uncertainty or missing data.
- Prefer concise bullets over prose.

Output format:
RAG Analyst Summary

Relevant Findings:
- ...

Structured Facts:
- ...

Source Hints:
- ...

Gaps / Follow-ups:
- ...
"""


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    provider: str = "dashscope"
    model: str
    dim: int = Field(default=1536, gt=0)
    api_key: str | None = None
    api_base: str | None = None


class LLMConfig(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    api_base: str | None = None


class RAGAddonConfig(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    enable: bool = True
    storage_dir: str = "data/indexes"
    output_dir: str = "data/raw"
    llm: LLMConfig = Field(default_factory=LLMConfig)
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


def _build_context_analysis_messages(query: str, raw_context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _RAG_CONTEXT_ANALYST_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"User query:\n{query}\n\n"
                "Raw RAG context:\n"
                f"{raw_context}\n\n"
                "Rewrite the raw RAG context into an agent-friendly briefing."
            ),
        },
    ]


def _summarize_rag_output(text: str, max_chars: int = _RAG_LOG_OUTPUT_MAX_CHARS) -> str:
    normalized = text.strip()
    if not normalized:
        return "(empty)"
    return truncate_text(normalized, max_chars) if len(normalized) > max_chars else normalized


@contextmanager
def suppress_rag_runtime_logs(enabled: bool = True):
    """Suppress noisy RAG runtime logs within the current async context."""
    token = _RAG_RUNTIME_LOGS_SUPPRESSED.set(enabled)
    try:
        yield
    finally:
        _RAG_RUNTIME_LOGS_SUPPRESSED.reset(token)


@contextmanager
def _silence_external_rag_loggers(enabled: bool):
    """Temporarily raise noisy third-party RAG loggers to ERROR."""
    if not enabled:
        yield
        return

    previous: dict[str, int] = {}
    try:
        for name in _RAG_NOISY_LOGGERS:
            logger_obj = logging.getLogger(name)
            previous[name] = logger_obj.level
            logger_obj.setLevel(logging.ERROR)
        yield
    finally:
        for name, level in previous.items():
            logging.getLogger(name).setLevel(level)


async def warmup_rag_addon(loop: Any) -> None:
    """Pre-initialize the LightRAG instance in the background at loop startup."""
    tool = loop.tools.get("rag_query")
    if tool is None or not isinstance(tool, RAGQueryTool):
        return
    try:
        with suppress_rag_runtime_logs(True):
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
            provider=loop.provider,
            model=loop.model,
            provider_retry_mode=getattr(loop, "provider_retry_mode", "standard"),
        )
    )


def _build_llm_func(loop: Any) -> Any:
    from nanobot.config.loader import load_config
    from nanobot.providers.anthropic_provider import AnthropicProvider
    from nanobot.providers.registry import find_by_name

    provider = loop.provider
    model = loop.model
    api_key = getattr(provider, "api_key", None) or ""
    # Prefer _effective_base (resolves spec.default_api_base) over the raw api_base attr.
    api_base = getattr(provider, "_effective_base", None) or getattr(provider, "api_base", None) or None

    rag = _load_addon_config()
    llm = rag.llm if rag else None
    rag_provider = (llm.provider if llm else "") or ""
    rag_provider = rag_provider.strip().lower()
    if rag_provider:
        provider_name = rag_provider.replace("-", "_")
        model = (llm.model if llm else None) or model
        nanobot_cfg = load_config()
        p = getattr(nanobot_cfg.providers, provider_name, None)
        api_key = (llm.api_key if llm else None) or (p.api_key if p else None) or ""
        api_base = (llm.api_base if llm else None) or (p.api_base if p else None) or None
    use_anthropic = rag_provider.replace("-", "_") == "anthropic" if rag_provider else isinstance(provider, AnthropicProvider)

    if use_anthropic:
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

    active_provider_name = rag_provider.replace("-", "_") if rag_provider else ""
    if not active_provider_name:
        nanobot_cfg = load_config()
        active_provider_name = (nanobot_cfg.get_provider_name(model) or "").replace("-", "_")
    active_spec = find_by_name(active_provider_name) if active_provider_name else None
    if active_spec and active_spec.is_oauth:

        async def oauth_llm(prompt, system_prompt=None, history_messages=[], **kwargs):
            kwargs.pop("model", None)
            response = await provider.chat_with_retry(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                tool_choice=None,
            )
            return response.content or ""

        return oauth_llm

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
    from nanobot.providers.registry import find_by_name

    emb = config.embedding
    provider_name = emb.provider.lower()
    normalized_provider_name = provider_name.replace("-", "_")

    nanobot_cfg = load_config()
    p = getattr(nanobot_cfg.providers, normalized_provider_name, None)
    spec = find_by_name(normalized_provider_name)
    api_key = emb.api_key or (p.api_key if p else None) or ""
    api_base = (
        emb.api_base
        or (p.api_base if p else None)
        or (spec.default_api_base if spec else None)
        or None
    )

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
                "description": (
                    "Retrieval query for the knowledge graph. Write this as short corpus-facing "
                    "terms, not a full task instruction: prefer exact field labels, headings, "
                    "entity aliases, programme names, years, and English source vocabulary; avoid "
                    "workflow phrases like 'first', 'then', 'summarize', and avoid asking for "
                    "ranking or final calculations in the query string."
                ),
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
        provider: Any | None = None,
        model: str | None = None,
        provider_retry_mode: str = "standard",
        analysis_threshold_chars: int = 2500,
    ) -> None:
        self._storage_dir = str(Path(storage_dir).resolve())
        self._llm_model_func = llm_model_func
        self._embedding_func = embedding_func
        self._provider = provider
        self._model = model
        self._provider_retry_mode = provider_retry_mode
        self._analysis_threshold_chars = max(1, analysis_threshold_chars)
        self._rag: Any = None
        self._lock = asyncio.Lock()

    async def _get_rag(self) -> Any:
        async with self._lock:
            if self._rag is None:
                from raganything import RAGAnything, RAGAnythingConfig
                quiet_logs = _RAG_RUNTIME_LOGS_SUPPRESSED.get()
                with _silence_external_rag_loggers(quiet_logs):
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
            "Always try this before searching online. Default mode='mix'. "
            "When writing the query, use short retrieval terms drawn from the corpus "
            "(field labels, headings, entity aliases, programme names, years, English table/header text), "
            "not a long natural-language workflow or ranking instruction."
        )

    @property
    def read_only(self) -> bool:
        return True

    async def _analyze_context(self, query: str, raw_context: str) -> str:
        if not raw_context or len(raw_context) < self._analysis_threshold_chars:
            return raw_context
        if self._provider is None or not self._model:
            return raw_context

        response: LLMResponse = await self._provider.chat_with_retry(
            messages=_build_context_analysis_messages(query, raw_context),
            tools=None,
            model=self._model,
            max_tokens=2200,
            temperature=0,
            reasoning_effort="minimal",
            retry_mode=self._provider_retry_mode,
        )
        analyzed = (response.content or "").strip()
        if response.finish_reason == "error" or not analyzed:
            return raw_context
        if len(analyzed) >= len(raw_context):
            return raw_context
        return analyzed

    async def execute(
        self,
        query: str,
        mode: str = "mix",
        top_k: int = 60,
        **kwargs: Any,
    ) -> str:
        quiet_logs = _RAG_RUNTIME_LOGS_SUPPRESSED.get()
        if not quiet_logs:
            logger.info("rag_query: query=%r mode=%s top_k=%d", query, mode, top_k)
        try:
            with _silence_external_rag_loggers(quiet_logs):
                rag = await self._get_rag()
        except Exception as exc:
            return (
                f"RAG initialisation failed — the index may not be built yet or the config is invalid: {exc}. "
                "Do not retry this tool. Fall back to grep on data/content for exact keyword matches."
            )
        try:
            with _silence_external_rag_loggers(quiet_logs):
                result = await rag.aquery(
                    query,
                    mode=mode,
                    top_k=top_k,
                    only_need_context=True,
                )
        except Exception as exc:
            return (
                f"RAG query failed (transient error — you may retry with a rephrased query): {exc}. "
                "Alternatively, fall back to grep on data/content for exact keyword matches."
            )
        if result is None or (isinstance(result, str) and not result.strip()):
            return (
                "No results found in the knowledge graph for this query. "
                "Suggested next steps: (1) try grep on data/content for exact keywords or course codes, "
                "(2) rephrase the query using different terminology, "
                "(3) break the question into smaller sub-queries."
            )
        result_str = str(result)
        if not quiet_logs:
            logger.debug("rag_query: returned %d chars", len(result_str))
        analyzed = await self._analyze_context(query, result_str)
        if not quiet_logs:
            logger.info("rag_query output -> {}", _summarize_rag_output(analyzed))
        return analyzed
