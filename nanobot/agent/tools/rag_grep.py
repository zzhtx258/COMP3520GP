"""Grep tool scoped to MinerU-extracted markdown files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nanobot.agent.tools.search import GrepTool


class RAGMarkdownGrepTool(GrepTool):
    """Search verbatim text in MinerU-parsed markdown files from ingested degree plan PDFs."""

    def __init__(self, output_dir: str | Path, workspace: Path | None = None) -> None:
        output_path = Path(output_dir).resolve()
        super().__init__(
            workspace=workspace,
            extra_allowed_dirs=[output_path],
        )
        self._output_dir = str(output_path)

    @property
    def name(self) -> str:
        return "rag_grep"

    @property
    def description(self) -> str:
        return (
            "Recursively search verbatim text across all MinerU-parsed markdown files "
            "under /data/raw/ (including all nested sub-directories). "
            "Use this when you need exact quoted source text, course codes such as COMP3516, "
            "credit counts, or prerequisite rules as literally written in the source documents. "
            "pattern supports Python regex. "
            "Set output_mode='content' to get matching lines with surrounding context."
        )

    @property
    def read_only(self) -> bool:
        return True

    async def execute(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        type: str | None = None,
        **kwargs: Any,
    ) -> str:
        return await super().execute(
            pattern=pattern,
            path=path or self._output_dir,
            glob=glob,
            type=type or "md",
            **kwargs,
        )
