"""Grep tool scoped to MinerU-extracted markdown files."""

import re
from pathlib import Path
from typing import Any

from nanobot.agent.tools.search import GrepTool


class RAGMarkdownGrepTool(GrepTool):
    """Search verbatim text in MinerU-parsed markdown files from ingested degree plan PDFs."""

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    _BLOCK_HEADER_RE = re.compile(r"^(.+):(\d+)$")

    def __init__(self, output_dir: str | Path, workspace: Path | None = None) -> None:
        output_path = Path(output_dir).expanduser().resolve()
        super().__init__(
            workspace=workspace,
            allowed_dir=output_path,
        )
        self._output_dir = output_path

    @property
    def parameters(self) -> dict[str, Any]:
        base = super().parameters
        props = dict(base["properties"])
        props.pop("path", None)
        props.pop("glob", None)
        props.pop("type", None)
        props["subpath"] = {
            "type": "string",
            "description": (
                "Optional relative subpath under the configured output directory to narrow search scope. "
                "Absolute paths are not allowed."
            ),
        }
        props["function_context"] = {
            "type": "boolean",
            "description": (
                "If true, annotate content-mode matches with nearest markdown heading path "
                "(section context)."
            ),
        }
        return {
            "type": "object",
            "properties": props,
            "required": ["pattern"],
            "additionalProperties": False,
        }

    @property
    def name(self) -> str:
        return "rag_grep"

    @property
    def description(self) -> str:
        return (
            "Search ingested degree-plan markdown for exact quoted text, course codes such as "
            "COMP3516, credit counts, or prerequisite rules as written in source documents. "
            "pattern supports Python regex. "
            "Use subpath to narrow to a specific programme/year folder. "
            "Set output_mode='content' to get matching lines with surrounding context. "
            "Set function_context=true to include nearest markdown section headings for each match."
        )

    @property
    def read_only(self) -> bool:
        return True

    async def execute(
        self,
        pattern: str,
        subpath: str | None = None,
        function_context: bool = False,
        **kwargs: Any,
    ) -> str:
        kwargs.pop("path", None)
        kwargs.pop("glob", None)
        kwargs.pop("type", None)

        if function_context:
            kwargs["output_mode"] = "content"
            kwargs.setdefault("context_before", 1)
            kwargs.setdefault("context_after", 1)

        try:
            search_root = self._resolve_search_root(subpath)
        except PermissionError as e:
            return f"Error: {e}"
        except FileNotFoundError as e:
            return f"Error: {e}"

        result = await super().execute(
            pattern=pattern,
            path=str(search_root),
            glob="*.md",
            **kwargs,
        )

        if (
            function_context
            and kwargs.get("output_mode", "files_with_matches") == "content"
            and result
            and not result.startswith("No matches found")
            and not result.startswith("Error:")
        ):
            return self._inject_markdown_function_context(result, search_root)
        return result

    def _resolve_search_root(self, subpath: str | None) -> Path:
        if not subpath:
            return self._output_dir

        rel = Path(subpath.strip())
        if rel.is_absolute():
            raise PermissionError("subpath must be relative to the configured output directory")

        candidate = (self._output_dir / rel).resolve()
        try:
            candidate.relative_to(self._output_dir)
        except ValueError as exc:
            raise PermissionError("subpath escapes the configured output directory") from exc

        if candidate.exists():
            if candidate.is_file() and candidate.suffix.lower() != ".md":
                raise PermissionError("subpath file must be a .md file")
            return candidate

        suggestions = self._suggest_subpaths(rel.name)

        if suggestions:
            sug = ", ".join(suggestions)
            raise FileNotFoundError(f"subpath not found: {subpath}. Did you mean: {sug}?")
        raise FileNotFoundError(f"subpath not found: {subpath}")

    def _suggest_subpaths(self, query: str) -> list[str]:
        root = self._output_dir
        if not root.exists() or not root.is_dir():
            return []

        normalized = query.lower().strip()
        if not normalized:
            return []

        choices = sorted(p.name for p in root.iterdir() if p.is_dir())
        if not choices:
            return []

        return [name for name in choices if normalized in name.lower() or name.lower() in normalized][
            :3
        ]

    def _inject_markdown_function_context(self, result: str, search_root: Path) -> str:
        heading_cache: dict[Path, list[str]] = {}
        out: list[str] = []
        base_dir = search_root.parent if search_root.is_file() else search_root

        for line in result.splitlines():
            m = self._BLOCK_HEADER_RE.match(line)
            if not m:
                out.append(line)
                continue

            display_path, line_no_raw = m.group(1), m.group(2)
            line_no = int(line_no_raw)
            file_path = self._resolve_display_path(display_path, base_dir)

            out.append(line)
            if not file_path or not file_path.exists() or file_path.suffix.lower() != ".md":
                continue

            headings = heading_cache.get(file_path)
            if headings is None:
                headings = self._build_heading_lookup(file_path)
                heading_cache[file_path] = headings

            if 0 < line_no < len(headings) and headings[line_no]:
                out.append(f"section: {headings[line_no]}")

        return "\n".join(out)

    def _resolve_display_path(self, display_path: str, base_dir: Path) -> Path | None:
        candidates = [(base_dir / display_path).resolve()]
        if self._workspace:
            candidates.append((self._workspace / display_path).resolve())

        for candidate in candidates:
            try:
                candidate.relative_to(self._output_dir)
            except ValueError:
                continue
            if candidate.exists():
                return candidate
        return None

    def _build_heading_lookup(self, file_path: Path) -> list[str]:
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return [""]

        stack = [""] * 6
        contexts = [""] * (len(lines) + 1)
        current = ""
        for idx, line in enumerate(lines, start=1):
            m = self._HEADING_RE.match(line.strip())
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                stack[level - 1] = title
                for i in range(level, 6):
                    stack[i] = ""
                current = " > ".join(part for part in stack if part)
            contexts[idx] = current
        return contexts
