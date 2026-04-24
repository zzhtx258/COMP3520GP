"""Manual bounded research loop for finding patterns and anomalies."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from loguru import logger

from nanobot.agent.runner import AgentRunResult, AgentRunSpec, AgentRunner
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import ResearchConfig
from nanobot.providers.base import LLMProvider
from nanobot.utils.helpers import ensure_dir

_FINDING_COMMENT_PREFIX = "research-finding:"
_FINDING_COMMENT_RE = re.compile(r"<!--\s*research-finding:\s*(\{.*?\})\s*-->")
_MAX_FINDING_FIELD_CHARS = 1200
_SCOPING_SYSTEM_PROMPT = """You are preparing a bounded research plan for a tool-using agent.

Turn a broad topic into:
- 3-6 concrete subquestions
- likely entities / programmes / years / actors
- likely files, fields, section titles, or table names to inspect
- a short note on what would count as an anomaly vs a stable pattern

Prefer corpus-facing retrieval phrases over general brainstorming. Keep it concise and operational."""
_ROUND_SYSTEM_PROMPT = """You are nanobot's research engine.

Your job is not to answer quickly. Your job is to discover potentially interesting findings.

Rules:
- Prefer local evidence first: rag_query, grep, glob, read_file, list_dir, exec.
- Only use web tools when they are available and only to validate a candidate finding that already has local support.
- Record every worthwhile discovery with record_finding.
- A finding must be either:
  - anomaly: surprising, counterintuitive, or unusual
  - pattern: stable recurring relationship worth noting
- Favor findings with specific evidence: file paths, years, programme names, employer names, table fields, or quoted snippets.
- Do not write files directly. The host system will persist findings for you.
- End each round by calling research_control with either continue or stop.

Quality bar:
- Interestingness: would a human researcher care?
- Novelty: is it non-obvious relative to the existing findings?
- Specificity: can someone verify it with the cited evidence?"""


def _now() -> datetime:
    return datetime.now()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "topic"


def _normalize_title(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _merge_unique(items: list[str], extra: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*items, *extra]:
        norm = item.strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        merged.append(norm)
    return merged


def _trim_text(value: str, limit: int = _MAX_FINDING_FIELD_CHARS) -> str:
    text = value.strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


@dataclass(slots=True)
class ResearchFinding:
    """Structured research discovery persisted across runs."""

    title: str
    kind: Literal["anomaly", "pattern"]
    claim: str
    why_interesting: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    next_checks: list[str] = field(default_factory=list)
    key: str | None = None

    def __post_init__(self) -> None:
        self.title = _trim_text(self.title, 200)
        self.claim = _trim_text(self.claim)
        self.why_interesting = _trim_text(self.why_interesting)
        self.evidence = _merge_unique([], [_trim_text(item, 300) for item in self.evidence])
        self.next_checks = _merge_unique([], [_trim_text(item, 300) for item in self.next_checks])
        self.confidence = max(0.0, min(float(self.confidence), 1.0))
        if not self.key:
            self.key = f"{self.kind}:{_slugify(_normalize_title(self.title))}"

    def to_json(self) -> dict[str, object]:
        return {
            "key": self.key,
            "title": self.title,
            "kind": self.kind,
            "claim": self.claim,
            "why_interesting": self.why_interesting,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "next_checks": self.next_checks,
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "ResearchFinding":
        return cls(
            key=str(data.get("key") or ""),
            title=str(data.get("title") or ""),
            kind=str(data.get("kind") or "pattern"),  # type: ignore[arg-type]
            claim=str(data.get("claim") or ""),
            why_interesting=str(data.get("why_interesting") or ""),
            confidence=float(data.get("confidence") or 0.0),
            evidence=[str(item) for item in data.get("evidence", []) or []],
            next_checks=[str(item) for item in data.get("next_checks", []) or []],
        )


@dataclass(slots=True)
class ResearchRoundResult:
    """Result of one research round."""

    findings: list[ResearchFinding]
    action: Literal["continue", "stop"]
    reason: str
    runner_result: AgentRunResult
    used_web_validation: bool = False


@dataclass(slots=True)
class ResearchRunResult:
    """Overall result returned to commands/UI."""

    topic: str
    topic_slug: str
    summary: str
    run_path: Path
    findings_path: Path
    findings: list[ResearchFinding]
    stop_reason: str
    rounds_run: int


class ResearchStore:
    """Persist research findings and run logs under workspace/research."""

    def __init__(self, workspace: Path):
        self.workspace = workspace

    def topic_slug(self, topic: str) -> str:
        return _slugify(topic)

    def research_dir(self) -> Path:
        return ensure_dir(self.workspace / "research")

    def topic_dir(self, topic: str) -> Path:
        return ensure_dir(self.research_dir() / self.topic_slug(topic))

    def findings_path(self, topic: str) -> Path:
        return self.topic_dir(topic) / "FINDINGS.md"

    def runs_dir(self, topic: str) -> Path:
        return ensure_dir(self.topic_dir(topic) / "runs")

    def latest_run_path(self, topic: str) -> Path | None:
        run_dir = self.topic_dir(topic) / "runs"
        if not run_dir.exists():
            return None
        runs = sorted(run_dir.glob("*.md"))
        return runs[-1] if runs else None

    def read_latest_run(self, topic: str) -> str | None:
        latest = self.latest_run_path(topic)
        if not latest or not latest.exists():
            return None
        return latest.read_text(encoding="utf-8")

    def load_findings(self, topic: str) -> list[ResearchFinding]:
        path = self.findings_path(topic)
        if not path.exists():
            return []
        content = path.read_text(encoding="utf-8")
        findings: list[ResearchFinding] = []
        for match in _FINDING_COMMENT_RE.finditer(content):
            try:
                findings.append(ResearchFinding.from_json(json.loads(match.group(1))))
            except Exception:
                logger.warning("Ignoring malformed research finding in {}", path)
        return findings

    def merge_findings(
        self,
        existing: list[ResearchFinding],
        new: list[ResearchFinding],
    ) -> list[ResearchFinding]:
        merged: dict[str, ResearchFinding] = {finding.key or "": finding for finding in existing}
        for finding in new:
            key = finding.key or ""
            if key and key in merged:
                old = merged[key]
                merged[key] = ResearchFinding(
                    key=key,
                    title=finding.title or old.title,
                    kind=finding.kind,
                    claim=finding.claim if len(finding.claim) >= len(old.claim) else old.claim,
                    why_interesting=(
                        finding.why_interesting
                        if len(finding.why_interesting) >= len(old.why_interesting)
                        else old.why_interesting
                    ),
                    confidence=max(old.confidence, finding.confidence),
                    evidence=_merge_unique(old.evidence, finding.evidence),
                    next_checks=_merge_unique(old.next_checks, finding.next_checks),
                )
            else:
                merged[key] = finding
        return sorted(merged.values(), key=lambda item: (item.kind, -item.confidence, item.title.lower()))

    def write_findings(self, topic: str, findings: list[ResearchFinding]) -> Path:
        path = self.findings_path(topic)
        topic_slug = self.topic_slug(topic)
        lines = [
            f"# Research Findings: {topic}",
            "",
            f"_Topic slug: `{topic_slug}`_",
            f"_Updated: {_now().strftime('%Y-%m-%d %H:%M:%S')}_",
            "",
        ]
        grouped = {
            "anomaly": [finding for finding in findings if finding.kind == "anomaly"],
            "pattern": [finding for finding in findings if finding.kind == "pattern"],
        }
        for kind in ("anomaly", "pattern"):
            lines.extend([f"## {kind.title()} Findings", ""])
            items = grouped[kind]
            if not items:
                lines.append("_None yet._")
                lines.append("")
                continue
            for finding in items:
                lines.extend([
                    f"### {finding.title}",
                    f"- Claim: {finding.claim}",
                    f"- Why it matters: {finding.why_interesting}",
                    f"- Confidence: {finding.confidence:.2f}",
                    "- Evidence:",
                ])
                if finding.evidence:
                    lines.extend([f"  - {item}" for item in finding.evidence])
                else:
                    lines.append("  - (none recorded)")
                lines.append("- Next checks:")
                if finding.next_checks:
                    lines.extend([f"  - {item}" for item in finding.next_checks])
                else:
                    lines.append("  - (none recorded)")
                lines.append(f"<!-- {_FINDING_COMMENT_PREFIX} {json.dumps(finding.to_json(), ensure_ascii=False)} -->")
                lines.append("")
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return path

    def write_run_log(
        self,
        topic: str,
        *,
        scoping_summary: str,
        findings: list[ResearchFinding],
        rounds: list[ResearchRoundResult],
        stop_reason: str,
    ) -> Path:
        run_dir = self.runs_dir(topic)
        timestamp = _now()
        run_path = run_dir / f"{timestamp.strftime('%Y%m%d-%H%M%S')}.md"
        lines = [
            f"# Research Run: {topic}",
            "",
            f"- Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- Stop reason: {stop_reason}",
            f"- Rounds: {len(rounds)}",
            "",
            "## Scope",
            "",
            scoping_summary.strip() or "_No scoping summary._",
            "",
            "## Findings From This Run",
            "",
        ]
        if findings:
            for finding in findings:
                lines.extend([
                    f"### [{finding.kind}] {finding.title}",
                    f"- Claim: {finding.claim}",
                    f"- Why it matters: {finding.why_interesting}",
                    f"- Confidence: {finding.confidence:.2f}",
                    "- Evidence:",
                ])
                if finding.evidence:
                    lines.extend([f"  - {item}" for item in finding.evidence])
                else:
                    lines.append("  - (none recorded)")
                lines.append("- Next checks:")
                if finding.next_checks:
                    lines.extend([f"  - {item}" for item in finding.next_checks])
                else:
                    lines.append("  - (none recorded)")
                lines.append("")
        else:
            lines.extend(["_No new findings recorded this run._", ""])

        lines.extend(["## Round Notes", ""])
        for idx, round_result in enumerate(rounds, start=1):
            lines.extend([
                f"### Round {idx}",
                f"- Decision: {round_result.action}",
                f"- Reason: {round_result.reason}",
                f"- Findings recorded: {len(round_result.findings)}",
                (
                    "- External validation: yes"
                    if round_result.used_web_validation
                    else "- External validation: no"
                ),
            ])
            if round_result.runner_result.tools_used:
                lines.append(
                    "- Tools used: "
                    + ", ".join(round_result.runner_result.tools_used)
                )
            if round_result.runner_result.final_content:
                lines.append(
                    "- Round summary: "
                    + _trim_text(round_result.runner_result.final_content, 400)
                )
            lines.append("")
        run_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return run_path


@dataclass(slots=True)
class _RoundState:
    findings: list[ResearchFinding] = field(default_factory=list)
    action: Literal["continue", "stop"] | None = None
    reason: str = ""


class _RecordFindingTool(Tool):
    """Virtual tool that stores a structured finding."""

    def __init__(self, state: _RoundState):
        self._state = state

    @property
    def name(self) -> str:
        return "record_finding"

    @property
    def description(self) -> str:
        return "Record one structured research finding worth preserving."

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "minLength": 3},
                "kind": {"type": "string", "enum": ["anomaly", "pattern"]},
                "claim": {"type": "string", "minLength": 10},
                "why_interesting": {"type": "string", "minLength": 10},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "next_checks": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "title",
                "kind",
                "claim",
                "why_interesting",
                "confidence",
                "evidence",
                "next_checks",
            ],
        }

    async def execute(
        self,
        *,
        title: str,
        kind: Literal["anomaly", "pattern"],
        claim: str,
        why_interesting: str,
        confidence: float,
        evidence: list[str],
        next_checks: list[str],
    ) -> str:
        finding = ResearchFinding(
            title=title,
            kind=kind,
            claim=claim,
            why_interesting=why_interesting,
            confidence=confidence,
            evidence=evidence,
            next_checks=next_checks,
        )
        self._state.findings.append(finding)
        return f"Recorded finding: {finding.kind} / {finding.title}"


class _ResearchControlTool(Tool):
    """Virtual tool that ends a round explicitly."""

    def __init__(self, state: _RoundState):
        self._state = state

    @property
    def name(self) -> str:
        return "research_control"

    @property
    def description(self) -> str:
        return "Decide whether the research loop should continue or stop after this round."

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["continue", "stop"]},
                "reason": {"type": "string", "minLength": 5},
            },
            "required": ["action", "reason"],
        }

    async def execute(
        self,
        *,
        action: Literal["continue", "stop"],
        reason: str,
    ) -> str:
        self._state.action = action
        self._state.reason = _trim_text(reason, 300)
        return f"Research action set to {action}"


class ResearchEngine:
    """Run bounded manual research loops and persist findings."""

    _LOCAL_TOOL_NAMES = ("rag_query", "grep", "glob", "read_file", "list_dir", "exec")
    _WEB_TOOL_NAMES = ("web_search", "web_fetch")

    def __init__(
        self,
        *,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        tool_source: ToolRegistry,
        config: ResearchConfig | None = None,
        provider_retry_mode: str = "standard",
    ) -> None:
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.tool_source = tool_source
        self.config = config or ResearchConfig()
        self.provider_retry_mode = provider_retry_mode
        self.store = ResearchStore(workspace)
        self._runner = AgentRunner(provider)

    @property
    def research_model(self) -> str:
        return self.config.model_override or self.model

    async def scope_topic(self, topic: str) -> str:
        """Create a concise scoping brief for the current research topic."""
        response = await self.provider.chat_with_retry(
            model=self.research_model,
            messages=[
                {"role": "system", "content": _SCOPING_SYSTEM_PROMPT},
                {"role": "user", "content": f"Research topic:\n{topic}"},
            ],
            tools=None,
            tool_choice=None,
            retry_mode=self.provider_retry_mode,
        )
        return (response.content or "").strip() or f"- Topic: {topic}\n- Start with local retrieval."

    def latest_run_log(self, topic: str) -> str | None:
        """Read the latest persisted research log for a topic."""
        return self.store.read_latest_run(topic)

    def _build_tool_registry(self, state: _RoundState, *, allow_web_validation: bool) -> ToolRegistry:
        tools = ToolRegistry()
        for name in self._LOCAL_TOOL_NAMES:
            if tool := self.tool_source.get(name):
                tools.register(tool)
        if allow_web_validation:
            for name in self._WEB_TOOL_NAMES:
                if tool := self.tool_source.get(name):
                    tools.register(tool)
        tools.register(_RecordFindingTool(state))
        tools.register(_ResearchControlTool(state))
        return tools

    def _format_existing_findings(self, findings: list[ResearchFinding]) -> str:
        if not findings:
            return "_No existing findings for this topic._"
        lines = []
        for finding in findings[:8]:
            lines.append(
                f"- [{finding.kind}] {finding.title}: {finding.claim} (confidence {finding.confidence:.2f})"
            )
        return "\n".join(lines)

    def _build_round_messages(
        self,
        *,
        topic: str,
        scoping_summary: str,
        existing_findings: list[ResearchFinding],
        round_index: int,
        findings_remaining: int,
        stale_rounds: int,
        allow_web_validation: bool,
    ) -> list[dict[str, str]]:
        existing_block = self._format_existing_findings(existing_findings)
        constraints = [
            f"- Round: {round_index} / {self.config.max_rounds}",
            f"- Findings remaining this run: {findings_remaining}",
            f"- Consecutive stale rounds so far: {stale_rounds}",
            "- Prefer query terms and file hints that match the corpus language.",
            "- Record only findings that have concrete evidence.",
            "- If you have enough strong findings or the search space is exhausted, stop.",
        ]
        if allow_web_validation:
            constraints.append("- Web validation is allowed this round, but only for locally-supported candidate findings.")
        else:
            constraints.append("- Web validation is not allowed this round. Stay local.")
        user_prompt = "\n".join([
            f"Research topic:\n{topic}",
            "",
            "Scoping summary:",
            scoping_summary.strip(),
            "",
            "Existing findings for this topic:",
            existing_block,
            "",
            "Round constraints:",
            *constraints,
            "",
            "Work the topic using tools. Call record_finding for each worthwhile finding, then call research_control to decide continue or stop.",
        ])
        return [
            {"role": "system", "content": _ROUND_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    async def _run_round(
        self,
        *,
        topic: str,
        scoping_summary: str,
        existing_findings: list[ResearchFinding],
        round_index: int,
        findings_remaining: int,
        stale_rounds: int,
        allow_web_validation: bool,
    ) -> ResearchRoundResult:
        state = _RoundState()
        result = await self._runner.run(
            AgentRunSpec(
                initial_messages=self._build_round_messages(
                    topic=topic,
                    scoping_summary=scoping_summary,
                    existing_findings=existing_findings,
                    round_index=round_index,
                    findings_remaining=findings_remaining,
                    stale_rounds=stale_rounds,
                    allow_web_validation=allow_web_validation,
                ),
                tools=self._build_tool_registry(state, allow_web_validation=allow_web_validation),
                model=self.research_model,
                max_iterations=max(6, self.config.max_rounds * 3),
                max_tool_result_chars=16_000,
                provider_retry_mode=self.provider_retry_mode,
                fail_on_tool_error=False,
                workspace=self.workspace,
            )
        )
        action = state.action or "stop"
        reason = state.reason or "Model ended the round without calling research_control."
        used_web_validation = any(
            tool_name in self._WEB_TOOL_NAMES for tool_name in (result.tools_used or [])
        )
        return ResearchRoundResult(
            findings=state.findings,
            action=action,
            reason=reason,
            runner_result=result,
            used_web_validation=used_web_validation,
        )

    def _format_run_summary(
        self,
        *,
        topic: str,
        findings: list[ResearchFinding],
        stop_reason: str,
        run_path: Path,
        findings_path: Path,
        rounds_run: int,
    ) -> str:
        lines = [
            f"Research completed for `{topic}`.",
            f"- Rounds: {rounds_run}",
            f"- Findings recorded this run: {len(findings)}",
            f"- Stop reason: {stop_reason}",
            f"- Findings file: `{findings_path}`",
            f"- Latest run log: `{run_path}`",
        ]
        if findings:
            lines.append("")
            lines.append("Top findings:")
            for finding in findings[:5]:
                lines.append(
                    f"- [{finding.kind}] {finding.title} ({finding.confidence:.2f}) — {finding.claim}"
                )
        return "\n".join(lines)

    async def run(self, topic: str) -> ResearchRunResult:
        """Run one bounded research session for the provided topic."""
        existing_findings = self.store.load_findings(topic)
        scoping_summary = await self.scope_topic(topic)
        run_findings: list[ResearchFinding] = []
        rounds: list[ResearchRoundResult] = []
        stale_rounds = 0
        stop_reason = "Reached max rounds."

        for round_index in range(1, self.config.max_rounds + 1):
            if len(run_findings) >= self.config.max_findings:
                stop_reason = f"Reached max findings ({self.config.max_findings})."
                break

            round_result = await self._run_round(
                topic=topic,
                scoping_summary=scoping_summary,
                existing_findings=self.store.merge_findings(existing_findings, run_findings),
                round_index=round_index,
                findings_remaining=max(self.config.max_findings - len(run_findings), 0),
                stale_rounds=stale_rounds,
                allow_web_validation=self.config.allow_web_validation and bool(run_findings),
            )
            rounds.append(round_result)
            run_findings = self.store.merge_findings(run_findings, round_result.findings)

            if round_result.findings:
                stale_rounds = 0
            else:
                stale_rounds += 1

            if len(run_findings) >= self.config.max_findings:
                stop_reason = f"Reached max findings ({self.config.max_findings})."
                break
            if round_result.action == "stop":
                stop_reason = round_result.reason
                break
            if stale_rounds >= self.config.max_stale_rounds:
                stop_reason = f"No new high-quality findings for {stale_rounds} round(s)."
                break

        merged_findings = self.store.merge_findings(existing_findings, run_findings)
        findings_path = self.store.write_findings(topic, merged_findings)
        run_path = self.store.write_run_log(
            topic,
            scoping_summary=scoping_summary,
            findings=run_findings,
            rounds=rounds,
            stop_reason=stop_reason,
        )
        summary = self._format_run_summary(
            topic=topic,
            findings=run_findings,
            stop_reason=stop_reason,
            run_path=run_path,
            findings_path=findings_path,
            rounds_run=len(rounds),
        )
        logger.info(
            "Research finished topic='{}' rounds={} findings={} stop_reason={}",
            topic,
            len(rounds),
            len(run_findings),
            stop_reason,
        )
        return ResearchRunResult(
            topic=topic,
            topic_slug=self.store.topic_slug(topic),
            summary=summary,
            run_path=run_path,
            findings_path=findings_path,
            findings=run_findings,
            stop_reason=stop_reason,
            rounds_run=len(rounds),
        )
