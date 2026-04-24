from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.research import ResearchEngine, ResearchFinding, ResearchRoundResult
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import ResearchConfig


def _runner_result(*, tools_used: list[str] | None = None, final_content: str | None = None) -> AgentRunResult:
    return AgentRunResult(
        final_content=final_content,
        messages=[],
        tools_used=tools_used or [],
        usage={},
        stop_reason="completed",
        tool_events=[],
    )


@pytest.mark.asyncio
async def test_research_run_persists_findings_and_run_log(tmp_path) -> None:
    engine = ResearchEngine(
        workspace=tmp_path,
        provider=MagicMock(),
        model="test-model",
        tool_source=ToolRegistry(),
        config=ResearchConfig(max_rounds=3, max_findings=4, max_stale_rounds=2),
    )
    engine.scope_topic = AsyncMock(return_value="- Search salary tables\n- Compare programmes")
    engine._run_round = AsyncMock(
        side_effect=[
            ResearchRoundResult(
                findings=[
                    ResearchFinding(
                        title="Finance-heavy placement cluster",
                        kind="pattern",
                        claim="Several programmes repeatedly surface finance employers.",
                        why_interesting="It suggests a stable finance-oriented employment pattern.",
                        confidence=0.82,
                        evidence=["data/content/2023/foo.md"],
                        next_checks=["Verify 2022 employer tables"],
                    ),
                    ResearchFinding(
                        title="Unexpected compensation spike",
                        kind="anomaly",
                        claim="One programme-year appears to have a much higher mean salary field.",
                        why_interesting="It may indicate an outlier cohort or parsing issue.",
                        confidence=0.73,
                        evidence=["data/content/2021/bar.md"],
                        next_checks=["Cross-check source table formatting"],
                    ),
                ],
                action="stop",
                reason="Enough high-value findings for this pass.",
                runner_result=_runner_result(tools_used=["grep", "rag_query"], final_content="Recorded two findings."),
            )
        ]
    )

    result = await engine.run("salary anomalies")

    assert result.rounds_run == 1
    assert len(result.findings) == 2
    assert result.findings_path.exists()
    assert result.run_path.exists()
    findings_text = result.findings_path.read_text(encoding="utf-8")
    run_text = result.run_path.read_text(encoding="utf-8")
    assert "## Anomaly Findings" in findings_text
    assert "## Pattern Findings" in findings_text
    assert "Unexpected compensation spike" in findings_text
    assert "Finance-heavy placement cluster" in findings_text
    assert "Stop reason: Enough high-value findings for this pass." in run_text
    assert "Research completed for `salary anomalies`." in result.summary


@pytest.mark.asyncio
async def test_research_run_stops_after_stale_rounds(tmp_path) -> None:
    engine = ResearchEngine(
        workspace=tmp_path,
        provider=MagicMock(),
        model="test-model",
        tool_source=ToolRegistry(),
        config=ResearchConfig(max_rounds=5, max_findings=4, max_stale_rounds=2),
    )
    engine.scope_topic = AsyncMock(return_value="- Search employer clusters")
    engine._run_round = AsyncMock(
        side_effect=[
            ResearchRoundResult(
                findings=[],
                action="continue",
                reason="Need another pass.",
                runner_result=_runner_result(final_content="No strong findings yet."),
            ),
            ResearchRoundResult(
                findings=[],
                action="continue",
                reason="Still scanning.",
                runner_result=_runner_result(final_content="Still no strong findings."),
            ),
        ]
    )

    result = await engine.run("broad employer trends")

    assert result.rounds_run == 2
    assert result.stop_reason == "No new high-quality findings for 2 round(s)."
    assert "No new high-quality findings" in result.summary


@pytest.mark.asyncio
async def test_research_findings_are_merged_across_runs(tmp_path) -> None:
    engine = ResearchEngine(
        workspace=tmp_path,
        provider=MagicMock(),
        model="test-model",
        tool_source=ToolRegistry(),
        config=ResearchConfig(max_rounds=2, max_findings=3, max_stale_rounds=1),
    )
    engine.scope_topic = AsyncMock(return_value="- Search same title twice")
    engine._run_round = AsyncMock(
        side_effect=[
            ResearchRoundResult(
                findings=[
                    ResearchFinding(
                        title="Recurring fintech crossover",
                        kind="pattern",
                        claim="The same programmes keep surfacing fintech-adjacent employers.",
                        why_interesting="It hints at a hybrid technical-finance employment lane.",
                        confidence=0.62,
                        evidence=["data/content/2022/a.md"],
                        next_checks=["Check 2023 tables"],
                    )
                ],
                action="stop",
                reason="Initial seed captured.",
                runner_result=_runner_result(tools_used=["grep"]),
            ),
            ResearchRoundResult(
                findings=[
                    ResearchFinding(
                        title="Recurring fintech crossover",
                        kind="pattern",
                        claim="The same programmes repeatedly surface fintech-adjacent employers across years.",
                        why_interesting="It looks persistent, not a one-off cohort effect.",
                        confidence=0.88,
                        evidence=["data/content/2023/b.md"],
                        next_checks=["Check job titles for analyst vs engineer split"],
                    )
                ],
                action="stop",
                reason="Updated evidence recorded.",
                runner_result=_runner_result(tools_used=["grep", "read_file"]),
            ),
        ]
    )

    await engine.run("fintech crossover")
    await engine.run("fintech crossover")

    findings_text = engine.store.findings_path("fintech crossover").read_text(encoding="utf-8")
    assert findings_text.count("### Recurring fintech crossover") == 1
    assert "data/content/2022/a.md" in findings_text
    assert "data/content/2023/b.md" in findings_text
    assert "0.88" in findings_text


@pytest.mark.asyncio
async def test_research_uses_existing_findings_to_enable_web_validation(tmp_path) -> None:
    engine = ResearchEngine(
        workspace=tmp_path,
        provider=MagicMock(),
        model="test-model",
        tool_source=ToolRegistry(),
        config=ResearchConfig(max_rounds=1, max_findings=3, max_stale_rounds=1, allow_web_validation=True),
    )
    engine.scope_topic = AsyncMock(return_value="- Validate existing findings against the web")
    engine.store.write_findings(
        "law topic",
        [
            ResearchFinding(
                title="Law graduates often queue through PCLL",
                kind="pattern",
                claim="Existing local evidence suggests further studies are dominated by PCLL.",
                why_interesting="It explains low direct-employment rates without implying weak outcomes.",
                confidence=0.81,
                evidence=["data/content/2023_law.md"],
                next_checks=["Validate with official PCLL references"],
            )
        ],
    )
    engine._run_round = AsyncMock(
        return_value=ResearchRoundResult(
            findings=[],
            action="stop",
            reason="Validated enough for now.",
            runner_result=_runner_result(tools_used=["web_search"]),
        )
    )

    await engine.run("law topic")

    assert engine._run_round.await_count == 1
    assert engine._run_round.await_args.kwargs["allow_web_validation"] is True


def test_round_messages_prompt_web_validation_from_existing_findings(tmp_path) -> None:
    engine = ResearchEngine(
        workspace=tmp_path,
        provider=MagicMock(),
        model="test-model",
        tool_source=ToolRegistry(),
        config=ResearchConfig(max_rounds=1, max_findings=3, max_stale_rounds=1, allow_web_validation=True),
    )
    messages = engine._build_round_messages(
        topic="law topic",
        scoping_summary="- Check further-studies pathways",
        existing_findings=[
            ResearchFinding(
                title="Law graduates often queue through PCLL",
                kind="pattern",
                claim="Existing local evidence suggests further studies are dominated by PCLL.",
                why_interesting="It explains low direct-employment rates without implying weak outcomes.",
                confidence=0.81,
                evidence=["data/content/2023_law.md"],
                next_checks=["Validate with official PCLL references"],
            )
        ],
        round_index=1,
        findings_remaining=3,
        stale_rounds=0,
        allow_web_validation=True,
    )
    joined = "\n".join(m["content"] for m in messages)
    assert "Existing findings from prior runs count as local support." in joined
    assert "Use at most 1-2 precise web_search calls" in joined


@pytest.mark.asyncio
async def test_research_collects_web_validation_brief_from_existing_findings(tmp_path) -> None:
    registry = ToolRegistry()
    fake_web = MagicMock()
    fake_web.name = "web_search"
    fake_web.execute = AsyncMock(return_value="Results for: official PCLL source")
    registry.register(fake_web)

    engine = ResearchEngine(
        workspace=tmp_path,
        provider=MagicMock(),
        model="test-model",
        tool_source=registry,
        config=ResearchConfig(max_rounds=1, max_findings=3, max_stale_rounds=1, allow_web_validation=True),
    )

    brief = await engine._collect_web_validation_brief(
        topic="law topic",
        findings=[
            ResearchFinding(
                title="Law graduates often queue through PCLL",
                kind="pattern",
                claim="Existing local evidence suggests further studies are dominated by PCLL.",
                why_interesting="It explains low direct-employment rates without implying weak outcomes.",
                confidence=0.81,
                evidence=["data/content/2023_law.md"],
                next_checks=["Validate with official HKU PLE / PCLL admissions references"],
            )
        ],
    )

    assert "Web validation brief:" in brief
    assert "official HKU PLE / PCLL admissions references" in brief
    assert "Results for: official PCLL source" in brief
    assert fake_web.execute.await_count >= 1
