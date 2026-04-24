from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.research import ResearchRunResult
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command.builtin import cmd_research, cmd_research_log, cmd_research_status, cmd_research_stop
from nanobot.command.router import CommandContext


def _make_ctx(
    raw: str,
    *,
    loop,
    args: str = "",
) -> CommandContext:
    msg = InboundMessage(channel="cli", sender_id="u1", chat_id="direct", content=raw)
    return CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, args=args, loop=loop)


@pytest.mark.asyncio
async def test_research_command_starts_task_and_publishes_summary(tmp_path) -> None:
    bus = MessageBus()
    research = SimpleNamespace(
        store=SimpleNamespace(topic_slug=lambda topic: topic.replace(" ", "-")),
        run=AsyncMock(
            return_value=ResearchRunResult(
                topic="salary patterns",
                topic_slug="salary-patterns",
                summary="Research completed for `salary patterns`.",
                run_path=tmp_path / "research" / "salary-patterns" / "runs" / "20260424-120000.md",
                findings_path=tmp_path / "research" / "salary-patterns" / "FINDINGS.md",
                findings=[],
                stop_reason="Done",
                rounds_run=1,
            )
        )
    )
    loop = SimpleNamespace(
        research=research,
        bus=bus,
        _active_tasks={},
        _research_tasks={},
        _research_status={},
    )

    out = await cmd_research(_make_ctx("/research salary patterns", loop=loop, args="salary patterns"))

    assert "Research started for `salary patterns`." in out.content
    final = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    assert final.content == "Research completed for `salary patterns`."


@pytest.mark.asyncio
async def test_research_log_handles_missing_and_existing_runs() -> None:
    loop = SimpleNamespace(research=SimpleNamespace(latest_run_log=lambda topic: None))

    missing = await cmd_research_log(_make_ctx("/research-log topic", loop=loop, args="topic"))
    assert "No research runs found yet" in missing.content

    loop = SimpleNamespace(research=SimpleNamespace(latest_run_log=lambda topic: "# Research Run: topic"))
    existing = await cmd_research_log(_make_ctx("/research-log topic", loop=loop, args="topic"))
    assert existing.content == "# Research Run: topic"


@pytest.mark.asyncio
async def test_research_status_reports_running_and_done_tasks(tmp_path) -> None:
    loop = SimpleNamespace(
        research=SimpleNamespace(store=SimpleNamespace(topic_slug=lambda topic: topic.replace(" ", "-"))),
        _research_status={
            "cli:direct": {
                "salary-trends": {
                    "topic": "salary trends",
                    "status": "running",
                    "started_at": 1.0,
                    "updated_at": 2.0,
                    "summary": "",
                    "run_path": None,
                    "findings_path": None,
                    "error": None,
                },
                "employer-clusters": {
                    "topic": "employer clusters",
                    "status": "done",
                    "started_at": 1.0,
                    "updated_at": 3.0,
                    "summary": "Research completed.",
                    "run_path": str(tmp_path / "research" / "employer-clusters" / "runs" / "latest.md"),
                    "findings_path": str(tmp_path / "research" / "employer-clusters" / "FINDINGS.md"),
                    "error": None,
                },
            }
        },
    )

    all_status = await cmd_research_status(_make_ctx("/research-status", loop=loop))
    assert "## Research Status" in all_status.content
    assert "`done`" in all_status.content
    assert "`running`" in all_status.content

    one_status = await cmd_research_status(
        _make_ctx("/research-status salary trends", loop=loop, args="salary trends")
    )
    assert "`salary trends`" in one_status.content
    assert "`running`" in one_status.content


@pytest.mark.asyncio
async def test_research_stop_cancels_session_research_tasks() -> None:
    cancelled = asyncio.Event()

    async def slow() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(slow())
    await asyncio.sleep(0)
    loop = SimpleNamespace(_research_tasks={"cli:direct": [task]})

    out = await cmd_research_stop(_make_ctx("/research-stop", loop=loop))

    assert cancelled.is_set()
    assert "Stopped 1 research task" in out.content
