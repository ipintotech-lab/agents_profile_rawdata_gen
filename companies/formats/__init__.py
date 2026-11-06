"""Renderers: the same runs written in each trace format profiler-client reads.

``node_name`` says what node a step becomes after ``profiler-client prepare
--keep-names`` -- the name the truth file gives each planted case, so a
report (re-identified with ``profiler-client open``) can be scored by name.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

FORMATS = ("generic", "otel", "openai-chat", "claude-code", "codex")
AGENT_LOOP_FORMATS = ("otel", "openai-chat", "claude-code", "codex")
# Formats whose tools are MCP tools named ``mcp__<server>__<tool>``.
MCP_FORMATS = ("claude-code", "codex")


def hex_id(*parts: Any, length: int = 32) -> str:
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:length]


def iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"


def dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def mcp_tool(server: str, name: str) -> str:
    return f"mcp__{server}__{name}"


def tool_node(fmt: str, server: str, name: str) -> str:
    return f"mcp {mcp_tool(server, name)}" if fmt in MCP_FORMATS else name


def node_name(fmt: str, server: str, *, tool: str | None = None, llm: str | None = None,
              after: str | None = None) -> str:
    if tool:
        return tool_node(fmt, server, tool)
    if llm:
        if fmt != "generic":
            raise ValueError("named LLM decisions are their own node only in the generic event log")
        return llm
    if fmt == "generic":
        raise ValueError("the generic event log has no 'LLM call after a tool' nodes")
    # profiler-client names an LLM call by the first thing it reacted to
    return f"llm_call ← {tool_node(fmt, server, after).split(': ')[0]}"
