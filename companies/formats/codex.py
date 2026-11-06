"""OpenAI Codex CLI rollouts (``sessions/YYYY/MM/DD/rollout-*.jsonl``):
one long-running session per agent per day, each run one turn in it. Tools
are MCP tools called with a ``namespace`` (``mcp__<server>``). Codex
records tokens (``token_count`` after each response) but no cost; a failed
tool's output carries a non-zero exit code.
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from ..model import Run, turns
from . import dumps, hex_id, iso, write_jsonl

CLI_VERSION = "0.46.0"


def _uuid(*parts) -> str:
    return str(uuid.UUID(hex_id(*parts)))


def _item(ts, payload: dict) -> dict:
    return {"timestamp": iso(ts), "type": "response_item", "payload": payload}


def _tokens(ts, llm) -> dict:
    usage = {"input_tokens": llm.tokens[0], "cached_input_tokens": 0, "output_tokens": llm.tokens[1],
             "reasoning_output_tokens": 0, "total_tokens": sum(llm.tokens)}
    return {"timestamp": iso(ts), "type": "event_msg",
            "payload": {"type": "token_count", "info": {"last_token_usage": usage}}}


def _session(runs: list[Run], server: str) -> tuple[str, list[dict]]:
    first = runs[0]
    session = _uuid(first.company, first.agent, first.start.date())
    rows: list[dict] = [{"timestamp": iso(first.start - timedelta(seconds=1)), "type": "session_meta",
                         "payload": {"id": session, "cwd": f"/srv/agents/{first.agent}", "originator": "codex_exec",
                                     "cli_version": CLI_VERSION, "instructions": None}}]
    for run in runs:
        turn = _uuid(run.run_id, "turn")
        t0 = run.start - timedelta(milliseconds=30)
        rows.append({"timestamp": iso(t0), "type": "turn_context",
                     "payload": {"turn_id": turn, "cwd": f"/srv/agents/{run.agent}", "model": run.model,
                                 "approval_policy": "never", "sandbox_policy": {"mode": "workspace-write"}}})
        rows.append({"timestamp": iso(t0), "type": "event_msg", "payload": {"type": "task_started", "turn_id": turn}})
        rows.append(_item(t0, {"type": "message", "role": "user",
                               "content": [{"type": "input_text", "text": f"Run {run.agent} for {run.run_id}."}]}))
        for index, (llm, tools) in enumerate(turns(run)):
            done = llm.start + timedelta(milliseconds=llm.latency_ms)
            rows.append(_item(done, {"type": "reasoning", "summary": [], "encrypted_content": None}))
            ids = [f"call_{hex_id(run.run_id, index, j, length=24)}" for j in range(len(tools))]
            for call_id, tool in zip(ids, tools):
                rows.append(_item(done, {"type": "function_call", "namespace": f"mcp__{server}", "name": tool.name,
                                         "arguments": dumps(tool.input), "call_id": call_id}))
            if not tools:
                rows.append(_item(done, {"type": "message", "role": "assistant",
                                         "content": [{"type": "output_text", "text": "Done."}]}))
            rows.append(_tokens(done, llm))
            for call_id, tool in zip(ids, tools):
                end = tool.start + timedelta(milliseconds=tool.latency_ms)
                output = f"Exit code: 1\n{tool.error}" if tool.error else dumps(tool.output)
                rows.append(_item(end, {"type": "function_call_output", "call_id": call_id, "output": output}))
        rows.append({"timestamp": iso(run.steps[-1].start + timedelta(milliseconds=run.steps[-1].latency_ms)),
                     "type": "event_msg", "payload": {"type": "task_complete", "turn_id": turn}})
    return session, rows


def render(runs: list[Run], out: Path, server: str) -> list[Path]:
    groups: dict[tuple[str, object], list[Run]] = defaultdict(list)
    for run in runs:
        groups[(run.agent, run.start.date())].append(run)
    paths = []
    for (agent, day), group in sorted(groups.items()):
        session, rows = _session(sorted(group, key=lambda r: r.start), server)
        start = rows[0]["timestamp"][:19].replace(":", "-")
        path = out / "sessions" / f"{day:%Y/%m/%d}" / f"rollout-{start}-{session}.jsonl"
        write_jsonl(path, rows)
        paths.append(path)
    return paths
