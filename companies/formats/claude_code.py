"""Claude Code session transcripts, as headless ``claude -p`` jobs write
them: one session per agent per day (``<sessionId>.jsonl``), each run one
prompt (``promptId``) in it, tools served over MCP (``mcp__<server>__*``)
and a closing ``cost-state`` record with the session's USD per model.
Each agent runs in its own directory, so it gets its own workflow.
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from ..model import Run, turns
from . import dumps, hex_id, iso, mcp_tool, write_jsonl

VERSION = "2.0.31"


def _uuid(*parts) -> str:
    return str(uuid.UUID(hex_id(*parts)))


def _session(runs: list[Run], server: str) -> tuple[str, list[dict]]:
    first = runs[0]
    session = _uuid(first.company, first.agent, first.start.date())
    base = {"sessionId": session, "cwd": f"/srv/agents/{first.agent}", "version": VERSION,
            "userType": "external", "isSidechain": False}
    rows: list[dict] = []
    cost: dict[str, float] = defaultdict(float)
    for run in runs:
        prompt = _uuid(run.run_id, "prompt")
        rows.append({**base, "type": "user", "uuid": _uuid(run.run_id, "u"), "promptId": prompt,
                     "timestamp": iso(run.start - timedelta(milliseconds=50)),
                     "message": {"role": "user", "content": f"Run the {run.agent} job for {run.run_id}."}})
        for index, (llm, tools) in enumerate(turns(run)):
            cost[llm.model] += llm.cost
            content: list[dict] = [{"type": "text", "text": "Working on it." if tools else "Done."}]
            ids = [f"toolu_{hex_id(run.run_id, index, j, length=24)}" for j in range(len(tools))]
            content += [{"type": "tool_use", "id": i, "name": mcp_tool(server, t.name), "input": t.input}
                        for i, t in zip(ids, tools)]
            rows.append({**base, "type": "assistant", "uuid": _uuid(run.run_id, index, "a"),
                         "timestamp": iso(llm.start + timedelta(milliseconds=llm.latency_ms)),
                         "message": {"id": f"msg_{hex_id(run.run_id, index, length=24)}", "role": "assistant",
                                     "model": llm.model, "content": content,
                                     "stop_reason": "tool_use" if tools else "end_turn",
                                     "usage": {"input_tokens": llm.tokens[0], "output_tokens": llm.tokens[1],
                                               "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}})
            if tools:
                end = max(t.start + timedelta(milliseconds=t.latency_ms) for t in tools)
                results = [{"type": "tool_result", "tool_use_id": i,
                            "content": t.error if t.error else dumps(t.output), "is_error": bool(t.error)}
                           for i, t in zip(ids, tools)]
                rows.append({**base, "type": "user", "uuid": _uuid(run.run_id, index, "r"), "promptId": prompt,
                             "timestamp": iso(end), "message": {"role": "user", "content": results}})
    rows.append({"type": "cost-state", "sessionId": session,
                 "modelUsage": {m: {"costUSD": round(c, 6)} for m, c in cost.items()}})
    return session, rows


def render(runs: list[Run], out: Path, server: str) -> list[Path]:
    groups: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for run in runs:
        groups[(run.agent, run.start.date().isoformat())].append(run)
    paths = []
    for (agent, _day), group in sorted(groups.items()):
        session, rows = _session(sorted(group, key=lambda r: r.start), server)
        path = out / f"-srv-agents-{agent}" / f"{session}.jsonl"
        write_jsonl(path, rows)
        paths.append(path)
    return paths
