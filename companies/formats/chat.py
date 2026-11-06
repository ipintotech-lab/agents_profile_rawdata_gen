"""OpenAI-style chat trajectories, one per run: ``{"id", "workflow",
"agent", "model", "messages": [...]}``, with ``tool_calls`` on assistant
messages, ``tool`` messages paired by ``tool_call_id`` and per-message
``usage``. Like real function-calling logs, they carry no cost or timing.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..model import Run, turns
from . import dumps, hex_id, write_jsonl


def _record(run: Run) -> dict:
    messages: list[dict] = [
        {"role": "system", "content": f"You are the {run.agent} assistant."},
        {"role": "user", "content": f"Handle request {run.run_id}."},
    ]
    for index, (llm, tools) in enumerate(turns(run)):
        message = {
            "role": "assistant",
            "content": None if tools else "Done.",
            "usage": {"prompt_tokens": llm.tokens[0], "completion_tokens": llm.tokens[1]},
            "finish_reason": "tool_calls" if tools else "stop",
        }
        calls = []
        for j, tool in enumerate(tools):
            calls.append({"id": f"call_{hex_id(run.run_id, index, j, length=24)}", "type": "function",
                          "function": {"name": tool.name, "arguments": dumps(tool.input)}})
        if calls:
            message["tool_calls"] = calls
        messages.append(message)
        for call, tool in zip(calls, tools):
            content = dumps({"error": tool.error}) if tool.error else dumps(tool.output)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
    return {"id": run.run_id, "workflow": run.workflow, "agent": run.agent, "model": run.model,
            "messages": messages}


def render(runs: list[Run], out: Path, server: str) -> list[Path]:
    by_day: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_day[run.start.date().isoformat()].append(_record(run))
    paths = []
    for day, rows in sorted(by_day.items()):
        path = out / f"trajectories-{day}.jsonl"
        write_jsonl(path, rows)
        paths.append(path)
    return paths
