"""A hand-instrumented agent's own event log: one JSON event per step, in
the profiler's event shape (the format profiler-client calls ``events``).

Only the agent's named decisions are logged as LLM events -- router turns
are an agent-loop detail this kind of log doesn't have. It is the only
format that records what an LLM decision saw (``input_state``) and chose
(``decision.selected``), the workflow version and declared side effects.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..model import Run
from . import write_jsonl


def _event(run: Run, step) -> dict:
    event = {
        "trace_id": f"trace-{run.run_id}",
        "run_id": run.run_id,
        "workflow": run.workflow,
        "agent": run.agent,
        "event_type": "decision" if step.kind == "llm" else "tool_call",
        "node": step.name,
        "timestamp": step.start.isoformat(),
        "input_state": step.input,
        "tokens": {"input": step.tokens[0], "output": step.tokens[1]},
        "latency_ms": step.latency_ms,
        "cost": step.cost,
        "workflow_version": run.version,
    }
    if step.kind == "llm":
        event["model"] = step.model
        event["decision"] = {"type": step.decision_type, "selected": step.output}
        event["result"] = step.output
    else:
        event["tool"] = {"name": step.name}
        if step.error:
            event["status"] = "error"
            event["error"] = step.error
        else:
            event["result"] = step.output
        if step.side_effect is not None:
            event["has_side_effects"] = step.side_effect
    return event


def render(runs: list[Run], out: Path, server: str) -> list[Path]:
    by_day: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_day[run.start.date().isoformat()].extend(_event(run, s) for s in run.steps if not s.implicit)
    paths = []
    for day, rows in sorted(by_day.items()):
        path = out / f"events-{day}.jsonl"
        write_jsonl(path, rows)
        paths.append(path)
    return paths
