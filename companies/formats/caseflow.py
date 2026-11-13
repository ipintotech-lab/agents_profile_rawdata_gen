"""Case flows (``caseflows.py``) in the event log and in OpenTelemetry.

Event log (``generic``): the same event shape as ``generic.py``, plus the
links a hand-instrumented orchestrator records -- every event has an
``event_id``; an agent's events name as ``parent`` the routing decision
that started its visit (or, for a KYC pass after document collection,
that collection's last step); each routing decision names as ``trigger``
the last step of every agent visit that returned to the supervisor since
its previous decision (both branches, after the parallel checks). Events are
written in time order, so the two parallel branches interleave.

OpenTelemetry (``otel``): one trace per run under an ``invoke_agent
intake-supervisor`` root span. Each agent visit is its own ``invoke_agent``
span, a child of the root, holding that visit's LLM and tool spans; the
parallel branches are sibling spans that overlap in time. A visit started
by a routing decision (or by document collection) carries a span link to
that step's span.
Routing decisions record their name, input and choice
(``traceloop.entity.*``), as in ``flows.py``'s OpenTelemetry output.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..model import Run, Step, turns
from . import dumps, hex_id, write_jsonl
from .generic import _event
from .otel import _attr, _nanos, _span


def _event_ids(run: Run) -> dict[int, str]:
    return {id(s): hex_id(run.run_id, i, length=16) for i, s in enumerate(run.steps)}


def _triggers(run: Run) -> dict[int, str]:
    """Each named supervisor decision's trigger: the last logged step of
    every other agent's visit since the previous decision that returned to
    the supervisor, not handed over to another agent."""
    supervisor = run.segments[0].agent
    last: dict[int, str] = {}
    out: dict[int, str] = {}
    for s in run.steps:
        if s.implicit:
            continue
        if s.agent == supervisor and s.kind == "llm":
            out[id(s)] = " + ".join(sorted(set(last.values()))) or "user_prompt"
            last = {}
        elif s.agent != supervisor:
            last[s.segment] = s.name
            by = run.segments[s.segment].by
            if by is not None and by.agent != supervisor:
                last.pop(by.segment, None)
    return out


def render_generic(runs: list[Run], out: Path) -> list[Path]:
    by_day: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        ids, triggers = _event_ids(run), _triggers(run)
        for s in run.steps:
            if s.implicit:
                continue
            event = {**_event(run, s), "event_id": ids[id(s)]}
            by = run.segments[s.segment].by
            if by is not None:
                event["parent"] = ids[id(by)]
            if id(s) in triggers:
                event["trigger"] = triggers[id(s)]
            by_day[run.start.date().isoformat()].append(event)
    paths = []
    for day, rows in sorted(by_day.items()):
        path = out / f"events-{day}.jsonl"
        write_jsonl(path, rows)
        paths.append(path)
    return paths


def _request(run: Run, service: str) -> dict:
    trace_id = hex_id(run.run_id)
    root = hex_id(run.run_id, "root", length=16)
    common = {"traceloop.workflow.name": run.workflow}
    span_of = {id(s): hex_id(run.run_id, i, length=16) for i, s in enumerate(run.steps, 1)}
    visit_span = {seg.id: root if seg.id == 0 else hex_id(run.run_id, "visit", seg.id, length=16)
                  for seg in run.segments}
    by_visit: dict[int, list[Step]] = defaultdict(list)
    for s in run.steps:
        by_visit[s.segment].append(s)

    spans = []
    for seg in run.segments[1:]:
        steps = by_visit[seg.id]
        span = {
            "traceId": trace_id, "spanId": visit_span[seg.id], "parentSpanId": root,
            "name": f"invoke_agent {seg.agent}", "kind": 1,
            "startTimeUnixNano": _nanos(steps[0])[0], "endTimeUnixNano": _nanos(steps[-1])[1],
            "attributes": [_attr(k, v) for k, v in {**common, "gen_ai.operation.name": "invoke_agent",
                                                     "gen_ai.agent.name": seg.agent}.items()],
            "status": {"code": 0},
        }
        if seg.by is not None:
            span["links"] = [{"traceId": trace_id, "spanId": span_of[id(seg.by)]}]
        spans.append(span)
    for seg in run.segments:
        visit = Run(run.company, run.agent, run.workflow, run.run_id, run.version, run.model, run.start,
                    steps=by_visit[seg.id])
        for llm, tools in turns(visit):
            named = {}
            if not llm.implicit:
                named = {"traceloop.entity.name": llm.name, "traceloop.entity.input": dumps(llm.input),
                         "traceloop.entity.output": dumps(llm.output)}
            spans.append(_with_id(_span(run, trace_id, visit_span[seg.id], 0, llm, {
                **common, **named,
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": llm.model,
                "gen_ai.response.model": llm.model,
                "gen_ai.usage.input_tokens": llm.tokens[0],
                "gen_ai.usage.output_tokens": llm.tokens[1],
                "gen_ai.usage.cost": llm.cost,
                "gen_ai.response.finish_reasons": "tool_calls" if tools else "stop",
            }, f"chat {llm.model}"), span_of[id(llm)]))
            for tool in tools:
                attrs = {
                    **common,
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": tool.name,
                    "gen_ai.tool.call.id": span_of[id(tool)][:12],
                    "gen_ai.tool.call.arguments": dumps(tool.input),
                }
                if not tool.error:
                    attrs["gen_ai.tool.call.result"] = dumps(tool.output)
                spans.append(_with_id(_span(run, trace_id, visit_span[seg.id], 0, tool, attrs,
                                            f"execute_tool {tool.name}"), span_of[id(tool)]))
    spans.sort(key=lambda s: (int(s["startTimeUnixNano"]), s["spanId"]))
    first, last = run.steps[0], run.steps[-1]
    supervisor = run.segments[0].agent
    agent_span = {
        "traceId": trace_id, "spanId": root, "name": f"invoke_agent {supervisor}", "kind": 1,
        "startTimeUnixNano": _nanos(first)[0], "endTimeUnixNano": _nanos(last)[1],
        "attributes": [_attr(k, v) for k, v in {**common, "gen_ai.operation.name": "invoke_agent",
                                                 "gen_ai.agent.name": supervisor}.items()],
        "status": {"code": 0},
    }
    return {"resourceSpans": [{
        "resource": {"attributes": [_attr("service.name", service), _attr("deployment.environment", "production")]},
        "scopeSpans": [{"scope": {"name": "opentelemetry.instrumentation.genai", "version": "0.47b0"},
                        "spans": [agent_span, *spans]}],
    }]}


def _with_id(span: dict, span_id: str) -> dict:
    span["spanId"] = span_id
    return span


def render_otel(runs: list[Run], out: Path, server: str) -> list[Path]:
    by_day: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_day[run.start.date().isoformat()].append(_request(run, server))
    paths = []
    for day, rows in sorted(by_day.items()):
        path = out / f"otlp-traces-{day}.jsonl"
        write_jsonl(path, rows)
        paths.append(path)
    return paths
