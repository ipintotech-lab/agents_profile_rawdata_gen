"""OpenTelemetry OTLP JSON, as a collector's file exporter writes it: one
export request per line, one trace per run, GenAI semantic conventions
(``gen_ai.*``). A root ``invoke_agent`` span holds an ``chat`` span per LLM
call and an ``execute_tool`` span per tool call, with arguments and results
recorded (``gen_ai.tool.call.arguments`` / ``.result``) and the LLM cost in
``gen_ai.usage.cost``. In a multi-agent flow the root span is the workflow
and each child span names the stage's agent in ``gen_ai.agent.name``.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..model import Run, Step, turns
from . import dumps, hex_id, write_jsonl


def _attr(key: str, value) -> dict:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _nanos(step: Step) -> tuple[str, str]:
    start = int(step.start.timestamp() * 1e9)
    return str(start), str(start + int(step.latency_ms * 1e6))


def _span(run: Run, trace_id: str, root: str, index: int, step: Step, attrs: dict, name: str) -> dict:
    start, end = _nanos(step)
    if step.agent:  # a stage of a multi-agent flow
        attrs = {**attrs, "gen_ai.agent.name": step.agent}
    span = {
        "traceId": trace_id, "spanId": hex_id(run.run_id, index, length=16), "parentSpanId": root,
        "name": name, "kind": 1, "startTimeUnixNano": start, "endTimeUnixNano": end,
        "attributes": [_attr(k, v) for k, v in attrs.items()],
        "status": {"code": 2, "message": step.error} if step.error else {"code": 0},
    }
    return span


def _request(run: Run, service: str) -> dict:
    trace_id = hex_id(run.run_id)
    root = hex_id(run.run_id, "root", length=16)
    common = {"traceloop.workflow.name": run.workflow, "gen_ai.agent.name": run.agent}
    spans = []
    index = 0
    for llm, tools in turns(run):
        index += 1
        spans.append(_span(run, trace_id, root, index, llm, {
            **common,
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": llm.model,
            "gen_ai.response.model": llm.model,
            "gen_ai.usage.input_tokens": llm.tokens[0],
            "gen_ai.usage.output_tokens": llm.tokens[1],
            "gen_ai.usage.cost": llm.cost,
            "gen_ai.response.finish_reasons": "tool_calls" if tools else "stop",
        }, f"chat {llm.model}"))
        for tool in tools:
            index += 1
            attrs = {
                **common,
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": tool.name,
                "gen_ai.tool.call.id": hex_id(run.run_id, index, length=12),
                "gen_ai.tool.call.arguments": dumps(tool.input),
            }
            if not tool.error:
                attrs["gen_ai.tool.call.result"] = dumps(tool.output)
            spans.append(_span(run, trace_id, root, index, tool, attrs, f"execute_tool {tool.name}"))
    first, last = run.steps[0], run.steps[-1]
    agent_span = {
        "traceId": trace_id, "spanId": root, "name": f"invoke_agent {run.agent}", "kind": 1,
        "startTimeUnixNano": _nanos(first)[0], "endTimeUnixNano": _nanos(last)[1],
        "attributes": [_attr(k, v) for k, v in {**common, "gen_ai.operation.name": "invoke_agent"}.items()],
        "status": {"code": 0},
    }
    return {"resourceSpans": [{
        "resource": {"attributes": [_attr("service.name", service), _attr("deployment.environment", "production")]},
        "scopeSpans": [{"scope": {"name": "opentelemetry.instrumentation.genai", "version": "0.47b0"},
                        "spans": [agent_span, *spans]}],
    }]}


def render(runs: list[Run], out: Path, server: str) -> list[Path]:
    by_day: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_day[run.start.date().isoformat()].append(_request(run, server))
    paths = []
    for day, rows in sorted(by_day.items()):
        path = out / f"otlp-traces-{day}.jsonl"
        write_jsonl(path, rows)
        paths.append(path)
    return paths
