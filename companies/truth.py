"""The answer key: every planted case, named as profiler-client will name
it, with what the generated traffic actually holds for it (calls, repeats,
LLM cost) -- so a report can be scored case by case, and a miss can be
told apart from a case the data was too thin to show.

Costs are the generator's own; ``cost_in_format`` says whether the trace
format carries them for the profiler to see.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .engine import Case, Company
from .formats import MCP_FORMATS, dumps, node_name, tool_node
from .model import Run, turns

COST_FORMATS = ("generic", "otel", "claude-code")


def _after(runs: list[Run], tool: str) -> list:
    """The LLM calls that react to ``tool``'s result."""
    out = []
    for run in runs:
        ts = turns(run)
        for (_, tools), (llm, _) in zip(ts, ts[1:]):
            if tools and sorted(t.name for t in tools)[0] == tool:
                out.append((run.version, llm))
    return out


def _choice(step) -> str:
    return dumps(step.output) if step.output is not None else "+".join(sorted(set(step.chose))) or "respond"


def _llm_stats(calls: list) -> dict:
    """``calls`` are (workflow version, LLM step); one input is compared
    within one version, as the profiler compares one configuration."""
    by_input: dict[tuple, set] = defaultdict(set)
    for version, s in calls:
        by_input[(version, dumps(s.input))].add(_choice(s))
    steps = [s for _, s in calls]
    return {"llm_calls": len(steps), "llm_cost_usd": round(sum(s.cost for s in steps), 4),
            "llm_tokens": sum(sum(s.tokens) for s in steps),
            "distinct_choices": len(Counter(_choice(s) for s in steps)),
            "versions": len({v for v, _ in calls}),
            "inputs_with_varying_choice": sum(len(v) > 1 for v in by_input.values())}


def _cache_stats(runs: list[Run], tool: str) -> dict:
    groups: dict[str, list] = defaultdict(list)
    for run in runs:
        for s in run.steps:
            if s.kind == "tool" and s.name == tool and not s.error:
                groups[dumps(s.input)].append(dumps(s.output))
    calls = sum(len(g) for g in groups.values())
    stable = [g for g in groups.values() if len(set(g)) == 1]
    return {"calls": calls, "distinct_inputs": len(groups),
            "repeat_calls": calls - len(groups),
            "repeat_calls_same_output": sum(len(g) - 1 for g in stable),
            "inputs_with_varying_output": len(groups) - len(stable)}


def _sequence_stats(runs: list[Run], steps: list[str], logs_routers: bool) -> dict:
    """``logs_routers``: whether the format records the router turns between
    tools (the generic event log doesn't, so there is nothing to save)."""
    occurrences, glue = 0, []
    for run in runs:
        names = [s for s in run.steps if s.kind == "tool"]
        for i in range(len(names) - len(steps) + 1):
            if [s.name for s in names[i:i + len(steps)]] == steps:
                occurrences += 1
                lo, hi = run.steps.index(names[i]), run.steps.index(names[i + len(steps) - 1])
                glue += [s for s in run.steps[lo:hi] if s.kind == "llm" and (logs_routers or not s.implicit)]
    return {"occurrences": occurrences, "llm_calls_between": len(glue),
            "llm_cost_between_usd": round(sum(s.cost for s in glue), 4)}


def _calibrate(case: Case, runs: list[Run], fmt: str) -> dict:
    if case.steps:
        return _sequence_stats(runs, case.steps, fmt != "generic")
    if case.tool:
        return _cache_stats(runs, case.tool)
    if case.llm:
        return _llm_stats([(r.version, s) for r in runs for s in r.steps if s.kind == "llm" and s.name == case.llm])
    return _llm_stats(_after(runs, case.after))


def case_rows(company: Company, runs: list[Run]) -> list[dict]:
    by_agent: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        by_agent[run.agent].append(run)
    rows = []
    for agent in company.agents:
        for n, case in enumerate(agent.cases, 1):
            row = {
                "id": f"{company.key}/{agent.name}/{n}",
                "company": company.key, "format": company.format, "agent": agent.name,
                # agent CLIs name the workflow after the agent's directory
                "workflow": f"{company.format}/{agent.name}" if company.format in MCP_FORMATS else agent.workflow,
                "category": case.category, "type": case.type, "expect": case.expect, "why": case.why,
            }
            if case.steps:
                row["steps"] = [tool_node(company.format, company.server, s) for s in case.steps]
            else:
                row["node"] = node_name(company.format, company.server, tool=case.tool, llm=case.llm,
                                        after=case.after)
            if case.field:
                row["field"] = case.field
            row["observed"] = {**_calibrate(case, by_agent[agent.name], company.format),
                               "cost_in_format": company.format in COST_FORMATS}
            rows.append(row)
    return rows


def _duplicate_stats(runs: list[Run], tool: str) -> dict:
    """Runs where two or more agents call ``tool`` with the same arguments,
    and whether every such group returned the same result in that run."""
    same = differing = 0
    for run in runs:
        groups: dict[str, list] = defaultdict(list)
        for s in run.steps:
            if s.kind == "tool" and s.name == tool and not s.error:
                groups[dumps(s.input)].append(s)
        shared = [g for g in groups.values() if len({s.agent for s in g}) > 1]
        if not shared:
            continue
        if all(len({dumps(s.output) for s in g}) == 1 for g in shared):
            same += 1
        else:
            differing += 1
    return {"runs_with_duplicate": same + differing, "runs_duplicate_same_output": same,
            "runs_duplicate_differing_output": differing}


def caseflow_rows(cases: dict, runs: list[Run], fmt: str, company: str, workflow: str) -> list[dict]:
    """The case flow's rows (``caseflows.CASES``): ``scope: "flow"``; named
    decisions are their own node in both formats."""
    rows = []
    for key, (case, agents) in cases.items():
        row = {"id": f"{company}/{workflow}/{key}", "company": company, "format": fmt, "scope": "flow",
               "agents": list(agents), "workflow": workflow, "category": case.category, "type": case.type,
               "expect": case.expect, "why": case.why, "observable": {"generic": True, "otel": True}}
        if case.steps:
            row["steps"] = list(case.steps)
        else:
            row["node"] = case.tool or case.llm
        if case.field:
            row["field"] = case.field
        if case.steps:
            observed = _sequence_stats(runs, case.steps, fmt != "generic")
        elif case.llm:
            observed = _llm_stats([(r.version, s) for r in runs for s in r.steps
                                   if s.kind == "llm" and s.name == case.llm])
        else:
            observed = {**_cache_stats(runs, case.tool), **_duplicate_stats(runs, case.tool)}
        row["observed"] = {**observed, "cost_in_format": fmt in COST_FORMATS}
        rows.append(row)
    return rows
