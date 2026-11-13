"""Five-agent flows: every run passes through its five agents in order, and
the generic event log names the agent on each event."""
from __future__ import annotations

import json
from pathlib import Path

from companies.flows import FLOWS, flow_runs
from companies.formats import generic
from companies.roster import COMPANIES


def test_every_company_has_a_five_agent_flow_of_its_own_agents():
    assert sorted(FLOWS) == sorted(COMPANIES)
    for key, flow in FLOWS.items():
        assert len(flow.agents) == len(set(flow.agents)) == 5
        assert set(flow.agents) <= {a.name for a in COMPANIES[key].agents}


def test_runs_visit_the_agents_in_order_and_hand_off(tmp_path: Path):
    for key, flow in FLOWS.items():
        runs = flow_runs(COMPANIES[key], flow, seed=7, days=3)
        assert runs
        generic.render(runs, tmp_path / key, COMPANIES[key].server)
        rows = [json.loads(line) for p in sorted((tmp_path / key).glob("*.jsonl")) for line in p.open()]
        by_run: dict[str, list[dict]] = {}
        for row in rows:
            assert row["workflow"] == flow.workflow
            by_run.setdefault(row["run_id"], []).append(row)
        for events in by_run.values():
            order = [e["agent"] for e in events]
            assert [a for i, a in enumerate(order) if i == 0 or a != order[i - 1]] == flow.agents
            handoffs = [e["result"] for e in events if e["node"] == "handoff"]
            assert handoffs == flow.agents[1:]


def test_same_seed_same_runs():
    flow = FLOWS["law_office"]
    a, b = (flow_runs(COMPANIES["law_office"], flow, seed=3, days=2) for _ in range(2))
    assert [(r.run_id, len(r.steps)) for r in a] == [(r.run_id, len(r.steps)) for r in b]


def test_otel_flows_name_each_stages_agent_on_its_spans(tmp_path: Path):
    from companies.formats import otel

    flow = FLOWS["fintech"]
    runs = flow_runs(COMPANIES["fintech"], flow, seed=7, days=2)
    otel.render(runs, tmp_path, COMPANIES["fintech"].server)
    requests = [json.loads(line) for p in sorted(tmp_path.glob("otlp-traces-*.jsonl")) for line in p.open()]
    assert len(requests) == len(runs)
    for request in requests:
        (scope,) = request["resourceSpans"][0]["scopeSpans"]
        root, *children = scope["spans"]
        assert root["name"] == f"invoke_agent {flow.workflow}"
        assert {s["parentSpanId"] for s in children} == {root["spanId"]}
        agents = [next(a["value"]["stringValue"] for a in s["attributes"] if a["key"] == "gen_ai.agent.name")
                  for s in children]
        assert [a for i, a in enumerate(agents) if i == 0 or a != agents[i - 1]] == flow.agents
