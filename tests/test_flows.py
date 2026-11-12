"""Five-agent flows: every run passes through its five agents in order, and
the generic event log names the agent on each event."""
from __future__ import annotations

import json
from pathlib import Path

from companies.flows import ESCALATE, FLOWS, flow_runs
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
            visited = [a for i, a in enumerate(order) if i == 0 or a != order[i - 1]]
            handoffs = [e for e in events if e["node"] == "handoff"]
            # stages run in flow order; only the optional one is ever skipped
            assert visited[0] == flow.agents[0]
            assert visited == [a for a in flow.agents if a in visited]
            assert set(flow.agents[:flow.agents.index(visited[-1]) + 1]) - set(visited) <= {flow.optional}
            # each handoff names the agent that ran next; escalate ends the run
            assert [h["result"] for h in handoffs if h["result"] != ESCALATE] == visited[1:]
            if visited[-1] != flow.agents[-1]:
                assert handoffs[-1]["result"] == ESCALATE
                assert handoffs[-1]["input_state"]["case"] == "review"


def test_handoff_varies_by_case():
    """Whole month: every route shows up, and the same input doesn't always
    get the same answer -- the handoff is a judgement, not a lookup."""
    for key, flow in FLOWS.items():
        runs = flow_runs(COMPANIES[key], flow, seed=7, days=30)
        seen: dict[tuple, set] = {}
        for run in runs:
            for s in run.steps:
                if s.name == "handoff":
                    seen.setdefault((s.input["from"], s.input["case"]), set()).add(s.output)
        answers = set().union(*seen.values())
        assert ESCALATE in answers
        i = flow.agents.index(flow.optional)
        assert flow.agents[i + 1] in seen[(flow.agents[i - 1], "expedited")]  # expedited skips the optional one
        assert any(len(v) > 1 for v in seen.values())
        lengths = {len({s.agent for s in r.steps}) for r in runs}
        assert lengths >= {4, 5} and min(lengths) < 4


def test_same_seed_same_runs():
    flow = FLOWS["law_office"]
    a, b = (flow_runs(COMPANIES["law_office"], flow, seed=3, days=2) for _ in range(2))
    assert [(r.run_id, len(r.steps)) for r in a] == [(r.run_id, len(r.steps)) for r in b]
