"""The eight-agent loan-application flow: same seed same bytes, an answer key
that matches the data, parallel checks that overlap, and the five-agent
flows left exactly as they were."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from companies.caseflows import AGENTS, CASES, CHECKS, SERVER, SUPERVISOR, caseflow_runs
from companies.flows import FLOWS, flow_runs
from companies.formats import caseflow, generic, otel
from companies.roster import COMPANIES
from companies.truth import caseflow_rows

MONTH = caseflow_runs(seed=7, days=30)


def _rows(directory: Path) -> list[dict]:
    return [json.loads(line) for p in sorted(directory.glob("*.jsonl")) for line in p.open()]


def _digest(directory: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*.jsonl")):
        h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()


@pytest.mark.parametrize("fmt", ["generic", "otel"])
def test_same_seed_same_bytes(tmp_path: Path, fmt: str):
    for name in ("a", "b"):
        runs = caseflow_runs(seed=7, days=3)
        if fmt == "otel":
            caseflow.render_otel(runs, tmp_path / name, SERVER)
        else:
            caseflow.render_generic(runs, tmp_path / name)
    assert _digest(tmp_path / "a") == _digest(tmp_path / "b")


def test_a_month_is_big_enough_and_every_agent_works():
    events = [s for r in MONTH for s in r.steps if not s.implicit]
    assert len(events) >= 5000
    assert 25 <= len({s.name for s in events}) <= 35
    per_run = [sum(not s.implicit for s in r.steps) for r in MONTH]
    assert 8 <= min(per_run) and max(per_run) <= 40
    assert {s.agent for s in events} == set(AGENTS)


def test_routing_follows_the_planted_rules():
    by_input: dict[tuple, Counter] = {}
    for run in MONTH:
        for s in run.steps:
            if s.kind == "llm" and not s.implicit:
                assert s.agent == SUPERVISOR
                given = {k: v for k, v in s.input.items() if k != "from"}
                by_input.setdefault((s.name, json.dumps(given, sort_keys=True)), Counter())[s.output] += 1
    checks = {k[1]: v for k, v in by_input.items() if k[0] == "route_after_checks"}
    for given, choices in checks.items():
        reject = '"high"' in given
        assert set(choices) == {"decision-letter" if reject else "pricing"}, given
    assert {k[1]: set(v) for k, v in by_input.items() if k[0] == "route_after_pricing"} == {
        '{"amount_band": "high"}': {"compliance-review"}, '{"amount_band": "standard"}': {"decision-letter"}}
    assert [set(v) for k, v in by_input.items() if k[0] == "route_after_compliance"] == [{"decision-letter"}]
    # route_after_kyc follows kyc_status about 90% of the time; a rejection always ends the run
    kyc = {k[1]: v for k, v in by_input.items() if k[0] == "route_after_kyc"}
    complete = kyc['{"kyc_status": "complete"}']
    assert 0.8 < complete["parallel-checks"] / sum(complete.values()) < 0.97
    rejected = kyc.get('{"kyc_status": "rejected"}', Counter())
    assert set(rejected) <= {"decision-letter"}
    assert sum(rejected.values()) < 10  # too thin a slice to call a rule


def test_rework_loops_at_most_twice_and_hands_back_to_kyc():
    for run in MONTH:
        visits = [seg.agent for seg in run.segments]
        assert visits.count("document-collection") <= 2
        for seg in run.segments:
            if seg.agent == "kyc-verification" and seg.by is not None:
                assert seg.by.agent == "document-collection"


def test_parallel_checks_overlap_and_interleave(tmp_path: Path):
    orders = set()
    for run in MONTH:
        spans = {seg.agent: [s for s in run.steps if s.segment == seg.id] for seg in run.segments
                 if seg.agent in CHECKS}
        if len(spans) < 2:
            continue
        fraud, credit = spans[CHECKS[0]], spans[CHECKS[1]]
        assert fraud[0].start == credit[0].start  # one fork
        orders.add(tuple(s.agent for s in run.steps if s.agent in CHECKS and not s.implicit))
    assert len(orders) > 10  # a different interleaving from run to run (F8)

    caseflow.render_otel(MONTH[:50], tmp_path, SERVER)
    overlapping = 0
    for request in _rows(tmp_path):
        spans = request["resourceSpans"][0]["scopeSpans"][0]["spans"]
        root = spans[0]
        assert root["name"] == f"invoke_agent {SUPERVISOR}" and "parentSpanId" not in root
        agents = {s["name"].split(" ", 1)[1]: s for s in spans[1:] if s["name"].startswith("invoke_agent ")}
        assert all(s["parentSpanId"] == root["spanId"] for s in agents.values())
        if set(CHECKS) <= set(agents):
            a, b = agents[CHECKS[0]], agents[CHECKS[1]]
            assert a["links"] == b["links"]  # both started by the same decision
            overlapping += int(a["startTimeUnixNano"]) < int(b["endTimeUnixNano"]) and \
                int(b["startTimeUnixNano"]) < int(a["endTimeUnixNano"])
    assert overlapping > 30


def test_event_log_links_agents_to_the_decision_that_started_them(tmp_path: Path):
    caseflow.render_generic(MONTH[:50], tmp_path)
    rows = _rows(tmp_path)
    by_id = {r["event_id"]: r for r in rows}
    assert len(by_id) == len(rows)
    for r in rows:
        if r["agent"] == SUPERVISOR:
            assert "parent" not in r
            assert ("trigger" in r) == (r["event_type"] != "tool_call")
        elif "parent" in r:
            parent = by_id[r["parent"]]
            assert parent["run_id"] == r["run_id"]
            assert parent["agent"] == SUPERVISOR or (parent["agent"], r["agent"]) == (
                "document-collection", "kyc-verification")
    triggers = {r["trigger"] for r in rows if r.get("node") == "route_after_checks"}
    assert triggers == {"get_credit_score + risk_score"}


def test_answer_key_matches_the_data():
    rows = {r["id"].rsplit("/", 1)[1]: r for r in caseflow_rows(CASES, MONTH, "generic", "fintech", "loan-application")}
    assert sorted(rows, key=lambda k: int(k[1:])) == [f"F{n}" for n in range(1, 13)]
    for key, row in rows.items():
        assert row["scope"] == "flow" and row["agents"]
        obs = row["observed"]
        if row["expect"]:  # every positive clears the detection minimum
            count = {"CROSS_AGENT_DUPLICATE": "runs_with_duplicate", "CACHE_RESULT": "repeat_calls_same_output"}
            assert obs[count.get(row["type"], "llm_calls")] >= 10, key
    o = {k: r["observed"] for k, r in rows.items()}
    assert o["F1"]["runs_duplicate_differing_output"] == 0 and o["F1"]["inputs_with_varying_output"] > 0
    assert o["F2"]["runs_duplicate_differing_output"] > 0
    assert o["F3"]["runs_duplicate_differing_output"] == 0 and o["F3"]["runs_with_duplicate"] >= 10
    assert o["F4"]["distinct_choices"] == 1
    assert o["F5"]["inputs_with_varying_choice"] == o["F11"]["inputs_with_varying_choice"] == 0
    assert o["F6"]["inputs_with_varying_choice"] > 0
    assert o["F7"]["inputs_with_varying_output"] == 0 and o["F7"]["runs_with_duplicate"] == 0
    assert o["F10"]["repeat_calls"] == 0
    assert o["F12"]["inputs_with_varying_output"] == 0 and o["F12"]["runs_with_duplicate"] >= 10
    # the counts are the data's own
    assert o["F7"]["calls"] == sum(s.name == "kyc_lookup" for r in MONTH for s in r.steps)


# sha256 of the five-agent flows (seed 7, 3 days) in both formats, as on main before case flows existed
FLOWS_DIGEST = {"generic": "870bc6ff798629992f793720070d630f3534fd71f4135eec3a5cb560a1a0fef1",
                "otel": "81c3f584dade4263a1ea66994fe3b057a718b96fa5e3c3d08ad75e4222287316"}


@pytest.mark.parametrize("fmt", ["generic", "otel"])
def test_five_agent_flows_are_unchanged(tmp_path: Path, fmt: str):
    for key, flow in FLOWS.items():
        runs = flow_runs(COMPANIES[key], flow, seed=7, days=3)
        if fmt == "otel":
            otel.render(runs, tmp_path / key, COMPANIES[key].server, named_decisions=True)
        else:
            generic.render(runs, tmp_path / key, COMPANIES[key].server)
    assert _digest(tmp_path) == FLOWS_DIGEST[fmt]


NOISY = caseflow_runs(seed=7, days=30, noise=0.1)


def test_noise_only_adds_failed_attempts_each_retried_with_the_same_arguments():
    def calls(run):
        # the balance depends on whether the parallel fraud check's hold came first (F2): timing, which retries shift
        return Counter((s.agent, s.kind, s.name, json.dumps(s.input, sort_keys=True),
                        None if s.name == "get_account_balance" else json.dumps(s.output, sort_keys=True))
                       for s in run.steps if not s.error)

    assert [r.run_id for r in NOISY] == [r.run_id for r in MONTH]
    assert all(calls(n) == calls(c) for n, c in zip(NOISY, MONTH))
    failed = [(r, i) for r in NOISY for i, s in enumerate(r.steps) if s.error]
    tools = sum(s.kind == "tool" for r in NOISY for s in r.steps if not s.error)
    assert 0.07 < len(failed) / tools < 0.13
    assert {s.error for r, i in failed for s in [r.steps[i]]} == {"timed out after 30 s", "503 service unavailable"}
    for run, i in failed:
        s = run.steps[i]
        retry = next(t for t in run.steps[i + 1:] if t.kind == "tool" and t.agent == s.agent and not t.error)
        assert (retry.name, retry.input) == (s.name, s.input)
    assert not any(s.error for r in caseflow_runs(seed=7, days=3, noise=0) for s in r.steps)


def test_with_noise_every_expected_value_stays_the_same():
    clean = {r["id"]: r for r in caseflow_rows(CASES, MONTH, "generic", "fintech", "loan-application")}
    noisy = {r["id"]: r for r in caseflow_rows(CASES, NOISY, "generic", "fintech", "loan-application")}
    for key, row in noisy.items():
        assert row["expect"] == clean[key]["expect"]
        if row["type"] in ("CROSS_AGENT_DUPLICATE", "CACHE_RESULT") and row["node"] != "get_account_balance":
            assert row["observed"] == clean[key]["observed"], key  # a failed call never hides a duplicate (F1, F3)
    o = {k.rsplit("/", 1)[1]: r["observed"] for k, r in noisy.items()}
    assert o["F2"]["runs_duplicate_differing_output"] > 0  # the trap is still a trap
    assert o["F3"]["runs_duplicate_differing_output"] == 0 and o["F3"]["runs_with_duplicate"] >= 10
