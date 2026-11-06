"""The company traffic generator: deterministic, well-formed, and its
planted cases really are in the data the way the truth file says."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from companies.engine import company_runs
from companies.formats import FORMATS
from companies.roster import COMPANIES
from companies.truth import case_rows
from generate_companies import RENDERERS


@pytest.fixture(scope="module")
def month():
    """Every company's 30 days and truth rows, generated once."""
    data = {}
    for key, company in COMPANIES.items():
        runs = company_runs(company, seed=7, days=30)
        data[key] = (runs, case_rows(company, runs))
    return data


def test_five_companies_ten_agents_every_format():
    assert len(COMPANIES) == 5
    assert all(len(c.agents) == 10 for c in COMPANIES.values())
    assert sorted(c.format for c in COMPANIES.values()) == sorted(FORMATS)


def test_same_seed_same_bytes(tmp_path: Path):
    for name in ("a", "b"):
        for key, company in COMPANIES.items():
            RENDERERS[company.format].render(company_runs(company, seed=3, days=2), tmp_path / name / key,
                                             company.server)
    files = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*.jsonl"))
    assert files
    for rel in files:
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes()


def test_every_file_is_jsonl(tmp_path: Path):
    for key, company in COMPANIES.items():
        for path in RENDERERS[company.format].render(company_runs(company, seed=1, days=1), tmp_path / key,
                                                     company.server):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert rows, path


def test_every_tool_call_is_chosen_by_an_llm_call(month):
    for runs, _ in month.values():
        for run in runs:
            assert run.steps[0].kind == "llm" and run.steps[-1].name == "respond"
            chosen = [n for s in run.steps if s.kind == "llm" for n in s.chose]
            assert chosen == [s.name for s in run.steps if s.kind == "tool"]


def test_volumes_are_realistic(month):
    for key, (runs, _) in month.items():
        events = sum(len(r.steps) for r in runs)
        assert 15_000 <= events <= 110_000, (key, events)


def _cases(month, **match):
    return [c for _, rows in month.values() for c in rows if all(c[k] == v for k, v in match.items())]


def test_cache_positives_repeat_with_one_output(month):
    for case in _cases(month, type="CACHE_RESULT", expect=True):
        seen = case["observed"]
        assert seen["repeat_calls_same_output"] >= 10, case["id"]
        assert seen["inputs_with_varying_output"] == 0, case["id"]


def test_noisy_and_moving_tools_vary(month):
    for cid in ("fintech/fraud-screening/1", "fintech/fx-quote/1", "supermarket/demand-forecast/1",
                "logistics/carrier-rate-quote/1"):
        (case,) = [c for c in _cases(month, type="CACHE_RESULT") if c["id"] == cid]
        assert case["observed"]["inputs_with_varying_output"] > 0, cid


def test_write_traps_repeat_identically(month):
    """A write trap only tests something if it looks cacheable."""
    for cid in ("fintech/payment-dispute/1", "fintech/card-activation/2", "food_factory/supplier-ordering/1",
                "food_factory/shipping-labels/1", "law_office/time-entry/2", "supermarket/waste-tracking/1",
                "logistics/proof-of-delivery/1", "logistics/eta-notification/1"):
        (case,) = [c for c in _cases(month) if c["id"] == cid]
        assert case["observed"]["repeat_calls_same_output"] >= 10, cid
        assert case["observed"]["inputs_with_varying_output"] == 0, cid


def test_bypass_cases(month):
    for case in _cases(month, type="CONDITIONAL_LLM_BYPASS", expect=True):
        assert case["observed"]["llm_calls"] >= 5, case["id"]
        if "node" in case and case["node"].startswith("llm_call"):
            assert case["observed"]["distinct_choices"] == 1, case["id"]
        else:  # a named decision: one choice per input
            assert case["observed"]["inputs_with_varying_choice"] == 0, case["id"]
    for cid in ("fintech/aml-report/1", "law_office/discovery-summary/1"):
        (case,) = [c for c in _cases(month) if c["id"] == cid]
        assert case["observed"]["llm_calls"] < 5, cid


def test_hidden_state_positives_vary_on_identical_input(month):
    for case in _cases(month, type="REVIEW_HIDDEN_STATE", expect=True):
        assert case["format"] == "generic"
        assert case["observed"]["inputs_with_varying_choice"] > 0, case["id"]


def test_sequences_occur(month):
    for case in _cases(month, type="EXTRACT_DETERMINISTIC_SUBSTEP"):
        assert case["observed"]["occurrences"] >= 20, case["id"]
        # the saving is the LLM calls between the steps the format records
        assert (case["observed"]["llm_calls_between"] > 0) == case["expect"], case["id"]


def test_scale_multiplies_traffic_but_rare_agents_stay_rare():
    company = next(c for c in COMPANIES.values() if any(a.rare for a in c.agents))
    rare = {a.name for a in company.agents if a.rare}

    def counts(scale):
        runs = company_runs(company, seed=7, days=30, scale=scale)
        return (sum(r.agent in rare for r in runs), sum(r.agent not in rare for r in runs))

    (rare_1x, busy_1x), (rare_5x, busy_5x) = counts(1), counts(5)
    assert rare_5x == rare_1x
    assert 4 * busy_1x < busy_5x < 6 * busy_1x
