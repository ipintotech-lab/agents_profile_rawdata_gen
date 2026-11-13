"""Multi-agent flows: one run that passes through five of a company's agents.

Each company's agents normally run alone, one agent per run. A flow chains
five of them into one workflow: every stage runs that agent's own steps
under its name, then makes a ``handoff`` LLM decision that picks the next
agent. The last stage answers instead. So one run, one trace and one
workflow hold up to five agents, the way an orchestrated pipeline looks.

The handoff routes on the run's case, which it sees in its input:

* ``standard`` -- the next agent.
* ``expedited`` -- skips the flow's ``optional`` agent when it is next.
* ``review`` -- ``ESCALATE_RATE`` of the time ``escalate``, which ends the
  flow there; otherwise the next agent.

``JUDGEMENT_RATE`` of the time the model picks another route the case
allows, for a reason the trace doesn't show -- so the handoff is a real
decision, not a rule a lookup table could replace.

Flows are written in the generic event log, which records an agent on
every event, and in OpenTelemetry, which names the agent on every span.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import timedelta

from .engine import START, Company, _poisson
from .model import Run, RunCtx


@dataclass
class Flow:
    company: str
    workflow: str
    agents: list[str]  # the five stages, in order
    optional: str  # the stage an ``expedited`` case skips
    runs_per_day: float
    hours: tuple[int, int] = (8, 20)


FLOWS = {f.company: f for f in (
    Flow("fintech", "customer-onboarding",
         ["kyc-verification", "fraud-screening", "loan-prequal", "card-activation", "onboarding-email"],
         "loan-prequal", 30),
    Flow("food_factory", "production-order",
         ["supplier-ordering", "recipe-scaling", "production-scheduling", "quality-inspection", "shipping-labels"],
         "recipe-scaling", 20, hours=(6, 22)),
    Flow("law_office", "new-matter",
         ["email-triage", "client-intake", "conflict-check", "document-drafting", "time-entry"],
         "document-drafting", 12),
    Flow("logistics", "shipment-lifecycle",
         ["carrier-rate-quote", "dispatch-assignment", "shipment-tracking", "eta-notification", "proof-of-delivery"],
         "eta-notification", 40, hours=(5, 22)),
    Flow("supermarket", "restock-cycle",
         ["demand-forecast", "stock-replenishment", "invoice-matching", "price-markdown", "waste-tracking"],
         "price-markdown", 24),
)}


CASES = {"standard": 0.6, "expedited": 0.25, "review": 0.15}
ESCALATE_RATE = 0.4
JUDGEMENT_RATE = 0.1
ESCALATE = "escalate"


def route(rng: random.Random, flow: Flow, n: int, case: str) -> str:
    """The agent stage ``n`` hands the run to, or ``escalate``."""
    after = flow.agents[n + 1:]
    options = [after[0]]
    if after[0] == flow.optional and len(after) > 1:
        options.append(after[1])
    if case == "review":
        options.append(ESCALATE)
    if case == "expedited" and len(options) > 1 and options[1] != ESCALATE:
        planned = options[1]
    elif case == "review" and rng.random() < ESCALATE_RATE:
        planned = ESCALATE
    else:
        planned = options[0]
    others = [o for o in options if o != planned]
    return rng.choice(others) if others and rng.random() < JUDGEMENT_RATE else planned


def flow_runs(company: Company, flow: Flow, seed: int, days: int, scale: float = 1.0) -> list[Run]:
    agents = {a.name: a for a in company.agents}
    stages = [agents[name] for name in flow.agents]
    rng = random.Random(f"{seed}:{company.key}:flow:{flow.workflow}")
    runs: list[Run] = []
    for day in range(days):
        date = START + timedelta(days=day)
        for index in range(_poisson(rng, flow.runs_per_day * scale)):
            start = date + timedelta(hours=rng.uniform(*flow.hours))
            run = Run(company.key, flow.workflow, flow.workflow, f"{flow.workflow}-d{day:02d}-{index:04d}",
                      "1.0.0", stages[0].model, start)
            ctx = RunCtx(run, rng, stages[0].llm_tokens)
            case = rng.choices(list(CASES), list(CASES.values()))[0]
            n = 0
            while True:
                stage = stages[n]
                ctx.agent, ctx.llm_tokens, run.model = stage.name, stage.llm_tokens, stage.model
                stage.fn(ctx)
                if n + 1 == len(stages):
                    break
                to = ctx.llm("handoff", {"from": stage.name, "case": case}, route(rng, flow, n, case),
                             tokens=(stage.llm_tokens[0], 30))
                if to == ESCALATE:
                    break
                n = flow.agents.index(to)
            ctx.finish()
            runs.append(run)
    return sorted(runs, key=lambda r: r.start)
