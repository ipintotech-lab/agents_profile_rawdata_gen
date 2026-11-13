"""Multi-agent flows: one run that passes through five of a company's agents.

Each company's agents normally run alone, one agent per run. A flow chains
five of them into one workflow: every stage runs that agent's own steps
under its name, then makes a ``handoff`` LLM decision that picks the next
agent. The last stage answers instead. So one run, one trace and one
workflow hold five agents, the way an orchestrated pipeline looks.

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
    runs_per_day: float
    hours: tuple[int, int] = (8, 20)


FLOWS = {f.company: f for f in (
    Flow("fintech", "customer-onboarding",
         ["kyc-verification", "fraud-screening", "loan-prequal", "card-activation", "onboarding-email"], 30),
    Flow("food_factory", "production-order",
         ["supplier-ordering", "recipe-scaling", "production-scheduling", "quality-inspection", "shipping-labels"],
         20, hours=(6, 22)),
    Flow("law_office", "new-matter",
         ["email-triage", "client-intake", "conflict-check", "document-drafting", "time-entry"], 12),
    Flow("logistics", "shipment-lifecycle",
         ["carrier-rate-quote", "dispatch-assignment", "shipment-tracking", "eta-notification", "proof-of-delivery"],
         40, hours=(5, 22)),
    Flow("supermarket", "restock-cycle",
         ["demand-forecast", "stock-replenishment", "invoice-matching", "price-markdown", "waste-tracking"], 24),
)}


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
            for n, stage in enumerate(stages):
                ctx.agent, ctx.llm_tokens, run.model = stage.name, stage.llm_tokens, stage.model
                stage.fn(ctx)
                if n + 1 < len(stages):
                    ctx.llm("handoff", {"from": stage.name}, stages[n + 1].name, tokens=(stage.llm_tokens[0], 30))
            ctx.finish()
            runs.append(run)
    return sorted(runs, key=lambda r: r.start)
