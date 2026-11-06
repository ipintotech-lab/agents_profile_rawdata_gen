"""Companies, their agents and the 30-day schedule that turns them into runs.

Every agent has its own random stream seeded by (seed, company, agent), so
adding or changing one agent never shifts another agent's data, and the same
seed always writes the same traffic.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .model import AgentFn, Run, RunCtx

START = datetime(2026, 10, 1, tzinfo=timezone.utc)


@dataclass
class Case:
    """One planted fact the profiler should (``expect``) or should not report.

    Name the step it is about with exactly one of ``tool`` (a tool step),
    ``llm`` (a named LLM decision -- visible as its own node only in the
    generic event log), ``after`` (the LLM call that reacts to this tool's
    result -- how agent-loop formats name LLM calls) or ``steps`` (a tool
    sequence, in order).
    """
    type: str
    expect: bool
    category: str  # positive | trap | edge
    why: str
    tool: str | None = None
    llm: str | None = None
    after: str | None = None
    steps: list[str] | None = None
    field: str | None = None  # DETERMINISTIC_REGION: the input field that marks the region

    def __post_init__(self) -> None:
        named = [x for x in (self.tool, self.llm, self.after, self.steps) if x]
        if len(named) != 1:
            raise ValueError(f"case {self.type} must name exactly one of tool/llm/after/steps")
        if self.category not in ("positive", "trap", "edge"):
            raise ValueError(f"unknown category {self.category}")


@dataclass
class AgentDef:
    name: str
    fn: AgentFn
    runs_per_day: float
    model: str
    llm_tokens: tuple[int, int] = (900, 120)
    weekdays_only: bool = False
    hours: tuple[int, int] = (8, 20)
    # (first day index, semver) -- a later entry is a release mid-month
    versions: list[tuple[int, str]] = field(default_factory=lambda: [(0, "1.0.0")])
    cases: list[Case] = field(default_factory=list)

    @property
    def workflow(self) -> str:
        return self.name


@dataclass
class Company:
    key: str
    name: str
    format: str  # generic | otel | openai-chat | claude-code | codex
    server: str  # the tool server's name (MCP formats name tools mcp__<server>__<tool>)
    agents: list[AgentDef]


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 30:
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def _version(agent: AgentDef, day: int) -> str:
    return [v for first, v in agent.versions if first <= day][-1]


def agent_runs(company: Company, agent: AgentDef, seed: int, days: int) -> list[Run]:
    rng = random.Random(f"{seed}:{company.key}:{agent.name}")
    runs: list[Run] = []
    for day in range(days):
        date = START + timedelta(days=day)
        if agent.weekdays_only and date.weekday() >= 5:
            continue
        for index in range(_poisson(rng, agent.runs_per_day)):
            start = date + timedelta(hours=rng.uniform(*agent.hours))
            run = Run(company.key, agent.name, agent.workflow, f"{agent.name}-d{day:02d}-{index:04d}",
                      _version(agent, day), agent.model, start)
            ctx = RunCtx(run, rng, agent.llm_tokens)
            agent.fn(ctx)
            ctx.finish()
            runs.append(run)
    return runs


def company_runs(company: Company, seed: int, days: int = 30) -> list[Run]:
    runs = [r for agent in company.agents for r in agent_runs(company, agent, seed, days)]
    return sorted(runs, key=lambda r: r.start)
