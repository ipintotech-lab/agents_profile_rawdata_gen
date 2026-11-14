"""Format-neutral runs: what an agent did, before any trace format.

A run is a list of steps. A step is an LLM call or a tool call, with its
input, output, timing, tokens and cost. The renderers in ``formats/`` turn
the same runs into each trace format, and ``truth.py`` reads them to count
what a planted case would save -- so every format and the truth come from
one set of facts.

In agent loops (Claude Code, Codex, OpenTelemetry agents, chat
function-calling) every tool call is chosen by an LLM call. ``RunCtx.tool``
therefore attaches each tool to the LLM call before it, or adds an
``implicit`` one (a router turn) when there is none. The generic event log
writes only the explicit LLM calls, the way a hand-instrumented agent logs
its named decisions.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

# USD per million tokens (input, output) -- list-price shaped, synthetic.
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
}


def price(model: str, tokens_in: int, tokens_out: int) -> float:
    rate_in, rate_out = PRICES[model]
    return round((tokens_in * rate_in + tokens_out * rate_out) / 1e6, 6)


@dataclass
class Step:
    kind: str  # "llm" | "tool"
    name: str
    input: dict
    output: Any  # tool: its result; llm: the decision it selected
    start: datetime
    latency_ms: float
    tokens: tuple[int, int] = (0, 0)
    cost: float = 0.0
    model: str | None = None
    error: str | None = None  # a tool's error message, when it failed
    side_effect: bool | None = None  # only what the agent declares, never inferred
    decision_type: str | None = None
    implicit: bool = False  # a router turn that only picks the next tool
    chose: list[str] = field(default_factory=list)  # llm: the tools it called
    agent: str | None = None  # the agent that took the step, in a multi-agent flow (else the run's)
    segment: int | None = None  # which visit of an agent it belongs to, in a case flow (``caseflows.py``)


@dataclass
class Run:
    company: str
    agent: str
    workflow: str
    run_id: str
    version: str
    model: str
    start: datetime
    steps: list[Step] = field(default_factory=list)
    segments: list = field(default_factory=list)  # a case flow's agent visits (``caseflows.Segment``)
    noise: float = 0.0  # share of tool calls that fail once and are retried (``RunCtx.tool``)


def stable(*parts: Any) -> random.Random:
    """A generator seeded only by ``parts``: the same arguments give the
    same "result" in every run, every format and every invocation."""
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def noisy(rng: random.Random, rate: float, canonical: str, alternatives: list[str]) -> str:
    """``canonical`` with probability ``rate``, else another alternative."""
    pool = [a for a in alternatives if a != canonical]
    return canonical if rate >= 1.0 or not pool or rng.random() < rate else rng.choice(pool)


class RunCtx:
    """Builds one run step by step; agent functions call ``llm`` and ``tool``."""

    def __init__(self, run: Run, rng: random.Random, llm_tokens: tuple[int, int]):
        self.run = run
        self.rng = rng
        self.now = run.start
        self.llm_tokens = llm_tokens
        self.agent: str | None = None  # set by a flow when it hands over to the next agent
        self.segment: int | None = None  # set by a case flow at each agent visit

    def _advance(self, latency_ms: float) -> datetime:
        start = self.now
        self.now += timedelta(milliseconds=latency_ms + self.rng.uniform(5, 40))
        return start

    def _llm_step(self, name: str, input: dict, output: Any, tokens: tuple[int, int] | None,
                  decision_type: str | None, implicit: bool) -> Step:
        base_in, base_out = tokens or self.llm_tokens
        t_in = max(1, int(base_in * self.rng.uniform(0.85, 1.15)))
        t_out = max(1, int(base_out * self.rng.uniform(0.7, 1.3)))
        latency = 250 + t_out * self.rng.uniform(12, 25)
        step = Step("llm", name, dict(input), output, self._advance(latency), round(latency, 1),
                    (t_in, t_out), price(self.run.model, t_in, t_out), self.run.model,
                    decision_type=decision_type, implicit=implicit, agent=self.agent, segment=self.segment)
        self.run.steps.append(step)
        return step

    def llm(self, name: str, input: dict, selected: str, *, tokens: tuple[int, int] | None = None,
            decision_type: str | None = None) -> str:
        """A named LLM decision (``selected`` is what it chose)."""
        self._llm_step(name, input, selected, tokens, decision_type or name, implicit=False)
        return selected

    def tool(self, name: str, args: dict, result: Any, *, error: str | None = None,
             side_effect: bool | None = None, latency: tuple[float, float] = (40, 400),
             parallel: bool = False) -> Any:
        """A tool call, chosen by the LLM call before it (or a new router turn)."""
        last = self.run.steps[-1] if self.run.steps else None
        chooser = None
        if parallel or (last is not None and last.kind == "llm" and not last.chose):
            chooser = next(s for s in reversed(self.run.steps) if s.kind == "llm")
        if chooser is None:
            chooser = self._llm_step("route", {}, None, (self.llm_tokens[0], 40), "next_action", implicit=True)
        chooser.chose.append(name)
        if self.run.noise and not error:
            self._fail_once(name, args, side_effect, latency)
        lat = self.rng.uniform(*latency)
        self.run.steps.append(Step("tool", name, dict(args), result, self._advance(lat), round(lat, 1),
                                   error=error, side_effect=side_effect, agent=self.agent, segment=self.segment))
        return result

    def _fail_once(self, name: str, args: dict, side_effect: bool | None, latency: tuple[float, float]) -> None:
        """With probability ``run.noise``, a failed attempt (a timeout or a
        server error) before the call, which the agent then retries with the
        same arguments. Its own random source, so the run's other steps stay
        exactly as they are without noise."""
        r = stable("noise", self.run.run_id, self.agent, len(self.run.steps), name)
        if r.random() >= self.run.noise:
            return
        timeout = r.random() < 0.5
        lat = 30000.0 if timeout else r.uniform(*latency)
        start = self.now
        self.now += timedelta(milliseconds=lat + r.uniform(5, 40))
        self.run.steps.append(Step("tool", name, dict(args), None, start, round(lat, 1),
                                   error="timed out after 30 s" if timeout else "503 service unavailable",
                                   side_effect=side_effect, agent=self.agent, segment=self.segment))

    def finish(self) -> None:
        """The closing LLM turn that answers instead of calling a tool."""
        self._llm_step("respond", {}, "respond", (self.llm_tokens[0], self.llm_tokens[1]), "next_action",
                       implicit=True)


def turns(run: Run) -> list[tuple[Step, list[Step]]]:
    """Each LLM call with the tool calls it made, in order."""
    out: list[tuple[Step, list[Step]]] = []
    for step in run.steps:
        if step.kind == "llm":
            out.append((step, []))
        elif out:
            out[-1][1].append(step)
    return out


AgentFn = Callable[[RunCtx], None]
