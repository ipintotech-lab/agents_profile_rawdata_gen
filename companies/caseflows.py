"""Case flows: eight agents work on one loan application.

Unlike ``flows.py`` (five agents in a line, each drawing its own data), one
case -- customer, application, amount, product, segment, risk tier -- is
drawn per run and every agent uses it in its tool arguments, so two agents
can fetch the same record. The supervisor decides who works next, with one
named LLM step per decision, each seeing only the fields it depends on::

    intake-supervisor: get_application, get_customer_profile
    kyc-verification                       <-------------------+
    route_after_kyc (from, kyc_status)                         |
      incomplete -> document-collection -> back to kyc  (at most twice)
      rejected   -> decision-letter (failed sanctions check)
      complete   -> fraud-screening || credit-assessment  (in parallel)
    route_after_checks (from, risk_tier, fraud_flag)
      high risk or high fraud score -> decision-letter (rejection)
      otherwise                     -> pricing
    route_after_pricing (from, amount_band)
      high -> compliance-review -> route_after_compliance (from) -> decision-letter
      else -> decision-letter

``route_after_kyc`` picks another allowed route ``JUDGEMENT_RATE`` of the
time, for a reason the trace doesn't show. The other three are rules.

Parallel branches run on their own clocks from the fork, so their steps
interleave by time; the join waits for the later one. Everything comes from
the seed, so the same seed writes the same bytes.

Each agent visit is a ``Segment``: the renderers in ``formats/caseflow.py``
turn it into an agent span (OpenTelemetry) or into parent links from the
routing decision that started it (event log).
"""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import timedelta

from .engine import START, Case, _poisson
from .model import Run, RunCtx, Step, stable
from .roster.kit import amount as draw_amount
from .roster.kit import band, fresh, pool_id, pure, ref, uid, world

COMPANY = "fintech"
SERVER = "payments-api"
WORKFLOW = "loan-application"
SUPERVISOR = "intake-supervisor"
# agent -> (model, typical LLM tokens in/out)
AGENTS: dict[str, tuple[str, tuple[int, int]]] = {
    SUPERVISOR: ("gpt-5", (1400, 60)),
    "kyc-verification": ("gpt-5-mini", (900, 120)),
    "document-collection": ("gpt-5-mini", (800, 140)),
    "fraud-screening": ("gpt-5-mini", (1100, 120)),
    "credit-assessment": ("gpt-5-mini", (1200, 150)),
    "pricing": ("gpt-5-mini", (900, 110)),
    "compliance-review": ("gpt-5", (1600, 160)),
    "decision-letter": ("gpt-5-mini", (1000, 400)),
}
CHECKS = ("fraud-screening", "credit-assessment")

RUNS_PER_DAY = 40
HOURS = (8, 20)
CUSTOMERS = 300  # a pool: customers come back, so some lookups repeat across runs
PRODUCTS = ["personal-loan", "auto-loan", "home-improvement"]
SEGMENTS = ["student", "salaried", "freelancer", "business"]
RISK_TIERS = {"low": 0.5, "medium": 0.35, "high": 0.15}
AMOUNT_MEAN = 15000  # lognormal, sigma 0.6: about 20% of applications are above HIGH_AMOUNT
HIGH_AMOUNT = 25000
INCOMPLETE = 0.25  # the first KYC pass finds documents missing
INCOMPLETE_AGAIN = 0.2  # ... and a later pass still does
MAX_LOOPS = 2
SANCTIONS_HIT = 0.002  # rare: the rejected slice of route_after_kyc stays under 10 runs a month
FRAUD_HIGH = 0.94  # risk_score above this is a high fraud score
JUDGEMENT_RATE = 0.1


@dataclass
class Segment:
    """One visit of an agent: its steps carry ``Step.segment == id``.
    ``by`` is the step that started it: a routing decision, or the last
    step of document collection for the KYC pass it hands back to (None
    for the supervisor's own segment and the first KYC pass)."""
    id: int
    agent: str
    by: Step | None = None


def _visit(ctx: RunCtx, agent: str, by: Step | None) -> None:
    """Start a visit of ``agent``: its own model, and its first LLM turn."""
    segment = Segment(len(ctx.run.segments), agent, by)
    ctx.run.segments.append(segment)
    ctx.agent, ctx.segment = agent, segment.id
    ctx.run.model, ctx.llm_tokens = AGENTS[agent]
    if agent != SUPERVISOR:
        ctx._llm_step("route", {}, None, (ctx.llm_tokens[0], 40), "next_action", implicit=True)


def _back_to_supervisor(ctx: RunCtx) -> None:
    ctx.agent, ctx.segment = SUPERVISOR, 0
    ctx.run.model, ctx.llm_tokens = AGENTS[SUPERVISOR]


def _decide(ctx: RunCtx, name: str, given: dict, selected: str) -> Step:
    _back_to_supervisor(ctx)
    ctx.llm(name, given, selected, tokens=(ctx.llm_tokens[0], 30))
    return ctx.run.steps[-1]


# ---- what each agent does with the case -------------------------------------------------------------

def _profile(ctx: RunCtx, case: dict) -> None:
    """Changes between runs (last activity), the same within one run."""
    args = {"customer_id": case["customer_id"]}
    r = stable("get_customer_profile", case["customer_id"], ctx.run.run_id)
    ctx.tool("get_customer_profile", args, {"ref": ref(r), "segment": case["segment"],
                                            "last_activity": r.randint(0, 10**9)})


def _transactions(ctx: RunCtx, case: dict) -> None:
    args = {"customer_id": case["customer_id"], "window_days": 90}
    r = stable("get_transaction_history", case["customer_id"], ctx.run.run_id)
    ctx.tool("get_transaction_history", args, {"ref": ref(r), "count": r.randint(20, 400)},
             latency=(300, 900))


def _balance(ctx: RunCtx, case: dict) -> None:
    """The balance a hold lowers: after ``place_hold`` it reads differently."""
    held = case["_hold_at"] is not None and ctx.now > case["_hold_at"]
    r = stable("get_account_balance", case["customer_id"], ctx.run.run_id, held)
    ctx.tool("get_account_balance", {"customer_id": case["customer_id"]},
             {"ref": ref(r), "available": round(r.uniform(200, 40000), 2), "held": held})


def supervisor_intake(ctx: RunCtx, case: dict) -> None:
    ctx.tool("get_application", {"application_id": case["application_id"]},
             fresh(ctx, product=case["product"], amount=case["amount"]))
    _profile(ctx, case)


def kyc_verification(ctx: RunCtx, case: dict) -> str:
    """The case's ``kyc_status`` after this pass. A rework pass already
    has the profile from the first one."""
    if case["_loops"] == 0:
        _profile(ctx, case)
    nid = case["national_id"]
    ctx.tool("kyc_lookup", {"national_id": nid}, pure("kyc_lookup", {"national_id": nid},
             status=lambda r: r.choice(["clear", "clear", "clear", "review"])), latency=(600, 1800))
    sanctions = world(ctx, "sanctions_check", {"national_id": nid}, 86400, list_version=lambda r: r.randint(1, 99))
    hit = ctx.rng.random() < SANCTIONS_HIT
    ctx.tool("sanctions_check", {"national_id": nid}, {**sanctions, "hit": hit}, latency=(400, 1200))
    if hit:
        return "rejected"
    missing = ctx.rng.random() < (INCOMPLETE if case["_loops"] == 0 else INCOMPLETE_AGAIN)
    return "incomplete" if missing and case["_loops"] < MAX_LOOPS else "complete"


def document_collection(ctx: RunCtx, case: dict) -> None:
    app = case["application_id"]
    docs = ctx.tool("list_documents", {"application_id": app},
                    fresh(ctx, missing=lambda r: r.choice(["payslip", "id_back", "bank_statement"])))
    ctx.tool("request_document", {"application_id": app, "document": docs["missing"]}, {"requested": True},
             side_effect=True)
    ctx.tool("validate_document", {"upload_id": uid(ctx, "doc")}, fresh(ctx, valid=True), latency=(500, 1500))


def fraud_screening(ctx: RunCtx, case: dict) -> None:
    _balance(ctx, case)
    ctx.tool("place_hold", {"application_id": case["application_id"], "amount": case["amount"]}, {"held": True},
             side_effect=True)
    case["_hold_at"] = ctx.run.steps[-1].start
    _transactions(ctx, case)
    score = fresh(ctx, score=lambda r: round(r.random(), 3))
    ctx.tool("risk_score", {"customer_id": case["customer_id"], "application_id": case["application_id"]}, score)
    case["fraud_flag"] = "high" if score["score"] > FRAUD_HIGH else "normal"


def credit_assessment(ctx: RunCtx, case: dict) -> None:
    _profile(ctx, case)
    _transactions(ctx, case)
    _balance(ctx, case)
    ctx.tool("calculate_affordability", {"application_id": case["application_id"], "amount": case["amount"]},
             fresh(ctx, ratio=lambda r: round(r.uniform(0.1, 0.6), 2)))
    # the hard bureau pull comes last, once affordability passes
    ctx.tool("get_credit_score", {"customer_id": case["customer_id"]},
             world(ctx, "get_credit_score", {"customer_id": case["customer_id"]}, 30 * 86400,
                   score=lambda r: r.randint(520, 830)), latency=(800, 2500))


def _rate_table(ctx: RunCtx, case: dict) -> None:
    args = {"product": case["product"], "risk_tier": case["risk_tier"]}
    ctx.tool("get_rate_table", args, pure("get_rate_table", args, apr=lambda r: round(r.uniform(4, 19), 2)))


def pricing(ctx: RunCtx, case: dict) -> None:
    _rate_table(ctx, case)
    ctx.tool("calculate_offer", {"application_id": case["application_id"], "amount": case["amount"],
                                 "risk_tier": case["risk_tier"]}, fresh(ctx, term_months=lambda r: r.choice([24, 36, 60])))


def compliance_review(ctx: RunCtx, case: dict) -> None:
    ctx.tool("get_policy", {"product": case["product"]}, pure("get_policy", {"product": case["product"]}))
    _rate_table(ctx, case)
    ctx.tool("check_policy", {"application_id": case["application_id"], "amount_band": case["amount_band"]},
             fresh(ctx, passed=lambda r: r.random() > 0.05))
    # a person approves: minutes to hours, no model involved
    ctx.tool("request_human_approval", {"application_id": case["application_id"]},
             fresh(ctx, approved=lambda r: r.random() > 0.1), latency=(120_000, 10_800_000))


def decision_letter(ctx: RunCtx, case: dict, outcome: str) -> None:
    args = {"template": outcome, "product": case["product"]}
    ctx.tool("render_template", args, pure("render_template", args))
    ctx.tool("send_letter", {"application_id": case["application_id"], "outcome": outcome}, {"queued": True},
             side_effect=True)


# ---- the run ----------------------------------------------------------------------------------------

def _case(ctx: RunCtx) -> dict:
    rng = ctx.rng
    customer = pool_id(rng, "cus", CUSTOMERS)
    amount = draw_amount(rng, AMOUNT_MEAN)
    return {
        "customer_id": customer, "national_id": "nid-" + customer[-4:],
        "application_id": uid(ctx, "app"), "amount": amount,
        "product": rng.choice(PRODUCTS), "segment": rng.choice(SEGMENTS),
        "risk_tier": rng.choices(list(RISK_TIERS), list(RISK_TIERS.values()))[0],
        "amount_band": band(amount, [HIGH_AMOUNT], ["standard", "high"]),
        "fraud_flag": None, "_hold_at": None, "_loops": 0,
    }


def _parallel(ctx: RunCtx, case: dict, by: Step, branches: list[tuple[str, Callable]]) -> None:
    """Run the branches from the same moment, each on its own clock; their
    steps interleave by time and the join waits for the later one."""
    fork, ends, steps = ctx.now, [], []
    for agent, fn in branches:
        sub = replace(ctx.run, start=fork, steps=[])  # shares ``segments``
        branch = RunCtx(sub, ctx.rng, ctx.llm_tokens)
        _visit(branch, agent, by)
        fn(branch, case)
        steps += sub.steps
        ends.append(branch.now)
    ctx.run.steps += sorted(steps, key=lambda s: s.start)
    ctx.now = max(ends)


def _route_after_kyc(rng: random.Random, status: str, loops: int) -> str:
    if status == "rejected":
        return "decision-letter"
    allowed = ["parallel-checks"] + (["document-collection"] if loops < MAX_LOOPS else [])
    planned = "document-collection" if status == "incomplete" and loops < MAX_LOOPS else "parallel-checks"
    others = [a for a in allowed if a != planned]
    return rng.choice(others) if others and rng.random() < JUDGEMENT_RATE else planned


def case_run(rng: random.Random, run: Run) -> None:
    ctx = RunCtx(run, rng, AGENTS[SUPERVISOR][1])
    case = _case(ctx)
    _visit(ctx, SUPERVISOR, None)
    supervisor_intake(ctx, case)

    by: Step | None = None
    while True:  # KYC, with the rework loop through document collection
        _visit(ctx, "kyc-verification", by)
        status = kyc_verification(ctx, case)
        to = _route_after_kyc(rng, status, case["_loops"])
        decision = _decide(ctx, "route_after_kyc", {"from": "kyc-verification", "kyc_status": status}, to)
        if to != "document-collection":
            break
        _visit(ctx, "document-collection", decision)
        document_collection(ctx, case)
        case["_loops"] += 1
        by = ctx.run.steps[-1]  # document collection hands the case straight back to KYC

    if to == "decision-letter":
        outcome = "rejected"
    else:
        _parallel(ctx, case, decision, [(CHECKS[0], fraud_screening), (CHECKS[1], credit_assessment)])
        reject = case["risk_tier"] == "high" or case["fraud_flag"] == "high"
        decision = _decide(ctx, "route_after_checks", {"from": "+".join(CHECKS), "risk_tier": case["risk_tier"],
                                                       "fraud_flag": case["fraud_flag"]},
                           "decision-letter" if reject else "pricing")
        outcome = "rejected"
        if not reject:
            _visit(ctx, "pricing", decision)
            pricing(ctx, case)
            high = case["amount_band"] == "high"
            decision = _decide(ctx, "route_after_pricing", {"from": "pricing", "amount_band": case["amount_band"]},
                               "compliance-review" if high else "decision-letter")
            if high:
                _visit(ctx, "compliance-review", decision)
                compliance_review(ctx, case)
                decision = _decide(ctx, "route_after_compliance", {"from": "compliance-review"}, "decision-letter")
            outcome = "offer"
    _visit(ctx, "decision-letter", decision)
    decision_letter(ctx, case, outcome)
    _back_to_supervisor(ctx)
    ctx.tool("update_application_status", {"application_id": case["application_id"], "status": outcome},
             {"updated": True}, side_effect=True)
    ctx.finish()


def caseflow_runs(seed: int, days: int, scale: float = 1.0) -> list[Run]:
    rng = random.Random(f"{seed}:{COMPANY}:caseflow:{WORKFLOW}")
    runs: list[Run] = []
    for day in range(days):
        date = START + timedelta(days=day)
        for index in range(_poisson(rng, RUNS_PER_DAY * scale)):
            start = date + timedelta(hours=rng.uniform(*HOURS))
            run = Run(COMPANY, WORKFLOW, WORKFLOW, f"{WORKFLOW}-d{day:02d}-{index:04d}", "1.0.0",
                      AGENTS[SUPERVISOR][0], start)
            case_run(rng, run)
            runs.append(run)
    return sorted(runs, key=lambda r: r.start)


# ---- the answer key ---------------------------------------------------------------------------------

# id -> (planted case, the agents whose steps it is about)
CASES: dict[str, tuple[Case, tuple[str, ...]]] = {
    "F1": (Case("CROSS_AGENT_DUPLICATE", True, "positive", "three agents fetch the same customer profile in one "
                "run; it changes between runs, so only the in-run repeat is reusable", tool="get_customer_profile"),
           (SUPERVISOR, "kyc-verification", "credit-assessment")),
    "F2": (Case("CROSS_AGENT_DUPLICATE", False, "trap", "fraud screening places a hold between the two balance "
                "reads, so they often differ", tool="get_account_balance"), CHECKS),
    "F3": (Case("CROSS_AGENT_DUPLICATE", True, "positive", "both parallel checks fetch the same 90-day history",
                tool="get_transaction_history"), CHECKS),
    "F4": (Case("CONDITIONAL_LLM_BYPASS", True, "positive", "after compliance review it always goes to the letter",
                llm="route_after_compliance"), (SUPERVISOR,)),
    "F5": (Case("DETERMINISTIC_REGION", True, "positive", "high-risk applications are always rejected",
                llm="route_after_checks", field="risk_tier"), (SUPERVISOR,)),
    "F6": (Case("CONDITIONAL_LLM_BYPASS", False, "trap", "10% of the time it picks another route, for a reason "
                "the trace doesn't show", llm="route_after_kyc"), (SUPERVISOR,)),
    "F7": (Case("CACHE_RESULT", True, "positive", "KYC is looked up again on every rework pass, and for returning "
                "customers", tool="kyc_lookup"), ("kyc-verification",)),
    "F8": (Case("EXTRACT_DETERMINISTIC_SUBSTEP", False, "trap", "the parallel checks interleave in a different "
                "order each run: no sequence runs across the two branches",
                steps=["get_account_balance", "get_customer_profile"]), CHECKS),
    "F9": (Case("EXTRACT_DETERMINISTIC_SUBSTEP", None, "edge", "a fixed pair across a handoff: the credit score "
                "always precedes the rate table", steps=["get_credit_score", "get_rate_table"]),
           ("credit-assessment", "pricing")),
    "F10": (Case("CACHE_RESULT", False, "trap", "a person approves: long, varying waits and a different "
                 "answer each time", tool="request_human_approval"), ("compliance-review",)),
    "F11": (Case("DETERMINISTIC_REGION", True, "positive", "above the amount threshold it always goes to "
                 "compliance review", llm="route_after_pricing", field="amount_band"), (SUPERVISOR,)),
    "F12": (Case("CACHE_RESULT", True, "edge", "pricing and compliance read the same rate table, which is the "
                 "same across runs too: one finding, with a note that agents repeat it", tool="get_rate_table"),
            ("pricing", "compliance-review")),
}
