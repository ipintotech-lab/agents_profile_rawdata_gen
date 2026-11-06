"""Law office -- OpenAI-style chat trajectories, gpt-5.

Chat logs carry no cost, so a rule-like LLM choice here can be seen but not
priced: the bypass positives are ``edge`` cases.
"""
from __future__ import annotations

from ..engine import AgentDef, Case, Company
from ..model import RunCtx
from .kit import fresh, pool_id, pure, uid

MODEL = "gpt-5"
AREAS = ["employment", "real-estate", "ip", "corporate", "tax", "litigation", "family", "immigration",
         "banking", "privacy", "construction", "insurance"]


def contract_review(ctx: RunCtx) -> None:
    area = ctx.rng.choice(AREAS)
    ctx.tool("fetch_clause_library", {"practice_area": area}, pure("fetch_clause_library", {"practice_area": area}))
    doc = uid(ctx, "doc")
    ctx.tool("extract_clauses", {"document_id": doc}, fresh(ctx), latency=(2000, 6000))
    ctx.tool("compare_clauses", {"document_id": doc, "practice_area": area}, fresh(ctx))


def legal_research(ctx: RunCtx) -> None:
    for _ in range(ctx.rng.randint(2, 4)):
        found = ctx.tool("search_case_law", {"query": uid(ctx, "q")}, fresh(ctx))
        ctx.tool("get_case", {"case_ref": found["ref"]}, fresh(ctx))


def client_intake(ctx: RunCtx) -> None:
    client = uid(ctx, "cli")
    ctx.tool("create_client_record", {"client_id": client}, fresh(ctx))
    ctx.tool("run_conflict_check", {"client_id": client}, fresh(ctx))
    ctx.tool("open_matter", {"client_id": client}, fresh(ctx))


def conflict_check(ctx: RunCtx) -> None:
    party = uid(ctx, "pty")
    hit = ctx.tool("search_party_db", {"party": party}, fresh(ctx, matches=lambda r: r.choice([0, 0, 0, 1, 2])))
    if hit["matches"]:
        ctx.tool("escalate_conflict", {"party": party}, fresh(ctx))
    else:
        ctx.tool("clear_conflict", {"party": party}, fresh(ctx))


def time_entry(ctx: RunCtx) -> None:
    keeper = pool_id(ctx.rng, "tk", 15, 0.4)
    ctx.tool("get_billing_rate", {"timekeeper": keeper}, pure("get_billing_rate", {"timekeeper": keeper}))
    ctx.tool("post_time_entry", {"timekeeper": keeper, "hours": ctx.rng.choice([0.5, 1.0, 1.5, 2.0])},
             {"status": "posted"})


def document_drafting(ctx: RunCtx) -> None:
    ctx.tool("get_matter", {"matter_id": uid(ctx, "mat")}, fresh(ctx))
    ctx.tool("save_draft", {"draft_id": uid(ctx, "drf")}, fresh(ctx))


def deadline_docketing(ctx: RunCtx) -> None:
    court = ctx.rng.choice(["district", "appeals", "supreme", "labor", "family", "magistrate"])
    ctx.tool("get_court_rules", {"court": court}, pure("get_court_rules", {"court": court}))
    ctx.tool("compute_deadline", {"court": court, "filed": ctx.now.date().isoformat()}, fresh(ctx))
    ctx.tool("add_calendar_event", {"event_ref": uid(ctx, "ev")}, fresh(ctx))


def discovery_summary(ctx: RunCtx) -> None:
    ctx.tool("load_document_batch", {"batch_id": uid(ctx, "dsc")}, fresh(ctx), latency=(4000, 9000))
    ctx.tool("summarize_batch", {"batch_id": uid(ctx, "dsc")}, fresh(ctx))


def email_triage(ctx: RunCtx) -> None:
    ctx.tool("fetch_inbox_message", {"message_id": uid(ctx, "msg")}, fresh(ctx))
    ctx.tool("label_email", {"message_id": uid(ctx, "msg"), "label": ctx.rng.choice(["client", "court", "admin"])},
             fresh(ctx))


def nda_generation(ctx: RunCtx) -> None:
    jurisdiction = ctx.rng.choice(["israel", "delaware", "england"])
    ctx.tool("render_nda_template", {"jurisdiction": jurisdiction},
             pure("render_nda_template", {"jurisdiction": jurisdiction}))
    ctx.tool("save_draft_nda", {"draft_id": uid(ctx, "nda")}, fresh(ctx))


COMPANY = Company("law_office", "Law office", "openai-chat", "firm", [
    AgentDef("contract-review", contract_review, 2, MODEL, weekdays_only=True, cases=[
        Case("CACHE_RESULT", True, "positive", "12 practice areas' clause libraries, fetched again",
             tool="fetch_clause_library")]),
    AgentDef("legal-research", legal_research, 3, MODEL, weekdays_only=True),
    AgentDef("client-intake", client_intake, 4, MODEL, weekdays_only=True, cases=[
        Case("EXTRACT_DETERMINISTIC_SUBSTEP", True, "positive", "always record -> conflict check -> open matter",
             steps=["create_client_record", "run_conflict_check", "open_matter"])]),
    AgentDef("conflict-check", conflict_check, 5, MODEL, weekdays_only=True, cases=[
        Case("REVIEW_HIDDEN_STATE", False, "trap", "escalate vs clear follows the matches it saw",
             after="search_party_db"),
        Case("CONDITIONAL_LLM_BYPASS", False, "trap", "the choice depends on the search result",
             after="search_party_db")]),
    AgentDef("time-entry", time_entry, 30, MODEL, weekdays_only=True, cases=[
        Case("CACHE_RESULT", True, "positive", "15 timekeepers' rates", tool="get_billing_rate"),
        Case("CACHE_RESULT", False, "trap", "the same entry posts the same reply -- posting is a write",
             tool="post_time_entry")]),
    AgentDef("document-drafting", document_drafting, 2, MODEL, weekdays_only=True),
    AgentDef("deadline-docketing", deadline_docketing, 3, MODEL, weekdays_only=True, cases=[
        Case("CONDITIONAL_LLM_BYPASS", True, "edge", "after the court rules it always computes the deadline -- "
             "but chat logs carry no cost to price it", after="get_court_rules")]),
    AgentDef("discovery-summary", discovery_summary, 0.2, MODEL, weekdays_only=True, cases=[
        Case("CONDITIONAL_LLM_BYPASS", False, "edge", "only a few runs a month", after="load_document_batch")]),
    AgentDef("email-triage", email_triage, 80, MODEL, hours=(7, 21)),
    AgentDef("nda-generation", nda_generation, 1, MODEL, weekdays_only=True, cases=[
        Case("CACHE_RESULT", True, "positive", "3 jurisdictions' templates", tool="render_nda_template")]),
])
