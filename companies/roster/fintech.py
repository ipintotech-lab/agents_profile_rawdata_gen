"""Fintech website -- OpenTelemetry GenAI traces, gpt-5-mini."""
from __future__ import annotations

from ..engine import AgentDef, Case, Company
from ..model import RunCtx
from .kit import amount, band, fresh, noisy_result, pool_id, pure, uid, world

MODEL = "gpt-5-mini"
CATEGORIES = ["groceries", "travel", "dining", "utilities", "entertainment", "transport", "health"]


def kyc_verification(ctx: RunCtx) -> None:
    customer = pool_id(ctx.rng, "nid", 400)
    ctx.tool("kyc_lookup", {"national_id": customer}, pure("kyc_lookup", {"national_id": customer},
             status=lambda r: r.choice(["clear", "clear", "clear", "review"])), latency=(600, 1800))
    ctx.tool("verify_document", {"upload_id": uid(ctx, "doc")}, fresh(ctx, match=lambda r: r.random() > 0.05))


def fraud_screening(ctx: RunCtx) -> None:
    txn = uid(ctx, "txn")
    ctx.tool("get_transaction", {"transaction_id": txn}, fresh(ctx))
    features = {"merchant_category": ctx.rng.choice(CATEGORIES),
                "amount_band": band(amount(ctx.rng, 60), [20, 100, 500], ["<20", "20-100", "100-500", ">500"])}
    score = noisy_result(ctx, 0.85, "risk_score", features, score=lambda r: round(r.random(), 3))
    ctx.tool("risk_score", features, score)
    if score["score"] > 0.7:
        ctx.tool("hold_transaction", {"transaction_id": txn}, {"held": True})
    else:
        ctx.tool("approve_transaction", {"transaction_id": txn}, {"approved": True})


def loan_prequal(ctx: RunCtx) -> None:
    applicant = uid(ctx, "app")
    report = ctx.tool("get_credit_report", {"applicant_id": applicant},
                      fresh(ctx, score=lambda r: r.randint(520, 830)), latency=(800, 2500))
    ctx.tool("compute_dti", {"applicant_id": applicant, "income": amount(ctx.rng, 5200)},
             fresh(ctx, dti=lambda r: round(r.uniform(0.1, 0.6), 2)))
    ctx.tool("check_policy_limits", {"score_band": band(report["score"], [620, 700, 760], ["d", "c", "b", "a"])},
             fresh(ctx, eligible=lambda r: r.random() > 0.3))


def support_chat(ctx: RunCtx) -> None:
    query = pool_id(ctx.rng, "faq", 40, 1.0)
    ctx.tool("search_help_center", {"query": query}, pure("search_help_center", {"query": query}))
    if ctx.rng.random() < 0.5:
        ctx.tool("get_account_summary", {"user_id": uid(ctx, "usr")}, fresh(ctx))


def payment_dispute(ctx: RunCtx) -> None:
    dispute = uid(ctx, "dsp")
    ctx.tool("get_dispute", {"dispute_id": dispute}, fresh(ctx))
    reason = ctx.rng.choice(["4837", "4853", "4863", "13.1", "10.4"])
    ctx.tool("submit_chargeback_response", {"reason_code": reason, "template": f"rebuttal-{reason}"},
             {"status": "submitted"})


def transaction_categorizer(ctx: RunCtx) -> None:
    merchant = pool_id(ctx.rng, "mer", 150)
    info = ctx.tool("get_merchant", {"merchant_id": merchant},
                    pure("get_merchant", {"merchant_id": merchant}, mcc=lambda r: r.choice(CATEGORIES)))
    ctx.tool("assign_category", {"transaction_id": uid(ctx, "txn"), "category": info["mcc"]}, {"ok": True})


def aml_report(ctx: RunCtx) -> None:
    ctx.tool("get_flagged_accounts", {"window_days": 30}, fresh(ctx, count=lambda r: r.randint(1, 9)))
    ctx.tool("compile_sar_draft", {"case_id": uid(ctx, "aml")}, fresh(ctx))


def onboarding_email(ctx: RunCtx) -> None:
    segment = ctx.rng.choice(["student", "salaried", "freelancer", "business"])
    ctx.tool("render_template", {"template": "welcome", "segment": segment},
             pure("render_template", {"segment": segment}))
    ctx.tool("send_welcome_email", {"customer_id": uid(ctx, "cus"), "segment": segment}, {"queued": True})


def fx_quote(ctx: RunCtx) -> None:
    pair = ctx.rng.choice(["EURUSD", "GBPUSD", "USDJPY", "USDILS", "EURILS", "USDCHF"])
    ctx.tool("get_fx_rate", {"pair": pair}, world(ctx, "get_fx_rate", {"pair": pair}, 60,
             rate=lambda r: round(r.uniform(0.5, 1.5), 5)))


def card_activation(ctx: RunCtx) -> None:
    card = pool_id(ctx.rng, "card", 1500, 0.6)
    ctx.tool("get_card_status", {"card_id": card}, fresh(ctx, status="inactive"))
    ctx.tool("activate_card", {"card_id": card}, {"status": "active"})
    if ctx.rng.random() < 0.3:  # the customer taps "activate" again
        ctx.tool("activate_card", {"card_id": card}, {"status": "active"})


COMPANY = Company("fintech", "Fintech website", "otel", "payments-api", [
    AgentDef("kyc-verification", kyc_verification, 40, MODEL, cases=[
        Case("CACHE_RESULT", True, "positive", "repeat customers: the same national id returns the same KYC record",
             tool="kyc_lookup")]),
    AgentDef("fraud-screening", fraud_screening, 120, MODEL, cases=[
        Case("CACHE_RESULT", False, "trap", "15% A/B rollout: the same features score differently",
             tool="risk_score"),
        Case("REVIEW_HIDDEN_STATE", False, "trap", "the router's choice follows the score it saw; the trace "
             "records only part of the LLM input", after="risk_score"),
        Case("CONDITIONAL_LLM_BYPASS", False, "trap", "hold vs approve depends on the score", after="risk_score")]),
    AgentDef("loan-prequal", loan_prequal, 25, MODEL, weekdays_only=True, cases=[
        Case("EXTRACT_DETERMINISTIC_SUBSTEP", True, "positive", "always credit report -> DTI -> policy limits",
             steps=["get_credit_report", "compute_dti", "check_policy_limits"]),
        Case("CACHE_RESULT", False, "trap", "every applicant is new", tool="get_credit_report")]),
    AgentDef("support-chat", support_chat, 100, MODEL, hours=(6, 23), cases=[
        Case("CACHE_RESULT", True, "positive", "40 FAQ topics answer most questions", tool="search_help_center")]),
    AgentDef("payment-dispute", payment_dispute, 15, MODEL, weekdays_only=True, cases=[
        Case("CACHE_RESULT", False, "trap", "same reason code, same reply -- but submitting is a write",
             tool="submit_chargeback_response")]),
    AgentDef("transaction-categorizer", transaction_categorizer, 150, MODEL, hours=(0, 24), cases=[
        Case("CACHE_RESULT", True, "positive", "150 merchants, looked up again and again", tool="get_merchant"),
        Case("CONDITIONAL_LLM_BYPASS", True, "positive", "after the merchant lookup it always assigns the category",
             after="get_merchant")]),
    AgentDef("aml-report", aml_report, 0.1, MODEL, rare=True, cases=[
        Case("CONDITIONAL_LLM_BYPASS", False, "edge", "consistent, but only ~3 runs a month: too few to trust",
             after="get_flagged_accounts")]),
    AgentDef("onboarding-email", onboarding_email, 30, MODEL, cases=[
        Case("CACHE_RESULT", True, "positive", "4 segments, the same rendered template", tool="render_template"),
        Case("CACHE_RESULT", False, "trap", "sending is a write (and every customer is new)",
             tool="send_welcome_email")]),
    AgentDef("fx-quote", fx_quote, 100, MODEL, hours=(0, 24), cases=[
        Case("CACHE_RESULT", False, "trap", "the rate moves every minute", tool="get_fx_rate")]),
    AgentDef("card-activation", card_activation, 60, MODEL, cases=[
        Case("EXTRACT_DETERMINISTIC_SUBSTEP", True, "positive", "always status -> activate",
             steps=["get_card_status", "activate_card"]),
        Case("CACHE_RESULT", False, "trap", "repeated with the same card and output, but activating is a write "
             "its name doesn't reveal", tool="activate_card")]),
])
