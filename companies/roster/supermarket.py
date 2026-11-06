"""Mid-size supermarket -- Claude Code headless jobs over an ERP MCP server,
claude-sonnet-4-5."""
from __future__ import annotations

from ..engine import AgentDef, Case, Company
from ..model import RunCtx
from .kit import fresh, noisy_result, pool_id, pure, uid, world

MODEL = "claude-sonnet-4-5"
DEPARTMENTS = ["produce", "dairy", "bakery", "meat", "frozen", "pantry", "drinks", "household", "deli", "fish"]


def stock_replenishment(ctx: RunCtx) -> None:
    sku = pool_id(ctx.rng, "sku", 200, 0.7)
    stock = ctx.tool("get_stock", {"sku": sku}, world(ctx, "get_stock", {"sku": sku}, 1800,
                     units=lambda r: r.randint(0, 120)))
    if stock["units"] < 40:
        ctx.tool("create_purchase_order", {"sku": sku, "qty": 48}, {"status": "created"})


def price_markdown(ctx: RunCtx) -> None:
    dept = ctx.rng.choice(DEPARTMENTS)
    ctx.tool("get_expiry_dates", {"department": dept, "date": ctx.now.date().isoformat()}, fresh(ctx))
    ctx.tool("get_sales_velocity", {"department": dept, "hour": ctx.now.hour}, fresh(ctx))
    ctx.tool("compute_markdown", {"department": dept, "batch": uid(ctx, "mk")}, fresh(ctx))


def planogram_check(ctx: RunCtx) -> None:
    aisle = f"aisle-{ctx.rng.randint(1, 30):02d}"
    ctx.tool("get_planogram", {"aisle": aisle}, pure("get_planogram", {"aisle": aisle}))
    ctx.tool("analyze_shelf_photo", {"photo_id": uid(ctx, "img")}, fresh(ctx), latency=(1500, 4000))


def invoice_matching(ctx: RunCtx) -> None:
    invoice = uid(ctx, "inv")
    ctx.tool("get_invoice", {"invoice_id": invoice}, fresh(ctx))
    result = ctx.tool("match_invoice_lines", {"invoice_id": invoice},
                      fresh(ctx, mismatches=lambda r: r.choice([0, 0, 0, 0, 1, 3])))
    if result["mismatches"]:
        ctx.tool("open_dispute", {"invoice_id": invoice}, fresh(ctx))
    else:
        ctx.tool("approve_invoice", {"invoice_id": invoice}, fresh(ctx))


def loyalty_offers(ctx: RunCtx) -> None:
    customer = pool_id(ctx.rng, "loy", 500)
    seg = ctx.tool("get_customer_segment", {"customer_id": customer},
                   pure("get_customer_segment", {"customer_id": customer},
                        segment=lambda r: r.choice(["family", "single", "senior", "student", "premium"])))
    ctx.tool("issue_offer", {"customer_id": customer, "segment": seg["segment"], "week": ctx.now.isocalendar()[1]},
             fresh(ctx))


def customer_chat(ctx: RunCtx) -> None:
    for _ in range(ctx.rng.randint(1, 3)):
        ctx.tool("search_products", {"query": uid(ctx, "q")}, fresh(ctx))
    if ctx.rng.random() < 0.3:
        ctx.tool("get_order_status", {"order_id": uid(ctx, "ord")}, fresh(ctx))


def staff_rostering(ctx: RunCtx) -> None:
    ctx.tool("get_staff_availability", {"week": ctx.now.isocalendar()[1]}, fresh(ctx))
    ctx.tool("publish_roster", {"roster_id": uid(ctx, "ros")}, fresh(ctx))


def demand_forecast(ctx: RunCtx) -> None:
    dept = ctx.rng.choice(DEPARTMENTS)
    args = {"department": dept, "horizon_days": 7}
    ctx.tool("run_forecast_model", args, noisy_result(ctx, 0.7, "run_forecast_model", args), latency=(2000, 5000))
    ctx.tool("save_forecast", {"department": dept, "ref": uid(ctx, "fc")}, fresh(ctx))


def waste_tracking(ctx: RunCtx) -> None:
    ctx.tool("log_waste", {"department": ctx.rng.choice(DEPARTMENTS),
                           "reason": ctx.rng.choice(["expired", "damaged", "recall"])}, {"logged": True})


def promo_compliance(ctx: RunCtx) -> None:
    store = f"store-{ctx.rng.randint(1, 4)}"
    ctx.tool("get_promo_rules", {"store": store, "week": ctx.now.isocalendar()[1]},
             pure("get_promo_rules", {"store": store, "week": ctx.now.isocalendar()[1]}))
    ctx.tool("check_shelf_tags", {"store": store, "scan_id": uid(ctx, "scan")}, fresh(ctx))


COMPANY = Company("supermarket", "Mid supermarket", "claude-code", "erp", [
    AgentDef("stock-replenishment", stock_replenishment, 48, MODEL, hours=(5, 23), cases=[
        Case("CACHE_RESULT", False, "trap", "stock moves every half hour", tool="get_stock"),
        Case("CACHE_RESULT", False, "trap", "creating an order is a write", tool="create_purchase_order")]),
    AgentDef("price-markdown", price_markdown, 24, MODEL, cases=[
        Case("EXTRACT_DETERMINISTIC_SUBSTEP", True, "positive", "always expiry -> velocity -> markdown",
             steps=["get_expiry_dates", "get_sales_velocity", "compute_markdown"])]),
    AgentDef("planogram-check", planogram_check, 12, MODEL, cases=[
        Case("CACHE_RESULT", True, "positive", "30 aisles' planograms", tool="get_planogram")]),
    AgentDef("invoice-matching", invoice_matching, 20, MODEL, weekdays_only=True, cases=[
        Case("REVIEW_HIDDEN_STATE", False, "trap", "dispute vs approve follows the mismatches it saw",
             after="match_invoice_lines"),
        Case("CONDITIONAL_LLM_BYPASS", False, "trap", "the choice depends on the match result",
             after="match_invoice_lines")]),
    AgentDef("loyalty-offers", loyalty_offers, 60, MODEL, cases=[
        Case("CACHE_RESULT", True, "positive", "regulars' segments looked up again", tool="get_customer_segment")]),
    AgentDef("customer-chat", customer_chat, 80, MODEL, hours=(7, 23)),
    AgentDef("staff-rostering", staff_rostering, 2, MODEL, weekdays_only=True),
    AgentDef("demand-forecast", demand_forecast, 24, MODEL, hours=(0, 24), cases=[
        Case("CACHE_RESULT", False, "trap", "the model is re-trained under it: the same request gives a different "
             "forecast 30% of the time", tool="run_forecast_model")]),
    AgentDef("waste-tracking", waste_tracking, 16, MODEL, cases=[
        Case("CACHE_RESULT", False, "trap", "logging is a write", tool="log_waste")]),
    AgentDef("promo-compliance", promo_compliance, 6, MODEL, cases=[
        Case("CONDITIONAL_LLM_BYPASS", True, "positive", "after the promo rules it always checks the shelf tags",
             after="get_promo_rules")]),
])
