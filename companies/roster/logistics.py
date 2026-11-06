"""Logistics / delivery -- Codex CLI sessions over a TMS MCP server, gpt-5.

Codex records no cost, so rule-like choices here are ``edge`` cases.
"""
from __future__ import annotations

from ..engine import AgentDef, Case, Company
from ..model import RunCtx
from .kit import fresh, pool_id, pure, uid, world

MODEL = "gpt-5"
DEPOTS = ["north", "south", "east", "west", "port"]


def dispatch_assignment(ctx: RunCtx) -> None:
    depot = ctx.rng.choice(DEPOTS)
    ctx.tool("get_open_orders", {"depot": depot}, world(ctx, "get_open_orders", {"depot": depot}, 300))
    ctx.tool("get_driver_locations", {"depot": depot}, world(ctx, "get_driver_locations", {"depot": depot}, 60))
    ctx.tool("assign_driver", {"order_id": uid(ctx, "ord"), "driver": uid(ctx, "drv")}, fresh(ctx))


def route_optimization(ctx: RunCtx) -> None:
    route = uid(ctx, "rte")
    ctx.tool("get_stops", {"route_id": route}, fresh(ctx))
    ctx.tool("geocode_batch", {"route_id": route}, fresh(ctx))
    ctx.tool("solve_vrp", {"route_id": route}, fresh(ctx), latency=(3000, 9000))
    for _ in range(ctx.rng.randint(1, 3)):  # addresses the batch couldn't resolve
        address = pool_id(ctx.rng, "addr", 300)
        ctx.tool("geocode", {"address": address}, pure("geocode", {"address": address}))


def shipment_tracking(ctx: RunCtx) -> None:
    shipment = pool_id(ctx.rng, "shp", 1500, 0.8)
    ctx.tool("get_tracking_status", {"shipment_id": shipment},
             world(ctx, "get_tracking_status", {"shipment_id": shipment}, 1800))


def delivery_exception(ctx: RunCtx) -> None:
    case = uid(ctx, "exc")
    detail = ctx.tool("get_exception_details", {"exception_id": case},
                      fresh(ctx, kind=lambda r: r.choice(["not_home", "not_home", "damaged", "refused"])))
    if detail["kind"] == "not_home":
        ctx.tool("reschedule_delivery", {"exception_id": case}, fresh(ctx))
    else:
        ctx.tool("create_return", {"exception_id": case}, fresh(ctx))


def customs_docs(ctx: RunCtx) -> None:
    product = pool_id(ctx.rng, "prd", 60, 0.8)
    ctx.tool("get_hs_code", {"product_id": product}, pure("get_hs_code", {"product_id": product}))
    ctx.tool("generate_invoice_pdf", {"shipment_id": uid(ctx, "shp")}, fresh(ctx))


def proof_of_delivery(ctx: RunCtx) -> None:
    stop = uid(ctx, "stp")
    ctx.tool("upload_pod_photo", {"stop_id": stop}, {"stored": True})
    if ctx.rng.random() < 0.25:  # a flaky connection: the driver app uploads again
        ctx.tool("upload_pod_photo", {"stop_id": stop}, {"stored": True})


def carrier_rate_quote(ctx: RunCtx) -> None:
    lane = pool_id(ctx.rng, "lane", 20, 0.5)
    ctx.tool("get_carrier_rates", {"lane": lane}, world(ctx, "get_carrier_rates", {"lane": lane}, 3600))


def warehouse_slotting(ctx: RunCtx) -> None:
    ctx.tool("get_pick_frequency", {"warehouse": "main", "day": ctx.now.date().isoformat()}, fresh(ctx))
    ctx.tool("propose_slotting", {"plan_id": uid(ctx, "slt")}, fresh(ctx))


def driver_hours_compliance(ctx: RunCtx) -> None:
    driver = pool_id(ctx.rng, "drv", 120, 0.3)
    ctx.tool("get_driver_log", {"driver_id": driver, "date": ctx.now.date().isoformat()}, fresh(ctx))
    ctx.tool("check_hos_rules", {"driver_id": driver}, fresh(ctx))


def eta_notification(ctx: RunCtx) -> None:
    ctx.tool("send_sms", {"template": ctx.rng.choice(["eta-30", "eta-60", "arriving"]),
                          "lang": ctx.rng.choice(["en", "he", "ar"])}, {"queued": True})


COMPANY = Company("logistics", "Logistics / delivery", "codex", "tms", [
    AgentDef("dispatch-assignment", dispatch_assignment, 75, MODEL, hours=(5, 22), cases=[
        Case("CONDITIONAL_LLM_BYPASS", True, "edge", "after the open orders it always fetches driver locations -- "
             "but Codex records no cost to price it", after="get_open_orders")]),
    AgentDef("route-optimization", route_optimization, 20, MODEL, hours=(4, 10), cases=[
        Case("EXTRACT_DETERMINISTIC_SUBSTEP", True, "positive", "always stops -> geocode batch -> solve",
             steps=["get_stops", "geocode_batch", "solve_vrp"]),
        Case("CACHE_RESULT", True, "positive", "the same 300 addresses geocoded again", tool="geocode")]),
    AgentDef("shipment-tracking", shipment_tracking, 100, MODEL, hours=(0, 24), cases=[
        Case("CACHE_RESULT", False, "trap", "status moves every half hour", tool="get_tracking_status")]),
    AgentDef("delivery-exception", delivery_exception, 15, MODEL, cases=[
        Case("REVIEW_HIDDEN_STATE", False, "trap", "reschedule vs return follows the exception kind",
             after="get_exception_details"),
        Case("CONDITIONAL_LLM_BYPASS", False, "trap", "the choice depends on the exception",
             after="get_exception_details")]),
    AgentDef("customs-docs", customs_docs, 8, MODEL, weekdays_only=True, cases=[
        Case("CACHE_RESULT", True, "positive", "60 products' HS codes", tool="get_hs_code")]),
    AgentDef("proof-of-delivery", proof_of_delivery, 90, MODEL, hours=(7, 21), cases=[
        Case("CACHE_RESULT", False, "trap", "re-uploads repeat the same input and reply, but uploading is a write "
             "its name doesn't reveal", tool="upload_pod_photo")]),
    AgentDef("carrier-rate-quote", carrier_rate_quote, 30, MODEL, weekdays_only=True, cases=[
        Case("CACHE_RESULT", False, "trap", "rates change every hour", tool="get_carrier_rates")]),
    AgentDef("warehouse-slotting", warehouse_slotting, 4, MODEL, weekdays_only=True),
    AgentDef("driver-hours-compliance", driver_hours_compliance, 25, MODEL, cases=[
        Case("CONDITIONAL_LLM_BYPASS", True, "edge", "after the driver log it always checks the HOS rules -- "
             "no cost in Codex", after="get_driver_log")]),
    AgentDef("eta-notification", eta_notification, 150, MODEL, hours=(7, 21), cases=[
        Case("CACHE_RESULT", False, "trap", "sending is a write", tool="send_sms")]),
])
