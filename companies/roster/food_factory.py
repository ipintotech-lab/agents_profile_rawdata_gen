"""Food factory -- the plant's own instrumented event log, claude-haiku-4-5.

The only company whose log records each LLM decision's input and choice,
the workflow version and declared side effects -- so rule-like decisions,
hidden state, deterministic regions and a mid-month release are visible.
"""
from __future__ import annotations

from ..engine import AgentDef, Case, Company
from ..model import RunCtx, noisy, stable
from .kit import fresh, pool_id, pure, uid, world

MODEL = "claude-haiku-4-5"
FAMILIES = ["bakery", "dairy", "snacks", "sauces", "frozen"]
LINES = ["line-1", "line-2", "line-3", "line-4"]


def production_scheduling(ctx: RunCtx) -> None:
    order = uid(ctx, "ord")
    ctx.tool("get_order", {"order_id": order}, fresh(ctx))
    family, volume = ctx.rng.choice(FAMILIES), ctx.rng.choice(["small", "medium", "large"])
    line = LINES[(FAMILIES.index(family) + ["small", "medium", "large"].index(volume)) % 4]
    ctx.llm("assign_line", {"product_family": family, "volume_band": volume}, line)
    ctx.tool("book_line_slot", {"order_id": order, "line": line}, {"booked": True}, side_effect=True)


def quality_inspection(ctx: RunCtx) -> None:
    batch = uid(ctx, "bat")
    defect = ctx.rng.choice(["<1%"] * 3 + ["1-3%"] * 2 + [">3%"])
    ctx.tool("measure_batch", {"batch_id": batch}, fresh(ctx, defect_band=defect))
    grade = "pass" if defect == "<1%" else noisy(ctx.rng, 0.6, "hold" if defect == "1-3%" else "reject",
                                                ["pass", "hold", "reject"])
    ctx.llm("grade_batch", {"batch_id": batch, "defect_band": defect, "line": ctx.rng.choice(LINES)}, grade)


def recipe_scaling(ctx: RunCtx) -> None:
    recipe = pool_id(ctx.rng, "rcp", 25, 0.8)
    ctx.tool("load_recipe", {"recipe_id": recipe}, pure("load_recipe", {"recipe_id": recipe}))
    factor = round(ctx.rng.uniform(0.5, 8), 2)
    ctx.tool("scale_quantities", {"recipe_id": recipe, "factor": factor}, fresh(ctx))
    ctx.tool("convert_units", {"recipe_id": recipe, "factor": factor, "system": "metric"}, fresh(ctx))
    ingredient = ctx.rng.choice(["butter", "milk", "eggs", "sugar", "flour", "cocoa", "yeast", "salt"])
    shortage = ctx.rng.random() < 0.5
    # release 1.1.0 (day 15) changed the substitution rule
    table = "a" if ctx.run.version == "1.0.0" else "b"
    choice = stable("substitute", table, ingredient, shortage).choice(["none", "sub-1", "sub-2", "sub-3"])
    ctx.llm("choose_substitution", {"ingredient": ingredient, "shortage": shortage}, choice)


def supplier_ordering(ctx: RunCtx) -> None:
    sku = pool_id(ctx.rng, "sku", 10, 0.5)
    ctx.tool("get_supplier_quote", {"sku": sku}, world(ctx, "get_supplier_quote", {"sku": sku}, 86400))
    supplier = ctx.rng.choice(["acme-foods", "nordic-grain", "valley-dairy"])
    ctx.llm("choose_supplier", {"sku": sku, "quote_ref": uid(ctx, "q")}, supplier)
    ctx.tool("create_purchase_order", {"sku": sku, "qty": ctx.rng.choice([100, 200, 500])},
             {"status": "accepted"}, side_effect=True)


def haccp_log(ctx: RunCtx) -> None:
    zone = ctx.rng.choice(["chiller-a", "chiller-b", "freezer-1", "freezer-2", "oven-hall", "packing"])
    reading = ctx.tool("read_probe", {"zone": zone}, world(ctx, "read_probe", {"zone": zone}, 300,
                       high=lambda r: r.random() < 0.15))
    ctx.tool("record_temperature", {"zone": zone, "reading_ref": reading["ref"]}, {"recorded": True},
             side_effect=True)
    expected = "flag" if reading["high"] else "none"
    ctx.llm("flag_deviation", {"zone": zone, "reading_band": "high" if reading["high"] else "ok"},
            noisy(ctx.rng, 0.98, expected, ["flag", "none"]))


def predictive_maintenance(ctx: RunCtx) -> None:
    machine = pool_id(ctx.rng, "mch", 12, 0.3)
    vib = ctx.tool("read_vibration", {"machine_id": machine}, world(ctx, "read_vibration", {"machine_id": machine},
                   3600, band=lambda r: r.choice(["low", "low", "medium", "high"])))
    expected = {"low": "ok", "medium": "watch", "high": "service"}[vib["band"]]
    verdict = noisy(ctx.rng, 0.8, expected, ["ok", "watch", "service"])
    ctx.llm("assess_vibration", {"machine_id": machine, "vibration_band": vib["band"]}, verdict)
    if verdict == "service":
        ctx.tool("schedule_maintenance", {"machine_id": machine}, fresh(ctx), side_effect=True)


def batch_traceability(ctx: RunCtx) -> None:
    lot = pool_id(ctx.rng, "lot", 80)
    if ctx.rng.random() < 0.06:  # the ERP times out; the agent retries the same call
        ctx.tool("lookup_lot", {"lot_id": lot}, None, error="timeout after 30s", latency=(30000, 30100))
    ctx.tool("lookup_lot", {"lot_id": lot}, pure("lookup_lot", {"lot_id": lot}))
    ctx.llm("summarize_trace", {"lot_id": lot, "request_ref": uid(ctx, "req")}, "summary")


def shipping_labels(ctx: RunCtx) -> None:
    ctx.tool("get_shipment", {"shipment_id": uid(ctx, "shp")}, fresh(ctx))
    ctx.tool("print_label", {"carrier": ctx.rng.choice(["dhl", "ups", "fedex"]),
                             "service": ctx.rng.choice(["express", "ground"]),
                             "pallets": ctx.rng.randint(1, 4)}, {"printed": True})
    ctx.llm("confirm_dispatch", {"shipment_ref": uid(ctx, "s")}, "dispatched")


def inventory_reorder(ctx: RunCtx) -> None:
    sku = pool_id(ctx.rng, "ing", 30, 0.5)
    level = ctx.tool("get_stock_level", {"sku": sku}, world(ctx, "get_stock_level", {"sku": sku}, 3600,
                     units=lambda r: r.randint(0, 900)))
    ctx.llm("decide_reorder", {"sku": sku, "units": level["units"]}, "reorder" if level["units"] < 200 else "hold")


def shift_report(ctx: RunCtx) -> None:
    shift = ctx.rng.choice(["A", "B", "C"])
    metrics = ctx.tool("get_shift_metrics", {"shift": shift, "date": ctx.now.date().isoformat()}, fresh(ctx))
    ctx.llm("write_summary", {"shift": shift, "metrics_ref": metrics["ref"]}, "summary",
            tokens=(2400, 600))


COMPANY = Company("food_factory", "Food factory", "generic", "plant-mes", [
    AgentDef("production-scheduling", production_scheduling, 40, MODEL, weekdays_only=True, cases=[
        Case("CONDITIONAL_LLM_BYPASS", True, "positive", "the line is a fixed function of family and volume",
             llm="assign_line")]),
    AgentDef("quality-inspection", quality_inspection, 80, MODEL, hours=(6, 22), cases=[
        Case("DETERMINISTIC_REGION", True, "positive", "batches under 1% defects always pass; the rest flip 40%",
             llm="grade_batch", field="defect_band"),
        Case("CONDITIONAL_LLM_BYPASS", False, "trap", "every batch is new, and outside the region it's noisy",
             llm="grade_batch")]),
    AgentDef("recipe-scaling", recipe_scaling, 30, MODEL, versions=[(0, "1.0.0"), (15, "1.1.0")], cases=[
        Case("EXTRACT_DETERMINISTIC_SUBSTEP", False, "edge", "always load -> scale -> convert, but the event log "
             "records no LLM call between the steps, so scripting them saves nothing",
             steps=["load_recipe", "scale_quantities", "convert_units"]),
        Case("CACHE_RESULT", True, "positive", "25 recipes loaded again and again", tool="load_recipe"),
        Case("CONDITIONAL_LLM_BYPASS", True, "positive", "a fixed table per version", llm="choose_substitution"),
        Case("REVIEW_HIDDEN_STATE", False, "edge", "the answer changed with release 1.1.0 -- the version explains "
             "it, nothing hidden", llm="choose_substitution")]),
    AgentDef("supplier-ordering", supplier_ordering, 20, MODEL, weekdays_only=True, cases=[
        Case("CACHE_RESULT", False, "trap", "same order, same reply -- declared a write", tool="create_purchase_order")]),
    AgentDef("haccp-log", haccp_log, 96, MODEL, hours=(0, 24), cases=[
        Case("REVIEW_HIDDEN_STATE", True, "positive", "2% of identical readings get the other verdict",
             llm="flag_deviation"),
        Case("CONDITIONAL_LLM_BYPASS", False, "trap", "98% consistent is not a rule", llm="flag_deviation")]),
    AgentDef("predictive-maintenance", predictive_maintenance, 48, MODEL, hours=(0, 24), cases=[
        Case("REVIEW_HIDDEN_STATE", True, "positive", "the same band gets another verdict 20% of the time",
             llm="assess_vibration"),
        Case("CONDITIONAL_LLM_BYPASS", False, "trap", "noisy -- not a rule", llm="assess_vibration")]),
    AgentDef("batch-traceability", batch_traceability, 25, MODEL, cases=[
        Case("CACHE_RESULT", True, "edge", "80 lots looked up again; 6% time out first and are retried",
             tool="lookup_lot")]),
    AgentDef("shipping-labels", shipping_labels, 60, MODEL, weekdays_only=True, cases=[
        Case("CACHE_RESULT", False, "trap", "the same label prints the same reply, but printing is a write its "
             "name doesn't reveal (and nothing is declared)", tool="print_label")]),
    AgentDef("inventory-reorder", inventory_reorder, 24, MODEL, hours=(0, 24), cases=[
        Case("CACHE_RESULT", False, "trap", "stock moves every hour", tool="get_stock_level")]),
    AgentDef("shift-report", shift_report, 3, MODEL, hours=(6, 23), cases=[
        Case("CONDITIONAL_LLM_BYPASS", False, "trap", "a written summary, new every time", llm="write_summary"),
        Case("CACHE_RESULT", False, "trap", "every shift's metrics are new", tool="get_shift_metrics")]),
])
