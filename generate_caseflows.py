#!/usr/bin/env python3
"""Generate 30 days of the eight-agent loan-application flow (see
``companies/caseflows.py``): a supervisor routes each case through KYC,
document collection, parallel fraud and credit checks, pricing, compliance
review and a decision letter.

    python3 generate_caseflows.py --out out-caseflows
    -> out-caseflows/traffic/fintech/events-<day>.jsonl   generic event log, with
                                                         event ids, parents and triggers
    python3 generate_caseflows.py --format otel
    -> out-caseflows/traffic/fintech/otlp-traces-<day>.jsonl  OTLP JSON, one trace per
                                                         run, a span per agent visit
       out-caseflows/truth.json                          the answer key -- never upload it
    python3 generate_caseflows.py --noise 0.1
    -> the same flow with 10% of tool calls timing out or failing once, then retried
       with the same arguments; every expected value in truth.json stays the same
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from companies.caseflows import AGENTS, CASES, COMPANY, SERVER, WORKFLOW, caseflow_runs
from companies.formats import caseflow
from companies.truth import caseflow_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("out-caseflows"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--scale", type=float, default=1.0, help="multiply the runs per day")
    parser.add_argument("--format", choices=("generic", "otel"), default="generic")
    parser.add_argument("--noise", type=float, default=0.0, metavar="RATE",
                        help="share of tool calls that time out or fail once, then are retried with the "
                             "same arguments (default 0)")
    args = parser.parse_args()

    if args.scale <= 0:
        parser.error("--scale must be positive")
    if not 0 <= args.noise < 1:
        parser.error("--noise must be at least 0 and below 1")
    runs = caseflow_runs(args.seed, args.days, args.scale, args.noise)
    out = args.out / "traffic" / COMPANY
    if out.exists():
        shutil.rmtree(out)
    if args.format == "otel":
        files = caseflow.render_otel(runs, out, SERVER)
    else:
        files = caseflow.render_generic(runs, out)
    events = sum(1 for r in runs for s in r.steps if args.format != "generic" or not s.implicit)
    truth = {"seed": args.seed, "days": args.days, "scale": args.scale, "noise": args.noise, "format": args.format,
             "companies": {COMPANY: {"format": args.format, "server": SERVER, "workflow": WORKFLOW,
                                     "runs": len(runs), "events": events, "files": len(files),
                                     "agents": list(AGENTS)}},
             "cases": caseflow_rows(CASES, runs, args.format, COMPANY, WORKFLOW)}
    (args.out / "truth.json").write_text(json.dumps(truth, indent=1) + "\n")
    print(f"{COMPANY} {WORKFLOW} {args.format}: {len(runs)} runs {events} events {len(files)} files, "
          f"{len(truth['cases'])} planted cases -> {args.out / 'truth.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
