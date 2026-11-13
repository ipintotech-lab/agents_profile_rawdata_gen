#!/usr/bin/env python3
"""Generate 30 days of five-agent flows, one per company: each run passes
through five of the company's agents in order (see ``companies/flows.py``).

    python3 generate_flows.py --out out-flows
    -> out-flows/traffic/<company>/events-<day>.jsonl   generic event log,
                                                       for profiler-client prepare
    python3 generate_flows.py --format otel
    -> out-flows/traffic/<company>/otlp-traces-<day>.jsonl  OTLP JSON, one trace
                                                       per run, an agent per span
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from companies.flows import FLOWS, flow_runs
from companies.formats import generic, otel
from companies.roster import COMPANIES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("out-flows"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--scale", type=float, default=1.0, help="multiply the flow runs per day")
    parser.add_argument("--company", action="append", choices=sorted(FLOWS), help="default: all five")
    parser.add_argument("--format", choices=("generic", "otel"), default="generic")
    args = parser.parse_args()

    if args.scale <= 0:
        parser.error("--scale must be positive")
    for key in args.company or list(FLOWS):
        flow = FLOWS[key]
        runs = flow_runs(COMPANIES[key], flow, args.seed, args.days, args.scale)
        out = args.out / "traffic" / key
        if out.exists():
            shutil.rmtree(out)
        render = otel.render if args.format == "otel" else generic.render
        files = render(runs, out, COMPANIES[key].server)
        events = sum(1 for r in runs for s in r.steps if not s.implicit)
        print(f"{key:13} {flow.workflow:20} {len(runs):5} runs {events:7} events {len(files):3} files  "
              f"{' -> '.join(flow.agents)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
