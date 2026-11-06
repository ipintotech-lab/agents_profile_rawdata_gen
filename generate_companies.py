#!/usr/bin/env python3
"""Generate 30 days of traffic for five synthetic companies (ten agents
each, one trace format per company) with planted cases, plus the answer key.

    python3 generate_companies.py --out out
    python3 generate_companies.py --out out-5x --scale 5   # 5x the runs, same month
    -> out/traffic/<company>/...   the files to feed profiler-client prepare
       out/truth.json              the answer key -- never upload it
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from companies.engine import company_runs
from companies.formats import chat, claude_code, codex, generic, otel
from companies.roster import COMPANIES
from companies.truth import case_rows

RENDERERS = {"generic": generic, "otel": otel, "openai-chat": chat, "claude-code": claude_code, "codex": codex}


def events(fmt: str, runs) -> int:
    return sum(1 for r in runs for s in r.steps if fmt != "generic" or not s.implicit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--scale", type=float, default=1.0, help="multiply every agent's runs per day (e.g. 2, 5)")
    parser.add_argument("--company", action="append", choices=sorted(COMPANIES), help="default: all five")
    args = parser.parse_args()

    if args.scale <= 0:
        parser.error("--scale must be positive")
    truth = {"seed": args.seed, "days": args.days, "scale": args.scale, "companies": {}, "cases": []}
    for key in args.company or list(COMPANIES):
        company = COMPANIES[key]
        runs = company_runs(company, args.seed, args.days, args.scale)
        out = args.out / "traffic" / key
        if out.exists():
            shutil.rmtree(out)
        files = RENDERERS[company.format].render(runs, out, company.server)
        count = events(company.format, runs)
        truth["companies"][key] = {
            "name": company.name, "format": company.format, "server": company.server, "runs": len(runs),
            "events": count, "files": len(files),
            "agents": {a.name: sum(r.agent == a.name for r in runs) for a in company.agents},
        }
        truth["cases"] += case_rows(company, runs)
        print(f"{key:13} {company.format:12} {len(runs):6} runs {count:7} events {len(files):4} files",
              file=sys.stderr)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "truth.json").write_text(json.dumps(truth, indent=1) + "\n")
    print(f"{len(truth['cases'])} planted cases -> {args.out / 'truth.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
