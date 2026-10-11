"""Run mock agents and write normalized telemetry as JSON Lines."""
import copy
from pathlib import Path

from agents.common import write_jsonl
from agents.cve_remediation import generate as generate_cve
from agents.deployment_incident import generate as generate_deployment
from agents.support_triage import generate as generate_support


AGENT_GENERATORS = [generate_cve, generate_support, generate_deployment]


def generate_configured(agent_count: int, actions_per_agent: list[int]):
    """Create one run per selected agent with the requested action counts."""
    if not 1 <= agent_count <= len(AGENT_GENERATORS):
        raise ValueError(f"agent_count must be between 1 and {len(AGENT_GENERATORS)}")
    if len(actions_per_agent) != agent_count:
        raise ValueError("provide exactly one action count per agent")
    if any(not 4 <= count <= 7 for count in actions_per_agent):
        raise ValueError("each action count must be between 4 and 7")

    runs = []
    for index, (generator, action_count) in enumerate(zip(AGENT_GENERATORS, actions_per_agent), 1):
        template = generator()[0]
        template.run_id = f"custom-{index:02d}"
        template.trace_id = f"trace-{template.run_id}"
        template.events = template.events[:action_count]
        while len(template.events) < action_count:
            event = copy.deepcopy(template.events[-1])
            event["node"] = f"{event['node']}_followup"
            template.events.append(event)
        for event in template.events:
            event["run_id"] = template.run_id
            event["trace_id"] = template.trace_id
            if "parent" in event:
                event["parent"] = template.run_id
        runs.append(template)
    return runs


def main() -> None:
    runs = generate_cve() + generate_support() + generate_deployment()
    output = Path(__file__).parent / "data" / "agent_traces.jsonl"
    count = write_jsonl(runs, output)
    print(f"Wrote {count} events from {len(runs)} runs to {output}")
    for run in runs:
        print(f"  {run.workflow}/{run.run_id}: {len(run.events)} actions")


if __name__ == "__main__":
    main()
