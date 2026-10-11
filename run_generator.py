"""Interactive launcher for generating a configurable set of agent traces."""
from generate_traces import generate_configured
from agents.common import write_jsonl
from pathlib import Path


def ask_int(prompt: str, minimum: int, maximum: int | None = None) -> int:
    while True:
        try:
            value = int(input(prompt))
        except ValueError:
            print("Enter a whole number.")
            continue
        if value >= minimum and (maximum is None or value <= maximum):
            return value
        if maximum is None:
            print(f"Enter a number of at least {minimum}.")
        else:
            print(f"Enter a number from {minimum} to {maximum}.")


def main() -> None:
    count = ask_int("Number of agents to include (1-3): ", 1, 3)
    action_counts = [
        ask_int(f"Number of actions for agent {index} (4-7): ", 4, 7)
        for index in range(1, count + 1)
    ]
    total_lines = ask_int("Number of lines in raw data: ", 1)
    runs = generate_configured(count, action_counts, total_lines)
    output = Path(__file__).parent / "data" / "agent_traces.jsonl"
    total = write_jsonl(runs, output)
    print(f"Generated {total} lines from {len(runs)} runs across {count} agents: {output}")
    for run in runs[:10]:
        print(f"  {run.run_id} {run.agent} ({run.workflow}): {len(run.events)} actions")
    if len(runs) > 10:
        print(f"  ... and {len(runs) - 10} more runs")


if __name__ == "__main__":
    main()
