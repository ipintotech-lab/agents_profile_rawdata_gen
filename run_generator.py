"""Interactive launcher for generating a configurable set of agent traces."""
from generate_traces import generate_configured
from agents.common import write_jsonl
from pathlib import Path


def ask_int(prompt: str, minimum: int, maximum: int) -> int:
    while True:
        try:
            value = int(input(prompt))
        except ValueError:
            print("Enter a whole number.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"Enter a number from {minimum} to {maximum}.")


def main() -> None:
    count = ask_int("Number of agents to include (1-3): ", 1, 3)
    action_counts = [
        ask_int(f"Number of actions for agent {index} (4-7): ", 4, 7)
        for index in range(1, count + 1)
    ]
    runs = generate_configured(count, action_counts)
    output = Path(__file__).parent / "data" / "agent_traces.jsonl"
    total = write_jsonl(runs, output)
    print(f"Generated {total} events for {count} agents: {output}")
    for run in runs:
        print(f"  {run.agent} ({run.workflow}): {len(run.events)} actions")


if __name__ == "__main__":
    main()
