# agents_profile_rawdata_gen

Generates synthetic AI agent telemetry (raw trace data) as JSON Lines, for testing an AI agent profiler.
Every line is one event: a decision, tool call, or model call made by an agent during a workflow run.

This generator's source is also vendored into
[`agents_profiler`](https://github.com/ipintotech-lab/agents_profiler)
(`src/rawdata_gen/`, copied rather than imported, since this repo has no
packaging metadata), where it powers that dashboard's "Generate sample
data" tab and `/generate` endpoint — so the pipeline there can be exercised
without a real trace file. Everything below describes using this repo
standalone; the vendored copy is a point-in-time snapshot and does not stay
in sync automatically.

## Mock agents

| # | Agent | Workflow | Description |
|---|-------|----------|-------------|
| 1 | `security-agent` | `cve-remediation-agent` | Classifies CVE severity, routes it, and analyzes remediation |
| 2 | `support-agent` | `support-ticket-triage` | Classifies tickets, retrieves context, routes, and responds |
| 3 | `release-agent` | `deployment-incident-response` | Handles deployment incidents with a rule-based rollback choice |

## Requirements

- Python 3.10 or newer (standard library only, nothing to install)

## How to run

```sh
git clone https://github.com/ipintotech-lab/agents_profile_rawdata_gen.git
cd agents_profile_rawdata_gen
./run.sh
```

The script asks for:

1. **Number of agents** (1–3). Agents are picked in the order shown in the table above.
2. **Number of actions per agent** (4–7). This is the number of events in each of that agent's runs.
3. **Number of lines in raw data** (1 or more). This is the total number of events written.

The output is written to `data/agent_traces.jsonl`. Each run creates the file again, replacing what was there.

### Example

```text
$ ./run.sh
Number of agents to include (1-3): 2
Number of actions for agent 1 (4-7): 4
Number of actions for agent 2 (4-7): 6
Number of lines in raw data: 25
Generated 25 lines from 6 runs across 2 agents: .../data/agent_traces.jsonl
  custom-01-0001 security-agent (cve-remediation-agent): 4 actions
  custom-02-0001 support-agent (support-ticket-triage): 6 actions
  ...
```

To run it without typing the answers (for scripts or CI), pipe them in:

```sh
printf "2\n4\n6\n25\n" | ./run.sh
```

To use a specific Python interpreter:

```sh
PYTHON=/usr/bin/python3.12 ./run.sh
```

### How the line count is filled

Runs are created for each selected agent in turn (agent 1, 2, 3, then agent 1 again, and so on) until the requested number of lines is reached.
If the last run would go over the total, it is cut short.
Each run gets its own ID in the form `custom-<agent>-<cycle>`, for example `custom-01-0003`.

### Default dataset

`python3 generate_traces.py` writes the built-in sample runs for all three agents, with no prompts.

## Using it from Python

```python
from pathlib import Path
from agents.common import write_jsonl
from generate_traces import generate_configured

runs = generate_configured(agent_count=3, actions_per_agent=[4, 5, 7], total_lines=1000)
write_jsonl(runs, Path("data/agent_traces.jsonl"))
```

If you leave out `total_lines`, you get one run per agent.

## Event format

```json
{
  "trace_id": "trace-custom-01-0001",
  "run_id": "custom-01-0001",
  "workflow": "cve-remediation-agent",
  "agent": "security-agent",
  "event_type": "decision",
  "node": "classify_severity",
  "input_state": {"severity": "critical", "fix_available": false},
  "tokens": {"input": 769, "output": 37},
  "latency_ms": 402,
  "cost": 0.0039,
  "result": "success",
  "workflow_version": "v1.0",
  "parent": "custom-01-0001",
  "decision": {"type": "severity_mapping", "selected": "critical"},
  "model": "demo-llm-v1"
}
```

`parent`, `decision`, `tool` (`{"name": ...}`), and `model` only appear on events where they apply.
Prompts and tool arguments are never included.

## Project layout

```text
run.sh                 # launcher script
run_generator.py       # interactive prompts
generate_traces.py     # builds runs and handles the line count
agents/
  common.py            # Run class, event builder, JSONL writer
  cve_remediation.py
  support_triage.py
  deployment_incident.py
data/agent_traces.jsonl  # generated output
```
