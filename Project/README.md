# Log File Anomaly Explainer — Technical Documentation

A CLI tool that ingests `.log` files, detects anomalies (errors, crashes,
timeouts), extracts contextual log windows, and uses **Groq Cloud** (Llama 3.3 70B)
to explain the root cause with actionable remediation steps.

Built for on-call engineers who need to understand incidents *fast*.

---

## Features

- **Smart anomaly detection** — ranked severity scoring across 10+ error
  patterns (CRITICAL, FATAL, Traceback, HTTP 5xx, OOM, timeouts, etc.)
- **Context-aware extraction** — pulls ±20 lines around the primary error
  so the LLM has full system-state awareness
- **Structured LLM analysis** — Root Cause, Probable Cause, Remediation
  Steps, and Confidence level
- **Tool-using agent mode** — LLM autonomously calls tools to investigate
- **Multiple output formats** — Rich console, JSON, or Markdown reports
- **Scan mode** — quick anomaly inventory without burning an LLM call
- **Multi-anomaly support** — analyse all distinct error clusters in one run

---

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A free [Groq Cloud API key](https://console.groq.com/keys) (no credit card needed)

## Installation

```bash
# From this directory (Project/)
uv sync

# Or with pip
pip install -e .
```

## Configuration

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your-key
```

---

## Usage

### Analyse a log file (primary command)

```bash
uv run log-explainer explain sample_data/Thunderbird_2k.log
uv run log-explainer explain app.log --context 30
uv run log-explainer explain app.log --output json --out-file result.json
uv run log-explainer explain app.log --output markdown --out-file report.md
uv run log-explainer explain app.log --all-anomalies
```

### Agent investigation mode (tool-using agent)

```bash
uv run log-explainer investigate app.log
uv run log-explainer investigate app.log --query "focus on database errors"
uv run log-explainer investigate app.log --output json --out-file report.json
```

### Quick scan (no LLM call)

```bash
uv run log-explainer scan app.log
```

---

## Project Structure

```
Project/
├── src/
│   ├── __init__.py          # Package init
│   ├── cli.py               # Click CLI (explain, scan, investigate commands)
│   ├── parser.py            # Log scanning and context extraction
│   ├── llm_client.py        # Groq Cloud API integration (direct mode)
│   ├── tools.py             # Tool functions for the agent
│   └── agent.py             # Tool-using agent loop
├── tests/
│   ├── fixtures/            # Sample log files
│   ├── test_parser.py       # Parser unit tests
│   ├── test_llm_client.py   # LLM client tests (mocked)
│   ├── test_tools.py        # Tool function tests
│   └── test_agent.py        # Agent loop tests (mocked)
├── sample_data/             # Real-world sample log files
├── pyproject.toml           # Project config & dependencies
├── .env.example             # API key template
└── README.md                # This file
```

---

## How It Works

```
┌──────────┐     ┌───────────────┐     ┌───────────────┐     ┌──────────────┐
│ .log file│────>│  Log Parser   │────>│ Context Block │────>│  Groq Cloud  │
│          │     │ (severity     │     │ (error + ±20  │     │ (Llama 3.3   │
│          │     │  ranked scan) │     │  lines)       │     │  70B)        │
└──────────┘     └───────────────┘     └───────────────┘     └──────┬───────┘
                                                                    │
                                                          ┌─────────v─────────┐
                                                          │ Rich Console /    │
                                                          │ JSON / Markdown   │
                                                          └───────────────────┘
```

### Agent Mode (investigate)

```
User → "investigate app.log"
  │
  ▼  Agent Loop (max 10 iterations)
  LLM decides → calls scan_all_anomalies() → gets overview
  LLM decides → calls parse_log_file()     → gets context
  LLM decides → calls search_log_pattern() → finds patterns
  LLM decides → produces final analysis
  │
  ▼  Root Cause / Probable Cause / Remediation / Confidence
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

---

## License

MIT
