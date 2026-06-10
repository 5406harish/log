"""
Agent Module
============
Implements a tool-using agent loop where the LLM iteratively calls tools
(log parser, line reader, pattern searcher) to investigate anomalies before
producing a final structured analysis.

Uses the Groq Cloud API (free tier, Llama 3.3 70B) via the OpenAI-compatible SDK.
"""

from __future__ import annotations

import json
import os
import textwrap
import time
from typing import Any, Callable, Optional

from openai import OpenAI
from dotenv import load_dotenv

from src.tools import TOOL_REGISTRY
from src.llm_client import _call_with_retry

load_dotenv()

# ---------------------------------------------------------------------------
# Maximum iterations to prevent infinite loops
# ---------------------------------------------------------------------------
MAX_AGENT_ITERATIONS = 10

# ---------------------------------------------------------------------------
# Agent system instruction
# ---------------------------------------------------------------------------
AGENT_SYSTEM_INSTRUCTION = textwrap.dedent("""\
    You are an expert Site Reliability Engineer (SRE) agent with access to log
    analysis tools.  Your job is to investigate log files to find and explain
    anomalies.

    **Available tools:**
    1. **parse_log_file** — Parse a log file to find the primary (highest-severity)
       anomaly and extract surrounding context lines.
    2. **scan_all_anomalies** — Scan the entire log and list all distinct anomaly
       clusters with their severity scores.
    3. **read_log_lines** — Read a specific range of lines from the log file for
       deeper inspection.
    4. **search_log_pattern** — Search the log for a regex pattern and return
       matching lines with line numbers.

    **Investigation workflow:**
    1. Start by scanning the log to get an overview of all anomalies.
    2. Parse the log to get full context around the primary anomaly.
    3. If needed, read additional lines or search for related patterns (e.g.
       preceding warnings, correlated request IDs) to build a complete picture.
    4. Once you have enough information, provide your final analysis.

    **Your final response MUST use this exact structured format:**

    ## Root Cause Analysis
    <One concise paragraph explaining the technical root cause of the error.>

    ## Probable Cause
    <One concise paragraph explaining WHY this error likely occurred, referencing
    specific lines or values from the log.>

    ## Remediation Steps
    <A numbered list of 3–6 actionable steps an on-call engineer can take
    immediately to diagnose, mitigate, or permanently fix this issue.>

    ## Confidence
    <A single word: HIGH / MEDIUM / LOW>
""")

# ---------------------------------------------------------------------------
# OpenAI-style tool definitions for Groq function calling
# ---------------------------------------------------------------------------
TOOL_DECLARATIONS = [
    {
        "type": "function",
        "function": {
            "name": "parse_log_file",
            "description": (
                "Parse a log file to detect the primary (highest-severity) anomaly "
                "and extract surrounding context lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the log file to analyse.",
                    },
                    "context_window": {
                        "type": "integer",
                        "description": "Number of context lines before/after the anomaly. Default: 20.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_all_anomalies",
            "description": (
                "Scan a log file and return a summary of ALL distinct anomaly "
                "clusters with their severity scores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the log file.",
                    },
                    "context_window": {
                        "type": "integer",
                        "description": "Context window size for cluster de-duplication. Default: 20.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_log_lines",
            "description": (
                "Read a specific range of lines from the log file for deeper "
                "inspection. Lines are 1-indexed and inclusive."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the log file.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-indexed).",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (1-indexed, inclusive).",
                    },
                },
                "required": ["file_path", "start_line", "end_line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_log_pattern",
            "description": (
                "Search the log file for a regex pattern. Returns up to 50 "
                "matching lines with their line numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the log file.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for (case-insensitive).",
                    },
                },
                "required": ["file_path", "pattern"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------
class LogAnalysisAgent:
    """
    Tool-using agent that iteratively investigates log files.

    The agent loop:
      1. Sends the user request + tool declarations to Groq (Llama 3.3 70B).
      2. If the model returns tool_calls, executes them and sends results back.
      3. Repeats until the model produces a final text response (no more tool calls).
      4. Parses the structured response and returns it.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set.  "
                "Get a free key at https://console.groq.com/keys and add it "
                "to your .env file or export it as an environment variable."
            )
        self._client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model_name = model or "llama-3.3-70b-versatile"
        self.tool_calls_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def investigate(
        self,
        log_file_path: str,
        user_query: str | None = None,
        on_tool_call: Optional[Callable[[str, dict, dict], None]] = None,
    ) -> dict[str, Any]:
        """
        Run the agent loop to investigate a log file.

        Args:
            log_file_path:  Absolute path to the log file.
            user_query:     Optional extra context or question from the user.
            on_tool_call:   Optional callback(tool_name, args, result) for
                            live progress reporting.

        Returns:
            dict with keys: root_cause, probable_cause, remediation,
            confidence, raw_response, model, tool_calls_log, iterations.
        """
        self.tool_calls_log = []

        # Build initial user message
        initial_prompt = (
            f"Investigate the log file at: {log_file_path}\n\n"
            "Use your tools to scan for anomalies, read context, and search "
            "for related patterns. Then provide your structured analysis."
        )
        if user_query:
            initial_prompt += f"\n\nAdditional context from the engineer: {user_query}"

        # Conversation history (OpenAI message format)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_INSTRUCTION},
            {"role": "user", "content": initial_prompt},
        ]

        # --- Agent loop ---
        for iteration in range(MAX_AGENT_ITERATIONS):
            # Small delay between iterations to avoid bursting rate limits
            if iteration > 0:
                time.sleep(2)

            def _do_call():
                return self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=TOOL_DECLARATIONS,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=4096,
                )

            response = _call_with_retry(_do_call)

            choice = response.choices[0]
            assistant_message = choice.message

            # Check for tool calls
            tool_calls = assistant_message.tool_calls

            if not tool_calls:
                # ── Final text response ──
                raw = assistant_message.content or ""
                parsed = self._parse_response(raw)
                parsed["raw_response"] = raw
                parsed["model"] = self.model_name
                parsed["tool_calls_log"] = self.tool_calls_log
                parsed["iterations"] = iteration + 1
                try:
                    parsed["prompt_tokens"] = response.usage.prompt_tokens
                except AttributeError:
                    parsed["prompt_tokens"] = None
                return parsed

            # ── Execute tool calls ──
            # Append the assistant's message (containing tool_calls).
            # Build a clean dict — Groq rejects unsupported fields like 'annotations'.
            assistant_dict: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            messages.append(assistant_dict)

            for tool_call in tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except json.JSONDecodeError:
                    fn_args = {}

                # Cast numeric args that arrive as floats
                for k, v in fn_args.items():
                    if isinstance(v, float) and v == int(v):
                        fn_args[k] = int(v)

                # Dispatch
                tool_fn = TOOL_REGISTRY.get(fn_name)
                if tool_fn is not None:
                    result = tool_fn(**fn_args)
                else:
                    result = {"error": f"Unknown tool: {fn_name}"}

                # Log
                self.tool_calls_log.append({
                    "iteration": iteration + 1,
                    "tool": fn_name,
                    "args": fn_args,
                    "result_preview": str(result)[:300],
                })

                # Callback for live UI
                if on_tool_call:
                    on_tool_call(fn_name, fn_args, result)

                # Send tool result back to the model
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str),
                })

        # ── Exhausted iterations ──
        return {
            "root_cause": "Agent reached maximum iterations without completing analysis.",
            "probable_cause": "",
            "remediation": "",
            "confidence": "LOW",
            "raw_response": "",
            "model": self.model_name,
            "tool_calls_log": self.tool_calls_log,
            "iterations": MAX_AGENT_ITERATIONS,
            "prompt_tokens": None,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(text: str) -> dict[str, str]:
        """Extract the four structured sections from the LLM response."""
        sections: dict[str, str] = {
            "root_cause": "",
            "probable_cause": "",
            "remediation": "",
            "confidence": "UNKNOWN",
        }
        markers = {
            "root_cause": "## Root Cause Analysis",
            "probable_cause": "## Probable Cause",
            "remediation": "## Remediation Steps",
            "confidence": "## Confidence",
        }
        for key, header in markers.items():
            start = text.find(header)
            if start == -1:
                continue
            content_start = start + len(header)
            next_header_pos = len(text)
            for other_key, other_header in markers.items():
                if other_key == key:
                    continue
                pos = text.find(other_header, content_start)
                if pos != -1 and pos < next_header_pos:
                    next_header_pos = pos
            sections[key] = text[content_start:next_header_pos].strip()
        return sections
