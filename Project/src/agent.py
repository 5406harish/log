"""
Agent Module
============
Implements a tool-using agent loop where the LLM iteratively calls tools
(log parser, line reader, pattern searcher) to investigate anomalies before
producing a final structured analysis.

Uses the Groq Cloud API (free tier) via the OpenAI-compatible SDK.
Optimised for free-tier rate limits:
  - Uses llama-3.1-8b-instant for fast, cheap tool-calling iterations
  - Adds generous inter-iteration delays to avoid 429 bursts
  - Limits context size sent back to the model to conserve TPM
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
import time
from typing import Any, Callable, Optional

from openai import OpenAI
from dotenv import load_dotenv

from src.tools import TOOL_REGISTRY

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Maximum iterations to prevent infinite loops
# ---------------------------------------------------------------------------
MAX_AGENT_ITERATIONS = 6

# ---------------------------------------------------------------------------
# Rate-limit-aware retry helper (standalone, no import needed)
# ---------------------------------------------------------------------------
AGENT_MAX_RETRIES = 5
AGENT_INITIAL_BACKOFF = 20  # seconds


def _agent_call_with_retry(call_fn, *, max_retries=AGENT_MAX_RETRIES, initial_backoff=AGENT_INITIAL_BACKOFF):
    """Execute *call_fn()* with exponential backoff on 429 errors.

    Parses the server-suggested retry delay when available.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return call_fn()
        except Exception as exc:
            exc_str = str(exc)
            # Only retry on rate-limit / quota errors
            if "429" not in exc_str and "rate" not in exc_str.lower():
                raise exc
            last_exc = exc
            if attempt == max_retries:
                break
            # Try to parse the server-suggested retry delay
            match = re.search(r"retry in ([\d.]+)s", exc_str, re.IGNORECASE)
            if match:
                wait = float(match.group(1)) + 2  # add 2s buffer
            else:
                # Also try "try again in Xm Ys" format
                match_min = re.search(r"try again in (\d+)m([\d.]+)s", exc_str, re.IGNORECASE)
                if match_min:
                    wait = int(match_min.group(1)) * 60 + float(match_min.group(2)) + 2
                else:
                    wait = initial_backoff * (2 ** attempt)
            # Cap wait at 120 seconds
            wait = min(wait, 120)
            logger.warning(
                "Rate limited (429). Retrying in %.1fs (attempt %d/%d)...",
                wait, attempt + 1, max_retries,
            )
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


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

    **IMPORTANT:** Be efficient with tool calls. Try to gather all needed info
    in as few calls as possible. After 2-3 tool calls you should have enough
    context to provide your analysis.

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


def _truncate_tool_result(result: dict, max_chars: int = 3000) -> str:
    """Serialise a tool result dict, truncating if too long to save tokens."""
    text = json.dumps(result, default=str)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated for brevity]"
    return text


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------
class LogAnalysisAgent:
    """
    Tool-using agent that iteratively investigates log files.

    The agent loop:
      1. Sends the user request + tool declarations to Groq.
      2. If the model returns tool_calls, executes them and sends results back.
      3. Repeats until the model produces a final text response (no more tool calls).
      4. Parses the structured response and returns it.

    Optimised for Groq free tier:
      - Uses llama-3.1-8b-instant for tool-calling (fast, high rate limits)
      - Adds inter-iteration delays to avoid 429 rate-limit bursts
      - Truncates large tool results to conserve token budget
    """

    # Use a small fast model for tool-calling to avoid rate limits
    DEFAULT_MODEL = "llama-3.1-8b-instant"

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
        self.model_name = model or self.DEFAULT_MODEL
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
            "for related patterns. Then provide your structured analysis.\n\n"
            "Be efficient: use scan_all_anomalies first, then parse_log_file, "
            "then provide your analysis. Minimise tool calls."
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
            # Generous delay between iterations to avoid rate-limit bursts
            if iteration > 0:
                delay = 5 + iteration * 2  # 7s, 9s, 11s, ...
                logger.info("Agent iteration %d — waiting %ds before next call...", iteration + 1, delay)
                time.sleep(delay)

            def _do_call():
                return self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=TOOL_DECLARATIONS,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=4096,
                )

            try:
                response = _agent_call_with_retry(_do_call)
            except Exception as exc:
                # If we've gathered any tool results, try to produce a partial analysis
                if self.tool_calls_log:
                    logger.warning("Rate limited after %d iterations, producing partial analysis", iteration)
                    return self._fallback_analysis(iteration)
                raise

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

                # Send tool result back to the model (truncated to save tokens)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": _truncate_tool_result(result),
                })

        # ── Exhausted iterations — try fallback ──
        return self._fallback_analysis(MAX_AGENT_ITERATIONS)

    # ------------------------------------------------------------------
    # Fallback analysis when rate-limited or max iterations reached
    # ------------------------------------------------------------------
    def _fallback_analysis(self, iterations: int) -> dict[str, Any]:
        """Produce a best-effort analysis from the tool results we gathered."""
        # Collect all tool results into a summary
        summary_parts = []
        for tc in self.tool_calls_log:
            summary_parts.append(
                f"Tool: {tc['tool']}, Args: {tc['args']}, "
                f"Result: {tc['result_preview']}"
            )
        tool_summary = "\n".join(summary_parts) if summary_parts else "No tool results gathered."

        # Try one more LLM call with just the tool summary (smaller context = less tokens)
        fallback_prompt = (
            "Based on the following tool investigation results, provide your "
            "structured analysis.\n\n"
            f"Tool Results:\n{tool_summary}\n\n"
            "Respond with:\n"
            "## Root Cause Analysis\n<analysis>\n\n"
            "## Probable Cause\n<cause>\n\n"
            "## Remediation Steps\n<steps>\n\n"
            "## Confidence\n<HIGH/MEDIUM/LOW>"
        )

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_INSTRUCTION},
            {"role": "user", "content": fallback_prompt},
        ]

        try:
            def _do_fallback():
                return self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2048,
                )

            response = _agent_call_with_retry(_do_fallback)
            raw = response.choices[0].message.content or ""
            parsed = self._parse_response(raw)
            parsed["raw_response"] = raw
            parsed["model"] = self.model_name
            parsed["tool_calls_log"] = self.tool_calls_log
            parsed["iterations"] = iterations
            try:
                parsed["prompt_tokens"] = response.usage.prompt_tokens
            except AttributeError:
                parsed["prompt_tokens"] = None
            return parsed
        except Exception:
            # Ultimate fallback: return whatever we have
            return {
                "root_cause": "Agent reached maximum iterations or was rate-limited before completing analysis.",
                "probable_cause": "",
                "remediation": "",
                "confidence": "LOW",
                "raw_response": "",
                "model": self.model_name,
                "tool_calls_log": self.tool_calls_log,
                "iterations": iterations,
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

        # Fallback: if confidence is still UNKNOWN, try to extract from text
        if sections["confidence"] == "UNKNOWN" or sections["confidence"] == "":
            conf_match = re.search(
                r"\b(confidence)\b[:\s]*(HIGH|MEDIUM|LOW)",
                text,
                re.IGNORECASE,
            )
            if conf_match:
                sections["confidence"] = conf_match.group(2).upper()
            elif sections["root_cause"]:
                # We have analysis but no confidence - default to MEDIUM
                sections["confidence"] = "MEDIUM"

        return sections

