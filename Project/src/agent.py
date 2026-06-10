"""
Agent Module
============
Implements a log investigation agent that gathers data using tools locally,
then makes a SINGLE LLM call for analysis.

This single-pass approach avoids rate-limit issues on the Groq free tier
by minimising API calls to exactly ONE.
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
# Simple retry helper
# ---------------------------------------------------------------------------
MAX_RETRIES = 2
RETRY_DELAY = 5  # seconds


def _call_with_retry(call_fn, *, max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Execute *call_fn()* with simple retry on transient errors."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return call_fn()
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            logger.warning(
                "API call failed. Retrying in %ds (attempt %d/%d)...",
                delay, attempt + 1, max_retries,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Agent system instruction
# ---------------------------------------------------------------------------
AGENT_SYSTEM_INSTRUCTION = textwrap.dedent("""\
    You are an expert Site Reliability Engineer (SRE) agent.
    You have been given the results of several log analysis tools.
    Analyse the data and provide a structured investigation report.

    **Your response MUST use this exact structured format:**

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


def _truncate(text: str, max_len: int = 2000) -> str:
    """Truncate text to save tokens."""
    if len(text) > max_len:
        return text[:max_len] + "\n... [truncated]"
    return text


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------
class LogAnalysisAgent:
    """
    Single-pass log investigation agent.

    Instead of an iterative LLM loop (which burns through rate limits),
    this agent:
      1. Runs ALL tools locally (no LLM needed) to gather data
      2. Makes ONE single LLM call with all gathered data for analysis

    This uses exactly 1 API call, avoiding rate-limit issues entirely.
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

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
        Run the agent to investigate a log file.

        Phase 1: Run tools locally (no API calls)
        Phase 2: Send all gathered data to LLM in ONE call

        Returns:
            dict with keys: root_cause, probable_cause, remediation,
            confidence, raw_response, model, tool_calls_log, iterations.
        """
        self.tool_calls_log = []

        # ── Phase 1: Gather data using tools locally ──
        gathered_data = self._gather_data(log_file_path, on_tool_call)

        # ── Phase 2: Single LLM call for analysis ──
        prompt = self._build_analysis_prompt(log_file_path, gathered_data, user_query)

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]

        def _do_call():
            return self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
            )

        try:
            response = _call_with_retry(_do_call)
        except Exception as exc:
            # If LLM fails, return a partial result with tool data
            return self._fallback_result(str(exc))

        raw = response.choices[0].message.content or ""
        parsed = self._parse_response(raw)
        parsed["raw_response"] = raw
        parsed["model"] = self.model_name
        parsed["tool_calls_log"] = self.tool_calls_log
        parsed["iterations"] = 1
        try:
            parsed["prompt_tokens"] = response.usage.prompt_tokens
        except AttributeError:
            parsed["prompt_tokens"] = None
        return parsed

    # ------------------------------------------------------------------
    # Phase 1: Local tool execution (no API calls needed)
    # ------------------------------------------------------------------
    def _gather_data(
        self,
        log_file_path: str,
        on_tool_call: Optional[Callable] = None,
    ) -> dict[str, Any]:
        """Run tools locally to gather all investigation data."""
        gathered = {}

        # Step 1: Scan all anomalies
        scan_fn = TOOL_REGISTRY["scan_all_anomalies"]
        scan_result = scan_fn(file_path=log_file_path, context_window=20)
        self._log_tool_call("scan_all_anomalies", {"file_path": log_file_path}, scan_result, on_tool_call)
        gathered["scan"] = scan_result

        # Step 2: Parse primary anomaly
        parse_fn = TOOL_REGISTRY["parse_log_file"]
        parse_result = parse_fn(file_path=log_file_path, context_window=20)
        self._log_tool_call("parse_log_file", {"file_path": log_file_path}, parse_result, on_tool_call)
        gathered["primary"] = parse_result

        # Step 3: If primary anomaly found, search for related patterns
        if parse_result.get("status") == "anomaly_found":
            primary_line = parse_result.get("primary_line_content", "")
            # Extract key error terms to search for
            error_terms = re.findall(r'\b(?:error|exception|fail|fatal|critical|timeout|refused)\b', primary_line, re.IGNORECASE)
            if error_terms:
                search_fn = TOOL_REGISTRY["search_log_pattern"]
                pattern = error_terms[0]
                search_result = search_fn(file_path=log_file_path, pattern=pattern)
                self._log_tool_call("search_log_pattern", {"file_path": log_file_path, "pattern": pattern}, search_result, on_tool_call)
                gathered["search"] = search_result

        return gathered

    def _log_tool_call(self, name: str, args: dict, result: dict, callback: Optional[Callable] = None):
        """Log a tool call."""
        self.tool_calls_log.append({
            "iteration": 1,
            "tool": name,
            "args": args,
            "result_preview": str(result)[:300],
        })
        if callback:
            callback(name, args, result)

    # ------------------------------------------------------------------
    # Build the analysis prompt from gathered data
    # ------------------------------------------------------------------
    def _build_analysis_prompt(
        self,
        log_file_path: str,
        gathered: dict[str, Any],
        user_query: str | None,
    ) -> str:
        """Build a single comprehensive prompt from all gathered tool data."""
        parts = [f"## Log File: {log_file_path}\n"]

        # Scan results
        scan = gathered.get("scan", {})
        if scan.get("status") == "anomalies_found":
            parts.append(f"### Anomaly Scan Summary")
            parts.append(f"Total log lines: {scan.get('total_log_lines', '?')}")
            parts.append(f"Anomaly clusters found: {scan.get('anomaly_count', 0)}\n")
            for a in scan.get("anomalies", [])[:10]:
                parts.append(
                    f"- Cluster {a['cluster']}: Line {a['primary_line_number']} "
                    f"(severity {a['severity']}/6) — {a['primary_line_content'][:100]}"
                )
            parts.append("")
        elif scan.get("status") == "clean":
            parts.append("### Anomaly Scan: No anomalies detected.\n")

        # Primary anomaly context
        primary = gathered.get("primary", {})
        if primary.get("status") == "anomaly_found":
            parts.append(f"### Primary Anomaly (Highest Severity)")
            parts.append(f"- Line: {primary.get('primary_line_number', '?')}")
            parts.append(f"- Severity: {primary.get('severity', '?')}/6")
            parts.append(f"- Content: {primary.get('primary_line_content', '?')}")
            parts.append(f"- Context range: lines {primary.get('context_start', '?')}–{primary.get('context_end', '?')}\n")
            ctx = primary.get("formatted_context", "")
            parts.append(f"### Context Window (±20 lines around primary error)")
            parts.append(f"```\n{_truncate(ctx, 2500)}\n```\n")

        # Search results
        search = gathered.get("search", {})
        if search.get("match_count", 0) > 0:
            parts.append(f"### Pattern Search Results (pattern: '{search.get('pattern', '?')}')")
            parts.append(f"Matches found: {search['match_count']}")
            for m in search.get("matches", [])[:15]:
                parts.append(f"  Line {m['line_number']}: {m['content'][:100]}")
            parts.append("")

        if user_query:
            parts.append(f"\n### Additional Context from Engineer\n{user_query}\n")

        parts.append("\nPlease analyse the above log data and provide your structured response.")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Fallback when LLM fails
    # ------------------------------------------------------------------
    def _fallback_result(self, error_msg: str) -> dict[str, Any]:
        """Return a result dict when LLM call fails."""
        return {
            "root_cause": f"LLM analysis failed: {error_msg}",
            "probable_cause": "",
            "remediation": "Try again in a few minutes or use Quick Scan mode.",
            "confidence": "LOW",
            "raw_response": "",
            "model": self.model_name,
            "tool_calls_log": self.tool_calls_log,
            "iterations": 1,
            "prompt_tokens": None,
        }

    # ------------------------------------------------------------------
    # Parse structured response
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

        # Fallback confidence detection
        if sections["confidence"] in ("UNKNOWN", ""):
            conf_match = re.search(
                r"\b(confidence)\b[:\s]*(HIGH|MEDIUM|LOW)",
                text,
                re.IGNORECASE,
            )
            if conf_match:
                sections["confidence"] = conf_match.group(2).upper()
            elif sections["root_cause"]:
                sections["confidence"] = "MEDIUM"

        return sections
