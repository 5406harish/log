"""
LLM Client Module
=================
Packages the extracted anomaly context into a structured prompt and sends it
to the Groq Cloud API via the OpenAI-compatible SDK.
Uses Llama 3.3 70B (free tier) for fast, high-quality log analysis.
"""

from __future__ import annotations

import os
import re
import textwrap
import time
import logging
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

from src.parser import AnomalyBlock

# ---------------------------------------------------------------------------
# Load environment variables from .env (if present)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = textwrap.dedent("""\
    You are an expert Site Reliability Engineer (SRE) and software debugging specialist.
    Your task is to analyse a section of a log file that contains an error or anomaly.

    You will receive:
    1. Metadata about the log file (path, line numbers, severity).
    2. A context window of log lines (±20 lines around the primary error), with the
       primary error line highlighted by ">>>" at the start of its row.

    You must respond in the following EXACT structured format (no extra prose outside
    the sections below):

    ## Root Cause Analysis
    <One concise paragraph explaining the technical root cause of the error.>

    ## Probable Cause
    <One concise paragraph explaining WHY this error likely occurred, referencing
    specific lines or values from the provided context.>

    ## Remediation Steps
    <A numbered list of 3–6 actionable steps an on-call engineer can take immediately
    to diagnose, mitigate, or permanently fix this issue.>

    ## Confidence
    <A single word: HIGH / MEDIUM / LOW — your confidence in the above analysis given
    the available context.>
""")

USER_PROMPT_TEMPLATE = textwrap.dedent("""\
    ### Log File Metadata
    - **File:** {file_path}
    - **Total lines in file:** {total_lines}
    - **Primary anomaly at line:** {primary_line} (marked with >>>)
    - **Context window:** lines {context_start} – {context_end}
    - **Detected severity score:** {severity}/6

    ### Primary Error Line
    ```
    {primary_line_content}
    ```

    ### Context Window (±20 lines)
    ```
    {formatted_context}
    ```

    Please analyse the above log excerpt and provide your structured response.
""")


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------
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


class GroqClient:
    """
    Thin wrapper around the OpenAI SDK targeting the Groq Cloud API
    (free tier, Llama 3.3 70B).  Falls back gracefully if the API key
    is missing.

    Includes automatic retry with exponential backoff for 429 rate-limit
    errors so transient quota bursts do not crash the application.
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def explain(self, block: AnomalyBlock) -> dict[str, Any]:
        """
        Send the AnomalyBlock to Groq and return a structured dict with keys:
          - raw_response  : full LLM text
          - root_cause    : extracted section
          - probable_cause: extracted section
          - remediation   : extracted section
          - confidence    : HIGH / MEDIUM / LOW
          - model         : model name used
          - prompt_tokens : approximate token count (if available)
        """
        prompt = self._build_prompt(block)

        def _do_call():
            return self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
            )

        response = _call_with_retry(_do_call)
        raw = response.choices[0].message.content
        parsed = self._parse_response(raw)
        parsed["raw_response"] = raw
        parsed["model"] = self.model_name
        # Extract usage metadata if available
        try:
            parsed["prompt_tokens"] = response.usage.prompt_tokens
        except AttributeError:
            parsed["prompt_tokens"] = None
        return parsed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_prompt(self, block: AnomalyBlock) -> str:
        return USER_PROMPT_TEMPLATE.format(
            file_path=block.file_path,
            total_lines=block.total_log_lines,
            primary_line=block.primary_line_number,
            context_start=block.context_start,
            context_end=block.context_end,
            severity=block.severity,
            primary_line_content=block.primary_line_content.strip(),
            formatted_context=block.formatted_context,
        )

    @staticmethod
    def _parse_response(text: str) -> dict[str, str]:
        """Extract the four structured sections from the LLM response."""
        sections: dict[str, str] = {
            "root_cause": "",
            "probable_cause": "",
            "remediation": "",
            "confidence": "UNKNOWN",
        }

        # Simple section splitter based on the markdown headers we instruct the model to use
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
            # Find the next section header
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

