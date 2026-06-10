"""
Tools Module
============
Defines callable tool functions that the LLM agent can invoke during its
investigation loop.  Each tool returns a plain dict serialisable to JSON.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.parser import LogParser


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def parse_log_file(file_path: str, context_window: int = 20) -> dict[str, Any]:
    """Parse a log file and return the primary anomaly with surrounding context."""
    parser = LogParser(context_window=context_window)
    try:
        block = parser.parse(file_path)
    except (FileNotFoundError, ValueError) as exc:
        return {"status": "error", "message": str(exc)}

    if block is None:
        return {"status": "clean", "message": "No anomalies detected in the log file."}

    return {
        "status": "anomaly_found",
        "primary_line_number": block.primary_line_number,
        "primary_line_content": block.primary_line_content.strip(),
        "severity": block.severity,
        "severity_max": 6,
        "context_start": block.context_start,
        "context_end": block.context_end,
        "total_log_lines": block.total_log_lines,
        "formatted_context": block.formatted_context,
    }


def scan_all_anomalies(file_path: str, context_window: int = 20) -> dict[str, Any]:
    """Scan a log file and return a summary of ALL distinct anomaly clusters."""
    parser = LogParser(context_window=context_window)
    try:
        blocks = parser.scan_all(file_path)
    except (FileNotFoundError, ValueError) as exc:
        return {"status": "error", "message": str(exc)}

    if not blocks:
        return {"status": "clean", "anomaly_count": 0, "anomalies": []}

    return {
        "status": "anomalies_found",
        "anomaly_count": len(blocks),
        "total_log_lines": blocks[0].total_log_lines,
        "anomalies": [
            {
                "cluster": i,
                "primary_line_number": b.primary_line_number,
                "primary_line_content": b.primary_line_content.strip(),
                "severity": b.severity,
                "context_range": f"{b.context_start}–{b.context_end}",
            }
            for i, b in enumerate(blocks, 1)
        ],
    }


def read_log_lines(file_path: str, start_line: int, end_line: int) -> dict[str, Any]:
    """Read specific lines from a log file (1-indexed, inclusive)."""
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    lines = text.splitlines()
    start = max(1, start_line) - 1
    end = min(len(lines), end_line)
    selected = lines[start:end]

    return {
        "file": file_path,
        "start_line": start + 1,
        "end_line": end,
        "total_lines": len(lines),
        "line_count": len(selected),
        "content": "\n".join(f"{start + 1 + i:>6} | {line}" for i, line in enumerate(selected)),
    }


def search_log_pattern(file_path: str, pattern: str) -> dict[str, Any]:
    """Search for a regex pattern in the log file and return matching lines."""
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    lines = text.splitlines()
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return {"error": f"Invalid regex pattern: {exc}"}

    matches = []
    for i, line in enumerate(lines, 1):
        if compiled.search(line):
            matches.append({"line_number": i, "content": line.strip()})
            if len(matches) >= 50:
                break

    return {
        "file": file_path,
        "pattern": pattern,
        "match_count": len(matches),
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# Tool registry — maps function names to callables
# ---------------------------------------------------------------------------
TOOL_REGISTRY: dict[str, Any] = {
    "parse_log_file": parse_log_file,
    "scan_all_anomalies": scan_all_anomalies,
    "read_log_lines": read_log_lines,
    "search_log_pattern": search_log_pattern,
}
