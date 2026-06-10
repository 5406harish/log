"""
Log Parser Module
=================
Scans a .log file for error/anomaly indicators and extracts the primary
error block along with a configurable context window (default ±20 lines).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Patterns that indicate an anomalous log line
# ---------------------------------------------------------------------------
ERROR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(CRITICAL|FATAL)\b", re.IGNORECASE),
    re.compile(r"\bERROR\b", re.IGNORECASE),
    re.compile(r"\bEXCEPTION\b", re.IGNORECASE),
    re.compile(r"\bTraceback\s+\(most recent call last\)", re.IGNORECASE),
    re.compile(r"\bSTACK\s+TRACE\b", re.IGNORECASE),
    re.compile(r"\b(WARN(?:ING)?)\b", re.IGNORECASE),
    # HTTP 4xx / 5xx status codes in access logs (e.g. "HTTP/1.1" 500)
    re.compile(r'"\s+[45]\d{2}\s+'),
    # Panic / segfault style messages
    re.compile(r"\b(panic|segfault|killed|oom-kill|out of memory)\b", re.IGNORECASE),
    # Timeout / connection refused patterns
    re.compile(r"\b(timed?\s*out|connection refused|connection reset)\b", re.IGNORECASE),
]

# Severity ordering (higher = worse) used to pick the *primary* error
SEVERITY_RANK: dict[str, int] = {
    "fatal": 6,
    "critical": 6,
    "panic": 6,
    "segfault": 6,
    "oom-kill": 6,
    "out of memory": 6,
    "killed": 5,
    "exception": 4,
    "traceback": 4,
    "stack trace": 4,
    "error": 3,
    "timed out": 2,
    "connection refused": 2,
    "connection reset": 2,
    "warning": 1,
    "warn": 1,
    "http_4xx_5xx": 2,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class AnomalyBlock:
    """Represents a detected anomaly with surrounding context."""

    file_path: str
    primary_line_number: int          # 1-indexed
    primary_line_content: str
    context_start: int                # 1-indexed
    context_end: int                  # 1-indexed
    context_lines: list[str] = field(default_factory=list)
    severity: int = 0
    total_log_lines: int = 0

    @property
    def formatted_context(self) -> str:
        """Return context as a numbered block ready for prompt injection."""
        lines = []
        for idx, line in enumerate(self.context_lines):
            lineno = self.context_start + idx
            marker = ">>>" if lineno == self.primary_line_number else "   "
            lines.append(f"{marker} {lineno:>6} | {line.rstrip()}")
        return "\n".join(lines)

    @property
    def summary(self) -> str:
        return (
            f"File: {self.file_path}\n"
            f"Primary anomaly at line {self.primary_line_number} "
            f"(context: lines {self.context_start}–{self.context_end})\n"
            f"Severity score: {self.severity}"
        )


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------
class LogParser:
    """
    Reads a log file, detects all anomalous lines, picks the highest-severity
    primary error, and returns an AnomalyBlock with ±context_window lines.
    """

    def __init__(self, context_window: int = 20) -> None:
        self.context_window = context_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def parse(self, log_path: str | Path) -> Optional[AnomalyBlock]:
        """
        Parse *log_path* and return the primary AnomalyBlock, or None if no
        anomaly is detected.
        """
        path = Path(log_path)
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {log_path}")

        lines = self._read_lines(path)
        if not lines:
            return None

        candidate = self._find_primary_anomaly(lines)
        if candidate is None:
            return None

        primary_idx, severity = candidate          # 0-indexed
        start_idx = max(0, primary_idx - self.context_window)
        end_idx = min(len(lines) - 1, primary_idx + self.context_window)

        return AnomalyBlock(
            file_path=str(path.resolve()),
            primary_line_number=primary_idx + 1,   # convert to 1-indexed
            primary_line_content=lines[primary_idx],
            context_start=start_idx + 1,
            context_end=end_idx + 1,
            context_lines=lines[start_idx : end_idx + 1],
            severity=severity,
            total_log_lines=len(lines),
        )

    def scan_all(self, log_path: str | Path) -> list[AnomalyBlock]:
        """
        Return *all* detected anomaly blocks (de-duplicated by context overlap).
        Useful for reports that need to surface multiple distinct error clusters.
        """
        path = Path(log_path)
        lines = self._read_lines(path)
        if not lines:
            return []

        anomaly_indices = self._find_all_anomalies(lines)
        if not anomaly_indices:
            return []

        blocks: list[AnomalyBlock] = []
        last_end = -1

        for idx, severity in anomaly_indices:
            start_idx = max(0, idx - self.context_window)
            # Skip if this anomaly's context overlaps the previous block
            if start_idx <= last_end:
                continue
            end_idx = min(len(lines) - 1, idx + self.context_window)
            blocks.append(
                AnomalyBlock(
                    file_path=str(path.resolve()),
                    primary_line_number=idx + 1,
                    primary_line_content=lines[idx],
                    context_start=start_idx + 1,
                    context_end=end_idx + 1,
                    context_lines=lines[start_idx : end_idx + 1],
                    severity=severity,
                    total_log_lines=len(lines),
                )
            )
            last_end = end_idx

        return blocks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_lines(path: Path) -> list[str]:
        """Read file trying UTF-8 first, falling back to latin-1."""
        for encoding in ("utf-8", "latin-1"):
            try:
                return path.read_text(encoding=encoding).splitlines(keepends=True)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot decode file: {path}")

    @staticmethod
    def _score_line(line: str) -> int:
        """Return a severity score for a single log line (0 = not anomalous)."""
        line_lower = line.lower()
        best = 0
        for pattern in ERROR_PATTERNS:
            if pattern.search(line):
                # Determine which keyword matched and look up its rank
                for keyword, rank in SEVERITY_RANK.items():
                    if keyword in line_lower:
                        best = max(best, rank)
                # HTTP status code pattern is a special case
                if pattern.pattern.startswith('"'):
                    best = max(best, SEVERITY_RANK["http_4xx_5xx"])
        return best

    def _find_all_anomalies(self, lines: list[str]) -> list[tuple[int, int]]:
        """Return sorted list of (0-indexed line number, severity) for every anomalous line."""
        results = []
        for idx, line in enumerate(lines):
            score = self._score_line(line)
            if score > 0:
                results.append((idx, score))
        return results

    def _find_primary_anomaly(self, lines: list[str]) -> Optional[tuple[int, int]]:
        """
        Return the (0-indexed line number, severity) of the single highest-severity
        anomaly.  Ties are broken by picking the *first* occurrence.
        """
        anomalies = self._find_all_anomalies(lines)
        if not anomalies:
            return None
        return max(anomalies, key=lambda t: t[1])
