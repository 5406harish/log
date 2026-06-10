"""
Tests for src/tools.py — tool functions used by the agent.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.tools import (
    parse_log_file,
    scan_all_anomalies,
    read_log_lines,
    search_log_pattern,
    TOOL_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
SAMPLE_LOG = textwrap.dedent("""\
    2024-01-15 10:00:01 INFO  Starting application
    2024-01-15 10:00:02 INFO  Connected to database
    2024-01-15 10:00:03 INFO  Listening on port 8080
    2024-01-15 10:01:00 WARN  Slow query detected: 2500ms
    2024-01-15 10:02:00 ERROR Database connection timeout after 30000ms
    2024-01-15 10:02:01 ERROR Retrying connection (1/3)
    2024-01-15 10:02:02 CRITICAL Max retries exceeded — shutting down
    2024-01-15 10:02:03 INFO  Shutdown complete
""")

CLEAN_LOG = textwrap.dedent("""\
    2024-01-15 10:00:01 INFO  All good
    2024-01-15 10:00:02 INFO  Heartbeat OK
    2024-01-15 10:00:03 DEBUG Cache hit ratio: 99%
""")


@pytest.fixture
def log_file(tmp_path: Path):
    def _make(content: str, name: str = "test.log") -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p
    return _make


# ---------------------------------------------------------------------------
# Tests: parse_log_file
# ---------------------------------------------------------------------------
class TestParseLogFile:
    def test_finds_anomaly(self, log_file):
        path = log_file(SAMPLE_LOG)
        result = parse_log_file(str(path))
        assert result["status"] == "anomaly_found"
        assert result["severity"] >= 5  # CRITICAL
        assert "formatted_context" in result

    def test_clean_log_returns_clean(self, log_file):
        path = log_file(CLEAN_LOG)
        result = parse_log_file(str(path))
        assert result["status"] == "clean"

    def test_file_not_found(self):
        result = parse_log_file("/nonexistent/file.log")
        assert result["status"] == "error"

    def test_custom_context_window(self, log_file):
        path = log_file(SAMPLE_LOG)
        result = parse_log_file(str(path), context_window=3)
        assert result["status"] == "anomaly_found"


# ---------------------------------------------------------------------------
# Tests: scan_all_anomalies
# ---------------------------------------------------------------------------
class TestScanAllAnomalies:
    def test_finds_multiple(self, log_file):
        multi = "INFO ok\n" * 50 + "ERROR first\n" + "INFO ok\n" * 50 + "CRITICAL second\n"
        path = log_file(multi)
        result = scan_all_anomalies(str(path), context_window=5)
        assert result["status"] == "anomalies_found"
        assert result["anomaly_count"] == 2

    def test_clean_log(self, log_file):
        path = log_file(CLEAN_LOG)
        result = scan_all_anomalies(str(path))
        assert result["status"] == "clean"
        assert result["anomaly_count"] == 0


# ---------------------------------------------------------------------------
# Tests: read_log_lines
# ---------------------------------------------------------------------------
class TestReadLogLines:
    def test_reads_range(self, log_file):
        path = log_file(SAMPLE_LOG)
        result = read_log_lines(str(path), 2, 4)
        assert result["start_line"] == 2
        assert result["end_line"] == 4
        assert result["line_count"] == 3
        assert "Connected to database" in result["content"]

    def test_clamps_to_file_bounds(self, log_file):
        path = log_file(SAMPLE_LOG)
        result = read_log_lines(str(path), 1, 9999)
        assert result["end_line"] <= result["total_lines"]

    def test_file_not_found(self):
        result = read_log_lines("/nonexistent/file.log", 1, 10)
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: search_log_pattern
# ---------------------------------------------------------------------------
class TestSearchLogPattern:
    def test_finds_matches(self, log_file):
        path = log_file(SAMPLE_LOG)
        result = search_log_pattern(str(path), r"ERROR|CRITICAL")
        assert result["match_count"] >= 3

    def test_no_matches(self, log_file):
        path = log_file(CLEAN_LOG)
        result = search_log_pattern(str(path), r"FATAL")
        assert result["match_count"] == 0

    def test_invalid_regex(self, log_file):
        path = log_file(SAMPLE_LOG)
        result = search_log_pattern(str(path), r"[invalid")
        assert "error" in result

    def test_file_not_found(self):
        result = search_log_pattern("/nonexistent/file.log", "ERROR")
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: TOOL_REGISTRY
# ---------------------------------------------------------------------------
class TestToolRegistry:
    def test_all_tools_registered(self):
        expected = {"parse_log_file", "scan_all_anomalies", "read_log_lines", "search_log_pattern"}
        assert set(TOOL_REGISTRY.keys()) == expected

    def test_all_tools_callable(self):
        for name, fn in TOOL_REGISTRY.items():
            assert callable(fn), f"Tool {name} is not callable"
