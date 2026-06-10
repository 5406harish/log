"""
Tests for src/parser.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.parser import LogParser, AnomalyBlock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
SIMPLE_ERROR_LOG = textwrap.dedent("""\
    2024-01-15 10:00:01 INFO  Starting application…
    2024-01-15 10:00:02 INFO  Connecting to database host=db.local port=5432
    2024-01-15 10:00:03 INFO  Connection pool initialised (size=10)
    2024-01-15 10:00:04 INFO  Loading configuration from /etc/app/config.yaml
    2024-01-15 10:00:05 DEBUG Config loaded successfully
    2024-01-15 10:00:06 INFO  Starting HTTP server on 0.0.0.0:8080
    2024-01-15 10:01:22 INFO  GET /api/health 200 12ms
    2024-01-15 10:01:55 ERROR Database query timeout after 30000ms — query=SELECT * FROM orders
    2024-01-15 10:01:55 ERROR Retrying connection (attempt 1/3)…
    2024-01-15 10:01:56 ERROR Retrying connection (attempt 2/3)…
    2024-01-15 10:01:57 CRITICAL Max retries exceeded. Shutting down worker.
    2024-01-15 10:01:57 INFO  Worker shutdown complete.
""")

NO_ERROR_LOG = textwrap.dedent("""\
    2024-01-15 10:00:01 INFO  All systems nominal.
    2024-01-15 10:00:02 INFO  Heartbeat OK
    2024-01-15 10:00:03 DEBUG Cache hit ratio: 98.2%
""")

TRACEBACK_LOG = textwrap.dedent("""\
    2024-01-15 10:05:00 INFO  Processing request id=abc123
    2024-01-15 10:05:01 INFO  Fetching user data
    Traceback (most recent call last):
      File "/app/handlers/user.py", line 42, in get_user
        return db.query(User).filter_by(id=user_id).one()
      File "/venv/lib/sqlalchemy/orm/query.py", line 3473, in one
        raise NoResultFound("No row was found when one was required")
    sqlalchemy.orm.exc.NoResultFound: No row was found when one was required
    2024-01-15 10:05:01 ERROR  Request failed with unhandled exception
""")


@pytest.fixture
def log_file(tmp_path: Path):
    """Factory fixture that writes content to a temp file and returns its path."""
    def _make(content: str, name: str = "test.log") -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p
    return _make


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestLogParserBasic:
    def test_detects_error_line(self, log_file):
        path = log_file(SIMPLE_ERROR_LOG)
        block = LogParser().parse(path)
        assert block is not None

    def test_no_anomaly_returns_none(self, log_file):
        path = log_file(NO_ERROR_LOG)
        block = LogParser().parse(path)
        assert block is None

    def test_detects_traceback(self, log_file):
        path = log_file(TRACEBACK_LOG)
        block = LogParser().parse(path)
        assert block is not None
        assert "Traceback" in block.primary_line_content or "NoResultFound" in block.primary_line_content

    def test_primary_anomaly_is_highest_severity(self, log_file):
        """CRITICAL should beat ERROR in severity ranking."""
        path = log_file(SIMPLE_ERROR_LOG)
        block = LogParser().parse(path)
        # CRITICAL line is line 11 ("Max retries exceeded…")
        assert block is not None
        assert "CRITICAL" in block.primary_line_content or block.severity >= 5

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            LogParser().parse("/nonexistent/path/to/file.log")


class TestContextWindow:
    def test_context_window_size(self, log_file):
        """Context should not exceed ±20 lines (or file bounds)."""
        path = log_file(SIMPLE_ERROR_LOG)
        block = LogParser(context_window=20).parse(path)
        assert block is not None
        total = block.context_end - block.context_start + 1
        assert total <= 41  # max 20 before + error + 20 after

    def test_custom_context_window(self, log_file):
        path = log_file(SIMPLE_ERROR_LOG)
        block = LogParser(context_window=5).parse(path)
        assert block is not None
        # Window should be at most 11 lines (5 before + error + 5 after)
        total = block.context_end - block.context_start + 1
        assert total <= 11

    def test_context_bounded_by_file_start(self, log_file):
        """Error near the start of file should not produce negative line numbers."""
        short_log = "ERROR Something went wrong immediately\nINFO Next line\n"
        path = log_file(short_log)
        block = LogParser(context_window=20).parse(path)
        assert block is not None
        assert block.context_start >= 1

    def test_context_bounded_by_file_end(self, log_file):
        short_log = "INFO Normal\nINFO Normal\nCRITICAL End of file error\n"
        path = log_file(short_log)
        block = LogParser(context_window=20).parse(path)
        assert block is not None
        total_lines = 3
        assert block.context_end <= total_lines


class TestScanAll:
    def test_scan_all_finds_multiple_clusters(self, log_file):
        multi_log = (
            "INFO  ok\n" * 50
            + "ERROR first error cluster\n"
            + "INFO  ok\n" * 50
            + "CRITICAL second error cluster\n"
            + "INFO  ok\n" * 5
        )
        path = log_file(multi_log)
        blocks = LogParser(context_window=5).scan_all(path)
        assert len(blocks) == 2

    def test_scan_all_empty_on_clean_log(self, log_file):
        path = log_file(NO_ERROR_LOG)
        blocks = LogParser().scan_all(path)
        assert blocks == []


class TestAnomalyBlock:
    def test_formatted_context_contains_marker(self, log_file):
        path = log_file(SIMPLE_ERROR_LOG)
        block = LogParser().parse(path)
        assert block is not None
        assert ">>>" in block.formatted_context

    def test_summary_contains_file_path(self, log_file):
        path = log_file(SIMPLE_ERROR_LOG)
        block = LogParser().parse(path)
        assert block is not None
        assert str(path) in block.summary or path.name in block.summary
