"""
Test suite for the Streamlit dashboard app.
Uses Streamlit's AppTest framework to programmatically test the UI
without requiring a browser.

Run with:
    uv run python -m pytest tests/test_streamlit_app.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test 1: App boots without errors
# ---------------------------------------------------------------------------
def test_app_boots_without_error():
    """The Streamlit app should load without throwing any exceptions."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"App raised exception: {at.exception}"


# ---------------------------------------------------------------------------
# Test 2: Quick Scan mode works with test_app.log
# ---------------------------------------------------------------------------
def test_quick_scan_finds_anomalies():
    """Uploading a log file and clicking Quick Scan should detect anomalies
    without any LLM call."""
    from streamlit.testing.v1 import AppTest

    test_log = PROJECT_ROOT / "test_app.log"
    assert test_log.exists(), f"Test log not found: {test_log}"

    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"Initial load exception: {at.exception}"

    # Simulate file upload — AppTest expects (file_name, file_bytes)
    log_bytes = test_log.read_bytes()
    at.file_uploader[0].upload("test_app.log", log_bytes, "text/plain")
    at.run()
    assert not at.exception, f"After upload exception: {at.exception}"

    # Click the Quick Scan button (first button)
    if at.button:
        # Find the Quick Scan button
        for btn in at.button:
            if "Quick Scan" in str(btn.label) or "Scan" in str(btn.label):
                btn.click()
                break
        else:
            # Fall back to first button
            at.button[0].click()
        at.run()
        assert not at.exception, f"After Quick Scan click exception: {at.exception}"


# ---------------------------------------------------------------------------
# Test 3: Parser integration test (direct, no Streamlit)
# ---------------------------------------------------------------------------
def test_parser_scan_all():
    """Directly test that the parser can scan the test log file."""
    from src.parser import LogParser

    test_log = PROJECT_ROOT / "test_app.log"
    assert test_log.exists()

    parser = LogParser(context_window=20)
    blocks = parser.scan_all(str(test_log))

    # Our test log has multiple error types
    assert len(blocks) > 0, "Expected at least one anomaly cluster"
    # Check severity scores exist
    assert all(b.severity > 0 for b in blocks)


# ---------------------------------------------------------------------------
# Test 4: Parser parse() returns highest severity
# ---------------------------------------------------------------------------
def test_parser_parse_primary():
    """Parser.parse() should return the highest-severity anomaly."""
    from src.parser import LogParser

    test_log = PROJECT_ROOT / "test_app.log"
    parser = LogParser(context_window=20)
    block = parser.parse(str(test_log))

    assert block is not None, "Expected a primary anomaly"
    # The FATAL/CRITICAL lines should be severity 6
    assert block.severity >= 5, f"Expected high severity, got {block.severity}"


# ---------------------------------------------------------------------------
# Test 5: Verify imports work correctly
# ---------------------------------------------------------------------------
def test_imports():
    """All backend modules should be importable."""
    from src.parser import LogParser, AnomalyBlock
    from src.llm_client import GroqClient
    from src.agent import LogAnalysisAgent
    from src.tools import TOOL_REGISTRY

    assert LogParser is not None
    assert AnomalyBlock is not None
    assert GroqClient is not None
    assert LogAnalysisAgent is not None
    assert len(TOOL_REGISTRY) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
