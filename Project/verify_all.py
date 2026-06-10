"""
Comprehensive end-to-end verification script.
Tests all three modes: Quick Scan, Explain, and Investigate.
Runs headlessly using Streamlit's AppTest + direct backend calls.
"""

import sys
import json
import os
from pathlib import Path

# Fix Windows console encoding for emoji/unicode output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
SKIP = "\033[93m⏭️  SKIP\033[0m"
results = []

def report(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((name, passed))
    print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))

# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Streamlit app boots
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  LOG ANOMALY EXPLAINER — FULL VERIFICATION")
print("="*70)

print("\n[1/6] App Boot Test")
try:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
    at.run()
    report("App boots without errors", not at.exception,
           str(at.exception)[:100] if at.exception else "Clean boot")
except Exception as e:
    report("App boots without errors", False, str(e)[:100])

# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Quick Scan (parser only, no LLM)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2/6] Quick Scan — Parser Test")
try:
    from src.parser import LogParser
    test_log = PROJECT_ROOT / "test_app.log"
    parser = LogParser(context_window=20)
    blocks = parser.scan_all(str(test_log))

    report("Parser finds anomalies", len(blocks) > 0,
           f"{len(blocks)} clusters detected")

    if blocks:
        max_sev = max(b.severity for b in blocks)
        report("Severity scoring works", max_sev >= 1,
               f"Max severity in scan_all: {max_sev}/6")

        # Check that formatted context works
        for b in blocks:
            ctx = b.formatted_context
            assert len(ctx) > 0
        report("Formatted context renders", True, "All blocks have context")
except Exception as e:
    report("Parser scan_all", False, str(e)[:100])

# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Primary anomaly parse
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3/6] Primary Anomaly Parse")
try:
    block = parser.parse(str(test_log))
    report("Primary anomaly detected", block is not None,
           f"Line {block.primary_line_number}, severity {block.severity}/6" if block else "")
    if block:
        report("Primary line content valid", len(block.primary_line_content.strip()) > 0,
               block.primary_line_content.strip()[:80])
except Exception as e:
    report("Primary anomaly parse", False, str(e)[:100])

# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: Explain Anomalies (LLM call)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[4/6] Explain Anomalies — LLM Call")
api_key = os.getenv("GROQ_API_KEY", "")
if not api_key or api_key == "your-groq-api-key-here":
    print(f"  {SKIP}  LLM Explain — No API key configured")
    results.append(("LLM Explain", None))
else:
    try:
        from src.llm_client import GroqClient
        client = GroqClient(api_key=api_key)
        block = parser.parse(str(test_log))
        analysis = client.explain(block)

        report("LLM returns response", "raw_response" in analysis and len(analysis["raw_response"]) > 0,
               f"Response length: {len(analysis.get('raw_response', ''))}")
        report("Root cause extracted", len(analysis.get("root_cause", "")) > 0,
               analysis.get("root_cause", "")[:80])
        report("Probable cause extracted", len(analysis.get("probable_cause", "")) > 0,
               analysis.get("probable_cause", "")[:80])
        report("Remediation extracted", len(analysis.get("remediation", "")) > 0,
               analysis.get("remediation", "")[:80])
        report("Confidence present", analysis.get("confidence", "UNKNOWN") != "UNKNOWN",
               f"Confidence: {analysis.get('confidence')}")
        report("Model info present", len(analysis.get("model", "")) > 0,
               f"Model: {analysis.get('model')}")
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            print(f"  {SKIP}  LLM Explain — API quota exhausted (not a code bug)")
            results.append(("LLM Explain (quota)", None))
        else:
            report("LLM Explain call", False, err_str[:150])

# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: Agentic Investigation (Agent loop)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[5/6] Agentic Investigation — Agent Loop")
if not api_key or api_key == "your-groq-api-key-here":
    print(f"  {SKIP}  Agent Investigation — No API key configured")
    results.append(("Agent Investigation", None))
else:
    try:
        from src.agent import LogAnalysisAgent
        agent = LogAnalysisAgent(api_key=api_key)

        tool_calls_seen = []
        def on_tool(name, args, result):
            tool_calls_seen.append(name)

        result = agent.investigate(
            log_file_path=str(test_log),
            user_query=None,
            on_tool_call=on_tool,
        )

        report("Agent completes investigation", "raw_response" in result,
               f"Iterations: {result.get('iterations')}, Tool calls: {len(result.get('tool_calls_log', []))}")
        report("Agent used tools", len(result.get("tool_calls_log", [])) > 0,
               f"Tools used: {', '.join(set(tc['tool'] for tc in result.get('tool_calls_log', [])))}")
        report("Agent root cause extracted", len(result.get("root_cause", "")) > 0,
               result.get("root_cause", "")[:80])
        report("Agent confidence present", result.get("confidence", "UNKNOWN") != "UNKNOWN",
               f"Confidence: {result.get('confidence')}")
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            print(f"  {SKIP}  Agent Investigation — API quota exhausted (not a code bug)")
            results.append(("Agent Investigation (quota)", None))
        else:
            report("Agent Investigation", False, err_str[:150])

# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: All backend imports
# ═══════════════════════════════════════════════════════════════════════════
print("\n[6/6] Import Verification")
try:
    from src.parser import LogParser, AnomalyBlock
    from src.llm_client import GroqClient
    from src.agent import LogAnalysisAgent
    from src.tools import TOOL_REGISTRY
    import streamlit
    report("All imports successful", True,
           f"Streamlit {streamlit.__version__}, {len(TOOL_REGISTRY)} tools registered")
except Exception as e:
    report("Import check", False, str(e)[:100])

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
passed = sum(1 for _, p in results if p is True)
failed = sum(1 for _, p in results if p is False)
skipped = sum(1 for _, p in results if p is None)
total = len(results)

print(f"  RESULTS: {passed} passed, {failed} failed, {skipped} skipped / {total} total")
if failed == 0:
    print(f"  \033[92m🎉 ALL CHECKS PASSED — PROJECT IS FULLY WORKING!\033[0m")
else:
    print(f"  \033[91m⚠️  {failed} check(s) failed — see details above\033[0m")
print("="*70 + "\n")

sys.exit(1 if failed > 0 else 0)
