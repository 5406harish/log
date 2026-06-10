"""
Log Anomaly Explainer — Streamlit Dashboard
=============================================
A modern web interface for the Log File Anomaly Explainer backend.
Provides three analysis modes: Quick Scan, Explain Anomalies, and
Agentic Investigation, with rich output rendering and download support.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Ensure the project root is importable (for `src.*` imports)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser import LogParser, AnomalyBlock
from src.llm_client import GroqClient
from src.agent import LogAnalysisAgent

# ---------------------------------------------------------------------------
# Load .env for fallback API key
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Log Anomaly Explainer Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a premium dark-themed UI
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* --- Import Google Font --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* --- Global Styles --- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* --- Header gradient --- */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1.05rem;
        margin: 0;
        font-weight: 300;
    }

    /* --- Stat cards --- */
    .stat-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
    }
    .stat-card .stat-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-card .stat-label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.6);
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* --- Anomaly severity badges --- */
    .severity-critical { color: #ff4757; font-weight: 700; }
    .severity-high { color: #ff6b6b; font-weight: 600; }
    .severity-medium { color: #ffa502; font-weight: 600; }
    .severity-low { color: #2ed573; font-weight: 500; }

    /* --- Section headers --- */
    .section-header {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.15) 0%, transparent 100%);
        border-left: 4px solid #667eea;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        font-size: 1.1rem;
    }

    /* --- Tool call trace badge --- */
    .tool-badge {
        display: inline-block;
        background: rgba(102, 126, 234, 0.2);
        border: 1px solid rgba(102, 126, 234, 0.4);
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        color: #667eea;
    }

    /* --- Smooth animations --- */
    .stButton > button {
        transition: all 0.3s ease;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    /* --- Sidebar styling --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #667eea;
    }

    /* --- File uploader area --- */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                           HEADER                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
st.markdown("""
<div class="main-header">
    <h1>🔍 Log Anomaly Explainer Dashboard</h1>
    <p>AI-powered log triage — upload a log file, detect anomalies, and get actionable root-cause analysis powered by Groq Cloud.</p>
</div>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                           SIDEBAR                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    # --- API Key ---
    env_key = os.getenv("GROQ_API_KEY", "")
    api_key_input = st.text_input(
        "🔑 Groq API Key",
        value=env_key,
        type="password",
        help="Enter your Groq Cloud API key (free at console.groq.com). Falls back to the `.env` file if left blank.",
        placeholder="Enter your API key...",
    )
    # Use whichever is available
    api_key = api_key_input.strip() or env_key.strip()

    if api_key:
        st.success("✅ API key configured", icon="🔐")
    else:
        st.warning("⚠️ API key not set — LLM features disabled", icon="🔑")

    st.divider()

    # --- Context window slider ---
    context_window = st.slider(
        "📐 Context Window Size",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
        help="Number of lines before/after the anomaly to include as context (±N lines).",
    )

    st.divider()

    # --- About section ---
    st.markdown("### 📖 About")
    st.markdown(
        "This dashboard is the graphical interface for the "
        "**Log File Anomaly Explainer** project.  \n\n"
        "It uses the `src/` backend modules (parser, LLM client, agent) "
        "to detect and explain anomalies in `.log` files."
    )
    st.markdown("---")
    st.caption("Built with Streamlit • Powered by Groq Cloud")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        FILE UPLOAD                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
st.markdown('<div class="section-header">📁 Upload Log File</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a .log or .txt file",
    type=["log", "txt"],
    help="Upload a server/application log file to analyse.",
    label_visibility="collapsed",
)


# ---------------------------------------------------------------------------
# Helper: save uploaded file to a temporary path so the backend can read it
# ---------------------------------------------------------------------------
def save_uploaded_file(uploaded) -> Path:
    """Persist the uploaded file to a temp directory and return its path."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="log_explainer_"))
    tmp_path = tmp_dir / uploaded.name
    tmp_path.write_bytes(uploaded.getvalue())
    return tmp_path


# ---------------------------------------------------------------------------
# Helper: build a severity badge
# ---------------------------------------------------------------------------
def severity_badge(score: int) -> str:
    """Return a colored severity string."""
    if score >= 5:
        return f"🔴 **{score}/6** (CRITICAL)"
    elif score >= 4:
        return f"🟠 **{score}/6** (HIGH)"
    elif score >= 2:
        return f"🟡 **{score}/6** (MEDIUM)"
    else:
        return f"🟢 **{score}/6** (LOW)"


# ---------------------------------------------------------------------------
# Helper: render structured LLM analysis output
# ---------------------------------------------------------------------------
def render_analysis(result: dict, title_prefix: str = "") -> None:
    """Render a structured analysis dict (from GeminiClient or Agent) using
    Streamlit markdown and expanders."""

    if title_prefix:
        st.markdown(f"### {title_prefix}")

    # --- Root Cause ---
    st.markdown("#### 🔍 Root Cause Analysis")
    root_cause = result.get("root_cause", "_No data available._")
    st.markdown(root_cause if root_cause else "_No data available._")

    # --- Probable Cause ---
    st.markdown("#### 🤔 Probable Cause")
    probable_cause = result.get("probable_cause", "_No data available._")
    st.markdown(probable_cause if probable_cause else "_No data available._")

    # --- Remediation Steps ---
    st.markdown("#### 🛠️ Remediation Steps")
    remediation = result.get("remediation", "_No data available._")
    st.markdown(remediation if remediation else "_No data available._")

    # --- Confidence ---
    confidence = result.get("confidence", "UNKNOWN").strip().upper()
    conf_colors = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
    conf_icon = conf_colors.get(confidence, "⚪")
    st.markdown(f"**Confidence:** {conf_icon} **{confidence}**")

    # --- Metadata expander ---
    with st.expander("📊 Metadata & Raw Output", expanded=False):
        meta_cols = st.columns(3)
        meta_cols[0].metric("Model", result.get("model", "N/A"))
        meta_cols[1].metric("Prompt Tokens", str(result.get("prompt_tokens", "N/A")))
        if "iterations" in result:
            meta_cols[2].metric("Agent Iterations", str(result.get("iterations", "N/A")))

        st.markdown("**Raw LLM Response:**")
        st.code(result.get("raw_response", ""), language="markdown")

        st.markdown("**Full JSON Output:**")
        # Create a JSON-safe copy (remove non-serialisable items)
        json_safe = {
            k: v for k, v in result.items()
            if k != "raw_response"
        }
        st.json(json_safe)


# ---------------------------------------------------------------------------
# Helper: generate a downloadable markdown report
# ---------------------------------------------------------------------------
def build_markdown_report(results: list[dict], mode: str, file_name: str) -> str:
    """Generate a markdown report from analysis results."""
    lines = [
        f"# Log Anomaly Explainer Report",
        f"",
        f"- **File:** `{file_name}`",
        f"- **Mode:** {mode}",
        f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        "---",
        "",
    ]

    for i, r in enumerate(results, 1):
        lines.append(f"## Analysis #{i}")
        lines.append("")

        meta = r.get("anomaly_metadata", {})
        if meta:
            lines.append(f"- **Primary Error Line:** {meta.get('primary_line', 'N/A')}")
            lines.append(f"- **Context Range:** {meta.get('context_start', '?')} – {meta.get('context_end', '?')}")
            lines.append(f"- **Severity:** {meta.get('severity', '?')}/6")
            lines.append("")

        lines.append(f"- **Model:** {r.get('model', 'N/A')}")
        lines.append(f"- **Confidence:** {r.get('confidence', 'UNKNOWN')}")
        if "iterations" in r:
            lines.append(f"- **Agent Iterations:** {r.get('iterations', 'N/A')}")
        lines.append("")

        lines.append("### 🔍 Root Cause Analysis")
        lines.append("")
        lines.append(r.get("root_cause", "_No data_"))
        lines.append("")

        lines.append("### 🤔 Probable Cause")
        lines.append("")
        lines.append(r.get("probable_cause", "_No data_"))
        lines.append("")

        lines.append("### 🛠️ Remediation Steps")
        lines.append("")
        lines.append(r.get("remediation", "_No data_"))
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     MAIN CONTENT AREA                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if uploaded_file is not None:
    # Save uploaded file to disk for backend consumption
    tmp_path = save_uploaded_file(uploaded_file)
    tmp_path_str = str(tmp_path)

    st.success(f"✅ File uploaded: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")

    # Show a preview of the uploaded log
    with st.expander("👁️ Preview uploaded log (first 50 lines)", expanded=False):
        raw_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
        preview_lines = raw_text.splitlines()[:50]
        st.code("\n".join(preview_lines), language="log", line_numbers=True)

    st.divider()

    # ─── Action Buttons ────────────────────────────────────────────────
    st.markdown('<div class="section-header">🚀 Analysis Actions</div>', unsafe_allow_html=True)

    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        scan_clicked = st.button(
            "🔎 Quick Scan",
            help="Scan for all anomalies without calling the LLM (fast, offline).",
            use_container_width=True,
            type="secondary",
        )
    with btn_col2:
        explain_clicked = st.button(
            "🧠 Explain Anomalies",
            help="Parse the primary anomaly and send to Groq for root-cause analysis.",
            use_container_width=True,
            type="primary",
            disabled=not api_key,
        )
    with btn_col3:
        investigate_clicked = st.button(
            "🤖 Agentic Investigation",
            help="Launch the autonomous agent loop — LLM iteratively uses tools to investigate.",
            use_container_width=True,
            type="primary",
            disabled=not api_key,
        )

    st.divider()

    # ════════════════════════════════════════════════════════════════════
    # MODE 1: QUICK SCAN
    # ════════════════════════════════════════════════════════════════════
    if scan_clicked:
        st.markdown('<div class="section-header">🔎 Quick Scan Results</div>', unsafe_allow_html=True)

        with st.spinner("Scanning log file for anomalies..."):
            parser = LogParser(context_window=context_window)
            try:
                blocks = parser.scan_all(tmp_path_str)
            except Exception as exc:
                st.error(f"❌ Error scanning log file: {exc}")
                blocks = []

        if not blocks:
            st.info("✅ No anomalies detected in the log file.")
        else:
            # Summary stats
            stat_cols = st.columns(4)
            with stat_cols[0]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{len(blocks)}</div>
                    <div class="stat-label">Anomaly Clusters</div>
                </div>
                """, unsafe_allow_html=True)
            with stat_cols[1]:
                max_sev = max(b.severity for b in blocks)
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{max_sev}/6</div>
                    <div class="stat-label">Max Severity</div>
                </div>
                """, unsafe_allow_html=True)
            with stat_cols[2]:
                total_lines = blocks[0].total_log_lines if blocks else 0
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{total_lines:,}</div>
                    <div class="stat-label">Total Log Lines</div>
                </div>
                """, unsafe_allow_html=True)
            with stat_cols[3]:
                crit_count = sum(1 for b in blocks if b.severity >= 5)
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{crit_count}</div>
                    <div class="stat-label">Critical Issues</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")

            # Anomaly table
            import pandas as pd
            table_data = []
            for i, blk in enumerate(blocks, 1):
                table_data.append({
                    "Cluster": i,
                    "Line #": blk.primary_line_number,
                    "Severity": f"{blk.severity}/6",
                    "Context Range": f"{blk.context_start} – {blk.context_end}",
                    "Error Snippet": blk.primary_line_content.strip()[:120],
                })

            df = pd.DataFrame(table_data)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Cluster": st.column_config.NumberColumn(width="small"),
                    "Line #": st.column_config.NumberColumn(width="small"),
                    "Severity": st.column_config.TextColumn(width="small"),
                    "Context Range": st.column_config.TextColumn(width="medium"),
                    "Error Snippet": st.column_config.TextColumn(width="large"),
                },
            )

            # Expandable details for each anomaly
            for i, blk in enumerate(blocks, 1):
                with st.expander(
                    f"📋 Cluster {i} — Line {blk.primary_line_number} — {severity_badge(blk.severity)}",
                    expanded=False,
                ):
                    st.markdown(f"**Primary Error Line:**")
                    st.code(blk.primary_line_content.strip(), language="log")
                    st.markdown(f"**Context (lines {blk.context_start}–{blk.context_end}):**")
                    st.code(blk.formatted_context, language="log", line_numbers=False)

    # ════════════════════════════════════════════════════════════════════
    # MODE 2: EXPLAIN ANOMALIES (LLM)
    # ════════════════════════════════════════════════════════════════════
    if explain_clicked:
        st.markdown('<div class="section-header">🧠 LLM Anomaly Explanation</div>', unsafe_allow_html=True)

        # Parse first
        with st.spinner("Scanning log for the primary anomaly..."):
            parser = LogParser(context_window=context_window)
            try:
                block = parser.parse(tmp_path_str)
            except Exception as exc:
                st.error(f"❌ Error parsing log file: {exc}")
                block = None

        if block is None:
            st.info("✅ No anomalies detected — nothing to explain.")
        else:
            # Show what was found
            st.markdown(f"**Primary anomaly found at line {block.primary_line_number}** — {severity_badge(block.severity)}")
            st.code(block.primary_line_content.strip(), language="log")

            # Call the LLM
            with st.spinner("🧠 Sending to Groq Cloud for analysis... This may take a moment."):
                try:
                    client = GroqClient(api_key=api_key)
                    analysis = client.explain(block)
                    analysis["anomaly_metadata"] = {
                        "file": uploaded_file.name,
                        "primary_line": block.primary_line_number,
                        "context_start": block.context_start,
                        "context_end": block.context_end,
                        "severity": block.severity,
                        "primary_line_content": block.primary_line_content.strip(),
                    }
                except Exception as exc:
                    exc_str = str(exc)
                    if "429" in exc_str or "rate" in exc_str.lower():
                        st.error(
                            "❌ **Rate Limited** — Groq API rate limit hit. "
                            "The app retried automatically but the limit persists.\n\n"
                            "**Options:**\n"
                            "- ⏳ Wait a moment and try again\n"
                            "- 💳 Check your plan at "
                            "[console.groq.com](https://console.groq.com)\n"
                            "- 🔎 Use **Quick Scan** (no API key needed) in the meantime"
                        )
                    else:
                        st.error(f"❌ LLM Error: {exc}")
                    analysis = None

            if analysis:
                st.success("✅ Analysis complete!")
                render_analysis(analysis)

                # Download buttons
                st.divider()
                st.markdown("#### 📥 Download Report")
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    md_report = build_markdown_report([analysis], "Explain Anomalies", uploaded_file.name)
                    st.download_button(
                        label="📄 Download as Markdown",
                        data=md_report,
                        file_name=f"anomaly_report_{uploaded_file.name}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with dl_col2:
                    json_safe = {k: v for k, v in analysis.items() if k != "raw_response"}
                    st.download_button(
                        label="📋 Download as JSON",
                        data=json.dumps(json_safe, indent=2, ensure_ascii=False),
                        file_name=f"anomaly_report_{uploaded_file.name}.json",
                        mime="application/json",
                        use_container_width=True,
                    )

    # ════════════════════════════════════════════════════════════════════
    # MODE 3: AGENTIC INVESTIGATION
    # ════════════════════════════════════════════════════════════════════
    if investigate_clicked:
        st.markdown('<div class="section-header">🤖 Agentic Investigation</div>', unsafe_allow_html=True)

        # Tool call log placeholder
        tool_call_container = st.container()
        tool_call_container.markdown("**Agent tool calls will appear here as they execute...**")

        # Collect tool calls in session state for live display
        if "agent_tool_calls" not in st.session_state:
            st.session_state.agent_tool_calls = []
        st.session_state.agent_tool_calls = []

        with st.spinner("🤖 Agent is investigating the log file... This may take a while."):
            try:
                agent = LogAnalysisAgent(api_key=api_key)

                # Run the agent investigation
                result = agent.investigate(
                    log_file_path=tmp_path_str,
                    user_query=None,
                )
            except Exception as exc:
                    exc_str = str(exc)
                    if "429" in exc_str or "rate" in exc_str.lower():
                        st.error(
                            "❌ **Rate Limited** — Groq API rate limit hit. "
                            "The agent retried automatically but the limit persists.\n\n"
                            "**Options:**\n"
                            "- ⏳ Wait a moment and try again\n"
                            "- 💳 Check your plan at "
                            "[console.groq.com](https://console.groq.com)\n"
                            "- 🔎 Use **Quick Scan** (no API key needed) in the meantime"
                        )
                    else:
                        st.error(f"❌ Agent Error: {exc}")
                    result = None

        if result:
            st.success(
                f"✅ Investigation complete! "
                f"({result.get('iterations', '?')} iterations, "
                f"{len(result.get('tool_calls_log', []))} tool calls)"
            )

            # Show tool call trace
            tool_log = result.get("tool_calls_log", [])
            if tool_log:
                with st.expander("🔧 Agent Tool Call Trace", expanded=True):
                    import pandas as pd
                    trace_data = []
                    for i, tc in enumerate(tool_log, 1):
                        args_str = ", ".join(
                            f"{k}={v!r}" for k, v in tc["args"].items()
                            if k != "file_path"
                        )
                        trace_data.append({
                            "#": i,
                            "Iteration": tc["iteration"],
                            "Tool": tc["tool"],
                            "Arguments": args_str or "(file only)",
                            "Result Preview": tc.get("result_preview", "")[:100],
                        })
                    trace_df = pd.DataFrame(trace_data)
                    st.dataframe(trace_df, use_container_width=True, hide_index=True)

            # Render the analysis
            render_analysis(result)

            # Download buttons
            st.divider()
            st.markdown("#### 📥 Download Report")
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                md_report = build_markdown_report([result], "Agentic Investigation", uploaded_file.name)
                st.download_button(
                    label="📄 Download as Markdown",
                    data=md_report,
                    file_name=f"agent_report_{uploaded_file.name}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with dl_col2:
                json_safe = {k: v for k, v in result.items() if k != "raw_response"}
                st.download_button(
                    label="📋 Download as JSON",
                    data=json.dumps(json_safe, indent=2, ensure_ascii=False),
                    file_name=f"agent_report_{uploaded_file.name}.json",
                    mime="application/json",
                    use_container_width=True,
                )

else:
    # ─── No file uploaded yet — show instructions ─────────────────────
    st.markdown("""
    <div style="
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        border: 2px dashed rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        margin: 2rem 0;
    ">
        <h2 style="color: #667eea; margin-bottom: 0.5rem;">📂 Upload a Log File to Get Started</h2>
        <p style="color: rgba(255,255,255,0.6); font-size: 1.05rem; max-width: 600px; margin: 0 auto;">
            Drag and drop a <code>.log</code> or <code>.txt</code> file above, then choose an analysis mode.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    st.markdown("")
    feat_cols = st.columns(3)

    with feat_cols[0]:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔎</div>
            <div class="stat-value" style="font-size: 1.2rem;">Quick Scan</div>
            <div class="stat-label" style="text-transform: none; letter-spacing: normal; margin-top: 0.5rem;">
                Detect all anomalies instantly.<br/>No API key required.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with feat_cols[1]:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🧠</div>
            <div class="stat-value" style="font-size: 1.2rem;">Explain</div>
            <div class="stat-label" style="text-transform: none; letter-spacing: normal; margin-top: 0.5rem;">
                Get LLM-powered root cause<br/>analysis with remediation steps.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with feat_cols[2]:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🤖</div>
            <div class="stat-value" style="font-size: 1.2rem;">Investigate</div>
            <div class="stat-label" style="text-transform: none; letter-spacing: normal; margin-top: 0.5rem;">
                Autonomous agent uses tools<br/>to deeply analyse your logs.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          FOOTER                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: rgba(255,255,255,0.4); font-size: 0.85rem;">'
    '🔍 Log Anomaly Explainer Dashboard v0.1.0 &nbsp;•&nbsp; '
    'Powered by <a href="https://console.groq.com" style="color: #667eea;">Groq Cloud</a> '
    '&nbsp;•&nbsp; Built with <a href="https://streamlit.io" style="color: #667eea;">Streamlit</a>'
    '</div>',
    unsafe_allow_html=True,
)
