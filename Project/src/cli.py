"""
CLI Module
==========
Entry point for the `log-explainer` command.  Uses Click for argument handling
and Rich for beautiful terminal output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

from src.parser import LogParser
from src.llm_client import GroqClient
from src.agent import LogAnalysisAgent

console = Console()


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------
@click.group()
@click.version_option("0.1.0", prog_name="log-explainer")
def main() -> None:
    """🔍 Log File Anomaly Explainer — powered by Groq Cloud (Llama 3.3)."""


# ---------------------------------------------------------------------------
# `explain` command — the primary workflow
# ---------------------------------------------------------------------------
@main.command("explain")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--context",
    "-c",
    default=20,
    show_default=True,
    help="Number of lines of context to extract before/after the anomaly.",
)
@click.option(
    "--model",
    "-m",
    default="llama-3.3-70b-versatile",
    show_default=True,
    help="Groq model to use.",
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["console", "json", "markdown"], case_sensitive=False),
    default="console",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--out-file",
    "-f",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Save output to a file (works with --output json or markdown).",
)
@click.option(
    "--all-anomalies",
    "-a",
    is_flag=True,
    default=False,
    help="Analyse ALL distinct anomaly blocks instead of only the primary one.",
)
def explain(
    file: Path,
    context: int,
    model: str,
    output: str,
    out_file: str | None,
    all_anomalies: bool,
) -> None:
    """
    Analyse FILE for anomalies and explain them using an LLM.

    \b
    Example:
        log-explainer explain app.log
        log-explainer explain app.log --output json --out-file result.json
        log-explainer explain app.log --all-anomalies --context 30
    """
    console.print(
        Panel.fit(
            f"[bold cyan]Log File Anomaly Explainer[/bold cyan]\n"
            f"[dim]File:[/dim] {file}\n"
            f"[dim]Context window:[/dim] ±{context} lines\n"
            f"[dim]Model:[/dim] {model}",
            border_style="cyan",
        )
    )

    # ------------------------------------------------------------------
    # Phase 1: Parse
    # ------------------------------------------------------------------
    parser = LogParser(context_window=context)
    console.print(Rule("[yellow]Scanning log file…[/yellow]"))

    if all_anomalies:
        blocks = parser.scan_all(file)
        if not blocks:
            console.print("[green]✅ No anomalies detected in the log file.[/green]")
            sys.exit(0)
        console.print(
            f"[yellow]⚠[/yellow]  Found [bold]{len(blocks)}[/bold] distinct anomaly cluster(s)."
        )
    else:
        block = parser.parse(file)
        if block is None:
            console.print("[green]✅ No anomalies detected in the log file.[/green]")
            sys.exit(0)
        blocks = [block]
        console.print(
            f"[yellow]⚠[/yellow]  Primary anomaly at line [bold]{block.primary_line_number}[/bold] "
            f"(severity {block.severity}/6)."
        )

    # ------------------------------------------------------------------
    # Phase 2: LLM
    # ------------------------------------------------------------------
    console.print(Rule("[yellow]Sending to Groq Cloud…[/yellow]"))

    try:
        client = GroqClient(model=model)
    except EnvironmentError as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        sys.exit(1)

    results = []
    for i, blk in enumerate(blocks, start=1):
        if len(blocks) > 1:
            console.print(f"\n[bold]Anomaly {i}/{len(blocks)}[/bold] (line {blk.primary_line_number})")
        with console.status("[bold green]Waiting for Groq response…[/bold green]"):
            analysis = client.explain(blk)
        analysis["anomaly_metadata"] = {
            "file": blk.file_path,
            "primary_line": blk.primary_line_number,
            "context_start": blk.context_start,
            "context_end": blk.context_end,
            "severity": blk.severity,
            "primary_line_content": blk.primary_line_content.strip(),
        }
        results.append(analysis)

    # ------------------------------------------------------------------
    # Phase 3: Output
    # ------------------------------------------------------------------
    console.print(Rule("[bold green]Analysis Complete[/bold green]"))

    if output == "console":
        _render_console(results)
    elif output == "json":
        _render_json(results, out_file)
    elif output == "markdown":
        _render_markdown(results, out_file)


# ---------------------------------------------------------------------------
# `scan` command — quick anomaly scan without LLM
# ---------------------------------------------------------------------------
@main.command("scan")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--context", "-c", default=20, show_default=True)
def scan(file: Path, context: int) -> None:
    """
    Quick scan of FILE to list all detected anomalies (no LLM call).

    \b
    Example:
        log-explainer scan app.log
    """
    parser = LogParser(context_window=context)
    blocks = parser.scan_all(file)

    if not blocks:
        console.print("[green]✅ No anomalies found.[/green]")
        return

    table = Table(
        title=f"Anomalies in {file.name}",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("Cluster", justify="center", style="dim", width=8)
    table.add_column("Line #", justify="right", style="bold yellow")
    table.add_column("Severity", justify="center")
    table.add_column("Context Range", justify="center")
    table.add_column("Error Snippet", no_wrap=False, max_width=60)

    for i, blk in enumerate(blocks, 1):
        severity_color = "red" if blk.severity >= 4 else "yellow" if blk.severity >= 2 else "blue"
        table.add_row(
            str(i),
            str(blk.primary_line_number),
            f"[{severity_color}]{blk.severity}/6[/{severity_color}]",
            f"{blk.context_start}–{blk.context_end}",
            blk.primary_line_content.strip()[:120],
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _render_console(results: list[dict]) -> None:
    for i, r in enumerate(results, 1):
        meta = r.get("anomaly_metadata", {})
        if len(results) > 1:
            console.print(f"\n[bold cyan]─── Anomaly #{i} — Line {meta.get('primary_line')} ───[/bold cyan]")

        # Metadata table
        meta_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        meta_table.add_column("Key", style="dim")
        meta_table.add_column("Value")
        meta_table.add_row("File", str(meta.get("file", "N/A")))
        meta_table.add_row("Primary Error Line", str(meta.get("primary_line", "N/A")))
        meta_table.add_row("Context Range", f"{meta.get('context_start')} – {meta.get('context_end')}")
        meta_table.add_row("Severity", f"{meta.get('severity')}/6")
        meta_table.add_row("Model", r.get("model", "N/A"))
        meta_table.add_row("Prompt Tokens", str(r.get("prompt_tokens", "N/A")))
        console.print(meta_table)

        console.print(Panel(meta.get("primary_line_content", ""), title="Primary Error Line", border_style="red"))

        console.print(Panel(Markdown(r.get("root_cause", "_No data_")),   title="🔍 Root Cause Analysis",  border_style="red"))
        console.print(Panel(Markdown(r.get("probable_cause", "_No data_")), title="🤔 Probable Cause",       border_style="yellow"))
        console.print(Panel(Markdown(r.get("remediation", "_No data_")),  title="🛠  Remediation Steps",   border_style="green"))

        confidence = r.get("confidence", "UNKNOWN")
        conf_color = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}.get(confidence.upper(), "dim")
        console.print(f"\n[bold]Confidence:[/bold] [{conf_color}]{confidence}[/{conf_color}]\n")


def _render_json(results: list[dict], out_file: str | None) -> None:
    # Remove raw_response from JSON output (too verbose)
    clean = [{k: v for k, v in r.items() if k != "raw_response"} for r in results]
    payload = json.dumps(clean, indent=2, ensure_ascii=False)
    if out_file:
        Path(out_file).write_text(payload, encoding="utf-8")
        console.print(f"[green]JSON saved to[/green] {out_file}")
    else:
        console.print_json(payload)


def _render_markdown(results: list[dict], out_file: str | None) -> None:
    lines = ["# Log Anomaly Explainer Report\n"]
    for i, r in enumerate(results, 1):
        meta = r.get("anomaly_metadata", {})
        lines.append(f"## Anomaly #{i}\n")
        lines.append(f"- **File:** `{meta.get('file')}`")
        lines.append(f"- **Primary Error Line:** {meta.get('primary_line')}")
        lines.append(f"- **Context:** Lines {meta.get('context_start')} – {meta.get('context_end')}")
        lines.append(f"- **Severity:** {meta.get('severity')}/6")
        lines.append(f"- **Model:** {r.get('model')}")
        lines.append(f"- **Confidence:** {r.get('confidence')}\n")
        lines.append("### 🔍 Root Cause Analysis\n")
        lines.append(r.get("root_cause", "_No data_") + "\n")
        lines.append("### 🤔 Probable Cause\n")
        lines.append(r.get("probable_cause", "_No data_") + "\n")
        lines.append("### 🛠 Remediation Steps\n")
        lines.append(r.get("remediation", "_No data_") + "\n")
        lines.append("---\n")

    content = "\n".join(lines)
    if out_file:
        Path(out_file).write_text(content, encoding="utf-8")
        console.print(f"[green]Markdown report saved to[/green] {out_file}")
    else:
        console.print(Markdown(content))


# ---------------------------------------------------------------------------
# `investigate` command — tool-using agent loop
# ---------------------------------------------------------------------------
@main.command("investigate")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--model",
    "-m",
    default="llama-3.3-70b-versatile",
    show_default=True,
    help="Groq model to use.",
)
@click.option(
    "--query",
    "-q",
    default=None,
    help="Optional question or context to guide the investigation.",
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["console", "json", "markdown"], case_sensitive=False),
    default="console",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--out-file",
    "-f",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Save output to a file (works with --output json or markdown).",
)
def investigate(
    file: Path,
    model: str,
    query: str | None,
    output: str,
    out_file: str | None,
) -> None:
    """
    🤖 Agent mode — LLM autonomously investigates FILE using tools.

    The agent iteratively calls log analysis tools (parse, scan, read lines,
    search patterns) and reasons about the results before producing a final
    structured analysis.

    \b
    Example:
        log-explainer investigate app.log
        log-explainer investigate app.log --query "focus on database errors"
        log-explainer investigate app.log --output json --out-file report.json
    """
    console.print(
        Panel.fit(
            f"[bold magenta]🤖 Agent Investigation Mode[/bold magenta]\n"
            f"[dim]File:[/dim] {file}\n"
            f"[dim]Model:[/dim] {model}\n"
            f"[dim]Query:[/dim] {query or '(auto)'}\n"
            f"[dim]Max iterations:[/dim] 10",
            border_style="magenta",
        )
    )

    try:
        agent = LogAnalysisAgent(model=model)
    except EnvironmentError as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        sys.exit(1)

    # Live progress callback
    def _on_tool_call(tool_name: str, args: dict, result: dict) -> None:
        status = result.get("status", result.get("match_count", "done"))
        console.print(
            f"  [dim]→[/dim] [cyan]{tool_name}[/cyan]"
            f"({', '.join(f'{k}={v!r}' for k, v in args.items() if k != 'file_path')})"
            f" [dim]» {status}[/dim]"
        )

    console.print(Rule("[yellow]Agent is investigating…[/yellow]"))

    with console.status("[bold green]Agent loop running…[/bold green]"):
        result = agent.investigate(
            log_file_path=str(file.resolve()),
            user_query=query,
            on_tool_call=_on_tool_call,
        )

    console.print(Rule("[bold green]Investigation Complete[/bold green]"))

    # Show tool call trace
    if result.get("tool_calls_log"):
        trace_table = Table(
            title="Agent Tool Call Trace",
            box=box.ROUNDED,
            header_style="bold cyan",
            show_lines=True,
        )
        trace_table.add_column("#", justify="center", width=4)
        trace_table.add_column("Iter", justify="center", width=5)
        trace_table.add_column("Tool", style="cyan")
        trace_table.add_column("Arguments", max_width=40)
        for i, tc in enumerate(result["tool_calls_log"], 1):
            args_str = ", ".join(
                f"{k}={v!r}" for k, v in tc["args"].items() if k != "file_path"
            )
            trace_table.add_row(
                str(i),
                str(tc["iteration"]),
                tc["tool"],
                args_str or "(file only)",
            )
        console.print(trace_table)
        console.print(
            f"[dim]Completed in {result.get('iterations', '?')} iteration(s)[/dim]\n"
        )

    # Render analysis
    results_list = [result]
    if output == "console":
        _render_agent_console(result)
    elif output == "json":
        _render_json(results_list, out_file)
    elif output == "markdown":
        _render_markdown(results_list, out_file)


def _render_agent_console(r: dict) -> None:
    """Render agent investigation results to the console."""
    meta_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    meta_table.add_column("Key", style="dim")
    meta_table.add_column("Value")
    meta_table.add_row("Model", r.get("model", "N/A"))
    meta_table.add_row("Iterations", str(r.get("iterations", "N/A")))
    meta_table.add_row("Tool Calls", str(len(r.get("tool_calls_log", []))))
    meta_table.add_row("Prompt Tokens", str(r.get("prompt_tokens", "N/A")))
    console.print(meta_table)

    console.print(Panel(Markdown(r.get("root_cause", "_No data_")),    title="🔍 Root Cause Analysis", border_style="red"))
    console.print(Panel(Markdown(r.get("probable_cause", "_No data_")), title="🤔 Probable Cause",      border_style="yellow"))
    console.print(Panel(Markdown(r.get("remediation", "_No data_")),   title="🛠  Remediation Steps",  border_style="green"))

    confidence = r.get("confidence", "UNKNOWN")
    conf_color = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}.get(confidence.upper(), "dim")
    console.print(f"\n[bold]Confidence:[/bold] [{conf_color}]{confidence}[/{conf_color}]\n")


if __name__ == "__main__":
    main()
