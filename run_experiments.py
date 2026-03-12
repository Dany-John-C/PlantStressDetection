import os
import sys
import argparse
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich.rule import Rule
from rich import box
from rich.columns import Columns
from rich.live import Live

console = Console()

BANNER = """
[bold green]
  ____  _             _     ____  _
 |  _ \\| | __ _ _ __ | |_  / ___|| |_ _ __ ___  ___ ___
 | |_) | |/ _` | '_ \\| __| \\___ \\| __| '__/ _ \\/ __/ __|
 |  __/| | (_| | | | | |_   ___) | |_| | |  __/\\__ \\__ \\
 |_|   |_|\\__,_|_| |_|\\__| |____/ \\__|_|  \\___||___/___/
[/bold green]
[bold cyan]         Detection  ·  Classification  ·  Analytics[/bold cyan]
"""

def print_banner():
    console.print(BANNER)
    console.print(Panel.fit(
        "[bold white]Early Plant Stress Detection Framework[/bold white]\n"
        "[dim]Texture Entropy Drift · Deep Feature Embedding · MobileNetV3[/dim]",
        border_style="green",
        padding=(0, 4),
    ))
    console.print()

def run_step(label: str, func, *args, **kwargs):
    console.print(Rule(f"[bold yellow]{label}[/bold yellow]", style="yellow"))
    start = time.perf_counter()
    result = None
    try:
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        console.print(f"  [bold green]✓[/bold green] Completed in [cyan]{elapsed:.1f}s[/cyan]\n")
    except Exception as e:
        elapsed = time.perf_counter() - start
        console.print(f"  [bold red]✗[/bold red] Failed after [cyan]{elapsed:.1f}s[/cyan]: [red]{e}[/red]\n")
    return result

def print_summary(results: dict):
    console.print()
    console.print(Rule("[bold green]Experiment Summary[/bold green]", style="green"))
    console.print()

    table = Table(
        title="[bold]Pipeline & Study Results[/bold]",
        box=box.ROUNDED,
        border_style="green",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Step", style="white", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Duration", justify="right", style="cyan")

    icons = {"ok": "[bold green]✓ Done[/bold green]", "fail": "[bold red]✗ Failed[/bold red]", "skip": "[dim]– Skipped[/dim]"}

    for name, (status, dur) in results.items():
        table.add_row(name, icons.get(status, status), f"{dur:.1f}s" if dur else "—")

    console.print(table)
    console.print()
    console.print(Panel(
        "[bold green]All experiments finished.[/bold green]\n"
        "[dim]Results saved under [cyan]results/[/cyan] · Models saved under [cyan]models/[/cyan][/dim]\n"
        "[dim]Run [cyan]streamlit run app.py[/cyan] to open the interactive dashboard.[/dim]",
        border_style="green",
        padding=(0, 2),
    ))

def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="Run Early Plant Stress Detection Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate dummy data and run a quick test")
    parser.add_argument("--skip-data", action="store_true", help="Skip dataset generation")
    parser.add_argument("--only", nargs="+", choices=["p1","p2","p3","p4","p5","p6","cross","robust","deploy","stat","paper"],
                        help="Run only specific steps (e.g. --only p1 p2 p6)")
    args = parser.parse_args()

    only = set(args.only) if args.only else None
    results = {}

    def should_run(key: str) -> bool:
        return only is None or key in only

    # ── Dataset Generation ─────────────────────────────────────────────────────
    if args.dry_run and not args.skip_data:
        from tools.generate_dummy_data import create_dummy_dataset
        t0 = time.perf_counter()
        try:
            console.print(Rule("[bold yellow]Generating Dummy Dataset[/bold yellow]", style="yellow"))
            create_dummy_dataset()
            d = time.perf_counter() - t0
            console.print(f"  [bold green]✓[/bold green] Dataset ready in [cyan]{d:.1f}s[/cyan]\n")
            results["Dataset Generation"] = ("ok", d)
        except Exception as e:
            d = time.perf_counter() - t0
            console.print(f"  [bold red]✗[/bold red] Failed: [red]{e}[/red]\n")
            results["Dataset Generation"] = ("fail", d)

    # ── Pipelines ──────────────────────────────────────────────────────────────
    if should_run("p1"):
        from experiments.texture_svm import run_texture_svm
        t0 = time.perf_counter()
        try:
            console.print(Rule("[bold yellow]Pipeline 1 · Classical Texture + SVM[/bold yellow]", style="yellow"))
            run_texture_svm()
            d = time.perf_counter() - t0
            console.print(f"  [bold green]✓[/bold green] Done in [cyan]{d:.1f}s[/cyan]\n")
            results["P1 · Texture + SVM"] = ("ok", d)
        except Exception as e:
            d = time.perf_counter() - t0
            console.print(f"  [bold red]✗[/bold red] [red]{e}[/red]\n")
            results["P1 · Texture + SVM"] = ("fail", d)

    if should_run("p2"):
        from experiments.mobilenet_classifier import run_mobilenet_pipeline
        t0 = time.perf_counter()
        try:
            console.print(Rule("[bold yellow]Pipeline 2 · MobileNetV3 Classification[/bold yellow]", style="yellow"))
            run_mobilenet_pipeline()
            d = time.perf_counter() - t0
            console.print(f"  [bold green]✓[/bold green] Done in [cyan]{d:.1f}s[/cyan]\n")
            results["P2 · MobileNetV3"] = ("ok", d)
        except Exception as e:
            d = time.perf_counter() - t0
            console.print(f"  [bold red]✗[/bold red] [red]{e}[/red]\n")
            results["P2 · MobileNetV3"] = ("fail", d)

    if should_run("p3"):
        from experiments.severity_regression import run_regression_pipeline
        t0 = time.perf_counter()
        try:
            console.print(Rule("[bold yellow]Pipeline 3 · Severity Regression[/bold yellow]", style="yellow"))
            run_regression_pipeline()
            d = time.perf_counter() - t0
            console.print(f"  [bold green]✓[/bold green] Done in [cyan]{d:.1f}s[/cyan]\n")
            results["P3 · Severity Regression"] = ("ok", d)
        except Exception as e:
            d = time.perf_counter() - t0
            console.print(f"  [bold red]✗[/bold red] [red]{e}[/red]\n")
            results["P3 · Severity Regression"] = ("fail", d)

    if should_run("p4"):
        from experiments.embedding_anomaly import run_anomaly_pipeline
        t0 = time.perf_counter()
        try:
            console.print(Rule("[bold yellow]Pipeline 4 · Deep Embedding Anomaly Detection[/bold yellow]", style="yellow"))
            run_anomaly_pipeline()
            d = time.perf_counter() - t0
            console.print(f"  [bold green]✓[/bold green] Done in [cyan]{d:.1f}s[/cyan]\n")
            results["P4 · Anomaly Detection"] = ("ok", d)
        except Exception as e:
            d = time.perf_counter() - t0
            console.print(f"  [bold red]✗[/bold red] [red]{e}[/red]\n")
            results["P4 · Anomaly Detection"] = ("fail", d)

    if should_run("p5"):
        from experiments.entropy_drift import run_entropy_drift
        t0 = time.perf_counter()
        try:
            console.print(Rule("[bold yellow]Pipeline 5 · Texture Entropy Drift[/bold yellow]", style="yellow"))
            run_entropy_drift()
            d = time.perf_counter() - t0
            console.print(f"  [bold green]✓[/bold green] Done in [cyan]{d:.1f}s[/cyan]\n")
            results["P5 · Entropy Drift"] = ("ok", d)
        except Exception as e:
            d = time.perf_counter() - t0
            console.print(f"  [bold red]✗[/bold red] [red]{e}[/red]\n")
            results["P5 · Entropy Drift"] = ("fail", d)

    if should_run("p6"):
        from experiments.alexnet_classifier import run_alexnet_pipeline
        t0 = time.perf_counter()
        try:
            console.print(Rule("[bold yellow]Pipeline 6 · Custom AlexNet Classification[/bold yellow]", style="yellow"))
            run_alexnet_pipeline()
            d = time.perf_counter() - t0
            console.print(f"  [bold green]✓[/bold green] Done in [cyan]{d:.1f}s[/cyan]\n")
            results["P6 · AlexNet"] = ("ok", d)
        except Exception as e:
            d = time.perf_counter() - t0
            console.print(f"  [bold red]✗[/bold red] [red]{e}[/red]\n")
            results["P6 · AlexNet"] = ("fail", d)

    # ── Studies ────────────────────────────────────────────────────────────────
    if should_run("cross"):
        from studies.cross_crop import run_cross_crop_study
        t0 = time.perf_counter()
        try:
            console.print(Rule("[bold yellow]Study · Cross-Crop Generalisation[/bold yellow]", style="yellow"))
            run_cross_crop_study()
            d = time.perf_counter() - t0
            console.print(f"  [bold green]✓[/bold green] Done in [cyan]{d:.1f}s[/cyan]\n")
            results["Study · Cross-Crop"] = ("ok", d)
        except Exception as e:
            d = time.perf_counter() - t0
            console.print(f"  [bold red]✗[/bold red] [red]{e}[/red]\n")
            results["Study · Cross-Crop"] = ("fail", d)

    if should_run("robust"):
        from studies.robustness import run_robustness_study
        t0 = time.perf_counter()
        try:
            console.print(Rule("[bold yellow]Study · Robustness to Perturbations[/bold yellow]", style="yellow"))
            run_robustness_study()
            d = time.perf_counter() - t0
            console.print(f"  [bold green]✓[/bold green] Done in [cyan]{d:.1f}s[/cyan]\n")
            results["Study · Robustness"] = ("ok", d)
        except Exception as e:
            d = time.perf_counter() - t0
            console.print(f"  [bold red]✗[/bold red] [red]{e}[/red]\n")
            results["Study · Robustness"] = ("fail", d)

    if should_run("deploy"):
        from studies.deployment import run_deployment_analysis
        t0 = time.perf_counter()
        try:
            console.print(Rule("[bold yellow]Study · Edge Deployment Feasibility[/bold yellow]", style="yellow"))
            run_deployment_analysis()
            d = time.perf_counter() - t0
            console.print(f"  [bold green]✓[/bold green] Done in [cyan]{d:.1f}s[/cyan]\n")
            results["Study · Deployment"] = ("ok", d)
        except Exception as e:
            d = time.perf_counter() - t0
            console.print(f"  [bold red]✗[/bold red] [red]{e}[/red]\n")
            results["Study · Deployment"] = ("fail", d)

    if should_run("stat"):
        from experiments.statistical_validation import run_statistical_validation
        t0 = time.perf_counter()
        try:
            console.print(Rule("[bold yellow]Statistical Validation[/bold yellow]", style="yellow"))
            run_statistical_validation()
            d = time.perf_counter() - t0
            console.print(f"  [bold green]✓[/bold green] Done in [cyan]{d:.1f}s[/cyan]\n")
            results["Statistical Validation"] = ("ok", d)
        except Exception as e:
            d = time.perf_counter() - t0
            console.print(f"  [bold red]✗[/bold red] [red]{e}[/red]\n")
            results["Statistical Validation"] = ("fail", d)

    if should_run("paper"):
        from experiments.generate_paper_outputs import generate_paper_outputs
        t0 = time.perf_counter()
        try:
            console.print(Rule("[bold yellow]Generating Research Outputs[/bold yellow]", style="yellow"))
            generate_paper_outputs()
            d = time.perf_counter() - t0
            console.print(f"  [bold green]✓[/bold green] Outputs written to [cyan]results/[/cyan] in {d:.1f}s\n")
            results["Paper Generation"] = ("ok", d)
        except Exception as e:
            d = time.perf_counter() - t0
            console.print(f"  [bold red]✗[/bold red] [red]{e}[/red]\n")
            results["Paper Generation"] = ("fail", d)

    print_summary(results)


if __name__ == "__main__":
    main()
