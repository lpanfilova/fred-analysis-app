"""Export economic comparison figures without opening a GUI."""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure
import pandas as pd

from comparison import CPI_RATE, GDP_RATE


def create_comparison_figure(report):
    """Plot quarterly rates and their paired relationship; preserve missing gaps."""
    history = report["history"]
    if not history:
        raise ValueError("Comparison history is required to create a chart.")
    frame = pd.DataFrame(history)
    dates = pd.to_datetime(frame["quarter_start"])
    for column in (GDP_RATE, CPI_RATE):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    paired = frame.dropna(subset=[GDP_RATE, CPI_RATE])
    figure = Figure(figsize=(12, 8.5), facecolor="white")
    FigureCanvasAgg(figure)
    time_axis, scatter_axis = figure.subplots(2, 1, gridspec_kw={"height_ratios": [1.5, 1]})
    figure.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.25, hspace=0.55)
    figure.suptitle("Nominal GDP growth and Core CPI inflation", fontsize=17, fontweight="bold", y=0.97)
    figure.text(0.5, 0.925, f"Quarterly year-over-year rates | {history[0]['quarter']}–{history[-1]['quarter']}", ha="center", fontsize=11)
    time_axis.plot(dates, frame[GDP_RATE], color="#2455a4", linewidth=2, marker=".", label="Nominal GDP growth")
    time_axis.plot(dates, frame[CPI_RATE], color="#bc4b19", linewidth=2, marker=".", label="Core CPI inflation (quarterly average index)")
    time_axis.axhline(0, color="#666666", linewidth=0.8)
    time_axis.set_ylabel("Year-over-year change (%)")
    locator = AutoDateLocator(minticks=3, maxticks=10)
    time_axis.xaxis.set_major_locator(locator)
    time_axis.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    time_axis.legend(loc="best", fontsize=9)
    scatter_axis.scatter(paired[GDP_RATE], paired[CPI_RATE], color="#2455a4", alpha=0.75, s=30)
    scatter_axis.set_xlabel("Nominal GDP growth, year over year (%)")
    scatter_axis.set_ylabel("Core CPI inflation (%)")
    relation = report["relationship"]
    correlation = relation["pearson_correlation"]
    value = f"{correlation:.3f}" if correlation is not None else "unavailable"
    scatter_axis.set_title(f"Same-quarter relationship | Pearson r: {value} | paired quarters: {relation['paired_quarters']}", fontsize=10)
    for axis in (time_axis, scatter_axis):
        axis.grid(True, alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.text(0.09, 0.095,
                "GDP: nominal, current-dollar GDP. Core CPI: year-over-year change in the quarterly average of three monthly indices.\n"
                "Both source series are seasonally adjusted. Missing or incomplete quarters remain gaps; correlation uses complete pairs.\n"
                "Correlation describes association and does not establish causation.", fontsize=9, linespacing=1.5)
    figure.text(0.09, 0.025, "Sources: BEA and BLS via FRED | fred.stlouisfed.org/series/GDP | fred.stlouisfed.org/series/CPILFESL", fontsize=8, color="#555555")
    return figure


def save_comparison_chart(report, path):
    """Save a standalone PNG, SVG, or PDF comparison figure."""
    destination = Path(path)
    if destination.suffix.lower() not in {".png", ".svg", ".pdf"}:
        raise ValueError("Chart output must use a .png, .svg, or .pdf extension.")
    figure = create_comparison_figure(report)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160, format=destination.suffix[1:].lower())
    finally:
        figure.clear()
    return destination.resolve()
