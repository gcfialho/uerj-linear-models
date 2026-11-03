"""Figures for the README: scatter plots and residual diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from linear_models.analysis import fit_linear, fit_loglog
from linear_models.data import positive_rows

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
LINEAR = "#2a78d6"
LOGLOG = "#eb6834"

LABELLED_LINEAR = ("Russia", "Canada", "China", "India")
LABELLED_LOG = ("Russia", "Canada", "China", "India", "Monaco", "Iceland")


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.size": 9,
        }
    )


def _millions(x: float, _pos: int) -> str:
    return "0" if x == 0 else f"{x / 1e6:g}M"


def _dots(ax: plt.Axes, x, y, color: str) -> None:
    ax.scatter(x, y, s=16, color=color, alpha=0.75, edgecolors=SURFACE, linewidths=0.6, zorder=3)


def _label(ax: plt.Axes, df: pd.DataFrame, x: str, y: str, names: tuple[str, ...]) -> None:
    for _, row in df[df["country"].isin(names)].iterrows():
        ax.annotate(
            row["country"],
            (row[x], row[y]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7.5,
            color=INK_2,
        )


def scatter(df: pd.DataFrame, path: Path) -> Path:
    """Land area against each regressor, on linear (top) and log-log (bottom) axes."""
    _style()
    pos = positive_rows(df)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), layout="constrained")
    for col, (xvar, xlabel) in enumerate(
        [("population", "Population (2020)"), ("density", "Density (people / km²)")]
    ):
        top, bottom = axes[0, col], axes[1, col]
        _dots(top, df[xvar], df["land_area"], LINEAR)
        extra = ("Monaco",) if xvar == "density" else ()
        _label(top, df, xvar, "land_area", LABELLED_LINEAR + extra)
        top.set(xlabel=xlabel, ylabel="Land area (km²)", title=f"Land area vs {xvar}, linear axes")
        top.yaxis.set_major_formatter(FuncFormatter(_millions))
        if xvar == "population":
            top.xaxis.set_major_formatter(FuncFormatter(_millions))

        _dots(bottom, pos[xvar], pos["land_area"], LOGLOG)
        _label(bottom, pos, xvar, "land_area", LABELLED_LOG)
        bottom.set(
            xscale="log",
            yscale="log",
            xlabel=xlabel,
            ylabel="Land area (km²)",
            title=f"Land area vs {xvar}, log-log axes",
        )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def residuals(df: pd.DataFrame, path: Path) -> Path:
    """Residuals vs fitted values and residual histograms for both models."""
    _style()
    pos = positive_rows(df)
    linear, loglog = fit_linear(df), fit_loglog(df)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), layout="constrained")

    rows = [
        (linear, df, LINEAR, "Linear model: area ~ population + density", "km²"),
        (loglog, pos, LOGLOG, "Log-log model: log area ~ log population + log density", "log"),
    ]
    for r, (model, data, color, title, unit) in enumerate(rows):
        ax_fit, ax_hist = axes[r]
        _dots(ax_fit, model.fitted, model.resid, color)
        ax_fit.axhline(0, color=INK_2, linewidth=1, zorder=2)
        ax_fit.set(
            xlabel="Fitted value" + (" (km²)" if unit == "km²" else " (log km²)"),
            ylabel="Residual" + (" (km²)" if unit == "km²" else " (log km²)"),
            title=f"{title}\nresiduals vs fitted, R² = {model.r2:.4f}",
        )
        order = np.argsort(-np.abs(model.resid))[:2]
        for i in order:
            ax_fit.annotate(
                data["country"].iloc[i],
                (model.fitted[i], model.resid[i]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7.5,
                color=INK_2,
            )
        if unit == "km²":
            ax_fit.xaxis.set_major_formatter(FuncFormatter(_millions))
            ax_fit.yaxis.set_major_formatter(FuncFormatter(_millions))

        ax_hist.hist(model.resid, bins=40, color=color, edgecolor=SURFACE, linewidth=1, zorder=3)
        ax_hist.axvline(0, color=INK_2, linewidth=1, zorder=4)
        ax_hist.set(
            xlabel="Residual" + (" (km²)" if unit == "km²" else " (log km²)"),
            ylabel="Countries",
            title="Residual histogram",
        )
        if unit == "km²":
            ax_hist.xaxis.set_major_formatter(FuncFormatter(_millions))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def all_figures(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [scatter(df, out_dir / "scatter.png"), residuals(df, out_dir / "residuals.png")]
