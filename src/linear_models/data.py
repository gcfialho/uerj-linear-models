"""Loading the 2020 population-by-country table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_FILE = "population_by_country_2020.csv"

COLUMNS = {
    "Country (or dependency)": "country",
    "Population (2020)": "population",
    "Density (P/Km²)": "density",
    "Land Area (Km²)": "land_area",
}
"""Source column -> name used in the code. Y is ``land_area``; X1, X2 are the other two."""


def default_path() -> Path:
    """``data/`` at the repository root (src layout), falling back to ``./data``."""
    repo_data = Path(__file__).resolve().parents[2] / "data" / DATA_FILE
    return repo_data if repo_data.exists() else Path.cwd() / "data" / DATA_FILE


def load(path: str | Path | None = None) -> pd.DataFrame:
    """Return ``country``, ``population``, ``density`` and ``land_area``, one row per country."""
    df = pd.read_csv(path or default_path(), usecols=list(COLUMNS))
    df = df.rename(columns=COLUMNS)[list(COLUMNS.values())]
    for col in ("population", "density", "land_area"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
    if df[["population", "density", "land_area"]].lt(0).any().any():
        raise ValueError("negative population, density or land area")
    return df.reset_index(drop=True)


def positive_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where all three variables are strictly positive (needed for logarithms)."""
    mask = (df[["population", "density", "land_area"]] > 0).all(axis=1)
    return df[mask].reset_index(drop=True)
