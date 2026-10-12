"""Shared helpers: bundled-data loading and chart styling.

Chart labels are kept in ASCII/English on purpose: it keeps the figures
reproducible on any machine without installing CJK fonts.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
IMAGE_DIR = REPO_ROOT / "images"
REPORT_DIR = REPO_ROOT / "reports"

SAMPLE_CSV = DATA_DIR / "BTC-USDT_1H.csv"

#: Fee schedule used throughout (OKX spot, base tier).
FEE_TAKER = 0.0010   # 0.10% - you cross the spread
FEE_MAKER = 0.0008   # 0.08% - you provide liquidity
HALF_SPREAD = 0.00005


def load_csv(path: str | os.PathLike = SAMPLE_CSV) -> pd.DataFrame:
    """Load an OHLC CSV with a ``dt`` column; returns a sorted DataFrame."""
    df = pd.read_csv(path)
    if "dt" not in df.columns:
        raise ValueError(f"{path} has no 'dt' column")
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    df = df.sort_values("dt").reset_index(drop=True)
    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def style() -> None:
    """Apply a consistent, readable matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "legend.frameon": False,
    })


def savefig(fig, name: str) -> Path:
    """Save a figure into ``images/`` and return the path."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGE_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def save_csv(df: pd.DataFrame, name: str) -> Path:
    """Save a result table into ``reports/`` and return the path."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / name
    df.to_csv(path, index=False)
    return path


def fmt_pct(x: float, digits: int = 2) -> str:
    """Format a fraction as a percentage string."""
    return f"{x * 100:.{digits}f}%"
