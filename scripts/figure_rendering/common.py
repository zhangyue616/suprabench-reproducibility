"""Portable output and rendering helpers for frozen display inputs."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DISPLAY = REPO / "data" / "figure_display"
OUTPUT = None
FORMATS = ("png", "pdf", "svg")
FIGURE1_FORMATS = ("png", "pdf")
FONT_FAMILY = "Arial"
FONT_FALLBACK = False
PDF_RASTERIZER_PYTHON = None


def configure(output, formats, pdf_rasterizer_python=None):
    global OUTPUT, FORMATS, FONT_FAMILY, FONT_FALLBACK, PDF_RASTERIZER_PYTHON
    OUTPUT, FORMATS = Path(output), tuple(formats)
    PDF_RASTERIZER_PYTHON = pdf_rasterizer_python
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError:
        FONT_FAMILY, FONT_FALLBACK = "DejaVu Sans", True
        print("Arial is unavailable; rendering uses DejaVu Sans. Font metrics may differ.",
              file=sys.stderr)


def save_figure(fig, stem):
    if OUTPUT is None:
        raise RuntimeError("Output must be configured by the CLI")
    for suffix in FORMATS:
        fig.savefig(OUTPUT / f"{stem}.{suffix}", dpi=600, facecolor="white",
                    edgecolor="none")
    plt.close(fig)


def render_figure1():
    """Export the Figure 1 assets without rebuilding or converting them."""
    if OUTPUT is None:
        raise RuntimeError("Output must be configured by the CLI")
    formats = [suffix for suffix in FORMATS if suffix in FIGURE1_FORMATS]
    if not formats:
        raise ValueError("Figure 1 provides PNG/PDF assets only")
    sources = [REPO / "figures" / f"Figure_1.{suffix}" for suffix in formats]
    for source in sources:
        if not source.is_file() or not source.stat().st_size:
            raise RuntimeError("Figure 1 asset is missing or empty: " + str(source))
    for source in sources:
        shutil.copyfile(source, OUTPUT / source.name)
    return {"kind": "asset_export",
            "sources": [source.relative_to(REPO).as_posix() for source in sources],
            "limitation": "Exports PNG/PDF bytes; does not rebuild the production sources."}
