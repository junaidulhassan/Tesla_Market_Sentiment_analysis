"""Shared colour tokens (validated categorical order from the dataviz palette) and matplotlib style."""

from __future__ import annotations

# Categorical slots, in fixed order. Candles use blue/orange (colour-vision safe) rather than green/red.
PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEUTRAL = "#8a8a85"
GRID = "rgba(128,128,128,0.18)"

ACTUAL, FORECAST, BAND = PALETTE["blue"], PALETTE["orange"], PALETTE["orange"]


def apply_matplotlib_style() -> None:
    """Minimal, recessive-grid matplotlib/seaborn style consistent with the Plotly figures."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.figsize": (11, 4.6),
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.7,
        "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.labelsize": 10.5,
        "axes.prop_cycle": mpl.cycler(color=list(PALETTE.values())),
        "legend.frameon": False, "lines.linewidth": 1.8, "font.size": 10.5,
    })
