"""
Shared dark-theme matplotlib style for all pipeline plots.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
BORDER     = "#30363d"
TEXT       = "#e6edf3"
MUTED      = "#8b949e"
GREEN      = "#3fb950"
BLUE       = "#388bfd"
YELLOW     = "#d29922"
RED        = "#f85149"
PALETTE    = [GREEN, BLUE, YELLOW, RED, "#bc8cff", "#79c0ff", "#ffa657"]


def apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":    DARK_BG,
        "axes.facecolor":      PANEL_BG,
        "axes.edgecolor":      BORDER,
        "axes.labelcolor":     TEXT,
        "axes.titlecolor":     TEXT,
        "axes.titlesize":      13,
        "axes.labelsize":      11,
        "axes.titlepad":       12,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "xtick.color":         MUTED,
        "ytick.color":         MUTED,
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "text.color":          TEXT,
        "legend.facecolor":    PANEL_BG,
        "legend.edgecolor":    BORDER,
        "legend.labelcolor":   TEXT,
        "legend.fontsize":     9,
        "grid.color":          BORDER,
        "grid.linestyle":      "--",
        "grid.alpha":          0.6,
        "lines.linewidth":     2.0,
        "patch.linewidth":     0,
        "font.family":         "DejaVu Sans",
        "savefig.facecolor":   DARK_BG,
        "savefig.bbox":        "tight",
        "savefig.dpi":         150,
    })

apply_dark_style()


def new_fig(ncols=1, nrows=1, w=12, h=5):
    return plt.subplots(nrows, ncols, figsize=(w, h))


def bar_label(ax, rects, fmt="{:.4f}", pad=0.005):
    for r in rects:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2., h + pad,
                fmt.format(h), ha="center", va="bottom",
                fontsize=9, color=TEXT)


def save_close(path):
    plt.tight_layout()
    plt.savefig(path)
    plt.close("all")
