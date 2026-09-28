"""One chart style for the notebooks and the deck: a single palette, thin marks, recessive axes.

Palette validated with the dataviz validator against SURFACE (blue/orange/aqua pass all-pairs CVD checks;
aqua is below 3:1 contrast, so it is only used with direct labels). Emphasis = BLUE for the story, GRAY for context.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import FuncFormatter

from bank_marketing.config import FIGURES_DIR

SURFACE = "#FAF9F6"
INK = "#0B0B0B"
INK_2 = "#52514E"
MUTED = "#898781"
GRID = "#E1E0D9"
BASELINE = "#C3C2B7"

BLUE = "#2A78D6"
ORANGE = "#EB6834"
AQUA = "#1BAF7A"
GRAY = MUTED
BLUE_RAMP = ["#CDE2FB", "#9EC5F4", "#6DA7EC", "#3987E5", "#256ABF", "#184F95", "#0D366B"]
MONTHS_PT = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.titlecolor": INK,
            "axes.labelsize": 13,
            "axes.labelcolor": INK_2,
            "text.color": INK,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.color": BASELINE,
            "ytick.color": BASELINE,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "grid.linestyle": "-",
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 7,
            "patch.linewidth": 0,
            "axes.prop_cycle": cycler(color=[BLUE, ORANGE, AQUA]),
            "legend.frameon": False,
            "legend.fontsize": 12,
        }
    )


def fmt_num(x: float, decimals: int = 0) -> str:
    """Brazilian number format: 124.956 and 1.234,5."""
    text = f"{x:,.{decimals}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_pct(x: float, decimals: int = 0) -> str:
    return f"{fmt_num(x * 100, decimals)}%"


def fmt_times(x: float, decimals: int = 2) -> str:
    return f"{fmt_num(x, decimals)}×"


def period_label(period: str) -> str:
    """"2009-05" -> "mai/09"."""
    return f"{MONTHS_PT[int(period[5:7]) - 1]}/{period[2:4]}"


def percent_axis(ax, axis: str = "y", decimals: int = 0) -> None:
    formatter = FuncFormatter(lambda v, _: fmt_pct(v, decimals))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def decimal_axis(ax, axis: str = "y", decimals: int = 2) -> None:
    formatter = FuncFormatter(lambda v, _: fmt_num(v, decimals))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def thousands_axis(ax, axis: str = "y") -> None:
    formatter = FuncFormatter(lambda v, _: fmt_num(v))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def save(fig, name: str) -> None:
    """Save a figure as reports/figures/<name>.png (the deck uploads these files)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")


def new_figure(width: float = 11, height: float = 5.5, **kwargs):
    return plt.subplots(figsize=(width, height), layout="constrained", **kwargs)
