"""ABADE chart theme: the visual contract every figure in the portfolio obeys.

Why this file exists
--------------------
The portfolio presents itself as one system, so a chart that arrives in a
README has to look like it belongs to the same object as the brand panels
above it. Those panels are dark (``#050505``, 24px corner radius), so the
charts are too: a light chart dropped between them reads as a screenshot
someone pasted in.

Colour rules, in order of precedence
------------------------------------
1. **Neutrals carry structure, one accent carries the finding.** Where a chart
   has a subject and a comparison, the subject gets ``COBALT`` and everything
   else gets ``SLATE``. This is the default and covers most charts here.
2. **Where two or three series are genuinely peers**, use ``SERIES``. It is a
   categorical palette validated against the ``#050505`` surface: every pair
   clears the colour-vision-deficiency and normal-vision separation floors
   with all pairs in play, so it is safe for scatter and small multiples, not
   just bars.
3. **Never encode by colour alone.** Every series is direct-labelled or
   legended. The amber/teal pair sits in the 6-8 CVD band, which is admissible
   only with that secondary encoding.

``SERIES`` is the brand palette re-stepped, not a new one: slot 1 is the brand
indigo unchanged, slot 2 is the brand gold moved into the lightness band the
dark surface requires (the published ``#C8B680`` is too light to sit on black
next to indigo without failing separation).

Text is outlined to paths on save (``svg.fonttype = "path"``), which is what
the hand-built brand SVGs do. It costs a few KB and buys a figure that renders
identically on GitHub, in both themes, with no webfont available.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyBboxPatch, PathPatch

# --- Brand surfaces and ink ------------------------------------------------
CARBON = "#050505"    # chart surface, straight from the brand panels
GRAPHITE = "#1B1C1F"  # raised surface / recessive fill
SLATE = "#3A3E46"     # the comparison bar in the brand's own chart panel
STEEL = "#9BA2AA"     # secondary ink
IVORY = "#F6F5F0"     # primary ink
GRID = "#24262B"      # grid lines: present, never competing

# --- Accents ---------------------------------------------------------------
COBALT = "#5B6CFF"    # the finding
AMBER = "#BD8620"     # brand gold, stepped for the dark surface
TEAL = "#1F9B73"      # third series, used only when three are unavoidable
CRIMSON = "#D9485F"   # loss / "this is the number that hurts"

#: Validated categorical order. Assign in this order, never cycle past it.
SERIES = (COBALT, AMBER, TEAL)

#: Sequential ramp for magnitude (one hue, dark to light, on the dark surface).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "cobalt_on_carbon", ["#0B0D18", "#23306F", "#3B4CB8", COBALT, "#A7B0FF"]
)

_CORNER_RADIUS = 24  # matches rx="24" on the brand panels
_FONT_STACK = ("Roboto", "Lato", "Helvetica Neue", "Arial", "DejaVu Sans")


def apply_theme() -> None:
    """Install the theme into matplotlib's global state.

    Called by :func:`panel`, so figure code never has to remember it.
    """
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    stack = [n for n in _FONT_STACK if n in available] or ["DejaVu Sans"]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": stack,
            "svg.fonttype": "path",
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.transparent": True,
            "axes.facecolor": "none",
            "axes.edgecolor": GRID,
            "axes.labelcolor": STEEL,
            "axes.labelsize": 10,
            "axes.titlecolor": IVORY,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.linewidth": 1.0,
            "text.color": IVORY,
            "xtick.color": STEEL,
            "ytick.color": STEEL,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "legend.labelcolor": STEEL,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "lines.solid_capstyle": "round",
            "grid.color": GRID,
            "grid.linewidth": 0.9,
        }
    )


def panel(width: float = 12.0, height: float = 5.6, subtitle: str | None = None,
          title: str | None = None, nrows: int = 1, ncols: int = 1):
    """A dark brand panel with rounded corners, and the axes to draw in.

    The rounded rectangle is drawn as a patch rather than a figure facecolor
    because matplotlib cannot round a figure's own background, and square
    corners next to the brand panels is exactly the tell this theme exists to
    remove.

    Returns ``(fig, ax)``, where ``ax`` is an array when a grid is requested.
    """
    apply_theme()
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))

    backdrop = FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle=f"round,pad=0,rounding_size={_CORNER_RADIUS / (72 * width):.5f}",
        transform=fig.transFigure, facecolor=CARBON, edgecolor="none",
        mutation_aspect=width / height, zorder=-10,
    )
    fig.patches.append(backdrop)

    if title:
        fig.suptitle(title, x=0.045, y=0.965, ha="left", color=IVORY,
                     fontsize=16, fontweight="bold")
    if subtitle:
        fig.text(0.045, 0.900, subtitle, ha="left", color=STEEL, fontsize=10.5)
    if title or subtitle:
        # Reserve the band the heading occupies; otherwise the top y-tick and
        # the subtitle land on the same pixels.
        fig.subplots_adjust(top=0.80 if subtitle else 0.86)

    return fig, axes


def clean(ax, axis: str = "y", spines=("top", "right", "left")) -> None:
    """Recessive grid on one axis, no chartjunk spines."""
    ax.grid(axis=axis, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines[list(spines)].set_visible(False)
    ax.spines[[s for s in ("top", "right", "bottom", "left") if s not in spines]].set_color(GRID)


def _rounded_bar_path(x0: float, x1: float, base: float, tip: float,
                      radius: float, horizontal: bool) -> MplPath:
    """A bar rounded only at the data end, square where it meets the baseline.

    Rounding both ends detaches the bar from its own axis and makes small
    values read as larger than they are; rounding only the tip keeps the
    baseline honest.
    """
    sign = 1.0 if tip >= base else -1.0
    span = abs(tip - base)
    r = float(min(radius, span, abs(x1 - x0) / 2))
    if r <= 0:
        verts = [(x0, base), (x0, tip), (x1, tip), (x1, base)]
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * 3
    else:
        shoulder = tip - sign * r
        verts = [
            (x0, base), (x0, shoulder),
            (x0, tip), (x0 + r, tip),            # quadratic corner
            (x1 - r, tip),
            (x1, tip), (x1, shoulder),           # quadratic corner
            (x1, base),
        ]
        codes = [
            MplPath.MOVETO, MplPath.LINETO,
            MplPath.CURVE3, MplPath.CURVE3,
            MplPath.LINETO,
            MplPath.CURVE3, MplPath.CURVE3,
            MplPath.LINETO,
        ]
    if horizontal:
        verts = [(y, x) for x, y in verts]
    return MplPath(verts + [verts[0]], codes + [MplPath.CLOSEPOLY])


def bars(ax, positions, values, colors, width: float = 0.62, base: float = 0.0,
         horizontal: bool = False, radius_px: float = 5.0, labels=None,
         label_fmt: str = "{:.2f}", label_color=IVORY, label_pad: float = 0.015,
         zorder: int = 3):
    """Thin bars with 4px rounded data-ends, drawn in data coordinates.

    ``colors`` may be a single colour or one per bar, which is how the
    highlight-the-subject rule is expressed: pass ``SLATE`` for every bar and
    ``COBALT`` for the one the sentence above the chart is about.
    """
    positions = np.asarray(positions, dtype=float)
    values = np.asarray(values, dtype=float)
    if isinstance(colors, str):
        colors = [colors] * len(values)

    span = (ax.get_xlim() if horizontal else ax.get_ylim())
    data_per_px = (span[1] - span[0]) / (
        ax.get_window_extent().width if horizontal else ax.get_window_extent().height
    )
    radius = radius_px * data_per_px

    for pos, value, color in zip(positions, values, colors):
        patch = PathPatch(
            _rounded_bar_path(pos - width / 2, pos + width / 2, base, value,
                              radius, horizontal),
            facecolor=color, edgecolor="none", zorder=zorder,
        )
        ax.add_patch(patch)

    if labels is not None:
        # A near-zero bar still needs its label clear of the tick row, so the
        # offset is a share of the axis span rather than of the bar.
        pad = label_pad * (span[1] - span[0])
        for pos, value, color, text in zip(positions, values, colors, labels):
            text = text if isinstance(text, str) else label_fmt.format(text)
            if horizontal:
                ax.text(value + np.sign(value or 1) * pad, pos, text, va="center",
                        ha="left" if value >= base else "right",
                        color=label_color, fontsize=10, fontweight="bold",
                        zorder=zorder + 1)
            else:
                ax.text(pos, value + np.sign(value or 1) * pad, text, ha="center",
                        va="bottom" if value >= base else "top",
                        color=label_color, fontsize=10, fontweight="bold",
                        zorder=zorder + 1)


def grouped_bars(ax, categories, series: dict[str, object], colors=SERIES,
                 width: float = 0.74, gap_px: float = 2.0, horizontal: bool = False,
                 label_fmt: str | None = None):
    """Two or three peer series, with a real 2px surface gap between fills.

    The gap is why adjacent bars never read as one shape, and it is sized in
    pixels rather than data units so it survives a change of figure width.
    """
    n = len(series)
    positions = np.arange(len(categories), dtype=float)
    span = ax.get_xlim() if horizontal else ax.get_ylim()
    del span  # the gap is computed on the categorical axis, below

    cat_span = ax.get_ylim() if horizontal else ax.get_xlim()
    px = (cat_span[1] - cat_span[0]) / (
        ax.get_window_extent().height if horizontal else ax.get_window_extent().width
    )
    gap = gap_px * px
    bar_w = (width - gap * (n - 1)) / n

    for index, ((name, values), color) in enumerate(zip(series.items(), colors)):
        offset = -width / 2 + bar_w / 2 + index * (bar_w + gap)
        values = np.asarray(list(values), dtype=float)
        bars(ax, positions + offset, values, color, width=bar_w,
             horizontal=horizontal,
             labels=values if label_fmt else None, label_fmt=label_fmt or "{:.2f}")
        # A proxy artist so the legend exists without drawing a second time.
        ax.plot([], [], color=color, linewidth=7, solid_capstyle="butt", label=name)

    if horizontal:
        ax.set_yticks(positions, categories)
    else:
        ax.set_xticks(positions, categories)
    return positions


def annotate(ax, text: str, xy, xytext, color=IVORY, fontsize: float = 10,
             arrow: bool = True, ha: str = "left") -> None:
    """A callout that names the finding, in ink rather than in a series colour."""
    ax.annotate(
        text, xy=xy, xytext=xytext, color=color, fontsize=fontsize, ha=ha,
        va="center", zorder=10, fontweight="medium",
        arrowprops=dict(arrowstyle="-", color=STEEL, linewidth=1.0,
                        shrinkA=2, shrinkB=4) if arrow else None,
    )


def reference_line(ax, value: float, label: str, horizontal: bool = True,
                   color=STEEL, where: float = 0.985, ha: str = "right") -> None:
    """A dashed target line that says what it is, inside the plot.

    ``where`` and ``ha`` place the label on whichever side of the plot is
    empty: its backing box is opaque, so parked over a bar it hides the bar's
    own value label.
    """
    # The line stops short of its own label rather than running under it.
    if horizontal:
        ax.axhline(value, xmin=0.0 if ha == "right" else where + 0.13,
                   xmax=where - 0.13 if ha == "right" else 1.0,
                   color=color, linewidth=1.1,
                   linestyle=(0, (4, 3)), alpha=0.75, zorder=2)
    else:
        ax.axvline(value, ymax=where - 0.11, color=color, linewidth=1.1,
                   linestyle=(0, (4, 3)), alpha=0.75, zorder=2)
    if horizontal:
        ax.text(where, value, f" {label} ", transform=ax.get_yaxis_transform(),
                ha=ha, va="center", fontsize=9, color=color, zorder=6,
                bbox=dict(facecolor=CARBON, edgecolor="none", pad=2.2))
    else:
        ax.text(value, where, f" {label} ", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color=color,
                bbox=dict(facecolor=CARBON, edgecolor="none", pad=1.6))


def kpi_strip(items, width: float = 12.0, height: float = 1.9,
              accent_index: int | None = 0):
    """The three numbers a reader should leave with, as a figure of its own.

    ``items`` is a sequence of ``(value, caption)``. This is the "is it even a
    chart?" case: three scalars have no shape worth plotting, so they are set
    as type instead of forced into bars.
    """
    apply_theme()
    items = list(items)
    fig, axes = plt.subplots(1, len(items), figsize=(width, height))
    axes = np.atleast_1d(axes)

    backdrop = FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle=f"round,pad=0,rounding_size={_CORNER_RADIUS / (72 * width):.5f}",
        transform=fig.transFigure, facecolor=CARBON, edgecolor="none",
        mutation_aspect=width / height, zorder=-10,
    )
    fig.patches.append(backdrop)

    for index, (ax, (value, caption)) in enumerate(zip(axes, items)):
        ax.axis("off")
        colour = COBALT if index == accent_index else IVORY
        ax.text(0.0, 0.70, value, ha="left", va="center", fontsize=30,
                fontweight="bold", color=colour, transform=ax.transAxes)
        ax.text(0.0, 0.22, caption, ha="left", va="center", fontsize=9.8,
                color=STEEL, transform=ax.transAxes, linespacing=1.45)
        if index:  # hairline divider between tiles
            ax.plot([-0.07, -0.07], [0.12, 0.88], transform=ax.transAxes,
                    color=GRID, linewidth=1.1, clip_on=False)

    fig.subplots_adjust(left=0.045, right=0.985, top=0.86, bottom=0.14, wspace=0.42)
    return fig, axes


def _check_glyphs(fig) -> None:
    """Fail the build on a character the active font cannot draw.

    Text is outlined to paths on save, so a missing glyph does not fall back —
    it renders as an empty box, and nothing in the build complains. This turns
    that into an error at the point where it is cheap to fix. (U+2192, the
    rightwards arrow, is the one that actually got through.)
    """
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return  # the check is a convenience, never a hard dependency

    family = mpl.rcParams["font.sans-serif"][0]
    try:
        cmap = TTFont(mpl.font_manager.findfont(
            mpl.font_manager.FontProperties(family=family)
        )).getBestCmap()
    except Exception:
        return

    missing = {
        character
        for artist in fig.findobj(mpl.text.Text)
        for character in artist.get_text()
        if character not in "\n\t" and ord(character) not in cmap
    }
    if missing:
        listed = ", ".join(f"{c!r} (U+{ord(c):04X})" for c in sorted(missing))
        raise ValueError(
            f"{family} has no glyph for {listed}. Outlined text does not fall "
            f"back, so these would render as empty boxes."
        )


def save(fig, path: Path | str, pad: float = 0.34) -> Path:
    """Write the figure as SVG with text outlined, and return the path."""
    _check_glyphs(fig)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=pad,
                transparent=True)
    plt.close(fig)
    return path
