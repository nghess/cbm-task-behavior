"""Shared matplotlib look for every notebook: one palette and one set of rcParams.

    from src.style import INK, MUTED, GRID, PAPER, use_style
    use_style()              # analysis notebooks
    use_style(slides=True)   # 16:9 slide figures (larger Arial text, 200 dpi exports)

PAPER is pure white (255, 255, 255), so figures pasted into PowerPoint or Photoshop sit
on a white slide or canvas without a visible box. Anything meant to blend into the
background (marker edges, lines drawn over maps) should use PAPER, not a literal colour.
"""
import matplotlib.pyplot as plt

INK, MUTED, GRID, PAPER = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'
SLIDE = (13.33, 7.5)  # inches: a 16:9 slide

_BASE = {
    'figure.facecolor': PAPER, 'axes.facecolor': PAPER, 'savefig.facecolor': PAPER,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUTED,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'legend.frameon': False,
}
_NOTEBOOK = {
    'axes.labelcolor': MUTED,
    'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'figure.dpi': 110,
}
_SLIDES = {
    'axes.labelcolor': INK, 'font.family': 'Arial', 'font.size': 14,
    'axes.titlesize': 18, 'axes.labelsize': 15, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 14,
    'figure.dpi': 80, 'savefig.dpi': 200,
}


def use_style(slides=False):
    """Apply the shared rcParams; slides=True for the slide-sized variant."""
    plt.rcParams.update({**_BASE, **(_SLIDES if slides else _NOTEBOOK)})


# One color per mouse: matplotlib's Dark2, which has eight colors, plus a deep blue (which Dark2 lacks) for a ninth
# mouse. The reference is MOUSE_PALETTE in cbm-qc's qc/colors.py (4_style_guide.ipynb).
MOUSE_PALETTE = [*plt.cm.Dark2.colors, '#1f3fbf']


def mouse_colors(mice):
    """{mouse: color}. Mice take MOUSE_PALETTE in sorted order, so a mouse keeps its color across notebooks."""
    mice = sorted(mice)
    if len(mice) > len(MOUSE_PALETTE):
        raise ValueError(f'{len(mice)} mice but only {len(MOUSE_PALETTE)} mouse colors')
    return dict(zip(mice, MOUSE_PALETTE))


def mouse_legend(ax, colors, ms=6, **kw):
    """One legend entry per mouse, by default just outside the right edge of `ax`. Replaces any legend on `ax`."""
    handles = [plt.Line2D([], [], ls='', marker='o', ms=ms, color=c, mec=PAPER) for c in colors.values()]
    opts = dict(title='Mouse', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, handletextpad=0.2, labelspacing=0.3)
    return ax.legend(handles, list(colors), **{**opts, **kw})
