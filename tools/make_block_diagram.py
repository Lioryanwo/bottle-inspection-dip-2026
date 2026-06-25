"""Render a clean two-phase system block diagram for the report."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = "#1d2433"; LINE = "#aab0bd"
OFF = "#eaf1f6"; OFF_E = "#3a6ea5"
ON = "#f3efe7"; ON_E = "#b9892f"
REF = "#e8f1ec"; REF_E = "#2e8b6f"
DEC = "#fbeef0"; DEC_E = "#d1495b"

plt.rcParams.update({"font.size": 11})


def box(ax, x, y, w, h, text, fc, ec, fs=10.5, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                       linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, fontweight="bold" if bold else "normal", zorder=3)
    return (x, y, w, h)


def arrow(ax, p1, p2, color=LINE, style="-|>", rad=0.0, lw=1.8, ls="-"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                        lw=lw, color=color, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(a)


def right(b):  return (b[0] + b[2], b[1] + b[3] / 2)
def left(b):   return (b[0], b[1] + b[3] / 2)
def top(b):    return (b[0] + b[2] / 2, b[1] + b[3])
def bottom(b): return (b[0] + b[2] / 2, b[1])


def main(out="article_assets/pipeline_diagram.png"):
    fig, ax = plt.subplots(figsize=(15, 7.2))
    ax.set_xlim(0, 15); ax.set_ylim(0, 7.2); ax.axis("off")
    ax.text(7.5, 6.95, "Classical Bottle-Inspection Pipeline",
            ha="center", fontsize=17, fontweight="bold", color=INK)

    # ---- Phase A: offline reference + ROI -------------------------------
    ax.text(0.2, 6.25, "A.  Offline — build reference & ROI (once)",
            fontsize=11.5, fontweight="bold", color=OFF_E)
    a1 = box(ax, 0.3, 5.25, 2.5, 0.8, "Good training\nimages", OFF, OFF_E)
    a2 = box(ax, 3.4, 5.25, 2.7, 0.8, "Register (ECC)\n+ pixel median", OFF, OFF_E)
    a3 = box(ax, 6.7, 5.25, 2.5, 0.8, "Reference\nimage", REF, REF_E, bold=True)
    a4 = box(ax, 9.8, 5.25, 2.7, 0.8, "Otsu mask\n+ erode", OFF, OFF_E)
    a5 = box(ax, 13.0, 5.25, 1.7, 0.8, "Inner\nROI", REF, REF_E, bold=True)
    for p, q in [(a1, a2), (a2, a3), (a3, a4), (a4, a5)]:
        arrow(ax, right(p), left(q), OFF_E)

    # ---- Phase B: online per-image inspection ---------------------------
    ax.text(0.2, 4.35, "B.  Online — inspect each image",
            fontsize=11.5, fontweight="bold", color=ON_E)
    y = 3.25
    b1 = box(ax, 0.3, y, 2.15, 0.85, "Inspection\nimage", ON, ON_E)
    b2 = box(ax, 2.75, y, 2.15, 0.85, "Preprocess\nCLAHE + blur", ON, ON_E)
    b3 = box(ax, 5.2, y, 2.15, 0.85, "Register\nto reference", ON, ON_E)
    b4 = box(ax, 7.65, y, 2.15, 0.85, "Photometric\nmatch", ON, ON_E)
    b5 = box(ax, 10.1, y, 2.25, 0.85, "LAB \u0394E\ndifference", ON, ON_E)
    b6 = box(ax, 12.6, y, 2.1, 0.85, "Threshold +\nmorphology", ON, ON_E)
    for p, q in [(b1, b2), (b2, b3), (b3, b4), (b4, b5), (b5, b6)]:
        arrow(ax, right(p), left(q), ON_E)

    y2 = 1.65
    c1 = box(ax, 12.6, y2, 2.1, 0.85, "Connected comp.\n+ features", ON, ON_E)
    c2 = box(ax, 9.0, y2, 3.0, 0.85, "Detect (area %ROI)\n+ classify type", ON, ON_E)
    c3 = box(ax, 5.0, y2, 3.4, 0.85, "Decision: good / broken_large /\nbroken_small / contamination",
             DEC, DEC_E, bold=True)
    c4 = box(ax, 0.3, y2, 4.0, 0.85, "Quantitative evaluation\n(confusion, IoU, sweep)", DEC, DEC_E)
    arrow(ax, bottom(b6), top(c1), ON_E)
    arrow(ax, left(c1), right(c2), ON_E)
    arrow(ax, left(c2), right(c3), DEC_E)
    arrow(ax, left(c3), right(c4), DEC_E)

    # reference / ROI feed into the online path (dashed)
    arrow(ax, bottom(a3), (6.27, y + 0.85), REF_E, rad=-0.05, lw=1.4, ls=(0, (4, 3)))
    arrow(ax, bottom(a5), (11.2, y + 0.85), REF_E, rad=0.05, lw=1.4, ls=(0, (4, 3)))
    ax.text(7.5, 4.62, "reference + ROI", fontsize=9, color=REF_E, ha="center", style="italic")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
