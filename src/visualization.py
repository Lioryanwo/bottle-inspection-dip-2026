"""Publication-quality figure helpers (matplotlib).

A single style is applied so every figure in the report shares typography,
colour and DPI.  Functions here are pure renderers: they take arrays / numbers
and write a PNG.
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

from .config import ALL_CLASSES

# ---- shared style --------------------------------------------------------
PALETTE = {
    "ink": "#1d2433", "muted": "#5b6472", "line": "#c7ccd6",
    "good": "#2e8b6f", "defect": "#d1495b", "accent": "#3a6ea5",
    "broken_large": "#d1495b", "broken_small": "#e6a23c",
    "contamination": "#7a5cc0", "panel": "#f6f7f9",
}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.edgecolor": PALETTE["line"], "axes.labelcolor": PALETTE["ink"],
    "text.color": PALETTE["ink"], "xtick.color": PALETTE["muted"],
    "ytick.color": PALETTE["muted"], "axes.grid": False,
})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# ---- per-image overlay (drawn with OpenCV for crisp pixels) --------------
def overlay_components(img_rgb, comps, binary, dtype=None):
    out = img_rgb.copy()
    color = (46, 139, 111) if binary == "good" else (209, 73, 91)
    for c in comps:
        x, y, w, h = c["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    label = binary if binary == "good" else f"defect: {dtype}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (255, 255, 255), -1)
    cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


def _show(ax, im, title):
    ax.axis("off"); ax.set_title(title)
    if im is None:
        return
    if im.ndim == 2:
        ax.imshow(im, cmap="gray")
    else:
        ax.imshow(im)


def stage_panel(path, ref, inspection, registered, diff, mask, overlay,
                gt_mask=None, title=""):
    """2x4 stage panel for one inspection image."""
    dn = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
    _show(axes[0, 0], ref, "Reference (median)")
    _show(axes[0, 1], inspection, "Inspection (raw)")
    _show(axes[0, 2], registered, "Registered")
    axes[0, 3].axis("off"); axes[0, 3].set_title("LAB distance")
    axes[0, 3].imshow(dn, cmap="inferno")
    _show(axes[1, 0], mask, "Anomaly mask")
    _show(axes[1, 1], overlay, "Detection overlay")
    if gt_mask is not None:
        _show(axes[1, 2], gt_mask, "Ground truth")
    else:
        axes[1, 2].axis("off"); axes[1, 2].set_title("Ground truth (n/a)")
    axes[1, 3].axis("off")
    axes[1, 3].text(0.0, 0.5, title, fontsize=11, va="center", wrap=True)
    fig.suptitle(title.split("\n")[0], fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, path)


def preprocessing_montage(path, rgb, gray, clahe, blurred):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, im, t in zip(axes, [rgb, gray, clahe, blurred],
                         ["Input RGB", "Grayscale", "CLAHE", "CLAHE + Gaussian"]):
        _show(ax, im, t)
    fig.suptitle("Preprocessing stages", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, path)


def reference_roi_figure(path, ref, full_mask, inner_roi):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    _show(axes[0], ref, "Median reference")
    _show(axes[1], full_mask, "Bottle mask (Otsu)")
    _show(axes[2], inner_roi, "Inner inspection ROI")
    fig.suptitle("Reference and region of interest", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, path)


def registration_figure(path, ref, raw, registered, diff_before, diff_after,
                         cc_before, cc_after):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    _show(axes[0], raw, "Misaligned (controlled rotation)")
    _show(axes[1], registered, "After ECC registration")
    axes[2].axis("off"); axes[2].set_title(f"|diff| before  (E={cc_before:.0f})")
    axes[2].imshow(diff_before, cmap="inferno")
    axes[3].axis("off"); axes[3].set_title(f"|diff| after  (E={cc_after:.0f})")
    axes[3].imshow(diff_after, cmap="inferno")
    fig.suptitle("ECC registration recovers a known rotation and reduces residual",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, path)


def confusion_figure(path, cm, labels, title, normalize=False):
    cm = np.asarray(cm, dtype=float)
    disp = cm.copy()
    if normalize:
        disp = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(1.2 + 1.1 * len(labels), 1.0 + 1.0 * len(labels)))
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=disp.max() if disp.max() else 1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth"); ax.set_title(title)
    thr = disp.max() / 2 if disp.max() else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{cm[i, j]:.0f}" if not normalize else f"{disp[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if disp[i, j] > thr else PALETTE["ink"],
                    fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, path)


def score_distribution_figure(path, df, threshold):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    order = list(ALL_CLASSES)
    rng = np.random.default_rng(0)
    for i, cls in enumerate(order):
        sub = df[df["gt_type"] == cls]["total_area_pct"].values
        if len(sub) == 0:
            continue
        jitter = rng.uniform(-0.16, 0.16, size=len(sub))
        ax.scatter(np.full(len(sub), i) + jitter, sub, s=40, alpha=0.8,
                   color=PALETTE.get(cls, PALETTE["accent"]),
                   edgecolor="white", linewidth=0.6, label=cls, zorder=3)
    ax.axhline(threshold, color=PALETTE["defect"], ls="--", lw=1.6, zorder=2,
               label=f"decision threshold = {threshold:.2f}%")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_ylabel("Anomalous area (% of ROI)")
    ax.set_title("Per-image anomaly score by class")
    ax.set_yscale("symlog", linthresh=0.3)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    _save(fig, path)


def threshold_sweep_figure(path, thresholds, prec, rec, f1, best_t):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(thresholds, prec, color=PALETTE["accent"], lw=2, label="precision")
    ax.plot(thresholds, rec, color=PALETTE["good"], lw=2, label="recall")
    ax.plot(thresholds, f1, color=PALETTE["defect"], lw=2.4, label="F1")
    ax.axvline(best_t, color=PALETTE["muted"], ls="--", lw=1.4,
               label=f"chosen = {best_t:.2f}%")
    ax.set_xlabel("Area threshold (% of ROI)"); ax.set_ylabel("Score")
    ax.set_title("Detection metrics vs. decision threshold")
    ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    _save(fig, path)


def gallery_figure(path, items, title, ncols=4):
    """items: list of (image_rgb, caption). Renders a grid."""
    n = len(items)
    if n == 0:
        items = [(np.full((64, 64, 3), 240, np.uint8), "none")]
        n = 1
    ncols = min(ncols, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (im, cap) in zip(axes, items):
        _show(ax, im, cap)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, path)
