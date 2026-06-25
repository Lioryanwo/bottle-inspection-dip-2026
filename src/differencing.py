"""Reference vs. aligned-image differencing in CIELAB.

`lab_distance` returns the per-pixel Euclidean distance between the reference
and the aligned image in LAB space (delta-E-like).  Crucially the map is NOT
per-image min-max normalised: keeping physical units is what makes a single
calibrated threshold comparable across all images.

A small ring-suppression term damps the thin high-gradient amber ring, the
dominant source of registration residual / false positives.
"""
import cv2
import numpy as np

from .config import CFG
from .preprocessing import resize_rgb, to_gray


def lab_distance(ref_rgb, aligned_rgb):
    ref = resize_rgb(ref_rgb)
    img = resize_rgb(aligned_rgb)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    d = np.sqrt(((ref_lab - img_lab) ** 2).sum(axis=2))           # delta-E
    d = cv2.GaussianBlur(d, (CFG.diff_blur, CFG.diff_blur), 0)

    if CFG.ring_suppress > 0:
        # locate the thin high-gradient ring on the reference and attenuate it
        gx = cv2.Sobel(to_gray(ref), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(to_gray(ref), cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        ring = (grad > np.percentile(grad, 97)).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * CFG.ring_suppress + 1,) * 2)
        ring = cv2.dilate(ring, k, iterations=1)
        d = np.where(ring > 0, d * 0.5, d)
    return d.astype(np.float32)
