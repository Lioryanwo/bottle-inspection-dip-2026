"""Thresholding + morphology + connected-component extraction.

The LAB-distance map is binarised with a fixed, calibrated absolute threshold
(`abs_threshold`).  Optionally an ROI-restricted Otsu level is OR-ed in so that
unusually high-contrast defects are still captured.  Morphological opening then
closing removes speckle and consolidates blobs before component labelling.
"""
import cv2
import numpy as np

from .config import CFG


def threshold_anomalies(diff, roi_mask, abs_threshold=None):
    if abs_threshold is None:
        abs_threshold = CFG.abs_threshold
    th = float(abs_threshold)

    binary = (diff >= th)
    if CFG.use_otsu_floor:
        vals = diff[roi_mask > 0]
        if vals.size:
            v = np.clip(vals, 0, 255).astype(np.uint8)
            otsu, _ = cv2.threshold(v, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = binary | (diff >= max(th, float(otsu)))

    mask = np.where(binary & (roi_mask > 0), 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (CFG.morph_kernel, CFG.morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask, th


def label_components(mask):
    num, labels, stats, cent = cv2.connectedComponentsWithStats(mask)
    comps = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < CFG.min_component_area:
            continue
        x, y, w, h = (int(stats[i, j]) for j in range(4))
        comps.append({"label": i, "area": area, "bbox": (x, y, w, h),
                      "centroid": (float(cent[i][0]), float(cent[i][1]))})
    return comps, labels
