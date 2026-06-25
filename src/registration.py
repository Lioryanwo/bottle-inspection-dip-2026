"""Geometric registration of an inspection image to the reference.

Strategy (classical, no learning):
  * ECC (MOTION_EUCLIDEAN) estimates a rotation + translation by iteratively
    maximising the enhanced correlation coefficient between the preprocessed
    inspection and reference images; this iteration is the critical loop,
    bounded by `ecc_iterations` / `ecc_eps`.
  * The estimated warp is applied with WARP_INVERSE_MAP, the convention that
    maps the inspection image onto the reference frame.
  * Safe fallback: if ECC fails to converge or returns a weak correlation, the
    identity transform is used, so a failed alignment never fabricates
    differences. Phase correlation is reported as a translation diagnostic.

Returns the aligned RGB image and a diagnostics dict (status, ecc_cc,
phase_response, dx, dy, angle_deg) logged per image for the report.
"""
import cv2
import numpy as np

from .config import CFG
from .preprocessing import preprocess_gray, resize_rgb


def register_to_reference(img_rgb, ref_rgb):
    src = resize_rgb(img_rgb)
    moving = preprocess_gray(src)
    fixed = preprocess_gray(ref_rgb)
    h, w = fixed.shape

    # translation diagnostic (not used to build the warp)
    try:
        (_, _), response = cv2.phaseCorrelate(moving.astype(np.float32),
                                              fixed.astype(np.float32))
    except cv2.error:
        response = 0.0

    warp = np.eye(2, 3, dtype=np.float32)
    status, ecc_cc, angle, dx, dy = "identity", float("nan"), 0.0, 0.0, 0.0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                CFG.ecc_iterations, CFG.ecc_eps)
    try:
        ecc_cc, warp_ecc = cv2.findTransformECC(
            fixed.astype(np.float32) / 255.0,
            moving.astype(np.float32) / 255.0,
            np.eye(2, 3, dtype=np.float32),
            cv2.MOTION_EUCLIDEAN, criteria, None, 5)
        if ecc_cc >= CFG.ecc_min_cc:
            warp = warp_ecc
            status = "ecc"
            angle = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
            dx, dy = float(warp[0, 2]), float(warp[1, 2])
    except cv2.error:
        ecc_cc = float("nan")

    aligned = cv2.warpAffine(src, warp, (w, h),
                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT)
    diag = {"status": status, "ecc_cc": float(ecc_cc),
            "phase_response": float(response),
            "dx": float(dx), "dy": float(dy), "angle_deg": float(angle)}
    return aligned, diag
