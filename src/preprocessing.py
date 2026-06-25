"""Preprocessing and reference / ROI construction.

Public functions
----------------
resize_rgb(img, size)               -> RGB image at (size,size)
to_gray(img)                        -> single-channel uint8
preprocess_gray(img)                -> CLAHE + Gaussian, used for registration
build_reference(good_imgs, register)-> robust median reference (uint8 RGB)
bottle_mask_from_reference(ref)     -> (full_mask, inner_roi)

Input/output limits: images are expected as uint8 RGB arrays.  Empty image
lists raise ValueError so the caller fails fast instead of producing a blank
reference.
"""
import cv2
import numpy as np

from .config import CFG


# ---- basic operators (kept tiny and side-effect free) --------------------
def resize_rgb(img, size: int = CFG.image_size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def preprocess_gray(img):
    """Grayscale + CLAHE (local contrast) + Gaussian (denoise).

    Used both as the input to registration and as a comparable photometric
    representation.  CLAHE is preferred over global equalisation because the
    bottle has a bright amber ring and a near-black interior.
    """
    g = to_gray(resize_rgb(img))
    clahe = cv2.createCLAHE(clipLimit=CFG.clahe_clip,
                            tileGridSize=(CFG.clahe_grid, CFG.clahe_grid))
    g = clahe.apply(g)
    g = cv2.GaussianBlur(g, (CFG.blur_ksize, CFG.blur_ksize), 0)
    return g


def _ecc_align_gray(moving_gray, fixed_gray):
    """Estimate a Euclidean (rotation+translation) warp aligning moving->fixed.

    Returns the 2x3 warp or None on failure.  Wrapped because findTransformECC
    raises cv2.error when its critical loop does not converge.
    """
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                CFG.ecc_iterations, CFG.ecc_eps)
    try:
        _, warp = cv2.findTransformECC(
            fixed_gray.astype(np.float32) / 255.0,
            moving_gray.astype(np.float32) / 255.0,
            warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
        return warp
    except cv2.error:
        return None


def build_reference(good_imgs, register: bool = True):
    """Construct a robust reference image as the pixel-wise median of good
    samples.  When `register` is True each sample is first aligned (ECC) to a
    common base, which sharpens the rotating amber ring and markedly reduces
    later false positives.

    Input: non-empty list of uint8 RGB images.  Output: uint8 RGB reference.
    """
    if not good_imgs:
        raise ValueError("build_reference requires at least one good image")

    imgs = [resize_rgb(im) for im in good_imgs[:CFG.ref_sample]]
    if register and len(imgs) > 1:
        base_gray = preprocess_gray(imgs[0])
        aligned = [imgs[0].astype(np.float32)]
        h, w = base_gray.shape
        for im in imgs[1:]:
            warp = _ecc_align_gray(preprocess_gray(im), base_gray)
            if warp is None:
                aligned.append(im.astype(np.float32))
            else:
                aligned.append(cv2.warpAffine(
                    im, warp, (w, h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REFLECT).astype(np.float32))
        stack = np.stack(aligned, axis=0)
    else:
        stack = np.stack([im.astype(np.float32) for im in imgs], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def bottle_mask_from_reference(ref_rgb):
    """Segment the bottle disc from the bright background and derive an inner
    inspection ROI by erosion (excludes the unreliable outer boundary band).

    Returns (full_mask, inner_roi) as uint8 {0,255} images.
    """
    gray = to_gray(ref_rgb)
    # bottle is darker than the bright background -> inverse Otsu
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    erode_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CFG.roi_erode, CFG.roi_erode))
    inner = cv2.erode(mask, erode_k, iterations=1)
    return mask, inner


def roi_geometry(roi_mask):
    """Centre and radius of the ROI, used for radial features."""
    ys, xs = np.where(roi_mask > 0)
    if xs.size == 0:
        return (roi_mask.shape[1] / 2, roi_mask.shape[0] / 2), 1.0
    cx, cy = float(xs.mean()), float(ys.mean())
    r = float(np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).mean())) + 1e-6
    return (cx, cy), r


def photometric_match(aligned_rgb, ref_rgb, roi_mask):
    """Robust gain+bias matching of the aligned image to the reference within
    the ROI (matches grey mean/std).  Removes global exposure / contrast
    differences, the dominant diffuse component of the difference image, so the
    fixed threshold reacts to genuine local defects rather than lighting.
    """
    a = to_gray(aligned_rgb).astype(np.float32)
    r = to_gray(ref_rgb).astype(np.float32)
    m = roi_mask > 0
    asd = a[m].std() + 1e-6
    gain = (r[m].std() + 1e-6) / asd
    bias = r[m].mean() - gain * a[m].mean()
    out = aligned_rgb.astype(np.float32) * gain + bias
    return np.clip(out, 0, 255).astype(np.uint8)
