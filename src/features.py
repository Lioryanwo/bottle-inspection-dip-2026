"""Region descriptors used for detection and defect-type classification.

For each surviving connected component we compute area, bounding box, and
shape descriptors (solidity, extent, eccentricity, aspect ratio) via
scikit-image regionprops, plus two domain features:
  * radial position  -- centroid distance from the ROI centre / ROI radius;
                        rim defects (broken) sit near 1.0, contamination is
                        more central.
  * brightness delta -- mean grey of the component on the aligned image minus
                        on the reference; positive = brighter (chipped glass),
                        negative = darker (contamination / dirt).
The pipeline summarises these into per-image aggregates plus the descriptors
of the dominant (largest) component, which drive the rule-based classifier.
"""
import numpy as np
from skimage.measure import regionprops

from .preprocessing import to_gray, roi_geometry


def _shape_props(labels, comp, aligned_gray, ref_gray):
    region = (labels == comp["label"]).astype(np.uint8)
    props = regionprops(region)
    p = props[0] if props else None
    x, y, w, h = comp["bbox"]
    aspect = max(w, h) / max(1, min(w, h))
    feats = {
        "solidity": float(p.solidity) if p else 1.0,
        "extent": float(p.extent) if p else 1.0,
        "eccentricity": float(p.eccentricity) if p else 0.0,
        "aspect": float(aspect),
    }
    m = region > 0
    feats["brightness_delta"] = float(aligned_gray[m].mean() - ref_gray[m].mean())
    return feats


def extract_features(comps, labels, roi_mask, aligned_rgb, ref_rgb):
    roi_area = int((roi_mask > 0).sum())
    (cx, cy), radius = roi_geometry(roi_mask)
    aligned_gray = to_gray(aligned_rgb).astype(np.float32)
    ref_gray = to_gray(ref_rgb).astype(np.float32)

    total = sum(c["area"] for c in comps)
    agg = {
        "num_components": len(comps),
        "total_area": int(total),
        "largest_area": 0,
        "total_area_pct": 100.0 * total / max(1, roi_area),
        "largest_area_pct": 0.0,
        "dom_radial": 0.0, "dom_solidity": 1.0, "dom_eccentricity": 0.0,
        "dom_aspect": 1.0, "dom_brightness_delta": 0.0,
    }
    if not comps:
        return agg

    dom = max(comps, key=lambda c: c["area"])
    sp = _shape_props(labels, dom, aligned_gray, ref_gray)
    ux, uy = dom["centroid"]
    agg.update({
        "largest_area": int(dom["area"]),
        "largest_area_pct": 100.0 * dom["area"] / max(1, roi_area),
        "dom_radial": float(np.hypot(ux - cx, uy - cy) / radius),
        "dom_solidity": sp["solidity"],
        "dom_eccentricity": sp["eccentricity"],
        "dom_aspect": sp["aspect"],
        "dom_brightness_delta": sp["brightness_delta"],
    })
    return agg
