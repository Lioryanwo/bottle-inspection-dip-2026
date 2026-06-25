"""End-to-end single-image inspection orchestration."""
from pathlib import Path

import cv2

from .preprocessing import resize_rgb, bottle_mask_from_reference, photometric_match
from .registration import register_to_reference
from .differencing import lab_distance
from .segmentation import threshold_anomalies, label_components
from .features import extract_features
from .classification import classify
from .visualization import overlay_components
from .io_utils import imwrite_rgb


class BottleInspectionPipeline:
    """Holds the reference + ROI once, then inspects images one by one."""

    def __init__(self, reference_rgb, roi_mask=None, abs_threshold=None):
        self.reference = resize_rgb(reference_rgb)
        full, inner = bottle_mask_from_reference(self.reference)
        self.full_mask = full
        self.roi_mask = roi_mask if roi_mask is not None else inner
        self.roi_area = int((self.roi_mask > 0).sum())
        self.abs_threshold = abs_threshold

    def run_one(self, image_rgb, out_dir=None, name="image", keep_arrays=False):
        img = resize_rgb(image_rgb)
        registered, reg = register_to_reference(img, self.reference)
        aligned = photometric_match(registered, self.reference, self.roi_mask)
        diff = lab_distance(self.reference, aligned)
        mask, th = threshold_anomalies(diff, self.roi_mask, self.abs_threshold)
        comps, labels = label_components(mask)
        feats = extract_features(comps, labels, self.roi_mask, aligned, self.reference)
        binary, dtype = classify(feats)
        overlay = overlay_components(registered, comps, binary, dtype)

        if out_dir:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / "diff.png"),
                        cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype("uint8"))
            cv2.imwrite(str(out_dir / "mask.png"), mask)
            cv2.imwrite(str(out_dir / "roi.png"), self.roi_mask)
            imwrite_rgb(out_dir / "registered.png", registered)
            imwrite_rgb(out_dir / "overlay.png", overlay)

        result = {"prediction": binary, "pred_type": dtype if dtype else "good",
                  "threshold": th, **reg, **feats}
        if keep_arrays:
            result["_arrays"] = {"registered": registered, "aligned": aligned,
                                 "diff": diff, "mask": mask, "overlay": overlay}
        return result
