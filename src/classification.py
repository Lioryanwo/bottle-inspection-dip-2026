"""Rule-based, two-stage classification (no learned model).

Stage 1 - detection: an image is 'defect' when the anomalous area exceeds a
          calibrated fraction of the ROI; otherwise 'good'.
Stage 2 - defect type: the dominant component is labelled using interpretable
          rules on brightness and shape:
            * contamination -> neutral/dark foreign material (low brightness delta)
            * broken_large  -> bright exposed glass, elongated or large blob
            * broken_small  -> bright exposed glass, compact blob

Thresholds come from `config.Config` and were set from the per-type feature
statistics; the rules are deliberately simple and auditable rather than
optimal.
"""
from .config import CFG


def detect(features):
    return "defect" if features["total_area_pct"] >= CFG.area_threshold_pct else "good"


def classify_type(features):
    """Assign a defect type to the dominant component using interpretable rules.

    * A bright exposed-glass edge (large positive brightness delta) indicates a
      *broken* bottle; broken_large is distinguished from broken_small by an
      elongated / larger dominant blob.
    * Otherwise the anomaly is foreign material -> contamination.

    Note: the three types overlap substantially in classical descriptors, so
    this stage is intentionally simple and its accuracy is reported honestly.
    """
    bright = features["dom_brightness_delta"]
    if bright < CFG.bright_glass_delta:
        return "contamination"
    if (features["largest_area_pct"] >= CFG.large_area_pct
            or (features["dom_aspect"] >= CFG.large_aspect
                and features["dom_eccentricity"] >= CFG.large_ecc)):
        return "broken_large"
    return "broken_small"


def classify(features):
    """Return (binary_label, type_label). type is None for 'good'."""
    binary = detect(features)
    if binary == "good":
        return "good", None
    return "defect", classify_type(features)
