"""Quantitative evaluation: image-level metrics, multi-class confusion,
pixel-level IoU against ground-truth masks, and threshold calibration."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import CFG, ALL_CLASSES


def gt_type_from_path(path):
    return Path(path).parent.name  # good / broken_large / broken_small / contamination


def gt_binary(gt_type):
    return "good" if gt_type == "good" else "defect"


def binary_metrics(df):
    gt = df["gt"].values
    pred = df["prediction"].values
    tp = int(((gt == "defect") & (pred == "defect")).sum())
    fp = int(((gt == "good") & (pred == "defect")).sum())
    tn = int(((gt == "good") & (pred == "good")).sum())
    fn = int(((gt == "defect") & (pred == "good")).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(1, len(df))
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "precision": prec,
            "recall": rec, "specificity": spec, "f1": f1, "accuracy": acc}


def binary_confusion(df):
    cm = np.zeros((2, 2), int)
    lab = ["good", "defect"]
    for _, r in df.iterrows():
        cm[lab.index(r["gt"]), lab.index(r["prediction"])] += 1
    return cm, lab


def multiclass_confusion(df):
    lab = list(ALL_CLASSES)
    cm = np.zeros((len(lab), len(lab)), int)
    for _, r in df.iterrows():
        gi = lab.index(r["gt_type"])
        pj = lab.index(r["pred_type"]) if r["pred_type"] in lab else lab.index("good")
        cm[gi, pj] += 1
    return cm, lab


def per_type_recall(df):
    out = {}
    for t in ALL_CLASSES:
        if t == "good":
            sub = df[df["gt_type"] == "good"]
            out[t] = float((sub["prediction"] == "good").mean()) if len(sub) else float("nan")
        else:
            sub = df[df["gt_type"] == t]
            out[t] = float((sub["prediction"] == "defect").mean()) if len(sub) else float("nan")
    return out


def calibrate_area_threshold(df, lo=0.05, hi=3.0, steps=160):
    """Sweep the area threshold and return curves + the operating point that
    maximises balanced accuracy (Youden's J = recall + specificity - 1).

    Balanced accuracy is used rather than F1 because the test set is imbalanced
    (far more defective than good images); F1 would push the threshold toward
    near-perfect recall at the cost of specificity.
    """
    ts = np.linspace(lo, hi, steps)
    prec, rec, f1, bal = [], [], [], []
    gt = (df["gt"] == "defect").values
    score = df["total_area_pct"].values
    for t in ts:
        pred = score >= t
        tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum())
        fn = int((~pred & gt).sum()); tn = int((~pred & ~gt).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        prec.append(p); rec.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
        bal.append(0.5 * (r + spec))
    best = int(np.argmax(bal))
    return ts, np.array(prec), np.array(rec), np.array(f1), float(ts[best])


def pixel_iou(pred_mask, gt_mask):
    """IoU and pixel precision/recall between two binary masks (any size)."""
    g = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]),
                   interpolation=cv2.INTER_NEAREST) > 0
    p = pred_mask > 0
    inter = int((p & g).sum()); union = int((p | g).sum())
    iou = inter / union if union else (1.0 if inter == 0 else 0.0)
    pp = inter / int(p.sum()) if p.sum() else 0.0
    pr = inter / int(g.sum()) if g.sum() else 0.0
    return {"iou": iou, "pixel_precision": pp, "pixel_recall": pr}


def mine_errors(df):
    fp = df[(df["gt"] == "good") & (df["prediction"] == "defect")]["image_name"].tolist()
    fn = df[(df["gt"] == "defect") & (df["prediction"] == "good")]["image_name"].tolist()
    return fp, fn


def write_tables(df, metrics, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out_dir / "summary_metrics.csv", index=False)
    cm, lab = binary_confusion(df)
    pd.DataFrame(cm, index=[f"gt_{l}" for l in lab],
                 columns=[f"pred_{l}" for l in lab]).to_csv(out_dir / "confusion_matrix.csv")
    mcm, mlab = multiclass_confusion(df)
    pd.DataFrame(mcm, index=[f"gt_{l}" for l in mlab],
                 columns=[f"pred_{l}" for l in mlab]).to_csv(out_dir / "confusion_multiclass.csv")
    df.to_csv(out_dir / "predictions.csv", index=False)
