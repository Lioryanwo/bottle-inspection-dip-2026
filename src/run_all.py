"""Full experiment: build reference, inspect all test images, calibrate the
decision threshold, evaluate (image- and pixel-level), and render every figure
used in the report.

Usage:  python -m src.run_all --data data/bottle --out results --assets article_assets
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import CFG, ALL_CLASSES
from .io_utils import imread_rgb, list_images, imwrite_rgb
from .preprocessing import (resize_rgb, to_gray, preprocess_gray,
                            build_reference, bottle_mask_from_reference)
from .differencing import lab_distance
from .pipeline import BottleInspectionPipeline
from .classification import classify_type
from . import evaluate as ev
from . import visualization as viz


def collect_test_images(data):
    return sorted(Path(data, "test").glob("*/*.png"))


def gt_mask_for(path, data):
    p = Path(path)
    if p.parent.name == "good":
        return None
    gt = Path(data, "ground_truth", p.parent.name, f"{p.stem}_mask.png")
    if gt.exists():
        m = cv2.imread(str(gt), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(m, (CFG.image_size, CFG.image_size),
                          interpolation=cv2.INTER_NEAREST)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bottle")
    ap.add_argument("--out", default="results")
    ap.add_argument("--assets", default="article_assets")
    args = ap.parse_args()
    data, out, assets = Path(args.data), Path(args.out), Path(args.assets)
    out.mkdir(parents=True, exist_ok=True); assets.mkdir(parents=True, exist_ok=True)

    # ---- 1. reference + ROI (offline, once) ------------------------------
    good_paths = list_images(data / "train" / "good")
    if not good_paths:
        raise SystemExit("No train/good images found")
    good_imgs = [imread_rgb(p) for p in good_paths]
    print(f"Building registered-median reference from {min(len(good_imgs),CFG.ref_sample)} good images...")
    ref = build_reference(good_imgs, register=True)
    full_mask, roi = bottle_mask_from_reference(ref)
    imwrite_rgb(assets / "reference_image.png", ref)
    imwrite_rgb(assets / "bottle_full_mask.png", full_mask)
    imwrite_rgb(assets / "bottle_inner_roi.png", roi)

    pipe = BottleInspectionPipeline(ref, roi, CFG.abs_threshold)

    # ---- 2. inspect every test image ------------------------------------
    rows, store = [], {}
    for p in collect_test_images(data):
        name = f"{p.parent.name}_{p.stem}"
        res = pipe.run_one(imread_rgb(p), out / "per_image" / name, name,
                           keep_arrays=True)
        arr = res.pop("_arrays")
        store[name] = {"path": p, "registered": arr["registered"],
                       "aligned": arr["aligned"], "diff": arr["diff"],
                       "mask": arr["mask"], "overlay": arr["overlay"]}
        gt_type = ev.gt_type_from_path(p)
        rows.append({"image_name": name, "path": str(p),
                     "gt": ev.gt_binary(gt_type), "gt_type": gt_type, **res})
    df = pd.DataFrame(rows)

    # ---- 3. calibrate the decision threshold ----------------------------
    ts, prec, rec, f1, best_t = ev.calibrate_area_threshold(df)
    df["prediction"] = np.where(df["total_area_pct"] >= best_t, "defect", "good")
    df["pred_type"] = [classify_type(r) if r_pred == "defect" else "good"
                       for r_pred, r in zip(df["prediction"],
                                            df.to_dict("records"))]
    print(f"Calibrated area threshold = {best_t:.3f} %ROI")

    # ---- 4. metrics -----------------------------------------------------
    metrics = ev.binary_metrics(df)
    metrics["area_threshold_pct"] = round(best_t, 4)

    # pixel-level IoU on defect images that have GT masks
    ious = []
    for _, r in df[df["gt"] == "defect"].iterrows():
        gm = gt_mask_for(r["path"], data)
        if gm is None:
            continue
        pm = store[r["image_name"]]["mask"]
        pix = ev.pixel_iou(pm, gm)
        ious.append({"image_name": r["image_name"], "gt_type": r["gt_type"], **pix})
    iou_df = pd.DataFrame(ious)
    if len(iou_df):
        metrics["mean_pixel_iou"] = round(float(iou_df["iou"].mean()), 4)
        metrics["mean_pixel_recall"] = round(float(iou_df["pixel_recall"].mean()), 4)
        iou_df.to_csv(out / "pixel_iou.csv", index=False)

    metrics["per_type_recall"] = {k: round(v, 3) for k, v in ev.per_type_recall(df).items()}
    fp_names, fn_names = ev.mine_errors(df)
    metrics["false_positives"] = fp_names
    metrics["false_negatives"] = fn_names

    ev.write_tables(df, {k: v for k, v in metrics.items()
                         if not isinstance(v, (list, dict))}, out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Image-level:", {k: metrics[k] for k in
          ("precision", "recall", "specificity", "f1", "accuracy")})
    print("Per-type recall:", metrics["per_type_recall"])
    print("FP:", fp_names, "| FN:", fn_names)

    # ---- 5. figures -----------------------------------------------------
    render_figures(df, store, ref, full_mask, roi, data, assets,
                   (ts, prec, rec, f1, best_t), iou_df)
    print("Figures written to", assets)
    return metrics


def render_figures(df, store, ref, full_mask, roi, data, assets, sweep, iou_df):
    ts, prec, rec, f1, best_t = sweep

    # preprocessing montage (a clean good image)
    g0 = imread_rgb(Path(data, "test", "good", "000.png"))
    rgb = resize_rgb(g0); gray = to_gray(rgb)
    clahe = cv2.createCLAHE(CFG.clahe_clip, (CFG.clahe_grid, CFG.clahe_grid)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (CFG.blur_ksize, CFG.blur_ksize), 0)
    viz.preprocessing_montage(assets / "preprocessing_stages.png", rgb, gray, clahe, blurred)

    viz.reference_roi_figure(assets / "reference_roi.png", ref, full_mask, roi)

    # Illustrate the registration stage with a *controlled* rotation of a good
    # image. Native misalignment in this fixed-camera dataset is small
    # (mean |angle| ~1 deg, max ~4 deg), so a known perturbation is applied to
    # show that ECC recovers rotation and reduces the resulting residual.
    from .registration import register_to_reference
    gname = df[df["gt_type"] == "good"]["image_name"].iloc[0]
    g = resize_rgb(imread_rgb(store[gname]["path"]))
    h, w = g.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 8.0, 1.0)
    M[0, 2] += 6; M[1, 2] -= 4
    misaligned = cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
    recovered, _ = register_to_reference(misaligned, ref)
    diff_before = lab_distance(ref, misaligned)
    diff_after = lab_distance(ref, recovered)
    e_before = float(diff_before[roi > 0].mean())
    e_after = float(diff_after[roi > 0].mean())
    viz.registration_figure(assets / "registration_example.png", ref, misaligned,
                            recovered, diff_before, diff_after, e_before, e_after)

    # confusion matrices
    cmb, labb = ev.binary_confusion(df)
    viz.confusion_figure(assets / "confusion_matrix.png", cmb, labb,
                         "Binary confusion (good vs defect)")
    cmm, labm = ev.multiclass_confusion(df)
    viz.confusion_figure(assets / "confusion_multiclass.png", cmm, labm,
                         "Multi-class confusion (defect type)")

    # score distribution + threshold sweep
    viz.score_distribution_figure(assets / "score_distribution.png", df, best_t)
    viz.threshold_sweep_figure(assets / "threshold_sweep.png", ts, prec, rec, f1, best_t)

    # representative stage panels: one correctly handled example per class
    for cls in ALL_CLASSES:
        want_pred = "good" if cls == "good" else "defect"
        exact = df[(df["gt_type"] == cls) & (df["pred_type"] == cls)]
        cand = exact if len(exact) else df[(df["gt_type"] == cls) & (df["prediction"] == want_pred)]
        if not len(cand):
            cand = df[df["gt_type"] == cls]
        if not len(cand):
            continue
        name = cand["image_name"].iloc[0]
        s = store[name]
        gm = gt_mask_for(s["path"], data)
        r = df[df["image_name"] == name].iloc[0]
        title = (f"{cls}  |  pred: {r['prediction']}"
                 + (f" ({r['pred_type']})" if r['prediction'] == 'defect' else "")
                 + f"\narea={r['total_area_pct']:.2f}%  ECC cc={r['ecc_cc']:.2f}"
                 + f"  angle={r['angle_deg']:.1f} deg")
        viz.stage_panel(assets / f"panel_{cls}.png", ref,
                        resize_rgb(imread_rgb(s["path"])), s["registered"],
                        s["diff"], s["mask"], s["overlay"], gm, title)

    # FP / FN galleries
    fp_names, fn_names = ev.mine_errors(df)
    fp_items = [(store[n]["overlay"],
                 f"{n}\narea={df[df.image_name==n]['total_area_pct'].iloc[0]:.2f}%")
                for n in fp_names]
    fn_items = [(store[n]["overlay"],
                 f"{n}\narea={df[df.image_name==n]['total_area_pct'].iloc[0]:.2f}%")
                for n in fn_names]
    viz.gallery_figure(assets / "false_positives.png", fp_items,
                       "False positives (good images flagged as defect)")
    viz.gallery_figure(assets / "false_negatives.png", fn_items,
                       "False negatives (defects missed)")

    # pixel IoU example: best-IoU defect, pred (red) vs GT (green)
    if len(iou_df):
        bestn = iou_df.sort_values("iou", ascending=False)["image_name"].iloc[0]
        s = store[bestn]
        gm = gt_mask_for(s["path"], data)
        comp = s["registered"].copy()
        gcont, _ = cv2.findContours(gm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pcont, _ = cv2.findContours(s["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(comp, gcont, -1, (46, 200, 90), 2)
        cv2.drawContours(comp, pcont, -1, (230, 60, 70), 2)
        iv = iou_df[iou_df.image_name == bestn]["iou"].iloc[0]
        viz.gallery_figure(assets / "pixel_iou_example.png",
                           [(comp, f"{bestn}\nGT=green  pred=red  IoU={iv:.2f}")],
                           "Pixel-level localisation example", ncols=1)


if __name__ == "__main__":
    main()
