# Classical Bottle Defect Inspection (DIP 2026)

A classical computer-vision pipeline that inspects the mouth of glass bottles and decides
whether each bottle is **good** or **defective**, and, when defective, classifies the defect as
**broken_large**, **broken_small**, or **contamination**. The system uses only classical
image-processing techniques — geometric registration, reference differencing, colour distance,
morphology, connected-component analysis and rule-based decisions. **No machine-learning
classifier and no synthetic data are used.**

The full method and results are written up in the report:
**`report/Bottle_Inspection_Report_DIP2026_Lior.pdf`**.

## Headline results (MVTec *bottle*, 83 test images)

| Metric | Value |
|---|---|
| Precision | **1.00** |
| Recall | **0.75** |
| Specificity | **1.00** (0 false positives on 20 good images) |
| F1 | **0.85** |
| Accuracy | **0.81** |
| Per-type recall | broken_large 1.00 · broken_small 0.86 · contamination 0.38 |

Detection is strong and raises no false alarms; the main limitation is subtle low-contrast
**contamination**, about half of which is near-invisible to colour differencing (an honest
negative result, discussed in the report).

## How it works

```
Offline (once):  good images ──ECC align──▶ pixel median ──▶ reference
                 reference ──Otsu + erode──▶ inner ROI

Online (per image):
  inspection ─▶ preprocess (CLAHE + blur) ─▶ register to reference (ECC, Euclidean)
            ─▶ photometric match (gain/bias) ─▶ CIELAB ΔE difference
            ─▶ threshold + morphology ─▶ connected components + features
            ─▶ detect (area % of ROI) ─▶ classify type ─▶ evaluate
```

Key design choices: a **registered-median reference** (sharp ring, no smear); **photometric
gain/bias matching** to remove global illumination (the main false-positive reducer); a
**CIELAB difference kept in physical units** so one fixed threshold is comparable across images;
and a decision threshold **calibrated by balanced accuracy** to respect class imbalance.

## Repository layout

```
src/                 pipeline source (well commented)
  config.py            frozen parameters
  io_utils.py          image loading / listing
  preprocessing.py     CLAHE, reference build, bottle mask + ROI, photometric match
  registration.py      ECC (Euclidean) alignment to the reference
  differencing.py      CIELAB ΔE difference map
  segmentation.py      thresholding + morphology + connected components
  features.py          per-component shape / brightness descriptors
  classification.py    detect + rule-based defect typing
  pipeline.py          per-image orchestration
  evaluate.py          metrics, confusion matrices, threshold calibration, pixel IoU
  visualization.py     all report figures
  run_all.py           end-to-end entry point
tools/
  make_block_diagram.py  pipeline block diagram for the report
report/
  build_report.py        builds the PDF from the result artifacts
  Bottle_Inspection_Report_DIP2026_Lior.pdf
article_assets/        publication-quality figures used in the report
results/               predictions.csv, metrics.json, confusion matrices, per_image/ outputs
data/bottle/           dataset (see "Data" below)
requirements.txt
```

## Run

```bash
pip install -r requirements.txt

# full pipeline: builds reference, runs all test images, calibrates,
# writes results/ and regenerates all figures in article_assets/
python -m src.run_all --data data/bottle --out results --assets article_assets

# (optional) rebuild the PDF report from the latest results
python report/build_report.py
```

Outputs:
- `results/predictions.csv` — per-image prediction, scores, registration diagnostics
- `results/metrics.json`, `results/summary_metrics.csv` — image-level metrics
- `results/confusion_matrix.csv`, `results/confusion_multiclass.csv`
- `results/pixel_iou.csv` — auxiliary pixel-level localisation vs ground truth
- `results/per_image/<class>_<id>/` — diff, mask, roi, registered, overlay PNGs
- `article_assets/*.png` — all report figures
- `report/Bottle_Inspection_Report_DIP2026_Lior.pdf` — the report

## Data

The pipeline is evaluated on the **bottle** category of the public
[MVTec Anomaly Detection dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
(Bergmann et al., CVPR 2019; licensed CC BY-NC-SA 4.0). Expected structure:

```
data/bottle/
  train/good/          good images used to build the reference
  test/good|broken_large|broken_small|contamination/
  ground_truth/        defect masks (used only for the auxiliary pixel metric)
```

The pipeline is dataset-agnostic: point `--data` at any folder with the same structure
(e.g. self-captured images) to run on different data.

## Notes
- All processing is at 512×512. Parameters are frozen in `src/config.py`.
- The course brief encourages original images; the public MVTec set was used for
  reproducibility and access to ground-truth masks (see the report's "Note on data").
