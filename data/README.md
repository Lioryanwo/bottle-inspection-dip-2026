# Data

This folder uses the **bottle** category of the public **MVTec Anomaly Detection (MVTec AD)**
dataset (Bergmann et al., CVPR 2019; licensed CC BY-NC-SA 4.0).

Only a **small sample** of images is bundled here (a few per class) so the code can be
inspected and smoke-tested. The full quantitative results in the report were computed on the
complete split (209 good training images; 83 test images: 20 good, 20 broken_large,
22 broken_small, 21 contamination).

## Get the full dataset
Download from: https://www.mvtec.com/company/research/datasets/mvtec-ad
and place it so the structure is:

```
data/bottle/
  train/good/*.png
  test/{good,broken_large,broken_small,contamination}/*.png
  ground_truth/{broken_large,broken_small,contamination}/*_mask.png
```

Then reproduce everything with:
```
python -m src.run_all --data data/bottle --out results --assets article_assets
```
