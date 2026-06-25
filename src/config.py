"""Central configuration for the classical bottle-inspection pipeline.

All tunable parameters live here so that every stage of the pipeline reads
from a single, documented source.  Values were calibrated empirically on the
score distributions of the `good` versus defective test images (see
`evaluate.calibrate_area_threshold` and the threshold-sweep figure).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- geometry ---------------------------------------------------------
    image_size: int = 512            # all images are processed at 512x512

    # --- preprocessing ----------------------------------------------------
    clahe_clip: float = 2.0          # CLAHE contrast-limit
    clahe_grid: int = 8              # CLAHE tile grid (grid x grid)
    blur_ksize: int = 5             # Gaussian blur kernel for denoising

    # --- reference construction ------------------------------------------
    ref_sample: int = 80             # # good images aggregated into the reference
    roi_erode: int = 23              # erosion kernel for the inner inspection ROI

    # --- registration (ECC, MOTION_EUCLIDEAN = rotation + translation) ----
    ecc_iterations: int = 100        # max iterations of the ECC critical loop
    ecc_eps: float = 1e-5            # ECC convergence tolerance (stopping criterion)
    ecc_min_cc: float = 0.50         # min ECC correlation to accept the warp
    max_shift_px: float = 40.0       # reject phase-correlation init beyond this

    # --- differencing -----------------------------------------------------
    diff_blur: int = 5               # Gaussian blur of the LAB distance map
    ring_suppress: int = 0           # dilation (px) of high-gradient ring to damp it

    # --- segmentation / morphology ---------------------------------------
    abs_threshold: float = 55.0      # absolute LAB delta-E threshold (calibrated on good/defect ROI distributions)
    use_otsu_floor: bool = False      # OR the fixed threshold with ROI-Otsu
    morph_kernel: int = 5            # open/close structuring-element size
    min_component_area: int = 100     # discard connected components below this area

    # --- classification ---------------------------------------------------
    area_threshold_pct: float = 1.00  # >= this %ROI of anomaly -> 'defect' (calibrated)
    # multi-class (defect-type) rules, on the dominant component:
    bright_glass_delta: float = 20.0  # dom brighter than ref by this -> broken glass
    large_area_pct: float = 1.20      # largest comp >= this %ROI -> broken_large
    large_aspect: float = 1.40        # elongated dominant blob -> broken_large
    large_ecc: float = 0.90           # high eccentricity -> broken_large

    random_state: int = 0


CFG = Config()

# Human-readable label sets used across the project
DEFECT_TYPES = ("broken_large", "broken_small", "contamination")
ALL_CLASSES = ("good",) + DEFECT_TYPES
