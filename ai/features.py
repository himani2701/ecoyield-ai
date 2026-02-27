"""
ai/features.py
Real feature extraction from wafer bin maps.
Extracts 47 hand-crafted spatial & statistical features
used by the WM-811K literature (matches production ViT preprocessing).
Works on:
  - .pkl files from WM-811K dataset (numpy arrays)
  - .png/.jpg wafer map images
  - synthetic arrays (for demo/training)
"""
import numpy as np
import pickle
import io
from PIL import Image
import cv2


DEFECT_CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-Full", "Scratch", "Random", "None"
]

SEVERITY = {
    "None":      "none",
    "Random":    "low",
    "Loc":       "medium",
    "Edge-Loc":  "medium",
    "Scratch":   "medium",
    "Donut":     "high",
    "Center":    "high",
    "Edge-Ring": "high",
    "Near-Full": "critical",
}

# Yield ranges per severity (min, max %)
YIELD_RANGE = {
    "none":     (95.0, 99.5),
    "low":      (86.0, 94.0),
    "medium":   (70.0, 85.0),
    "high":     (42.0, 69.0),
    "critical": (8.0,  41.0),
}


def load_wafer_map(filepath: str) -> np.ndarray:
    """Load wafer map from .pkl, .png, .jpg, or .bmp file.
    Returns a normalized 2D numpy array (values 0 or 1).
    """
    ext = filepath.lower().split(".")[-1]

    if ext == "pkl":
        with open(filepath, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        # WM-811K .pkl can be a dict with 'waferMap' key or a direct array
        if isinstance(data, dict):
            wm = data.get("waferMap") or data.get("wafer_map") or list(data.values())[0]
        elif isinstance(data, (list, np.ndarray)):
            wm = np.array(data)
        else:
            raise ValueError("Unrecognised .pkl structure")
        wm = np.array(wm, dtype=np.float32)
        # Normalize: WM-811K uses 0=no die, 1=pass, 2=fail
        # Convert to binary defect map: 1 where die failed
        if wm.max() > 1:
            wm = (wm == 2).astype(np.float32)
        return wm

    else:  # image
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read image: {filepath}")
        # Threshold: dark pixels = defects
        _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)
        return binary.astype(np.float32)


def extract_features(wafer_map: np.ndarray) -> np.ndarray:
    """
    Extract 47 real spatial features from a wafer map.
    Based on WM-811K literature feature engineering.
    """
    wm = np.array(wafer_map, dtype=np.float32)
    h, w = wm.shape
    features = []

    # ── 1. Global statistics (6 features) ──────────────────────────────
    total_dies = wm.size
    defect_count = wm.sum()
    defect_density = defect_count / (total_dies + 1e-9)
    features += [
        defect_density,
        defect_count,
        total_dies,
        wm.mean(),
        wm.std(),
        float(defect_count > 0),
    ]

    # ── 2. Radial zone features (8 features) ────────────────────────────
    cy, cx = h / 2, w / 2
    max_r = min(h, w) / 2
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_map = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    zones = 4
    for z in range(zones):
        r0 = z * max_r / zones
        r1 = (z + 1) * max_r / zones
        mask = (dist_map >= r0) & (dist_map < r1)
        zone_density = wm[mask].mean() if mask.sum() > 0 else 0.0
        features.append(float(zone_density))
    # Edge vs center ratio
    center_mask = dist_map < max_r * 0.3
    edge_mask = dist_map > max_r * 0.7
    center_d = wm[center_mask].mean() if center_mask.sum() > 0 else 0.0
    edge_d = wm[edge_mask].mean() if edge_mask.sum() > 0 else 0.0
    features += [
        center_d,
        edge_d,
        edge_d - center_d,
        float(edge_d > center_d * 1.5),
    ]

    # ── 3. Ring detection (6 features) ──────────────────────────────────
    ring_counts = []
    ring_step = max_r / 6
    for i in range(6):
        r0 = i * ring_step
        r1 = (i + 1) * ring_step
        ring_mask = (dist_map >= r0) & (dist_map < r1)
        ring_counts.append(wm[ring_mask].mean() if ring_mask.sum() > 0 else 0.0)
    features += ring_counts

    # ── 4. Quadrant analysis (8 features) ───────────────────────────────
    q_masks = [
        (y_idx < cy) & (x_idx < cx),   # top-left
        (y_idx < cy) & (x_idx >= cx),  # top-right
        (y_idx >= cy) & (x_idx < cx),  # bottom-left
        (y_idx >= cy) & (x_idx >= cx), # bottom-right
    ]
    q_densities = [wm[m].mean() if m.sum() > 0 else 0.0 for m in q_masks]
    features += q_densities
    features += [
        max(q_densities) - min(q_densities),  # quadrant imbalance
        np.std(q_densities),
        float(max(q_densities) > 0.3),
        float(np.std(q_densities) > 0.1),
    ]

    # ── 5. Linear / scratch detection (5 features) ──────────────────────
    wm_uint8 = (wm * 255).astype(np.uint8)
    edges = cv2.Canny(wm_uint8, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10,
                            minLineLength=min(h, w) // 4, maxLineGap=5)
    num_lines = len(lines) if lines is not None else 0
    features += [
        float(num_lines),
        float(num_lines > 2),
        float(num_lines > 5),
        edges.mean() / 255.0,
        float(edges.sum() > 0),
    ]

    # ── 6. Cluster / blob analysis (6 features) ──────────────────────────
    wm_uint8_2 = (wm * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(wm_uint8_2)
    num_blobs = num_labels - 1  # subtract background
    blob_areas = stats[1:, cv2.CC_STAT_AREA] if num_blobs > 0 else np.array([0])
    features += [
        float(num_blobs),
        float(blob_areas.max()) if num_blobs > 0 else 0.0,
        float(blob_areas.mean()) if num_blobs > 0 else 0.0,
        float(num_blobs > 5),
        float(num_blobs == 1 and blob_areas[0] > total_dies * 0.3),
        float(blob_areas.std()) if num_blobs > 0 else 0.0,
    ]

    # ── 7. Symmetry features (4 features) ────────────────────────────────
    wm_resized = cv2.resize(wm, (32, 32))
    flip_h = np.flip(wm_resized, axis=0)
    flip_v = np.flip(wm_resized, axis=1)
    rot180 = np.rot90(wm_resized, 2)
    features += [
        float(np.abs(wm_resized - flip_h).mean()),
        float(np.abs(wm_resized - flip_v).mean()),
        float(np.abs(wm_resized - rot180).mean()),
        float(np.abs(wm_resized - flip_h).mean() + np.abs(wm_resized - flip_v).mean()),
    ]

    # ── 8. Donut / ring shape score (4 features) ─────────────────────────
    # Donut = high edge density, low center density, ring-like radial profile
    radial_var = float(np.var(ring_counts))
    radial_peak_idx = int(np.argmax(ring_counts))
    features += [
        radial_var,
        float(radial_peak_idx),                      # which ring has most defects
        float(ring_counts[radial_peak_idx]),
        float(ring_counts[0] < 0.1 and ring_counts[2] > 0.2),  # center clear, mid-ring high
    ]

    assert len(features) == 47, f"Expected 47 features, got {len(features)}"
    return np.array(features, dtype=np.float32)


def generate_heatmap_8x8(wafer_map: np.ndarray) -> list:
    """Downsample wafer map to 8x8 defect density heatmap."""
    wm = np.array(wafer_map, dtype=np.float32)
    h, w = wm.shape
    out = []
    for r in range(8):
        row = []
        for c in range(8):
            r0 = int(r * h / 8)
            r1 = int((r + 1) * h / 8)
            c0 = int(c * w / 8)
            c1 = int((c + 1) * w / 8)
            patch = wm[r0:r1, c0:c1]
            val = float(patch.mean()) if patch.size > 0 else 0.0
            row.append(round(val, 3))
        out.append(row)
    return out
