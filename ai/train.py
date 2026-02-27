"""
ai/train.py  —  EcoYield AI Model Trainer

Two modes:
  1. SYNTHETIC (default) : ~1000 samples total (112 per class × 9 classes)
  2. REAL DATA           : python3 ai/train.py --data LSWMD.pkl

Flags:
  --data   LSWMD.pkl     use real WM-811K dataset
  --samples 200          override samples-per-class (default 112)
"""

import sys, os, pickle, time, json
import numpy as np

# ── Safe pandas import for old WM-811K pkl files ─────────────────────────
try:
    import pandas as pd
    if not hasattr(pd, 'indexes'):
        import pandas.core.indexes
        sys.modules['pandas.indexes'] = pd.core.indexes
except ImportError:
    pd = None

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ai.features import extract_features, DEFECT_CLASSES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ecoyield_model.pkl")
META_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "model_meta.json")

# 112 × 9 classes = 1008 total ≈ 1000
DEFAULT_SAMPLES_PER_CLASS = 112


# ── Wafer map generators ──────────────────────────────────────────────────
def _make_wafer_base(size=64):
    wm = np.zeros((size, size), dtype=np.float32)
    cy, cx = size // 2, size // 2
    r = size // 2 - 2
    y, x = np.ogrid[:size, :size]
    circle = (x - cx)**2 + (y - cy)**2 <= r**2
    return wm, circle, cy, cx, r

def gen_center(rng, size=64):
    wm, circle, cy, cx, r = _make_wafer_base(size)
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((y-cy)**2 + (x-cx)**2)
    wm[circle & (dist < r * rng.uniform(0.25, 0.45))] = 1.0
    wm[circle & (rng.random((size, size)) < 0.02)] = 1.0
    return wm

def gen_donut(rng, size=64):
    wm, circle, cy, cx, r = _make_wafer_base(size)
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((y-cy)**2 + (x-cx)**2)
    inner = r * rng.uniform(0.3, 0.5)
    outer = inner + r * rng.uniform(0.15, 0.3)
    wm[circle & (dist >= inner) & (dist <= outer)] = 1.0
    return wm

def gen_edge_loc(rng, size=64):
    wm, circle, cy, cx, r = _make_wafer_base(size)
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((y-cy)**2 + (x-cx)**2)
    angle = rng.uniform(0, 2*np.pi)
    arc_w = rng.uniform(np.pi/4, np.pi)
    pt_angle = np.arctan2((y-cy)/r, (x-cx)/r)
    adiff = np.minimum(np.abs(pt_angle - angle), 2*np.pi - np.abs(pt_angle - angle))
    wm[circle & (dist > r*0.75) & (adiff < arc_w/2)] = 1.0
    return wm

def gen_edge_ring(rng, size=64):
    wm, circle, cy, cx, r = _make_wafer_base(size)
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((y-cy)**2 + (x-cx)**2)
    wm[circle & (dist >= r * rng.uniform(0.75, 0.88))] = 1.0
    return wm

def gen_scratch(rng, size=64):
    wm, circle, cy, cx, r = _make_wafer_base(size)
    for _ in range(rng.randint(1, 3)):
        a = rng.uniform(0, np.pi)
        d = rng.uniform(-r*0.5, r*0.5)
        w = rng.uniform(1, 3)
        yi, xi = np.ogrid[:size, :size]
        wm[circle & (np.abs((xi-cx)*np.sin(a)-(yi-cy)*np.cos(a)-d) < w)] = 1.0
    return wm

def gen_loc(rng, size=64):
    wm, circle, cy, cx, r = _make_wafer_base(size)
    for _ in range(rng.randint(2, 6)):
        a = rng.uniform(0, 2*np.pi)
        dc = rng.uniform(0, r*0.8)
        ccy, ccx = cy+dc*np.sin(a), cx+dc*np.cos(a)
        cr = rng.uniform(2, 8)
        yi, xi = np.ogrid[:size, :size]
        wm[circle & (np.sqrt((yi-ccy)**2+(xi-ccx)**2) < cr)] = 1.0
    return wm

def gen_near_full(rng, size=64):
    wm, circle, cy, cx, r = _make_wafer_base(size)
    wm[circle & (rng.random((size, size)) < rng.uniform(0.6, 0.9))] = 1.0
    return wm

def gen_random(rng, size=64):
    wm, circle, cy, cx, r = _make_wafer_base(size)
    wm[circle & (rng.random((size, size)) < rng.uniform(0.03, 0.15))] = 1.0
    return wm

def gen_none(rng, size=64):
    wm, circle, cy, cx, r = _make_wafer_base(size)
    wm[circle & (rng.random((size, size)) < rng.uniform(0.001, 0.01))] = 1.0
    return wm

GENERATORS = {
    "Center": gen_center, "Donut": gen_donut, "Edge-Loc": gen_edge_loc,
    "Edge-Ring": gen_edge_ring, "Loc": gen_loc, "Near-Full": gen_near_full,
    "Scratch": gen_scratch, "Random": gen_random, "None": gen_none,
}


# ── Dataset builders ──────────────────────────────────────────────────────
def generate_synthetic_dataset(samples_per_class=DEFAULT_SAMPLES_PER_CLASS, seed=42):
    total = samples_per_class * len(DEFECT_CLASSES)
    print(f"\n  Generating {total} synthetic wafer maps  ({samples_per_class} per class × {len(DEFECT_CLASSES)} classes)...")
    rng = np.random.RandomState(seed)
    X, y = [], []
    for cls_idx, cls_name in enumerate(DEFECT_CLASSES):
        for _ in range(samples_per_class):
            wm = GENERATORS[cls_name](rng, size=64)
            wm = np.clip(wm + (rng.random((64, 64)) < 0.005).astype(np.float32), 0, 1)
            X.append(extract_features(wm))
            y.append(cls_idx)
        print(f"    ✔  {cls_name:12s}  {samples_per_class} samples")
    return np.array(X), np.array(y)


def load_real_wm811k(pkl_path: str):
    if pd is None:
        raise ImportError("pandas required: pip install pandas")
    print(f"\n  Loading WM-811K from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        df = pickle.load(f, encoding="latin1")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Expected pandas DataFrame")
    print(f"  Total records  : {len(df)}")
    labeled = df[df["failureType"].apply(lambda x: len(x) > 0 if hasattr(x, '__len__') else False)]
    print(f"  Labeled records: {len(labeled)}")
    X, y, skipped = [], [], 0
    class_map = {c: i for i, c in enumerate(DEFECT_CLASSES)}
    for _, row in labeled.iterrows():
        try:
            wm = np.array(row["waferMap"], dtype=np.float32)
            ft = row["failureType"]
            if isinstance(ft, (list, np.ndarray)):
                ft = ft[0][0] if hasattr(ft[0], '__len__') else ft[0]
            ft = str(ft).strip()
            ft = ft[0].upper() + ft[1:]  # none → None
            if ft not in class_map:
                skipped += 1; continue
            if wm.max() > 1:
                wm = (wm == 2).astype(np.float32)
            X.append(extract_features(wm))
            y.append(class_map[ft])
        except Exception:
            skipped += 1
    print(f"  Extracted {len(X)} samples  ({skipped} skipped)")
    return np.array(X), np.array(y)


# ── Train ─────────────────────────────────────────────────────────────────
def train(data_path=None, samples_per_class=DEFAULT_SAMPLES_PER_CLASS):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    t0 = time.time()

    if data_path and os.path.exists(data_path):
        print(f"\n[EcoYield AI] Training on REAL WM-811K data: {data_path}")
        X, y = load_real_wm811k(data_path)
        mode = "real_wm811k"
    else:
        if data_path:
            print(f"  [!] File not found: {data_path} — using synthetic data")
        print("\n[EcoYield AI] Synthetic training mode")
        print("  Tip: add --data LSWMD.pkl to train on real WM-811K data")
        X, y = generate_synthetic_dataset(samples_per_class)
        mode = "synthetic"

    print(f"\n  Dataset  : {X.shape[0]} samples × {X.shape[1]} features")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Split    : {len(X_train)} train  |  {len(X_test)} test")

    print("\n  Training GradientBoosting classifier...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.1, subsample=0.8,
            random_state=42, verbose=0,
        ))
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=DEFECT_CLASSES, output_dict=True)

    print(f"\n  ✔  Accuracy : {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=DEFECT_CLASSES))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "accuracy": round(acc, 4), "mode": mode,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_samples": int(X.shape[0]), "num_features": int(X.shape[1]),
        "samples_per_class": samples_per_class, "classes": DEFECT_CLASSES,
        "per_class": {
            cls: {"precision": round(report[cls]["precision"], 3),
                  "recall":    round(report[cls]["recall"], 3),
                  "f1":        round(report[cls]["f1-score"], 3)}
            for cls in DEFECT_CLASSES
        },
        "train_seconds": round(time.time() - t0, 1),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  ✔  Model    → {MODEL_PATH}")
    print(f"  ✔  Metadata → {META_PATH}")
    print(f"  ✔  Time     : {meta['train_seconds']}s")
    return model, meta


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data_path        = None
    samples_per_class = DEFAULT_SAMPLES_PER_CLASS  # 112 × 9 = 1008

    if "--data" in sys.argv:
        i = sys.argv.index("--data")
        if i + 1 < len(sys.argv):
            data_path = sys.argv[i + 1]

    if "--samples" in sys.argv:
        i = sys.argv.index("--samples")
        if i + 1 < len(sys.argv):
            samples_per_class = int(sys.argv[i + 1])

    print(f"  samples_per_class : {samples_per_class}")
    print(f"  total samples     : {samples_per_class * len(DEFECT_CLASSES)}")
    train(data_path, samples_per_class)
