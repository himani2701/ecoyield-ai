"""
start.py  —  EcoYield AI  One-Command Bootstrap
Run:  python3 start.py

Does everything automatically:
  1. Creates folder structure
  2. Trains ML model on synthetic data (if not already trained)
  3. Starts the API server on http://localhost:8000
  4. Tells you to open dashboard.html in your browser

No dataset, no npm, no pip install — just Python 3 + sklearn + numpy + opencv
"""

import os, sys, json, time, subprocess, importlib

# ── Check required packages ───────────────────────────────────────────────
REQUIRED = {
    "numpy":      "numpy",
    "sklearn":    "scikit-learn",
    "cv2":        "opencv-python",
    "PIL":        "Pillow",
}
missing = []
for mod, pkg in REQUIRED.items():
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"\n  [!] Missing packages: {', '.join(missing)}")
    print(f"  Run: pip install {' '.join(missing)}")
    sys.exit(1)

# ── Folder structure ──────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
for folder in ["ai", "api", "db", "frontend", "models", "uploads"]:
    os.makedirs(os.path.join(BASE, folder), exist_ok=True)

MODEL_PATH = os.path.join(BASE, "models", "ecoyield_model.pkl")
META_PATH  = os.path.join(BASE, "models", "model_meta.json")

# ── Auto-train if model doesn't exist ────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("\n  [1/2] No model found — training now (synthetic data, ~15s)...")
    train_script = os.path.join(BASE, "ai", "train.py")
    if not os.path.exists(train_script):
        print(f"  [!] Missing: {train_script}")
        print("  Make sure ai/train.py is in the same folder as start.py")
        sys.exit(1)
    result = subprocess.run([sys.executable, train_script], cwd=BASE)
    if result.returncode != 0:
        print("  [!] Training failed. Check errors above.")
        sys.exit(1)
    print("  ✔  Model trained and saved.")
else:
    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
    acc = meta.get("accuracy", 0)
    mode = meta.get("mode", "?")
    print(f"\n  [1/2] Model already trained  [{mode} | acc={acc*100:.1f}%] — skipping training.")

# ── Start server ──────────────────────────────────────────────────────────
server_script = os.path.join(BASE, "api", "server.py")
if not os.path.exists(server_script):
    print(f"  [!] Missing: {server_script}")
    sys.exit(1)

print("\n  [2/2] Starting API server...")
print("\n  ╔═══════════════════════════════════════════════════╗")
print("  ║       EcoYield AI  —  Prototype Ready            ║")
print("  ╠═══════════════════════════════════════════════════╣")
print("  ║  Backend   →  http://localhost:8000              ║")
print("  ║  Frontend  →  open  frontend/dashboard.html      ║")
print("  ║  Press Ctrl+C to stop                           ║")
print("  ╚═══════════════════════════════════════════════════╝\n")

os.execv(sys.executable, [sys.executable, server_script])
