"""
api/server.py  —  EcoYield AI  Full Stack API Server
Python stdlib only. No external web framework needed.

Routes:
  GET  /health                  - Server + model + DB status
  GET  /api/model-info          - Loaded model metadata
  POST /api/analyze             - Upload & analyze wafer map (real ML)
  GET  /api/scans               - Recent scan history
  GET  /api/scans/<scan_id>     - Single scan detail
  GET  /api/stats               - Cumulative sustainability stats
  GET  /api/distribution        - Defect type distribution
  GET  /api/trend               - Yield trend over time
  POST /api/retrain             - Retrain model (optional data path)
"""
import os, sys, json, time, uuid, pickle, traceback, io
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np

# Adjust path so we can import sibling packages
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ai.features import (
    extract_features, generate_heatmap_8x8,
    DEFECT_CLASSES, SEVERITY, YIELD_RANGE, load_wafer_map
)
from db.database import (
    init_db, save_scan, get_scan, get_recent_scans,
    get_totals, get_defect_distribution, get_yield_trend,
    upsert_model_meta, get_active_model_meta
)

MODEL_PATH  = os.path.join(BASE_DIR, "models", "ecoyield_model.pkl")
META_PATH   = os.path.join(BASE_DIR, "models", "model_meta.json")
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Global model state ────────────────────────────────────────────────────
_model = None
_model_meta = {}

def load_model():
    global _model, _model_meta
    if not os.path.exists(MODEL_PATH):
        print("  [!] No model found. Training now...")
        from ai.train import train
        _model, _model_meta = train()
        upsert_model_meta(_model_meta)
        return
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            _model_meta = json.load(f)
    print(f"  ✔  Model loaded  accuracy={_model_meta.get('accuracy', '?')}")


# ── Inference engine ──────────────────────────────────────────────────────
def run_inference(wafer_map: np.ndarray, filename: str) -> dict:
    global _model
    t0 = time.perf_counter()

    features = extract_features(wafer_map)
    features_2d = features.reshape(1, -1)

    # Real ML prediction
    pred_idx = int(_model.predict(features_2d)[0])
    proba = _model.predict_proba(features_2d)[0]

    primary = DEFECT_CLASSES[pred_idx]
    confidence = float(proba[pred_idx])
    all_conf = {DEFECT_CLASSES[i]: round(float(p), 4) for i, p in enumerate(proba)}

    severity = SEVERITY[primary]
    yield_range = YIELD_RANGE[severity]

    # Yield estimation: blend model confidence + defect density
    defect_density = float(wafer_map.mean())
    base_yield = yield_range[0] + (yield_range[1] - yield_range[0]) * (1 - defect_density)
    yield_pct = round(min(99.5, max(1.0, base_yield + (confidence - 0.5) * 5)), 2)

    heatmap = generate_heatmap_8x8(wafer_map)
    inference_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "primary_defect": primary,
        "confidence": round(confidence, 4),
        "all_confidences": all_conf,
        "severity": severity,
        "yield_percent": yield_pct,
        "defect_density": round(defect_density, 4),
        "heatmap_8x8": heatmap,
        "inference_ms": inference_ms,
        "model": _model_meta.get("mode", "ecoyield-v1"),
        "model_accuracy": _model_meta.get("accuracy", 0),
        "hardware": "AMD ROCm™ (CPU fallback for demo)",
        "features_extracted": len(features),
    }


def compute_sustainability(yield_pct: float, severity: str, defect_type: str) -> dict:
    WATER_PER_WAFER = 2200
    ENERGY_PER_WAFER = 4.8
    COST_PER_WAFER = 3200
    SILICON_PER_WAFER = 185
    baseline = 78.0
    lot_size = 25

    baseline_frac = baseline / 100
    current_frac = yield_pct / 100
    cascade = severity not in ("none", "low")

    wafers_scrapped_baseline = int(lot_size * (1 - baseline_frac))
    wafers_scrapped_current  = int(lot_size * (1 - current_frac))

    if cascade:
        wafers_rescued = max(0, int(wafers_scrapped_baseline * 0.4))
        early_bonus = max(0, int(wafers_scrapped_baseline * 0.3))
    else:
        wafers_rescued = max(0, wafers_scrapped_baseline - wafers_scrapped_current)
        early_bonus = 0

    total_rescued = wafers_rescued + early_bonus
    save_factor = 0.85

    water_saved    = round(total_rescued * WATER_PER_WAFER * save_factor)
    energy_saved   = round(total_rescued * ENERGY_PER_WAFER * save_factor, 1)
    cost_saved     = round(total_rescued * COST_PER_WAFER * save_factor)
    silicon_saved  = round(total_rescued * SILICON_PER_WAFER * save_factor)
    co2_saved      = round(energy_saved * 0.82, 1)
    homes_powered  = round(energy_saved / 3.5, 1)
    yield_delta    = round(yield_pct - baseline, 2)

    return {
        "lot_size": lot_size,
        "yield_pct": yield_pct,
        "baseline_yield_pct": baseline,
        "yield_delta": yield_delta,
        "wafers_rescued": total_rescued,
        "water_saved_liters": water_saved,
        "energy_saved_kwh": energy_saved,
        "cost_saved_usd": cost_saved,
        "silicon_saved_grams": silicon_saved,
        "co2_saved_kg": co2_saved,
        "homes_powered_days": homes_powered,
        "cascade_prevention": cascade,
        "amd_advantage": "4x energy reduction via INT8 quantization on AMD ROCm™",
    }


def full_analysis(filepath: str, filename: str) -> dict:
    wafer_map = load_wafer_map(filepath)
    inference = run_inference(wafer_map, filename)
    sustainability = compute_sustainability(
        inference["yield_percent"], inference["severity"], inference["primary_defect"]
    )
    scan_id = str(uuid.uuid4())[:8].upper()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    result = {
        "scan_id": scan_id,
        "filename": filename,
        "timestamp": timestamp,
        "inference": inference,
        "defect_info": {
            "type": inference["primary_defect"],
            "severity": inference["severity"],
            "color": {
                "Center":"#FF4444","Donut":"#FF8800","Edge-Loc":"#FFCC00",
                "Edge-Ring":"#FF4488","Loc":"#AA44FF","Near-Full":"#FF0000",
                "Scratch":"#4488FF","Random":"#44BBFF","None":"#00DD88"
            }.get(inference["primary_defect"], "#888888"),
        },
        "sustainability": sustainability,
        "wafer_shape": list(wafer_map.shape),
    }

    save_scan(result)
    return result


# ── HTTP Handler ──────────────────────────────────────────────────────────
class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  [{self.log_date_time_string()}] {fmt % args}")

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path).path.rstrip("/")

        if p == "/health":
            model_ok = _model is not None
            db_ok = os.path.exists(os.path.join(BASE_DIR, "db", "ecoyield.db"))
            totals = get_totals()
            self.send_json({
                "status": "ok",
                "model_loaded": model_ok,
                "db_connected": db_ok,
                "model_accuracy": _model_meta.get("accuracy"),
                "model_mode": _model_meta.get("mode"),
                "total_scans": totals.get("total_scans", 0),
                "version": "2.0.0",
            })

        elif p in ("/api/model-info", "/api/model_info"):
            self.send_json({
                "loaded": _model is not None,
                "meta": _model_meta,
                "accuracy": _model_meta.get("accuracy"),
                "mode": _model_meta.get("mode", "unknown"),
                "classes": DEFECT_CLASSES,
            })

        elif p == "/api/scans":
            qs = parse_qs(urlparse(self.path).query)
            limit = int(qs.get("limit", ["20"])[0])
            self.send_json({"scans": get_recent_scans(limit)})

        elif p.startswith("/api/scans/"):
            scan_id = p.split("/api/scans/")[1]
            scan = get_scan(scan_id)
            if scan:
                self.send_json(scan)
            else:
                self.send_json({"error": "Not found"}, 404)

        elif p == "/api/stats":
            totals = get_totals()
            dist = get_defect_distribution()
            trend = get_yield_trend()
            self.send_json({
                "totals": totals,
                "distribution": dist,
                "trend": trend,
            })

        elif p == "/api/distribution":
            self.send_json({"distribution": get_defect_distribution()})

        elif p == "/api/trend":
            self.send_json({"trend": get_yield_trend()})

        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        p = urlparse(self.path).path.rstrip("/")

        if p == "/api/analyze":
            content_type = self.headers.get("Content-Type", "")
            content_length = int(self.headers.get("Content-Length", 0))

            if "multipart/form-data" in content_type:
                # File upload
                try:
                    body = self.rfile.read(content_length)
                    # Parse multipart manually
                    boundary = content_type.split("boundary=")[1].encode()
                    parts = body.split(b"--" + boundary)
                    filename = "upload.pkl"
                    file_data = None
                    for part in parts:
                        if b"filename=" in part:
                            header, _, data = part.partition(b"\r\n\r\n")
                            data = data.rstrip(b"\r\n--")
                            fn_start = header.find(b'filename="') + 10
                            fn_end = header.find(b'"', fn_start)
                            filename = header[fn_start:fn_end].decode()
                            file_data = data

                    if file_data is None:
                        self.send_json({"error": "No file in request"}, 400)
                        return

                    save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
                    with open(save_path, "wb") as f:
                        f.write(file_data)

                    result = full_analysis(save_path, filename)
                    os.remove(save_path)  # clean up
                    self.send_json(result)

                except Exception as e:
                    traceback.print_exc()
                    self.send_json({"error": str(e)}, 500)

            elif "application/json" in content_type:
                # JSON with base64 image or demo request
                try:
                    body = self.rfile.read(content_length)
                    payload = json.loads(body.decode())
                    filename = payload.get("filename", "wafer.png")
                    demo = payload.get("demo", False)

                    if demo or not payload.get("image_data"):
                        # Generate a synthetic wafer for demo
                        import random
                        from ai.train import GENERATORS
                        rng = np.random.RandomState(int(time.time()) % 1000)
                        cls_name = random.choice(DEFECT_CLASSES)
                        wm = GENERATORS[cls_name](rng)
                        filename = f"demo_{cls_name}.png"
                        # Save synthetic as image
                        from PIL import Image as PILImage
                        img = PILImage.fromarray((wm * 255).astype(np.uint8))
                        save_path = os.path.join(UPLOAD_DIR, f"demo_{uuid.uuid4().hex}.png")
                        img.save(save_path)
                    else:
                        # Decode base64 image
                        import base64
                        img_data = base64.b64decode(payload["image_data"])
                        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
                        save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.{ext}")
                        with open(save_path, "wb") as f:
                            f.write(img_data)

                    result = full_analysis(save_path, filename)
                    os.remove(save_path)
                    self.send_json(result)

                except Exception as e:
                    traceback.print_exc()
                    self.send_json({"error": str(e)}, 500)
            else:
                self.send_json({"error": "Unsupported content type"}, 415)

        elif p == "/api/retrain":
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                payload = json.loads(body.decode()) if body else {}
                data_path = payload.get("data_path")
                from ai.train import train
                model, meta = train(data_path)
                global _model, _model_meta
                _model = model
                _model_meta = meta
                upsert_model_meta(meta)
                self.send_json({"status": "ok", "accuracy": meta["accuracy"], "mode": meta["mode"]})
            except Exception as e:
                traceback.print_exc()
                self.send_json({"error": str(e)}, 500)

        else:
            self.send_json({"error": "Not found"}, 404)


# ── Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PORT = 8000
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║        EcoYield AI  —  Full Stack Server v2.0           ║")
    print("║        AMD Slingshot Hackathon Prototype                 ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Initializing database...")
    init_db()
    print("║  ✔  SQLite DB ready")
    print("║  Loading ML model...")
    load_model()
    print(f"║  ✔  Model ready  [{_model_meta.get('mode','?')} | acc={_model_meta.get('accuracy','?')}]")
    print(f"╠══════════════════════════════════════════════════════════╣")
    print(f"║  Server  →  http://localhost:{PORT}                        ║")
    print(f"║  Analyze →  POST /api/analyze  (multipart or JSON)      ║")
    print(f"║  History →  GET  /api/scans                             ║")
    print(f"║  Stats   →  GET  /api/stats                             ║")
    print(f"╚══════════════════════════════════════════════════════════╝\n")

    server = HTTPServer(("0.0.0.0", PORT), APIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()
