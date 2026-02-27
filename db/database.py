"""
db/database.py
SQLite database layer for EcoYield AI.
Stores: scan results, sustainability metrics, batch history, model metadata.
In production this maps 1:1 to a MongoDB schema.
"""
import sqlite3
import json
import time
import os
from typing import Optional, List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "ecoyield.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_conn()
    c = conn.cursor()

    # ── Scans table ────────────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS scans (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id         TEXT UNIQUE NOT NULL,
        filename        TEXT NOT NULL,
        uploaded_at     TEXT NOT NULL,
        defect_type     TEXT NOT NULL,
        confidence      REAL NOT NULL,
        severity        TEXT NOT NULL,
        yield_percent   REAL NOT NULL,
        inference_ms    REAL NOT NULL,
        heatmap_json    TEXT,
        confidences_json TEXT,
        features_json   TEXT,
        model_version   TEXT,
        hardware        TEXT DEFAULT 'CPU (AMD ROCm™ target)'
    )
    """)

    # ── Sustainability table ───────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS sustainability (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id             TEXT NOT NULL REFERENCES scans(scan_id),
        lot_size            INTEGER NOT NULL,
        yield_pct           REAL NOT NULL,
        baseline_yield_pct  REAL NOT NULL,
        yield_delta         REAL NOT NULL,
        wafers_rescued      INTEGER NOT NULL,
        water_saved_liters  REAL NOT NULL,
        energy_saved_kwh    REAL NOT NULL,
        cost_saved_usd      REAL NOT NULL,
        silicon_saved_g     REAL NOT NULL,
        co2_saved_kg        REAL NOT NULL,
        homes_powered_days  REAL NOT NULL,
        cascade_prevention  INTEGER NOT NULL
    )
    """)

    # ── Cumulative sustainability (running totals) ─────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS sustainability_totals (
        id                  INTEGER PRIMARY KEY CHECK (id = 1),
        total_scans         INTEGER DEFAULT 0,
        total_water_l       REAL DEFAULT 0,
        total_energy_kwh    REAL DEFAULT 0,
        total_cost_usd      REAL DEFAULT 0,
        total_co2_kg        REAL DEFAULT 0,
        total_wafers_rescued INTEGER DEFAULT 0,
        last_updated        TEXT
    )
    """)
    c.execute("""
    INSERT OR IGNORE INTO sustainability_totals (id) VALUES (1)
    """)

    # ── Model registry ─────────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS model_registry (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        version     TEXT NOT NULL,
        accuracy    REAL,
        mode        TEXT,
        trained_at  TEXT,
        num_samples INTEGER,
        meta_json   TEXT,
        active      INTEGER DEFAULT 1
    )
    """)

    conn.commit()
    conn.close()
    return True


def save_scan(scan_data: Dict[str, Any]) -> str:
    """Save a complete scan result to database."""
    conn = get_conn()
    try:
        c = conn.cursor()
        inf = scan_data["inference"]
        sus = scan_data["sustainability"]
        scan_id = scan_data["scan_id"]

        # Insert scan
        c.execute("""
        INSERT INTO scans
            (scan_id, filename, uploaded_at, defect_type, confidence,
             severity, yield_percent, inference_ms, heatmap_json,
             confidences_json, model_version, hardware)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scan_id,
            scan_data["filename"],
            scan_data["timestamp"],
            inf["primary_defect"],
            inf["confidence"],
            inf["severity"],
            inf["yield_percent"],
            inf["inference_ms"],
            json.dumps(inf["heatmap_8x8"]),
            json.dumps(inf["all_confidences"]),
            inf.get("model", "ecoyield-v1"),
            inf.get("hardware", "AMD ROCm™ target"),
        ))

        # Insert sustainability
        c.execute("""
        INSERT INTO sustainability
            (scan_id, lot_size, yield_pct, baseline_yield_pct, yield_delta,
             wafers_rescued, water_saved_liters, energy_saved_kwh,
             cost_saved_usd, silicon_saved_g, co2_saved_kg,
             homes_powered_days, cascade_prevention)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scan_id,
            sus["lot_size"],
            sus["yield_pct"],
            sus["baseline_yield_pct"],
            sus["yield_delta"],
            sus["wafers_rescued"],
            sus["water_saved_liters"],
            sus["energy_saved_kwh"],
            sus["cost_saved_usd"],
            sus["silicon_saved_grams"],
            sus["co2_saved_kg"],
            sus["homes_powered_days"],
            int(sus["cascade_prevention"]),
        ))

        # Update running totals
        c.execute("""
        UPDATE sustainability_totals SET
            total_scans         = total_scans + 1,
            total_water_l       = total_water_l + ?,
            total_energy_kwh    = total_energy_kwh + ?,
            total_cost_usd      = total_cost_usd + ?,
            total_co2_kg        = total_co2_kg + ?,
            total_wafers_rescued = total_wafers_rescued + ?,
            last_updated        = ?
        WHERE id = 1
        """, (
            sus["water_saved_liters"],
            sus["energy_saved_kwh"],
            sus["cost_saved_usd"],
            sus["co2_saved_kg"],
            sus["wafers_rescued"],
            scan_data["timestamp"],
        ))

        conn.commit()
        return scan_id
    finally:
        conn.close()


def get_scan(scan_id: str) -> Optional[Dict]:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT s.*, sus.* FROM scans s JOIN sustainability sus ON s.scan_id = sus.scan_id WHERE s.scan_id = ?",
            (scan_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_recent_scans(limit: int = 20) -> List[Dict]:
    conn = get_conn()
    try:
        rows = conn.execute("""
        SELECT s.scan_id, s.filename, s.uploaded_at, s.defect_type,
               s.confidence, s.severity, s.yield_percent, s.inference_ms,
               sus.water_saved_liters, sus.energy_saved_kwh,
               sus.cost_saved_usd, sus.wafers_rescued, sus.cascade_prevention
        FROM scans s
        JOIN sustainability sus ON s.scan_id = sus.scan_id
        ORDER BY s.id DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_totals() -> Dict:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM sustainability_totals WHERE id=1").fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def get_defect_distribution() -> Dict:
    conn = get_conn()
    try:
        rows = conn.execute("""
        SELECT defect_type, COUNT(*) as count,
               AVG(yield_percent) as avg_yield,
               AVG(confidence) as avg_confidence
        FROM scans
        GROUP BY defect_type
        ORDER BY count DESC
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_yield_trend(days: int = 7) -> List[Dict]:
    conn = get_conn()
    try:
        rows = conn.execute("""
        SELECT DATE(uploaded_at) as date,
               AVG(yield_percent) as avg_yield,
               COUNT(*) as scan_count,
               SUM(CASE WHEN cascade_prevention=1 THEN 1 ELSE 0 END) as cascades_prevented
        FROM scans s
        JOIN sustainability sus ON s.scan_id = sus.scan_id
        GROUP BY DATE(uploaded_at)
        ORDER BY date DESC LIMIT ?
        """, (days,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def upsert_model_meta(meta: Dict):
    conn = get_conn()
    try:
        conn.execute("UPDATE model_registry SET active=0")
        conn.execute("""
        INSERT INTO model_registry (version, accuracy, mode, trained_at, num_samples, meta_json, active)
        VALUES (?, ?, ?, ?, ?, ?, 1)
        """, (
            f"v{meta.get('trained_at','')[:10]}",
            meta.get("accuracy"),
            meta.get("mode"),
            meta.get("trained_at"),
            meta.get("num_samples"),
            json.dumps(meta),
        ))
        conn.commit()
    finally:
        conn.close()


def get_active_model_meta() -> Optional[Dict]:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM model_registry WHERE active=1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            d = dict(row)
            d["meta"] = json.loads(d.get("meta_json", "{}"))
            return d
        return None
    finally:
        conn.close()
