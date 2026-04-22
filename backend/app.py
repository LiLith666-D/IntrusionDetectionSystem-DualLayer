import os
import time
import random
import threading
import pandas as pd
import numpy as np
import joblib
from flask import Flask, jsonify
from flask_cors import CORS

# ── Paths (env-overridable) ───────────────────────────────────────────────────
LIVE_CSV   = os.environ.get("FLOW_CSV",    "/app/data/live_flows.csv")
MODEL_PATH = os.environ.get("MODEL_PATH",  "/app/models")
RELOAD_SEC = int(os.environ.get("RELOAD_SEC", "5"))

# ── Load models (once at startup) ────────────────────────────────────────────
print("[*] Loading models from:", MODEL_PATH)
binary_model  = joblib.load(os.path.join(MODEL_PATH, "binary_ids.pkl"))
multi_model   = joblib.load(os.path.join(MODEL_PATH, "random_forest_ids.pkl"))
scaler        = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))
print("[✓] Models loaded.")

# ── Live dataframe — reloaded by background thread ───────────────────────────
_df_lock   = threading.Lock()
_live_df   = pd.DataFrame()
_csv_mtime = 0.0


def _reload_loop():
    """Background thread: reload CSV whenever it changes on disk."""
    global _live_df, _csv_mtime
    while True:
        time.sleep(RELOAD_SEC)
        if not os.path.exists(LIVE_CSV):
            continue
        try:
            mtime = os.path.getmtime(LIVE_CSV)
            if mtime <= _csv_mtime:
                continue
            df = pd.read_csv(LIVE_CSV)
            df.columns = df.columns.str.strip()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            if df.empty:
                continue
            with _df_lock:
                _live_df   = df
                _csv_mtime = mtime
            print(f"[+] CSV reloaded — {len(df)} rows")
        except Exception as e:
            print(f"[!] CSV reload error: {e}")


threading.Thread(target=_reload_loop, daemon=True).start()

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    with _df_lock:
        rows = len(_live_df)
    return jsonify({"status": "ok", "live_rows": rows})


@app.route("/predict", methods=["GET"])
def predict():
    with _df_lock:
        df = _live_df.copy()

    if df.empty:
        return jsonify({
            "level1":     "WAITING",
            "level2":     None,
            "confidence": 0,
            "flow":       {},
            "message":    "Capture starting — no flows yet"
        }), 202

    # Pick randomly from the last 10 rows (freshest captured flows)
    sample = df.tail(10).sample(1)

    try:
        sample_scaled = scaler.transform(sample)
    except Exception as e:
        return jsonify({"error": f"Scaler failed: {e}"}), 500

    # ── Level 1: Binary ───────────────────────────────────────────────────────
    binary_pred = binary_model.predict(sample_scaled)[0]

    row = sample.iloc[0]

    base_flow = {
        "flow_bytes_per_s":  float(row.get("Flow Bytes/s",         0)),
        "flow_pkts_per_s":   float(row.get("Flow Packets/s",        0)),
        "flow_duration":     float(row.get("Flow Duration",          0)),
        "dst_port":          int(row.get("Destination Port",         0)),
        "tot_fwd_pkts":      int(row.get("Total Fwd Packets",        0)),
        "tot_bwd_pkts":      int(row.get("Total Backward Packets",   0)),
        "pkt_len_mean":      float(row.get("Packet Length Mean",     0)),
        "pkt_len_var":       float(row.get("Packet Length Variance", 0)),
        "syn_flag_cnt":      int(row.get("SYN Flag Count",           0)),
        "rst_flag_cnt":      int(row.get("RST Flag Count",           0)),
        "ack_flag_cnt":      int(row.get("ACK Flag Count",           0)),
        "flow_iat_mean":     float(row.get("Flow IAT Mean",          0)),
        "active_mean":       float(row.get("Active Mean",            0)),
        "idle_mean":         float(row.get("Idle Mean",              0)),
    }

    if binary_pred == 0:
        return jsonify({
            "level1":     "BENIGN",
            "level2":     None,
            "confidence": 100,
            "flow":       base_flow
        })

    # ── Level 2: Multiclass ───────────────────────────────────────────────────
    multi_pred   = multi_model.predict(sample_scaled)[0]
    multi_proba  = multi_model.predict_proba(sample_scaled).max()
    attack_label = label_encoder.inverse_transform([multi_pred])[0]

    return jsonify({
        "level1":     "ATTACK",
        "level2":     attack_label,
        "confidence": round(float(multi_proba * 100), 2),
        "flow":       base_flow
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
