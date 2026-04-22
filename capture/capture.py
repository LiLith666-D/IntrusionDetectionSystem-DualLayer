"""
SentryAI — Live Flow Capture
Captures packets, computes CICIDS2017-compatible flow features,
appends them to live_flows.csv which the backend reads for prediction.
"""

import os
import csv
import time
import math
import threading
from collections import defaultdict
from scapy.all import sniff, IP, TCP, UDP

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_CSV   = os.environ.get("FLOW_CSV", "/app/data/live_flows.csv")
IFACE        = os.environ.get("CAPTURE_IFACE", "eth0")
FLOW_TIMEOUT = float(os.environ.get("FLOW_TIMEOUT", "10"))   # seconds of idle = flush
FLUSH_EVERY  = int(os.environ.get("FLUSH_EVERY",   "5"))     # flush every N seconds
MAX_ROWS     = int(os.environ.get("MAX_ROWS",       "5000"))  # rolling window size

# ── These are the exact columns the model was trained on ─────────────────────
COLUMNS = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_div(a, b, default=0.0):
    return a / b if b else default

def stats(lst):
    """Return mean, std, max, min of a list (or zeros if empty)."""
    if not lst:
        return 0.0, 0.0, 0.0, 0.0
    n   = len(lst)
    mu  = sum(lst) / n
    if n > 1:
        var = sum((x - mu) ** 2 for x in lst) / (n - 1)
        sd  = math.sqrt(var)
    else:
        sd  = 0.0
    return mu, sd, max(lst), min(lst)


# ── Flow store ────────────────────────────────────────────────────────────────
class Flow:
    __slots__ = [
        "start", "last", "dst_port", "protocol",
        "fwd_pkts", "bwd_pkts",
        "fwd_lens", "bwd_lens",
        "fwd_times", "bwd_times",
        "all_lens", "all_times",
        "tcp_flags",
        "fwd_hdr", "bwd_hdr",
        "init_win_fwd", "init_win_bwd",
        "active_start", "active_times", "idle_times",
        "last_active",
    ]

    def __init__(self, now, dst_port, protocol):
        self.start       = now
        self.last        = now
        self.dst_port    = dst_port
        self.protocol    = protocol
        self.fwd_pkts    = []   # packet lengths forward
        self.bwd_pkts    = []   # packet lengths backward
        self.fwd_times   = []   # inter-arrival times forward
        self.bwd_times   = []   # inter-arrival times backward
        self.all_lens    = []
        self.all_times   = []
        self.tcp_flags   = {"FIN":0,"SYN":0,"RST":0,"PSH":0,"ACK":0,"URG":0,"CWE":0,"ECE":0}
        self.fwd_hdr     = 0
        self.bwd_hdr     = 0
        self.init_win_fwd  = -1
        self.init_win_bwd  = -1
        self.active_start  = now
        self.active_times  = []
        self.idle_times    = []
        self.last_active   = now


flows = {}
flows_lock = threading.Lock()


def add_packet(flow_key, is_fwd, length, now, hdr_len, flags, win):
    with flows_lock:
        if flow_key not in flows:
            flows[flow_key] = Flow(now, flow_key[3], flow_key[4])
        f = flows[flow_key]

        # IAT
        if f.all_times:
            iat = now - f.last
            f.all_times.append(iat)
            if is_fwd:
                f.fwd_times.append(iat)
            else:
                f.bwd_times.append(iat)
        else:
            f.all_times.append(0)

        # Active / Idle tracking (2-second gap = idle)
        gap = now - f.last_active
        if gap > 2.0:
            f.active_times.append(f.last_active - f.active_start)
            f.idle_times.append(gap)
            f.active_start = now
        f.last_active = now

        f.last = now
        f.all_lens.append(length)

        if is_fwd:
            f.fwd_pkts.append(length)
            f.fwd_hdr += hdr_len
            if f.init_win_fwd == -1 and win is not None:
                f.init_win_fwd = win
        else:
            f.bwd_pkts.append(length)
            f.bwd_hdr += hdr_len
            if f.init_win_bwd == -1 and win is not None:
                f.init_win_bwd = win

        if flags:
            for fl in ["FIN","SYN","RST","PSH","ACK","URG"]:
                if fl in flags:
                    f.tcp_flags[fl] += 1


def flow_to_row(f, dst_port):
    duration = max(f.last - f.start, 1e-9)

    tot_fwd   = len(f.fwd_pkts)
    tot_bwd   = len(f.bwd_pkts)
    tot_fwd_b = sum(f.fwd_pkts)
    tot_bwd_b = sum(f.bwd_pkts)

    fwd_mu, fwd_sd, fwd_max, fwd_min = stats(f.fwd_pkts)
    bwd_mu, bwd_sd, bwd_max, bwd_min = stats(f.bwd_pkts)
    all_mu, all_sd, all_max, all_min = stats(f.all_lens)

    iat_mu, iat_sd, iat_max, iat_min = stats(f.all_times)
    fiat_mu, fiat_sd, fiat_max, fiat_min = stats(f.fwd_times)
    biat_mu, biat_sd, biat_max, biat_min = stats(f.bwd_times)

    act_mu, act_sd, act_max, act_min = stats(f.active_times)
    idl_mu, idl_sd, idl_max, idl_min = stats(f.idle_times)

    all_pkts   = tot_fwd + tot_bwd
    bytes_total = tot_fwd_b + tot_bwd_b
    pkt_var     = all_sd ** 2

    bps = safe_div(bytes_total, duration)
    pps = safe_div(all_pkts,    duration)

    down_up   = safe_div(tot_bwd, tot_fwd)
    avg_pkt   = safe_div(bytes_total, all_pkts)

    row = {
        "Destination Port":             dst_port,
        "Flow Duration":                int(duration * 1_000_000),  # microseconds
        "Total Fwd Packets":            tot_fwd,
        "Total Backward Packets":       tot_bwd,
        "Total Length of Fwd Packets":  tot_fwd_b,
        "Total Length of Bwd Packets":  tot_bwd_b,
        "Fwd Packet Length Max":        fwd_max,
        "Fwd Packet Length Min":        fwd_min,
        "Fwd Packet Length Mean":       round(fwd_mu, 4),
        "Fwd Packet Length Std":        round(fwd_sd, 4),
        "Bwd Packet Length Max":        bwd_max,
        "Bwd Packet Length Min":        bwd_min,
        "Bwd Packet Length Mean":       round(bwd_mu, 4),
        "Bwd Packet Length Std":        round(bwd_sd, 4),
        "Flow Bytes/s":                 round(bps, 4),
        "Flow Packets/s":               round(pps, 4),
        "Flow IAT Mean":                round(iat_mu * 1e6, 4),
        "Flow IAT Std":                 round(iat_sd * 1e6, 4),
        "Flow IAT Max":                 round(iat_max * 1e6, 4),
        "Flow IAT Min":                 round(iat_min * 1e6, 4),
        "Fwd IAT Total":                round(sum(f.fwd_times) * 1e6, 4),
        "Fwd IAT Mean":                 round(fiat_mu * 1e6, 4),
        "Fwd IAT Std":                  round(fiat_sd * 1e6, 4),
        "Fwd IAT Max":                  round(fiat_max * 1e6, 4),
        "Fwd IAT Min":                  round(fiat_min * 1e6, 4),
        "Bwd IAT Total":                round(sum(f.bwd_times) * 1e6, 4),
        "Bwd IAT Mean":                 round(biat_mu * 1e6, 4),
        "Bwd IAT Std":                  round(biat_sd * 1e6, 4),
        "Bwd IAT Max":                  round(biat_max * 1e6, 4),
        "Bwd IAT Min":                  round(biat_min * 1e6, 4),
        "Fwd PSH Flags":                f.tcp_flags["PSH"],
        "Bwd PSH Flags":                0,
        "Fwd URG Flags":                f.tcp_flags["URG"],
        "Bwd URG Flags":                0,
        "Fwd Header Length":            f.fwd_hdr,
        "Bwd Header Length":            f.bwd_hdr,
        "Fwd Packets/s":                round(safe_div(tot_fwd, duration), 4),
        "Bwd Packets/s":                round(safe_div(tot_bwd, duration), 4),
        "Min Packet Length":            all_min,
        "Max Packet Length":            all_max,
        "Packet Length Mean":           round(all_mu, 4),
        "Packet Length Std":            round(all_sd, 4),
        "Packet Length Variance":       round(pkt_var, 4),
        "FIN Flag Count":               f.tcp_flags["FIN"],
        "SYN Flag Count":               f.tcp_flags["SYN"],
        "RST Flag Count":               f.tcp_flags["RST"],
        "PSH Flag Count":               f.tcp_flags["PSH"],
        "ACK Flag Count":               f.tcp_flags["ACK"],
        "URG Flag Count":               f.tcp_flags["URG"],
        "CWE Flag Count":               f.tcp_flags["CWE"],
        "ECE Flag Count":               f.tcp_flags["ECE"],
        "Down/Up Ratio":                round(down_up, 4),
        "Average Packet Size":          round(avg_pkt, 4),
        "Avg Fwd Segment Size":         round(fwd_mu, 4),
        "Avg Bwd Segment Size":         round(bwd_mu, 4),
        "Fwd Header Length.1":          f.fwd_hdr,
        "Fwd Avg Bytes/Bulk":           0,
        "Fwd Avg Packets/Bulk":         0,
        "Fwd Avg Bulk Rate":            0,
        "Bwd Avg Bytes/Bulk":           0,
        "Bwd Avg Packets/Bulk":         0,
        "Bwd Avg Bulk Rate":            0,
        "Subflow Fwd Packets":          tot_fwd,
        "Subflow Fwd Bytes":            tot_fwd_b,
        "Subflow Bwd Packets":          tot_bwd,
        "Subflow Bwd Bytes":            tot_bwd_b,
        "Init_Win_bytes_forward":       f.init_win_fwd if f.init_win_fwd != -1 else 0,
        "Init_Win_bytes_backward":      f.init_win_bwd if f.init_win_bwd != -1 else 0,
        "act_data_pkt_fwd":             tot_fwd,
        "min_seg_size_forward":         int(fwd_min),
        "Active Mean":                  round(act_mu * 1e6, 4),
        "Active Std":                   round(act_sd * 1e6, 4),
        "Active Max":                   round(act_max * 1e6, 4),
        "Active Min":                   round(act_min * 1e6, 4),
        "Idle Mean":                    round(idl_mu * 1e6, 4),
        "Idle Std":                     round(idl_sd * 1e6, 4),
        "Idle Max":                     round(idl_max * 1e6, 4),
        "Idle Min":                     round(idl_min * 1e6, 4),
    }
    return [row[c] for c in COLUMNS]


# ── Packet handler ────────────────────────────────────────────────────────────
def handle_packet(pkt):
    if IP not in pkt:
        return

    src = pkt[IP].src
    dst = pkt[IP].dst
    now = time.time()

    if TCP in pkt:
        sport  = pkt[TCP].sport
        dport  = pkt[TCP].dport
        proto  = "TCP"
        hdr    = pkt[TCP].dataofs * 4 if pkt[TCP].dataofs else 20
        win    = pkt[TCP].window
        fl     = pkt[TCP].flags
        flags  = []
        if fl & 0x01: flags.append("FIN")
        if fl & 0x02: flags.append("SYN")
        if fl & 0x04: flags.append("RST")
        if fl & 0x08: flags.append("PSH")
        if fl & 0x10: flags.append("ACK")
        if fl & 0x20: flags.append("URG")
    elif UDP in pkt:
        sport  = pkt[UDP].sport
        dport  = pkt[UDP].dport
        proto  = "UDP"
        hdr    = 8
        win    = None
        flags  = []
    else:
        return

    length  = len(pkt)
    fwd_key = (src, dst, sport, dport, proto)
    bwd_key = (dst, src, dport, sport, proto)

    with flows_lock:
        is_fwd = fwd_key in flows or bwd_key not in flows

    key = fwd_key if is_fwd else bwd_key
    add_packet(key, is_fwd, length, now, hdr, flags, win)


# ── Periodic flusher ──────────────────────────────────────────────────────────
def flush_flows():
    """
    Every FLUSH_EVERY seconds:
      - Collect flows that have been idle > FLOW_TIMEOUT
      - Write their feature rows to the CSV
      - Keep CSV rolling at MAX_ROWS
    """
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Write header if file doesn't exist
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w", newline="") as f:
            csv.writer(f).writerow(COLUMNS)
        print(f"[✓] Created {OUTPUT_CSV}")

    while True:
        time.sleep(FLUSH_EVERY)
        now     = time.time()
        to_flush = []

        with flows_lock:
            expired = [k for k, v in flows.items() if now - v.last > FLOW_TIMEOUT]
            for k in expired:
                to_flush.append((flows.pop(k), k[3]))   # (flow, dst_port)

        if not to_flush:
            continue

        rows = []
        for flow, dst_port in to_flush:
            if len(flow.fwd_pkts) + len(flow.bwd_pkts) < 2:
                continue   # skip single-packet non-flows
            rows.append(flow_to_row(flow, dst_port))

        if not rows:
            continue

        # Append to CSV
        with open(OUTPUT_CSV, "a", newline="") as f:
            csv.writer(f).writerows(rows)

        print(f"[+] Flushed {len(rows)} flows → {OUTPUT_CSV}")

        # Rolling window — keep only last MAX_ROWS rows
        try:
            with open(OUTPUT_CSV, "r") as f:
                lines = f.readlines()
            if len(lines) > MAX_ROWS + 1:          # +1 for header
                with open(OUTPUT_CSV, "w") as f:
                    f.writelines([lines[0]] + lines[-(MAX_ROWS):])
        except Exception as e:
            print(f"[!] Rolling trim failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[*] SentryAI Capture — interface: {IFACE}")
    print(f"[*] Flow CSV: {OUTPUT_CSV}")
    print(f"[*] Flow timeout: {FLOW_TIMEOUT}s | Flush every: {FLUSH_EVERY}s")

    # Start flusher thread
    t = threading.Thread(target=flush_flows, daemon=True)
    t.start()

    # Start sniffing (blocking)
    sniff(iface=IFACE, filter="ip", prn=handle_packet, store=False)
