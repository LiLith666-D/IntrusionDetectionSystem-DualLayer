import csv
import os
import time
from collections import defaultdict
from scapy.all import IP, TCP, UDP

OUTPUT_FILE = "data/processed/PacketFlow.csv"

os.makedirs("data/processed", exist_ok=True)

flows = defaultdict(lambda: {
    "start": None,
    "last": None,
    "packets": 0,
    "bytes": 0,
    "lengths": []
})

csv_file = open(OUTPUT_FILE, "w", newline="")
writer = csv.writer(csv_file)

writer.writerow([
    "Flow Duration",
    "Total Packets",
    "Total Bytes",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Packet Length Mean",
    "Packet Length Max",
    "Packet Length Min",
    "Source Port",
    "Destination Port",
    "Protocol"
])

csv_file.flush()

print(f"Saving flow features to {OUTPUT_FILE}")

def process_packet(packet):

    if IP not in packet:
        return

    src = packet[IP].src
    dst = packet[IP].dst

    protocol = None
    sport = 0
    dport = 0

    if TCP in packet:
        protocol = "TCP"
        sport = packet[TCP].sport
        dport = packet[TCP].dport

    elif UDP in packet:
        protocol = "UDP"
        sport = packet[UDP].sport
        dport = packet[UDP].dport

    else:
        return

    flow_id = (src, dst, sport, dport, protocol)

    now = time.time()
    length = len(packet)

    flow = flows[flow_id]

    if flow["start"] is None:
        flow["start"] = now

    flow["last"] = now
    flow["packets"] += 1
    flow["bytes"] += length
    flow["lengths"].append(length)

    duration = flow["last"] - flow["start"]

    if duration == 0:
        return

    packets = flow["packets"]
    bytes_total = flow["bytes"]

    lengths = flow["lengths"]

    pkt_mean = sum(lengths) / len(lengths)
    pkt_max = max(lengths)
    pkt_min = min(lengths)

    bytes_per_sec = bytes_total / duration
    packets_per_sec = packets / duration

    writer.writerow([
        duration,
        packets,
        bytes_total,
        bytes_per_sec,
        packets_per_sec,
        pkt_mean,
        pkt_max,
        pkt_min,
        sport,
        dport,
        protocol
    ])

    csv_file.flush()

    print(f"{src}:{sport} → {dst}:{dport} | {protocol} | packets={packets}")