from scapy.all import sniff
from packet_parser import process_packet

def start_sniffing():
    print("🛡 SentryAI Flow Sniffer Started")
    print("Capturing packets and generating flow features...")
    print("Press CTRL+C to stop\n")

    sniff(
        iface="lo",     # captures localhost traffic
        filter="ip",
        prn=process_packet,
        store=False
    )

if __name__ == "__main__":
    start_sniffing()