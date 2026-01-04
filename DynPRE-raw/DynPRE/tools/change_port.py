from scapy.all import *
import sys


def change_port(pcap_file, old_port, new_port):
    packets = rdpcap(pcap_file)

    for packet in packets:
        if packet.haslayer(TCP):
            if packet[TCP].sport == old_port:
                packet[TCP].sport = new_port
            if packet[TCP].dport == old_port:
                packet[TCP].dport = new_port
            del packet.chksum
            del packet.len
            del packet[TCP].chksum
        if packet.haslayer(UDP):
            if packet[UDP].sport == old_port:
                packet[UDP].sport = new_port
            if packet[UDP].dport == old_port:
                packet[UDP].dport = new_port
            del packet[UDP].chksum
            del packet[UDP].len

    wrpcap(pcap_file + "_modified", packets)


if __name__ == "__main__":
    change_port(sys.argv[1], sys.argv[2], sys.argv[3])  # pcap_file, old_port, new_port
