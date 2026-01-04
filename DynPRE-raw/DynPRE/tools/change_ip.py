from scapy.all import *
import sys


def change_ip_address(pcap_file, old_ip, new_ip):
    packets = rdpcap(pcap_file)

    for packet in packets:
        if packet.haslayer(IP):
            if packet[IP].src == old_ip:
                packet[IP].src = new_ip
            if packet[IP].dst == old_ip:
                packet[IP].dst = new_ip
            del packet[IP].chksum

    wrpcap(pcap_file + "_modified", packets)


if __name__ == "__main__":
    change_ip_address(sys.argv[1], sys.argv[2], sys.argv[3])
