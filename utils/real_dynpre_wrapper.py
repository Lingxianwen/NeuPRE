import os
import csv
import time
import shutil
import struct
import subprocess
import logging
from typing import List

class RealDynPRERunner:
    def __init__(self, dynpre_path: str = "./DynPRE/DynPRE.py", work_dir: str = "./temp_dynpre"):
        self.dynpre_path = os.path.abspath(dynpre_path)
        self.work_dir = os.path.abspath(work_dir)
        self.input_dir = os.path.join(self.work_dir, "in")
        self.output_dir = os.path.join(self.work_dir, "out")

    def _write_pcap(self, messages: List[bytes], filename: str, port: int, is_tcp: bool):
        """
        生成带有完整 TCP/UDP + IP + Ethernet 头的 PCAP 文件。
        DynPRE 将会读取这个文件，并尝试连接这里指定的 Destination Port。
        """
        with open(filename, 'wb') as f:
            # PCAP Global Header (Little Endian)
            # Magic(4), Major(2), Minor(2), Zone(4), SigFigs(4), SnapLen(4), Network(4)
            # Network = 1 (Ethernet)
            f.write(struct.pack('<IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1))
            
            timestamp = int(time.time())
            
            for i, msg in enumerate(messages):
                # 1. Ethernet Header (14 bytes)
                # Dst(6) + Src(6) + Type(2). Type 0x0800 = IPv4
                eth_header = b'\x00\x00\x00\x00\x00\x00' + \
                             b'\x00\x00\x00\x00\x00\x00' + \
                             b'\x08\x00'
                
                # 2. IP Header (20 bytes)
                # Proto: 6 for TCP, 17 for UDP
                ip_proto = 6 if is_tcp else 17
                # 构造 IP 头
                ip_total_len = 20 + (20 if is_tcp else 8) + len(msg)
                # !BBHHHBBH4s4s: Ver/IHL, TOS, Len, ID, Frag, TTL, Proto, Checksum, SrcIP, DstIP
                # Src/Dst = 127.0.0.1
                ip_header = struct.pack('!BBHHHBBH4s4s', 
                                        0x45, 0, ip_total_len, i % 65535, 0, 64, ip_proto, 0, 
                                        b'\x7f\x00\x00\x01', b'\x7f\x00\x00\x01')
                
                l4_header = b""
                if is_tcp:
                    # 3. TCP Header (20 bytes)
                    # SrcPort(2), DstPort(2), Seq(4), Ack(4), Offset/Flags(2), Win(2), Check(2), Urg(2)
                    # Data Offset = 5 (20 bytes), Flags = 0x18 (PSH, ACK) -> 0x5018
                    # 注意：这里我们一定要把 DstPort 设为用户服务器的端口 (如 1502)
                    l4_header = struct.pack('!HHIIHHHH', 
                                            12345, port, i+1, 0, 0x5018, 65535, 0, 0)
                else:
                    # 3. UDP Header (8 bytes)
                    udp_len = 8 + len(msg)
                    l4_header = struct.pack('!HHHH', 12345, port, udp_len, 0)
                
                packet_data = eth_header + ip_header + l4_header + msg
                
                # PCAP Packet Header
                f.write(struct.pack('<IIII', timestamp, 0, len(packet_data), len(packet_data)))
                f.write(packet_data)

    def run(self, messages: List[bytes], port: int = 1502, is_tcp: bool = True) -> List[List[int]]:
        """
        运行 DynPRE。
        :param port: 目标服务器端口 (DynPRE 会尝试连接这个端口)
        :param is_tcp: 协议类型
        """
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)

        # 生成包含正确端口信息的流量包
        pcap_path = os.path.join(self.input_dir, "test.pcap")
        self._write_pcap(messages, pcap_path, port, is_tcp)

        # 运行 DynPRE
        cmd = [
            "python3", self.dynpre_path,
            "-i", self.input_dir,
            "-o", self.output_dir,
            "-s", str(len(messages))
        ]
        
        logging.info(f"Executing Real DynPRE (Target: 127.0.0.1:{port}/{'TCP' if is_tcp else 'UDP'})...")
        
        try:
            # 设置 cwd 以防止 import 错误
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=os.path.dirname(os.path.dirname(self.dynpre_path))
            )
            
            if result.returncode != 0:
                logging.error(f"DynPRE process failed. Stderr:\n{result.stderr}")
                return []
                
        except Exception as e:
            logging.error(f"Failed to execute DynPRE subprocess: {e}")
            return []

        # 解析结果
        csv_path = os.path.join(self.output_dir, "Segmentation.csv")
        if not os.path.exists(csv_path):
            logging.warning("DynPRE produced no Segmentation.csv (Probable cause: Connection refused by server)")
            return []

        segmentations = []
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i >= len(messages): break
                    lengths = []
                    for x in row:
                        if x.strip():
                            try: lengths.append(int(float(x)))
                            except: pass
                    
                    boundaries = [0]
                    current = 0
                    for l in lengths:
                        current += l
                        boundaries.append(current)
                    
                    msg_len = len(messages[i])
                    boundaries = sorted(list(set([b for b in boundaries if b <= msg_len])))
                    if not boundaries or boundaries[-1] != msg_len:
                        boundaries.append(msg_len)
                        
                    segmentations.append(boundaries)
            return segmentations
        except Exception as e:
            logging.error(f"Error parsing CSV: {e}")
            return []