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

    def _write_pcap(self, messages: List[bytes], filename: str):
        """简单的 PCAP 写入器，无需 scapy 依赖"""
        with open(filename, 'wb') as f:
            # Global Header
            f.write(struct.pack('IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1))
            # Packets
            timestamp = int(time.time())
            for msg in messages:
                # Packet Header: ts_sec, ts_usec, incl_len, orig_len
                f.write(struct.pack('IIII', timestamp, 0, len(msg), len(msg)))
                f.write(msg)

    def run(self, messages: List[bytes]) -> List[List[int]]:
        """运行真实的 DynPRE 进程并解析结果"""
        # 1. 准备目录
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)

        # 2. 写入数据
        pcap_path = os.path.join(self.input_dir, "test.pcap")
        self._write_pcap(messages, pcap_path)

        # 3. 构建命令
        # python3 ./DynPRE/DynPRE.py -i input -o output -s size
        cmd = [
            "python3", self.dynpre_path,
            "-i", self.input_dir,
            "-o", self.output_dir,
            "-s", str(len(messages))
        ]
        
        logging.info(f"Executing Real DynPRE: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=os.path.dirname(os.path.dirname(self.dynpre_path)) # 调整 CWD 以防 import 错误
            )
            
            if result.returncode != 0:
                logging.error(f"DynPRE failed:\n{result.stderr}")
                return []
                
        except Exception as e:
            logging.error(f"Failed to run DynPRE subprocess: {e}")
            return []

        # 4. 解析结果 (Segmentation.csv)
        # 格式通常是: len1, len2, len3... 
        csv_path = os.path.join(self.output_dir, "Segmentation.csv")
        if not os.path.exists(csv_path):
            logging.error("DynPRE did not produce Segmentation.csv")
            return []

        segmentations = []
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i >= len(messages): break
                    
                    # 解析字段长度
                    lengths = []
                    for x in row:
                        x = x.strip()
                        if x:
                            try:
                                lengths.append(int(float(x))) # 兼容可能的 float
                            except ValueError:
                                pass
                    
                    # 转换为边界
                    boundaries = [0]
                    current = 0
                    for l in lengths:
                        current += l
                        boundaries.append(current)
                    
                    # 修正：DynPRE 有时可能少算最后一段
                    msg_len = len(messages[i])
                    if boundaries[-1] < msg_len:
                        boundaries.append(msg_len)
                    elif boundaries[-1] > msg_len:
                        # 如果超出，截断
                        boundaries = [b for b in boundaries if b <= msg_len]
                        if boundaries[-1] != msg_len:
                             boundaries.append(msg_len)
                             
                    segmentations.append(sorted(list(set(boundaries))))
                    
            return segmentations
            
        except Exception as e:
            logging.error(f"Error parsing DynPRE CSV: {e}")
            return []