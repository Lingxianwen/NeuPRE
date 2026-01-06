"""
PCAP Data Loader with Ground Truth Extraction

This module loads pcap files and extracts ground truth field boundaries
based on protocol specifications (DHCP, DNS, Modbus, SMB2).
"""

import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path

try:
    from scapy.all import rdpcap, Raw, UDP, TCP, IP, DHCP, DNS, DNSQR, DNSRR
    from scapy.packet import Packet
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logging.warning("Scapy not available. Install with: pip install scapy")


class PCAPDataLoader:
    """
    Loads PCAP files and extracts protocol messages with ground truth boundaries.
    """

    def __init__(self, data_dir: str = './data'):
        """
        Args:
            data_dir: Directory containing pcap files
        """
        self.data_dir = Path(data_dir)
        if not SCAPY_AVAILABLE:
            raise ImportError("Scapy is required. Install with: pip install scapy")

        logging.info(f"PCAPDataLoader initialized. Data directory: {self.data_dir}")

    def load_protocol_data(self, protocol: str, max_messages: int = 100) -> Tuple[List[bytes], List[List[int]]]:
        """
        Load protocol data from pcap files.

        Args:
            protocol: Protocol name ('dhcp', 'dns', 'modbus', 'smb2')
            max_messages: Maximum number of messages to load

        Returns:
            Tuple of (messages, ground_truth)
            - messages: List of raw protocol messages (bytes)
            - ground_truth: List of field boundary lists
        """
        protocol = protocol.lower()

        if protocol == 'dhcp':
            return self.load_dhcp_data(max_messages)
        elif protocol == 'dns':
            return self.load_dns_data(max_messages)
        elif protocol == 'modbus':
            return self.load_modbus_data(max_messages)
        elif protocol == 'smb2':
            return self.load_smb2_data(max_messages)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

    def load_dhcp_data(self, max_messages: int = 100) -> Tuple[List[bytes], List[List[int]]]:
        """
        Load DHCP protocol data.

        DHCP message format (simplified):
        [Op(1)] [HType(1)] [HLen(1)] [Hops(1)] [XID(4)] [Secs(2)] [Flags(2)]
        [CIAddr(4)] [YIAddr(4)] [SIAddr(4)] [GIAddr(4)] [CHAddr(16)] [SName(64)]
        [File(128)] [Options(variable)]
        """
        pcap_file = self.data_dir / 'in-dhcp-pcaps' / 'BinInf_dhcp_1000.pcap'

        if not pcap_file.exists():
            raise FileNotFoundError(f"DHCP pcap file not found: {pcap_file}")

        logging.info(f"Loading DHCP data from {pcap_file}")
        packets = rdpcap(str(pcap_file))

        messages = []
        ground_truth = []

        for pkt in packets:
            if max_messages and len(messages) >= max_messages:
                break

            # Extract DHCP payload - handle both UDP encapsulated and raw ethernet frames
            if pkt.haslayer(Raw):
                payload = bytes(pkt[Raw].load)

                # DHCP minimum size is 236 bytes (without options)
                # Check if it looks like a DHCP message (starts with 0x01 or 0x02 for OP)
                if len(payload) >= 236 and payload[0] in [0x01, 0x02]:
                    messages.append(payload)

                    # Ground truth boundaries for DHCP
                    # Standard DHCP header fields
                    boundaries = [
                        0,    # Start
                        1,    # Op
                        2,    # HType
                        3,    # HLen
                        4,    # Hops
                        8,    # XID
                        10,   # Secs
                        12,   # Flags
                        16,   # CIAddr
                        20,   # YIAddr
                        24,   # SIAddr
                        28,   # GIAddr
                        44,   # CHAddr (16 bytes)
                        108,  # SName (64 bytes)
                        236,  # File (128 bytes)
                        len(payload)  # End (Options)
                    ]
                    ground_truth.append(boundaries)

        logging.info(f"Loaded {len(messages)} DHCP messages")
        return messages, ground_truth

    def load_dns_data(self, max_messages: int = 100) -> Tuple[List[bytes], List[List[int]]]:
        """
        Load DNS protocol data.

        DNS message format:
        [Header(12)] [Questions(variable)] [Answers(variable)] [Authority(variable)] [Additional(variable)]

        Header format:
        [ID(2)] [Flags(2)] [QDCount(2)] [ANCount(2)] [NSCount(2)] [ARCount(2)]
        """
        pcap_file = self.data_dir / 'in-dns-pcaps' / 'SMIA_DNS_1000.pcap'

        if not pcap_file.exists():
            raise FileNotFoundError(f"DNS pcap file not found: {pcap_file}")

        logging.info(f"Loading DNS data from {pcap_file}")
        packets = rdpcap(str(pcap_file))

        messages = []
        ground_truth = []

        for pkt in packets:
            if max_messages and len(messages) >= max_messages:
                break

            # Extract DNS payload
            if pkt.haslayer(DNS):
                dns_layer = pkt[DNS]
                # Get raw DNS bytes
                payload = bytes(dns_layer)

                if len(payload) >= 12:  # DNS header is 12 bytes
                    messages.append(payload)

                    # Ground truth boundaries for DNS
                    boundaries = [
                        0,   # Start
                        2,   # ID
                        4,   # Flags
                        6,   # QDCount
                        8,   # ANCount
                        10,  # NSCount
                        12,  # ARCount / Start of questions
                        len(payload)  # End
                    ]

                    # Note: Parsing variable-length questions/answers is complex
                    # This is a simplified version with just the header
                    ground_truth.append(boundaries)

        logging.info(f"Loaded {len(messages)} DNS messages")
        return messages, ground_truth

    def load_modbus_data(self, max_messages: int = 100) -> Tuple[List[bytes], List[List[int]]]:
        """
        Load Modbus TCP protocol data.

        Modbus TCP message format:
        [Transaction ID(2)] [Protocol ID(2)] [Length(2)] [Unit ID(1)] [Function Code(1)] [Data(variable)]
        """
        pcap_file = self.data_dir / 'in-modbus-pcaps' / 'libmodbus-bandwidth_server-rand_client.pcap'

        if not pcap_file.exists():
            raise FileNotFoundError(f"Modbus pcap file not found: {pcap_file}")

        logging.info(f"Loading Modbus data from {pcap_file}")
        packets = rdpcap(str(pcap_file))

        messages = []
        ground_truth = []

        for pkt in packets:
            if max_messages and len(messages) >= max_messages:
                break

            # Modbus typically uses TCP port 502, but can use other ports (like 1502)
            if pkt.haslayer(TCP):
                if pkt.haslayer(Raw):
                    payload = bytes(pkt[Raw].load)

                    # Modbus TCP minimum size is 8 bytes (MBAP header + function code)
                    # Check if it looks like Modbus TCP (Protocol ID should be 0x0000 at bytes 2-3)
                    if len(payload) >= 8:
                        # Verify Modbus protocol ID (bytes 2-3 should be 0x0000)
                        if payload[2:4] == b'\x00\x00':
                            messages.append(payload)

                            # Ground truth boundaries for Modbus TCP
                            boundaries = [
                                0,   # Start
                                2,   # Transaction ID
                                4,   # Protocol ID
                                6,   # Length
                                7,   # Unit ID
                                8,   # Function Code
                                len(payload)  # End (Data)
                            ]
                            ground_truth.append(boundaries)

        logging.info(f"Loaded {len(messages)} Modbus messages")
        return messages, ground_truth

    def load_smb2_data(self, max_messages: int = 100) -> Tuple[List[bytes], List[List[int]]]:
        """
        Load SMB2 protocol data.

        SMB2 message format (simplified):
        [ProtocolID(4)] [StructureSize(2)] [CreditCharge(2)] [Status(4)] [Command(2)]
        [Credits(2)] [Flags(4)] [NextCommand(4)] [MessageID(8)] [Reserved(4)]
        [TreeID(4)] [SessionID(8)] [Signature(16)] [Data(variable)]
        """
        pcap_file = self.data_dir / 'in-smb2-pcaps' / 'samba.pcap'

        if not pcap_file.exists():
            raise FileNotFoundError(f"SMB2 pcap file not found: {pcap_file}")

        logging.info(f"Loading SMB2 data from {pcap_file}")
        packets = rdpcap(str(pcap_file))

        messages = []
        ground_truth = []

        for pkt in packets:
            if max_messages and len(messages) >= max_messages:
                break

            # SMB2 typically uses TCP port 445 or 139 (netbios_ssn)
            if pkt.haslayer(TCP) and (pkt[TCP].sport == 445 or pkt[TCP].dport == 445 or
                                      pkt[TCP].sport == 139 or pkt[TCP].dport == 139):
                # Check if Scapy already parsed SMB2
                if 'SMB2_Header' in str(pkt.layers()):
                    # Get the raw bytes of SMB2 (skip NetBIOS Session Service header if present)
                    raw_bytes = bytes(pkt)

                    # Try to find SMB2 magic bytes
                    smb2_magic = b'\xfeSMB'
                    if smb2_magic in raw_bytes:
                        smb2_start = raw_bytes.index(smb2_magic)
                        # Extract from SMB2 header onwards
                        payload = raw_bytes[smb2_start:]

                        if len(payload) >= 64:
                            messages.append(payload)

                            # Ground truth boundaries for SMB2 header
                            boundaries = [
                                0,   # Start
                                4,   # ProtocolID
                                6,   # StructureSize
                                8,   # CreditCharge
                                12,  # Status
                                14,  # Command
                                16,  # Credits
                                20,  # Flags
                                24,  # NextCommand
                                32,  # MessageID
                                36,  # Reserved
                                40,  # TreeID
                                48,  # SessionID
                                64,  # Signature / Start of data
                                len(payload)  # End
                            ]
                            ground_truth.append(boundaries)

                elif pkt.haslayer(Raw):
                    payload = bytes(pkt[Raw].load)

                    # Check for SMB2 magic bytes (0xFE 'S' 'M' 'B')
                    # NetBIOS Session Service header is 4 bytes, SMB2 may follow
                    if len(payload) >= 68:  # NetBIOS(4) + SMB2 header(64)
                        smb2_offset = 0
                        # Check if NetBIOS header is present (0x00 at start)
                        if payload[0] == 0x00:
                            smb2_offset = 4

                        # Check for SMB2 magic
                        if payload[smb2_offset:smb2_offset+4] == b'\xfeSMB':
                            # Extract SMB2 message (with or without NetBIOS header)
                            if smb2_offset > 0:
                                smb2_payload = payload[smb2_offset:]
                            else:
                                smb2_payload = payload

                            messages.append(smb2_payload)

                            # Ground truth boundaries for SMB2 header
                            boundaries = [
                                0,   # Start
                                4,   # ProtocolID
                                6,   # StructureSize
                                8,   # CreditCharge
                                12,  # Status
                                14,  # Command
                                16,  # Credits
                                20,  # Flags
                                24,  # NextCommand
                                32,  # MessageID
                                36,  # Reserved
                                40,  # TreeID
                                48,  # SessionID
                                64,  # Signature / Start of data
                                len(smb2_payload)  # End
                            ]
                            ground_truth.append(boundaries)

        logging.info(f"Loaded {len(messages)} SMB2 messages")
        return messages, ground_truth

    def load_messages(self, pcap_path: str, max_messages: int = 1000) -> List[bytes]:
        """
        [新增] 通用加载器：从指定 PCAP 路径加载原始消息载荷。
        用于加载 DNP3 或其他未预定义协议的数据。
        """
        # 处理路径：如果传入的是相对路径，拼接到 data_dir
        full_path = self.data_dir / pcap_path
        
        # 如果拼接后不存在，尝试直接使用传入路径（兼容绝对路径）
        if not full_path.exists():
            if Path(pcap_path).exists():
                full_path = Path(pcap_path)
            else:
                logging.warning(f"PCAP file not found: {full_path}")
                return []

        logging.info(f"Loading raw messages from {full_path}")
        
        try:
            packets = rdpcap(str(full_path))
            messages = []

            for pkt in packets:
                if len(messages) >= max_messages:
                    break
                
                # 提取应用层载荷
                # 优先级 1: Raw 层 (Scapy 无法解析的部分通常在这里)
                if pkt.haslayer(Raw):
                    messages.append(bytes(pkt[Raw].load))
                # 优先级 2: TCP 载荷 (如果 Scapy 解析了 TCP 但没解析应用层)
                elif pkt.haslayer(TCP):
                    payload = bytes(pkt[TCP].payload)
                    if len(payload) > 0:
                        messages.append(payload)
                # 优先级 3: UDP 载荷
                elif pkt.haslayer(UDP):
                    payload = bytes(pkt[UDP].payload)
                    if len(payload) > 0:
                        messages.append(payload)
            
            return messages
            
        except Exception as e:
            logging.error(f"Error loading {full_path}: {e}")
            return []

    def get_available_protocols(self) -> List[str]:
        """Get list of available protocols in the data directory."""
        protocols = []

        protocol_dirs = {
            'dhcp': 'in-dhcp-pcaps',
            'dns': 'in-dns-pcaps',
            'modbus': 'in-modbus-pcaps',
            'smb2': 'in-smb2-pcaps'
        }

        for protocol, dirname in protocol_dirs.items():
            protocol_dir = self.data_dir / dirname
            if protocol_dir.exists() and any(protocol_dir.glob('*.pcap')):
                protocols.append(protocol)

        return protocols


def test_pcap_loader():
    """Test the PCAP loader."""
    loader = PCAPDataLoader(data_dir='./data')

    protocols = loader.get_available_protocols()
    print(f"Available protocols: {protocols}")

    for protocol in protocols:
        print(f"\nTesting {protocol.upper()}:")
        messages, ground_truth = loader.load_protocol_data(protocol, max_messages=5)
        print(f"  Loaded {len(messages)} messages")

        if messages:
            print(f"  First message length: {len(messages[0])} bytes")
            print(f"  First message boundaries: {ground_truth[0]}")
            print(f"  Number of fields: {len(ground_truth[0]) - 1}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_pcap_loader()
