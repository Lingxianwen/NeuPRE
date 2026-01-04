"""
Experiment 1: State Coverage Efficiency

Compares how quickly NeuPRE vs DYNpre discovers unique protocol states.

Expected result:
- NeuPRE curve should be steeper (reaches target coverage with fewer packets)
- Demonstrates Bayesian active learning is more efficient than random mutation
"""

import sys
import socket
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
from typing import List, Tuple
import time
from utils.pcap_loader import PCAPDataLoader

from neupre import NeuPRE, setup_logging
from utils.evaluator import NeuPREEvaluator
from modules.state_explorer import DeepKernelStateExplorer


class RealTargetProbe:
    """
    真实目标的探测器 (Modbus/HTTP/etc)
    """
    def __init__(self, host='127.0.0.1', port=1502):
        self.target = (host, port)
        self.sock = None
        
    def send_and_recv(self, message: bytes) -> bytes:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0) # Modbus 响应很快，1秒足够
            s.connect(self.target)
            s.send(message)
            
            # 接收响应
            try:
                response = s.recv(2048)
            except socket.timeout:
                response = b'TIMEOUT'
                
            s.close()
            return response
        except ConnectionRefusedError:
            return b'CONN_REFUSED'
        except Exception as e:
            return f'ERROR_{str(e)}'.encode()

    # [修复] 增加这个方法以兼容 DYNpre 的调用
    def handle_message(self, message: bytes) -> bytes:
        return self.send_and_recv(message)

class MockProtocolServer:
    """
    Mock protocol server with defined state machine for testing.
    """

    def __init__(self, num_states: int = 10):
        """
        Args:
            num_states: Number of unique states in the protocol
        """
        self.num_states = num_states
        self.state_responses = {
            i: f"STATE_{i}_RESPONSE".encode() + bytes([i] * 10)
            for i in range(num_states)
        }

    def handle_message(self, message: bytes) -> bytes:
        """
        Handle protocol message and return response.

        State is determined by first byte of message.
        """
        if not message:
            return b"ERROR_EMPTY"

        # Simple state determination: hash message to state ID
        state_id = sum(message) % self.num_states
        return self.state_responses[state_id]


def get_state_id(response: bytes) -> str:
    # 真实的逆向中，我们不知道状态ID，通常通过响应的相似度来聚类
    # 简单版本：取响应的前几字节作为状态标识
    if not response: return "NO_RESP"
    if response.startswith(b'HTTP'):
        # 提取状态码，如 "200", "404"
        return response.split(b' ')[1].decode(errors='ignore')
    return str(hash(response[:10])) # 仅哈希响应头部

def simulate_dynpre_exploration(server: MockProtocolServer,
                                base_messages: List[bytes],
                                num_iterations: int = 100,
                                mutation_rate: float = 0.1) -> Tuple[List[int], List[int]]:
    """
    Simulate DYNpre's random mutation exploration.

    Args:
        server: Mock protocol server
        base_messages: Seed messages
        num_iterations: Number of exploration iterations
        mutation_rate: Mutation probability per byte

    Returns:
        (messages_sent, unique_states) histories
    """
    logging.info("Simulating DYNpre exploration (random mutation)...")

    messages_sent = []
    unique_states = []
    discovered_states = set()
    total_messages = 0

    for iteration in range(num_iterations):
        # Random mutation
        base_msg = np.random.choice(base_messages)
        mutated = bytearray(base_msg)

        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                mutated[i] = np.random.randint(0, 256)

        # Send and observe
        response = server.handle_message(bytes(mutated))
        total_messages += 1

        # Track state
        state_hash = hash(response)
        discovered_states.add(state_hash)

        messages_sent.append(total_messages)
        unique_states.append(len(discovered_states))

        if iteration % 10 == 0:
            logging.debug(f"DYNpre iteration {iteration}: {len(discovered_states)} states")

    logging.info(f"DYNpre: {total_messages} messages → {len(discovered_states)} states")
    return messages_sent, unique_states


def simulate_neupre_exploration(server: RealTargetProbe,  # 参数类型变了
                               base_messages: List[bytes],
                               num_iterations: int = 100) -> Tuple[List[int], List[int]]:
    logging.info("Simulating NeuPRE exploration (Active Learning on Real Server)...")

    # 初始化 State Explorer
    # 真实环境下，我们可能需要更大的维度来捕捉 HTTP 文本特征
    explorer = DeepKernelStateExplorer(
        embedding_dim=64,
        hidden_dim=128,
        feature_dim=32,
        kappa=2.0
    )

    # 定义 Probe 回调：直接通过网络发送
    def probe_callback(msg: bytes) -> bytes:
        response = server.send_and_recv(msg)
        # 简单状态抽象：根据响应的哈希或状态码区分状态
        # 为了实验可视化，我们返回包含响应特征的 bytes
        return response

    stats = explorer.active_exploration(
        base_messages=base_messages,
        num_iterations=num_iterations,
        num_mutations=50,
        probe_callback=probe_callback
    )

    return stats['iterations'], stats['unique_states']


def run_experiment1(num_states: int = 10,
                   num_iterations: int = 100,
                   num_runs: int = 3,
                   output_dir: str = './experiment1_results'):
    """
    Run Experiment 1: State Coverage Efficiency.

    Args:
        num_states: Number of states in mock protocol
        num_iterations: Exploration iterations per run
        num_runs: Number of runs to average over
        output_dir: Output directory
    """
    setup_logging(level=logging.INFO)
    logging.info("=" * 80)
    logging.info("EXPERIMENT 1: State Coverage (Target: Modbus TCP)")
    logging.info("=" * 80)

    evaluator = NeuPREEvaluator(output_dir=output_dir)

    # 初始化真实目标探测器
    server = RealTargetProbe(host='127.0.0.1', port=1502)

    # 2. 加载真实的 Modbus 种子数据
    logging.info("Loading Modbus seed messages from PCAP...")
    loader = PCAPDataLoader(data_dir='./data') # 确保你的 data 目录路径正确
    
    # 尝试加载 Modbus 数据，如果失败则回退到简单的种子
    try:
        # 注意：这里调用的是 pcap_loader 里的方法
        base_messages, _ = loader.load_modbus_data(max_messages=50)
        logging.info(f"Loaded {len(base_messages)} Modbus seed messages")
    except Exception as e:
        logging.error(f"Failed to load PCAP: {e}")
        logging.warning("Falling back to synthetic Modbus seeds")
        # 简单的 Modbus Read Holding Registers (TransactionID, ProtoID, Len, UnitID, Func, Start, Count)
        base_messages = [
            b'\x00\x01\x00\x00\x00\x06\x01\x03\x00\x00\x00\x01',
            b'\x00\x02\x00\x00\x00\x06\x01\x03\x00\x00\x00\x0A'
        ]

    if not base_messages:
        logging.error("No base messages available!")
        return

    # Run multiple times and average
    neupre_all_runs = []
    dynpre_all_runs = []

    for run in range(num_runs):
        logging.info(f"\n--- Run {run + 1}/{num_runs} ---")

        # NeuPRE exploration
        neupre_msgs, neupre_states = simulate_neupre_exploration(
            server, base_messages, num_iterations
        )
        neupre_all_runs.append((neupre_msgs, neupre_states))

        # DYNpre exploration
        dynpre_msgs, dynpre_states = simulate_dynpre_exploration(
            server, base_messages, num_iterations
        )
        dynpre_all_runs.append((dynpre_msgs, dynpre_states))

    # Average results
    neupre_msgs_avg = np.mean([msgs for msgs, _ in neupre_all_runs], axis=0).astype(int).tolist()
    neupre_states_avg = np.mean([states for _, states in neupre_all_runs], axis=0).tolist()

    dynpre_msgs_avg = np.mean([msgs for msgs, _ in dynpre_all_runs], axis=0).astype(int).tolist()
    dynpre_states_avg = np.mean([states for _, states in dynpre_all_runs], axis=0).tolist()

    # Evaluate
    target_coverage = 5

    neupre_metrics = evaluator.evaluate_state_coverage(
        neupre_msgs_avg, neupre_states_avg, target_coverage
    )

    dynpre_metrics = evaluator.evaluate_state_coverage(
        dynpre_msgs_avg, dynpre_states_avg, target_coverage
    )

    # Compare
    comparison = evaluator.compare_methods(
        neupre_metrics, dynpre_metrics, 'coverage'
    )

    # Plot
    evaluator.plot_state_coverage_curve(
        (neupre_msgs_avg, neupre_states_avg),
        (dynpre_msgs_avg, dynpre_states_avg),
        filename='state_coverage_efficiency.png'
    )

    # Generate report
    evaluator.generate_report([comparison], filename='experiment1_report.txt')
    evaluator.save_metrics_json({
        'neupre': neupre_metrics.__dict__,
        'dynpre': dynpre_metrics.__dict__,
        'comparison': comparison
    }, filename='experiment1_metrics.json')

    # Summary
    logging.info("\n" + "=" * 80)
    logging.info("EXPERIMENT 1 SUMMARY")
    logging.info("=" * 80)
    if neupre_metrics.messages_to_target_coverage and dynpre_metrics.messages_to_target_coverage:
        improvement = (dynpre_metrics.messages_to_target_coverage -
                      neupre_metrics.messages_to_target_coverage) / \
                      dynpre_metrics.messages_to_target_coverage * 100
        logging.info(f"NeuPRE reached {target_coverage} states in {neupre_metrics.messages_to_target_coverage} messages")
        logging.info(f"DYNpre reached {target_coverage} states in {dynpre_metrics.messages_to_target_coverage} messages")
        logging.info(f"NeuPRE improvement: {improvement:.1f}% fewer messages")
    logging.info("=" * 80)


if __name__ == '__main__':
    run_experiment1(
        num_states=15,
        num_iterations=100,
        num_runs=3,
        output_dir='./experiments/experiment1_results'
    )
