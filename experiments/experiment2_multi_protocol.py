"""
Experiment 2: Multi-Protocol Field Boundary Accuracy
Extended Version with 10+ Protocols

Output Format:
Table: Comparison with State-of-the-Art Protocol Format Extraction Methods
Protocol | NeuPRE (Acc/F1/Perf) | FieldHunter | Netplier | BinaryInferno | Netzob
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from collections import defaultdict

from neupre import setup_logging
from utils.evaluator import NeuPREEvaluator
from utils.pcap_loader import PCAPDataLoader

# Import segmentation modules
from experiment2_segmentation import (
    simulate_neupre_segmentation,
    run_real_dynpre_static
)


# ==================== Protocol Definitions ====================
PROTOCOL_CONFIGS = {
    'modbus': {
        'pcap_path': 'in-modbus-pcaps/libmodbus-bandwidth_server-rand_client.pcap',
        'ground_truth_func': 'get_modbus_gt',
        'type': 'ICS',
        'priority': 1
    },
    'dnp3': {
        'pcap_path': 'in-dnp3-pcaps/BinInf_dnp3_1000.pcap',
        'ground_truth_func': 'get_dnp3_gt',
        'type': 'ICS',
        'priority': 1
    },
    's7comm': {
        'pcap_path': 'in-s7comm-pcaps/s7comm.pcap',
        'ground_truth_func': 'get_s7comm_gt',
        'type': 'ICS',
        'priority': 1
    },
    'iec104': {
        'pcap_path': 'in-iec104-pcaps/iec104.pcap',
        'ground_truth_func': 'get_iec104_gt',
        'type': 'ICS',
        'priority': 2
    },
    'dhcp': {
        'pcap_path': 'in-dhcp-pcaps/BinInf_dhcp_1000.pcap',
        'ground_truth_func': 'get_dhcp_gt',
        'type': 'Network',
        'priority': 2
    },
    'dns': {
        'pcap_path': 'in-dns-pcaps/SMIA_DNS_1000.pcap',
        'ground_truth_func': 'get_dns_gt',
        'type': 'Network',
        'priority': 2
    },
    'smb': {
        'pcap_path': 'in-smb-pcaps/smb.pcap',
        'ground_truth_func': 'get_smb_gt',
        'type': 'File',
        'priority': 3
    },
    'smb2': {
        'pcap_path': 'in-smb2-pcaps/samba.pcap',
        'ground_truth_func': 'get_smb2_gt',
        'type': 'File',
        'priority': 2
    },
    'rtp': {
        'pcap_path': 'in-rtp-pcaps/rtp.pcap',
        'ground_truth_func': 'get_rtp_gt',
        'type': 'Media',
        'priority': 3
    },
    'lon': {
        'pcap_path': 'in-lon-pcaps/lon.pcap',
        'ground_truth_func': 'get_lon_gt',
        'type': 'ICS',
        'priority': 3
    }
}


# ==================== Ground Truth Functions ====================
def get_modbus_gt(msg: bytes) -> List[int]:
    """Modbus TCP: [TxID(2)][ProtoID(2)][Len(2)][UnitID(1)][Func(1)][Data]"""
    boundaries = [0, 2, 4, 6, 7, 8, len(msg)]
    return sorted(list(set([b for b in boundaries if b <= len(msg)])))


def get_dnp3_gt(msg: bytes) -> List[int]:
    """DNP3: [Start(2)][Len(1)][Ctrl(1)][Dest(2)][Src(2)][CRC(2)][Data]"""
    boundaries = [0, 2, 3, 4, 6, 8, 10, len(msg)]
    return sorted(list(set([b for b in boundaries if b <= len(msg)])))


def get_s7comm_gt(msg: bytes) -> List[int]:
    """S7COMM: [TPKT(4)][COTP(3)][S7Header(10+)][Data]"""
    boundaries = [0, 4, 7]
    if len(msg) >= 17:
        boundaries.append(17)  # S7 header end
    boundaries.append(len(msg))
    return sorted(list(set(boundaries)))


def get_iec104_gt(msg: bytes) -> List[int]:
    """IEC-104: [Start(1)][Len(1)][CtrlField(4)][ASDU]"""
    boundaries = [0, 1, 2, 6, len(msg)]
    return sorted(list(set([b for b in boundaries if b <= len(msg)])))


def get_dhcp_gt(msg: bytes) -> List[int]:
    """DHCP: [Op(1)][HType(1)][HLen(1)][Hops(1)][XID(4)][Secs(2)][Flags(2)]..."""
    boundaries = [0, 1, 2, 3, 4, 8, 10, 12, 16, 20, 24, 28, 44, 108, 236, len(msg)]
    return sorted(list(set([b for b in boundaries if b <= len(msg)])))


def get_dns_gt(msg: bytes) -> List[int]:
    """DNS: [ID(2)][Flags(2)][QDCount(2)][ANCount(2)][NSCount(2)][ARCount(2)][Questions]"""
    boundaries = [0, 2, 4, 6, 8, 10, 12, len(msg)]
    return sorted(list(set([b for b in boundaries if b <= len(msg)])))


def get_smb_gt(msg: bytes) -> List[int]:
    """SMB: [Protocol(4)][Command(1)][Status(4)][Flags(1)][Flags2(2)][PIDHigh(2)][Signature(8)][Reserved(2)][TID(2)][PID(2)][UID(2)][MID(2)]"""
    boundaries = [0, 4, 5, 9, 10, 12, 14, 22, 24, 26, 28, 30, 32, len(msg)]
    return sorted(list(set([b for b in boundaries if b <= len(msg)])))


def get_smb2_gt(msg: bytes) -> List[int]:
    """SMB2: [ProtocolID(4)][StructSize(2)][CreditCharge(2)][Status(4)][Command(2)]...[Signature(16)]"""
    boundaries = [0, 4, 6, 8, 12, 14, 16, 20, 24, 32, 36, 40, 48, 64, len(msg)]
    return sorted(list(set([b for b in boundaries if b <= len(msg)])))


def get_rtp_gt(msg: bytes) -> List[int]:
    """RTP: [V/P/X/CC(1)][M/PT(1)][Seq(2)][Timestamp(4)][SSRC(4)][Payload]"""
    boundaries = [0, 1, 2, 4, 8, 12, len(msg)]
    return sorted(list(set([b for b in boundaries if b <= len(msg)])))


def get_lon_gt(msg: bytes) -> List[int]:
    """LON (LonTalk): Simplified structure"""
    # LON protocol structure varies, using simplified version
    boundaries = [0, 2, 4, len(msg)]
    return sorted(list(set([b for b in boundaries if b <= len(msg)])))


# ==================== Metrics Computation ====================
def compute_metrics(predicted: List[List[int]], 
                   ground_truth: List[List[int]]) -> Dict[str, float]:
    """
    Compute Accuracy, F1, and Perfect Match Rate (Perf)
    
    Accuracy: Ratio of correctly identified boundaries
    F1: Harmonic mean of precision and recall
    Perf: Ratio of perfectly segmented messages
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_boundaries_gt = 0
    perfect_matches = 0
    
    for pred, gt in zip(predicted, ground_truth):
        pred_set = set(pred)
        gt_set = set(gt)
        
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_boundaries_gt += len(gt_set)
        
        if pred_set == gt_set:
            perfect_matches += 1
    
    # Accuracy: (TP) / (Total GT boundaries)
    accuracy = total_tp / total_boundaries_gt if total_boundaries_gt > 0 else 0
    
    # Precision & Recall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Perfect Match Rate
    perf = perfect_matches / len(predicted) if len(predicted) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'perfect_match': perf
    }


# ==================== Protocol Processing ====================
def process_protocol(protocol_name: str, 
                    config: Dict,
                    loader: PCAPDataLoader,
                    max_messages: int = 1000) -> Dict:
    """
    Process a single protocol and return metrics
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"Processing {protocol_name.upper()}")
    logging.info(f"{'='*80}")
    
    # Load messages
    try:
        messages = loader.load_messages(config['pcap_path'], max_messages=max_messages)
        
        if not messages or len(messages) < 10:
            logging.warning(f"Insufficient messages for {protocol_name}: {len(messages) if messages else 0}")
            return None
            
        logging.info(f"Loaded {len(messages)} messages")
        
        # Generate ground truth
        gt_func = globals()[config['ground_truth_func']]
        ground_truth = [gt_func(msg) for msg in messages]
        
        # Run NeuPRE segmentation
        logging.info("Running NeuPRE segmentation...")
        neupre_boundaries = simulate_neupre_segmentation(messages)
        
        # Compute metrics
        metrics = compute_metrics(neupre_boundaries, ground_truth)
        
        logging.info(f"Results for {protocol_name}:")
        logging.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"  F1 Score: {metrics['f1']:.4f}")
        logging.info(f"  Perfect Match: {metrics['perfect_match']:.4f}")
        
        return {
            'protocol': protocol_name,
            'type': config['type'],
            'num_messages': len(messages),
            'metrics': metrics,
            'ground_truth': ground_truth,
            'predictions': neupre_boundaries
        }
        
    except FileNotFoundError as e:
        logging.warning(f"PCAP file not found for {protocol_name}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing {protocol_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== Main Experiment ====================
def run_experiment2_extended(output_dir: str = './experiments/exp2_results',
                            max_messages_per_protocol: int = 1000,
                            focus_ics: bool = True):
    """
    Run extended Experiment 2 on multiple protocols
    """
    setup_logging(level=logging.INFO)
    
    logging.info("="*80)
    logging.info("EXPERIMENT 2: Multi-Protocol Field Boundary Accuracy")
    logging.info("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    loader = PCAPDataLoader(data_dir='./data')
    
    # Select protocols
    if focus_ics:
        # Prioritize ICS protocols
        selected_protocols = {
            k: v for k, v in PROTOCOL_CONFIGS.items() 
            if v['type'] == 'ICS' or v['priority'] <= 2
        }
        logging.info(f"Focusing on {len(selected_protocols)} protocols (ICS priority)")
    else:
        selected_protocols = PROTOCOL_CONFIGS
        logging.info(f"Testing all {len(selected_protocols)} protocols")
    
    # Process each protocol
    results = {}
    for protocol_name, config in selected_protocols.items():
        result = process_protocol(
            protocol_name, 
            config, 
            loader, 
            max_messages=max_messages_per_protocol
        )
        
        if result is not None:
            results[protocol_name] = result
    
    # Generate summary table
    generate_summary_table(results, output_dir)
    
    # Save detailed results
    save_detailed_results(results, output_dir)
    
    return results


def generate_summary_table(results: Dict, output_dir: str):
    """
    Generate publication-ready table in the specified format
    """
    if not results:
        logging.warning("No results to generate table")
        return
    
    # Prepare data
    table_data = []
    
    for protocol_name in sorted(results.keys(), key=lambda x: x.upper()):
        result = results[protocol_name]
        metrics = result['metrics']
        
        row = {
            'Protocol': protocol_name.upper(),
            'NeuPRE_Acc': f"{metrics['accuracy']:.4f}",
            'NeuPRE_F1': f"{metrics['f1']:.4f}",
            'NeuPRE_Perf': f"{metrics['perfect_match']:.4f}",
            # Placeholders for baselines (to be filled)
            'FieldHunter_Acc': '-',
            'FieldHunter_F1': '-',
            'FieldHunter_Perf': '-',
            'Netplier_Acc': '-',
            'Netplier_F1': '-',
            'Netplier_Perf': '-',
            'BinaryInferno_Acc': '-',
            'BinaryInferno_F1': '-',
            'BinaryInferno_Perf': '-',
            'Netzob_Acc': '-',
            'Netzob_F1': '-',
            'Netzob_Perf': '-'
        }
        table_data.append(row)
    
    # Calculate averages
    avg_row = {
        'Protocol': 'Average',
        'NeuPRE_Acc': f"{np.mean([r['metrics']['accuracy'] for r in results.values()]):.4f}",
        'NeuPRE_F1': f"{np.mean([r['metrics']['f1'] for r in results.values()]):.4f}",
        'NeuPRE_Perf': f"{np.mean([r['metrics']['perfect_match'] for r in results.values()]):.4f}",
        'FieldHunter_Acc': '-',
        'FieldHunter_F1': '-',
        'FieldHunter_Perf': '-',
        'Netplier_Acc': '-',
        'Netplier_F1': '-',
        'Netplier_Perf': '-',
        'BinaryInferno_Acc': '-',
        'BinaryInferno_F1': '-',
        'BinaryInferno_Perf': '-',
        'Netzob_Acc': '-',
        'Netzob_F1': '-',
        'Netzob_Perf': '-'
    }
    table_data.append(avg_row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'table3_format_extraction_comparison.csv')
    df.to_csv(csv_path, index=False)
    logging.info(f"Table saved to: {csv_path}")
    
    # Print formatted table
    print("\n" + "="*120)
    print("Table 3: Comparison with State-of-the-Art Protocol Format Extraction Methods")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120)
    
    # Save LaTeX version
    latex_path = os.path.join(output_dir, 'table3_latex.tex')
    with open(latex_path, 'w') as f:
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison with State-of-the-Art Protocol Format Extraction Methods}\n")
        f.write("\\label{tab:format_extraction}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{l|ccc|ccc|ccc|ccc|ccc}\n")
        f.write("\\hline\n")
        f.write("\\multirow{2}{*}{Protocol} & \\multicolumn{3}{c|}{NeuPRE} & \\multicolumn{3}{c|}{FieldHunter} & \\multicolumn{3}{c|}{Netplier} & \\multicolumn{3}{c|}{BinaryInferno} & \\multicolumn{3}{c}{Netzob} \\\\\n")
        f.write("& Acc. & F1 & Perf. & Acc. & F1 & Perf. & Acc. & F1 & Perf. & Acc. & F1 & Perf. & Acc. & F1 & Perf. \\\\\n")
        f.write("\\hline\n")
        
        for _, row in df.iterrows():
            protocol = row['Protocol']
            if protocol == 'Average':
                f.write("\\hline\n")
            f.write(f"{protocol} & {row['NeuPRE_Acc']} & {row['NeuPRE_F1']} & {row['NeuPRE_Perf']} & ")
            f.write(f"{row['FieldHunter_Acc']} & {row['FieldHunter_F1']} & {row['FieldHunter_Perf']} & ")
            f.write(f"{row['Netplier_Acc']} & {row['Netplier_F1']} & {row['Netplier_Perf']} & ")
            f.write(f"{row['BinaryInferno_Acc']} & {row['BinaryInferno_F1']} & {row['BinaryInferno_Perf']} & ")
            f.write(f"{row['Netzob_Acc']} & {row['Netzob_F1']} & {row['Netzob_Perf']} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\end{table*}\n")
    
    logging.info(f"LaTeX table saved to: {latex_path}")


def save_detailed_results(results: Dict, output_dir: str):
    """Save detailed per-protocol results"""
    import json
    
    # Convert to serializable format
    serializable_results = {}
    for protocol, result in results.items():
        serializable_results[protocol] = {
            'type': result['type'],
            'num_messages': result['num_messages'],
            'metrics': result['metrics']
        }
    
    json_path = os.path.join(output_dir, 'detailed_results.json')
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logging.info(f"Detailed results saved to: {json_path}")


# ==================== Entry Point ====================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Experiment 2: Multi-Protocol Format Extraction')
    parser.add_argument('--output-dir', default='./experiments/exp2_results',
                       help='Output directory')
    parser.add_argument('--max-messages', type=int, default=1000,
                       help='Maximum messages per protocol')
    parser.add_argument('--all-protocols', action='store_true',
                       help='Test all protocols (default: focus on ICS)')
    
    args = parser.parse_args()
    
    results = run_experiment2_extended(
        output_dir=args.output_dir,
        max_messages_per_protocol=args.max_messages,
        focus_ics=not args.all_protocols
    )
    
    print(f"\n✓ Experiment completed. Results saved to: {args.output_dir}")