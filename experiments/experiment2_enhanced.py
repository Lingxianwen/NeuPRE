"""
Experiment 2: Multi-Protocol Enhanced Version

Updates:
1. ✅ 添加 BGP 和 ZigBee 协议
2. ✅ 优化 Ground Truth（渐进式GT策略）
3. ✅ 添加调试模式诊断低Perfection问题
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd
from typing import List, Dict
from ground_truth_definitions import *
from neupre_core import simulate_neupre_segmentation, compute_metrics
from neupre import setup_logging
from utils.pcap_loader import PCAPDataLoader

# ==================== 协议配置 ====================
PROTOCOL_CONFIGS = {
    # ICS 协议
    'modbus': {'pcap': 'in-modbus-pcaps/libmodbus-bandwidth_server-rand_client.pcap', 
               'gt_func': 'get_modbus_gt', 'type': 'ICS'},
    'dnp3': {'pcap': 'in-dnp3-pcaps/BinInf_dnp3_1000.pcap', 
             'gt_func': 'get_dnp3_gt', 'type': 'ICS'},
    's7comm': {'pcap': 'in-s7comm-pcaps/s7comm.pcap', 
               'gt_func': 'get_s7comm_gt', 'type': 'ICS'},
    'iec104': {'pcap': 'in-iec104-pcaps/iec104.pcap', 
               'gt_func': 'get_iec104_gt', 'type': 'ICS'},
    'lon': {'pcap': 'in-lon-pcaps/lon.pcap', 
            'gt_func': 'get_lon_gt', 'type': 'ICS'},
    
    # 网络协议
    # 'dhcp': {'pcap': 'in-dhcp-pcaps/BinInf_dhcp_1000.pcap', 
    #          'gt_func': 'get_dhcp_gt', 'type': 'Network'},
    'dns': {'pcap': 'in-dns-pcaps/SMIA_DNS_1000.pcap', 
            'gt_func': 'get_dns_gt', 'type': 'Network'},
    'rtp': {'pcap': 'in-rtp-pcaps/RTP_1000.pcap', 
            'gt_func': 'get_rtp_gt', 'type': 'Network'},
    'tcp': {'pcap': 'in-tcp-pcaps/SMIA_TCP_part_1000.pcap', 
            'gt_func': 'get_tcp_gt', 'type': 'Network'},
    'bgp': {'pcap': 'in-bgp-pcaps/bgp.pcap',  # ⭐ 新增
            'gt_func': 'get_bgp_gt', 'type': 'Network'},
    
    # 文件/应用协议
    # 'smb2': {'pcap': 'in-smb2-pcaps/samba.pcap', 
    #          'gt_func': 'get_smb2_gt', 'type': 'File'},
    # 'smb': {'pcap': 'in-smb-pcaps/BinInf_smb_1000.pcap', 
    #         'gt_func': 'get_smb_gt', 'type': 'File'},
    
    # # IoT 协议
    # 'zigbee': {'pcap': 'in-zigbee-pcaps/zigbeelxw.pcap',  # ⭐ 新增
    #            'gt_func': 'get_zigbee_gt', 'type': 'IoT'}
}


def find_pcap(pattern: str, data_dir: str) -> str:
    """查找 PCAP 文件"""
    import glob
    if '*' in pattern:
        matches = glob.glob(os.path.join(data_dir, pattern))
        return os.path.relpath(matches[0], data_dir) if matches else None
    return pattern


def process_protocol(name: str, config: Dict, loader: PCAPDataLoader, 
                     max_msg: int = 1000, debug: bool = False):
    """处理单个协议"""
    logging.info(f"\n{'='*80}\nProcessing {name.upper()}\n{'='*80}")
    
    try:
        pcap_path = find_pcap(config['pcap'], str(loader.data_dir))
        if not pcap_path:
            logging.warning(f"PCAP not found for {name}")
            return None
        
        messages = loader.load_messages(pcap_path, max_messages=max_msg)
        if not messages or len(messages) < 10:
            logging.warning(f"Insufficient data: {len(messages) if messages else 0} messages")
            return None
        
        logging.info(f"Loaded {len(messages)} messages")
        
        # Ground Truth
        gt_func = globals()[config['gt_func']]
        ground_truth = [gt_func(m) for m in messages]
        
        # 🔍 调试：检查GT分布
        if debug:
            avg_gt_fields = np.mean([len(gt) - 1 for gt in ground_truth])
            logging.info(f"📊 GT Stats: Avg {avg_gt_fields:.1f} fields/message")
            logging.info(f"   Sample GT: {ground_truth[0]}")
        
        # NeuPRE 分割
        predictions = simulate_neupre_segmentation(messages)
        
        # 🔍 调试：对比GT vs Pred
        if debug:
            avg_pred_fields = np.mean([len(p) - 1 for p in predictions])
            logging.info(f"📊 Pred Stats: Avg {avg_pred_fields:.1f} fields/message")
            logging.info(f"   Sample Pred: {predictions[0]}")
            
            # 详细对比
            debug_gt_coverage(name, predictions, ground_truth)
        
        # 计算指标
        metrics = compute_metrics(predictions, ground_truth)
        
        logging.info(f"Results: Acc={metrics['accuracy']:.3f} F1={metrics['f1']:.3f} "
                    f"Correct={metrics['correctness']:.3f} Perf={metrics['perfect_match']:.3f}")
        
        return {
            'protocol': name,
            'type': config['type'],
            'num_messages': len(messages),
            'metrics': metrics,
            'predictions': predictions if debug else None,
            'ground_truth': ground_truth if debug else None
        }
    
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_table(results: Dict, output_dir: str):
    """生成结果表格"""
    rows = []
    for name in sorted(results.keys()):
        r = results[name]
        m = r['metrics']
        rows.append({
            'Protocol': name.upper(),
            'Type': r['type'],
            'Msgs': r['num_messages'],
            'Acc': f"{m['accuracy']:.3f}",
            'P': f"{m['precision']:.3f}",
            'R': f"{m['recall']:.3f}",
            'F1': f"{m['f1']:.3f}",
            'Correct': f"{m['correctness']:.3f}",
            'Perf': f"{m['perfect_match']:.3f}"
        })
    
    # 分类平均
    for ptype in sorted(set(r['type'] for r in results.values())):
        subset = [r for r in results.values() if r['type'] == ptype]
        rows.append({
            'Protocol': f'--- {ptype} Avg ---',
            'Type': ptype,
            'Msgs': sum(r['num_messages'] for r in subset),
            'Acc': f"{np.mean([r['metrics']['accuracy'] for r in subset]):.3f}",
            'P': f"{np.mean([r['metrics']['precision'] for r in subset]):.3f}",
            'R': f"{np.mean([r['metrics']['recall'] for r in subset]):.3f}",
            'F1': f"{np.mean([r['metrics']['f1'] for r in subset]):.3f}",
            'Correct': f"{np.mean([r['metrics']['correctness'] for r in subset]):.3f}",
            'Perf': f"{np.mean([r['metrics']['perfect_match'] for r in subset]):.3f}"
        })
    
    # 总平均
    rows.append({
        'Protocol': '=== OVERALL ===',
        'Type': 'All',
        'Msgs': sum(r['num_messages'] for r in results.values()),
        'Acc': f"{np.mean([r['metrics']['accuracy'] for r in results.values()]):.3f}",
        'P': f"{np.mean([r['metrics']['precision'] for r in results.values()]):.3f}",
        'R': f"{np.mean([r['metrics']['recall'] for r in results.values()]):.3f}",
        'F1': f"{np.mean([r['metrics']['f1'] for r in results.values()]):.3f}",
        'Correct': f"{np.mean([r['metrics']['correctness'] for r in results.values()]):.3f}",
        'Perf': f"{np.mean([r['metrics']['perfect_match'] for r in results.values()]):.3f}"
    })
    
    df = pd.DataFrame(rows)
    
    # 保存
    csv_path = os.path.join(output_dir, 'results_enhanced.csv')
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*100)
    print("RESULTS (Optimized GT + DYNPRE Perfection)")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    logging.info(f"Saved to: {csv_path}")
    
    # 🎯 性能分析
    print("\n📊 Performance Analysis:")
    avg_perf = np.mean([r['metrics']['perfect_match'] for r in results.values()])
    high_perf = [n for n, r in results.items() if r['metrics']['perfect_match'] > 0.3]
    low_perf = [n for n, r in results.items() if r['metrics']['perfect_match'] < 0.1]
    
    print(f"  Average Perfection: {avg_perf:.3f}")
    print(f"  High performers (>30%): {high_perf}")
    print(f"  Low performers (<10%): {low_perf}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', default=None, help='ics/network/file/iot or protocol name')
    parser.add_argument('--max-messages', type=int, default=1000)
    parser.add_argument('--output-dir', default='./experiments/exp2_enhanced')
    parser.add_argument('--all', action='store_true', help='Test all protocols')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    setup_logging(level=logging.INFO)
    
    logging.info("="*80)
    logging.info("Experiment 2: Enhanced Multi-Protocol (Optimized GT)")
    if args.debug:
        logging.info("🔍 DEBUG MODE ENABLED")
    logging.info("="*80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    loader = PCAPDataLoader(data_dir='../data')
    
    # 选择协议
    if args.all or args.filter is None:
        selected = PROTOCOL_CONFIGS
    elif args.filter.lower() in ['ics', 'network', 'file', 'iot']:
        selected = {k: v for k, v in PROTOCOL_CONFIGS.items() 
                   if v['type'].lower() == args.filter.lower()}
    elif args.filter in PROTOCOL_CONFIGS:
        selected = {args.filter: PROTOCOL_CONFIGS[args.filter]}
    else:
        print(f"❌ Unknown filter: {args.filter}")
        print(f"Available: {list(PROTOCOL_CONFIGS.keys())}")
        return
    
    logging.info(f"Testing {len(selected)} protocols: {list(selected.keys())}")
    
    # 处理
    results = {}
    for name, config in selected.items():
        result = process_protocol(name, config, loader, args.max_messages, args.debug)
        if result:
            results[name] = result
    
    # 输出
    if results:
        generate_table(results, args.output_dir)
        print(f"\n✅ Processed {len(results)}/{len(selected)} protocols")
        
        # 🔍 如果有低性能协议，建议使用debug模式
        low_perf = [n for n, r in results.items() if r['metrics']['perfect_match'] < 0.1]
        if low_perf and not args.debug:
            print(f"\n💡 Tip: Low performance on {low_perf}")
            print(f"   Run with --debug to see detailed GT vs Pred comparison")
    else:
        print("❌ No results generated")


if __name__ == '__main__':
    main()