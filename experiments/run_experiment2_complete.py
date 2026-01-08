"""
Complete Runner for Experiment 2: Multi-Protocol Format Extraction

This script orchestrates the entire experimental pipeline:
1. Run NeuPRE on all protocols
2. Integrate baseline results
3. Generate tables and figures
4. Create publication-ready outputs

Usage:
    python run_experiment2_complete.py --data-dir ./data --output-dir ./exp2_results
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_directories(base_dir: str):
    """Create necessary directories"""
    dirs = [
        base_dir,
        os.path.join(base_dir, 'neupre_results'),
        os.path.join(base_dir, 'tables'),
        os.path.join(base_dir, 'figures'),
        os.path.join(base_dir, 'logs')
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    return {
        'base': base_dir,
        'neupre': os.path.join(base_dir, 'neupre_results'),
        'tables': os.path.join(base_dir, 'tables'),
        'figures': os.path.join(base_dir, 'figures'),
        'logs': os.path.join(base_dir, 'logs')
    }


def run_neupre_experiments(data_dir: str, output_dir: str, 
                          max_messages: int = 1000,
                          focus_ics: bool = True):
    """
    Step 1: Run NeuPRE on all protocols
    """
    print("\n" + "="*80)
    print("STEP 1: Running NeuPRE on Multiple Protocols")
    print("="*80)
    
    # Import the experiment module
    from experiment2_multi_protocol import run_experiment2_extended
    
    results = run_experiment2_extended(
        output_dir=output_dir,
        max_messages_per_protocol=max_messages,
        focus_ics=focus_ics
    )
    
    print(f"\n✓ NeuPRE experiments completed for {len(results)} protocols")
    return results


def integrate_baselines(neupre_results_path: str, 
                        baseline_data_path: str,
                        output_dir: str):
    """
    Step 2: Integrate baseline results
    """
    print("\n" + "="*80)
    print("STEP 2: Integrating Baseline Results")
    print("="*80)
    
    from baseline_integration import (
        integrate_baseline_results,
        format_table_for_publication,
        highlight_best_results,
        BASELINE_RESULTS
    )
    
    # Load baseline data
    if baseline_data_path and os.path.exists(baseline_data_path):
        with open(baseline_data_path, 'r') as f:
            baseline_data = json.load(f)
        print(f"Loaded custom baseline data from: {baseline_data_path}")
    else:
        baseline_data = BASELINE_RESULTS
        print("Using default baseline data (example values)")
    
    # Integrate
    df = integrate_baseline_results(
        neupre_results_path=neupre_results_path,
        baseline_data=baseline_data
    )
    
    # Format for publication
    format_table_for_publication(df, output_dir=output_dir)
    
    # Highlight best results
    highlight_path = os.path.join(output_dir, 'table3_highlighted.csv')
    highlight_best_results(df, output_path=highlight_path)
    
    print(f"\n✓ Baseline integration completed")
    
    return os.path.join(output_dir, 'table3_complete.csv')


def generate_visualizations(csv_path: str, output_dir: str):
    """
    Step 3: Generate all visualizations
    """
    print("\n" + "="*80)
    print("STEP 3: Generating Visualization Figures")
    print("="*80)
    
    from visualization_exp2 import generate_all_figures
    
    protocol_types = {
        'modbus': 'ICS',
        'dnp3': 'ICS',
        's7comm': 'ICS',
        'iec104': 'ICS',
        'lon': 'ICS',
        'dhcp': 'Network',
        'dns': 'Network',
        'smb': 'File Transfer',
        'smb2': 'File Transfer',
        'rtp': 'Media'
    }
    
    generate_all_figures(csv_path, output_dir, protocol_types)
    
    print(f"\n✓ Visualizations generated")


def generate_summary_report(dirs: dict, results: dict):
    """
    Step 4: Generate summary report
    """
    print("\n" + "="*80)
    print("STEP 4: Generating Summary Report")
    print("="*80)
    
    report_path = os.path.join(dirs['base'], 'EXPERIMENT2_REPORT.md')
    
    with open(report_path, 'w') as f:
        f.write("# Experiment 2: Multi-Protocol Format Extraction Results\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- Total Protocols Tested: {len(results)}\n")
        f.write(f"- Output Directory: `{dirs['base']}`\n\n")
        
        f.write("## Protocols Tested\n\n")
        for protocol, result in sorted(results.items()):
            f.write(f"### {protocol.upper()}\n")
            f.write(f"- Type: {result['type']}\n")
            f.write(f"- Messages Processed: {result['num_messages']}\n")
            metrics = result['metrics']
            f.write(f"- Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"- F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"- Perfect Match: {metrics['perfect_match']:.4f}\n\n")
        
        f.write("## Output Files\n\n")
        f.write("### Tables\n")
        f.write(f"- `{os.path.join(dirs['tables'], 'table3_complete.csv')}` - Complete results table\n")
        f.write(f"- `{os.path.join(dirs['tables'], 'table3_complete.tex')}` - LaTeX version\n")
        f.write(f"- `{os.path.join(dirs['tables'], 'table3_highlighted.csv')}` - With best method highlights\n\n")
        
        f.write("### Figures\n")
        f.write(f"- `{os.path.join(dirs['figures'], 'fig_exp2_*')}` - Various comparison plots\n\n")
        
        f.write("### Raw Data\n")
        f.write(f"- `{os.path.join(dirs['neupre'], 'detailed_results.json')}` - Detailed NeuPRE results\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Review the generated tables and figures\n")
        f.write("2. If you have actual baseline results, update `baseline_data.json` and re-run integration\n")
        f.write("3. Use the LaTeX table in your paper\n")
        f.write("4. Select the most relevant figures for publication\n\n")
        
        f.write("## Citation Format for Table 3\n\n")
        f.write("```latex\n")
        f.write("Table~\\ref{tab:format_extraction} shows the comparison of NeuPRE with \n")
        f.write("state-of-the-art protocol format extraction methods across multiple protocols.\n")
        f.write("NeuPRE achieves an average F1 score of X.XXX, outperforming the best baseline \n")
        f.write("by XX.X\\%.\n")
        f.write("```\n")
    
    print(f"\n✓ Summary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Complete Pipeline for Experiment 2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (focus on ICS protocols)
  python run_experiment2_complete.py
  
  # Test all protocols with custom data directory
  python run_experiment2_complete.py --data-dir ./data --all-protocols
  
  # Use custom baseline data
  python run_experiment2_complete.py --baseline-json ./my_baselines.json
  
  # Quick test with fewer messages
  python run_experiment2_complete.py --max-messages 100
        """
    )
    
    parser.add_argument('--data-dir', default='./data',
                       help='Directory containing PCAP files')
    parser.add_argument('--output-dir', default='./experiments/exp2_complete',
                       help='Base output directory')
    parser.add_argument('--max-messages', type=int, default=1000,
                       help='Maximum messages to process per protocol')
    parser.add_argument('--all-protocols', action='store_true',
                       help='Test all protocols (default: focus on ICS)')
    parser.add_argument('--baseline-json', default=None,
                       help='Path to custom baseline results JSON')
    parser.add_argument('--skip-neupre', action='store_true',
                       help='Skip NeuPRE experiments (use existing results)')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, 'logs', 'experiment2.log'))
        ]
    )
    
    # Setup directories
    print("\n" + "="*80)
    print("EXPERIMENT 2: Multi-Protocol Format Extraction")
    print("Complete Pipeline Runner")
    print("="*80)
    
    dirs = setup_directories(args.output_dir)
    print(f"\nOutput directory: {args.output_dir}")
    
    # Step 1: Run NeuPRE experiments
    if not args.skip_neupre:
        results = run_neupre_experiments(
            data_dir=args.data_dir,
            output_dir=dirs['neupre'],
            max_messages=args.max_messages,
            focus_ics=not args.all_protocols
        )
    else:
        print("\nSkipping NeuPRE experiments (using existing results)")
        # Load existing results
        with open(os.path.join(dirs['neupre'], 'detailed_results.json'), 'r') as f:
            results = json.load(f)
    
    # Step 2: Integrate baselines
    neupre_results_path = os.path.join(dirs['neupre'], 'detailed_results.json')
    complete_csv = integrate_baselines(
        neupre_results_path=neupre_results_path,
        baseline_data_path=args.baseline_json,
        output_dir=dirs['tables']
    )
    
    # Step 3: Generate visualizations
    if not args.skip_viz:
        generate_visualizations(
            csv_path=complete_csv,
            output_dir=dirs['figures']
        )
    else:
        print("\nSkipping visualization generation")
    
    # Step 4: Generate summary report
    generate_summary_report(dirs, results)
    
    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT 2 PIPELINE COMPLETED")
    print("="*80)
    print(f"\nResults Location: {args.output_dir}")
    print("\nKey Outputs:")
    print(f"  📊 Main Table: {os.path.join(dirs['tables'], 'table3_complete.csv')}")
    print(f"  📈 Figures: {dirs['figures']}")
    print(f"  📝 Report: {os.path.join(dirs['base'], 'EXPERIMENT2_REPORT.md')}")
    print("\nFor paper submission:")
    print(f"  1. Copy LaTeX table from: {os.path.join(dirs['tables'], 'table3_complete.tex')}")
    print(f"  2. Select figures from: {dirs['figures']}")
    print("="*80)


if __name__ == '__main__':
    main()