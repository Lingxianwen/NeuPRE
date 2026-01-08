"""
Baseline Methods Integration for Experiment 2

This module provides templates for integrating baseline methods:
- Netzob (already implemented)
- FieldHunter
- Netplier
- BinaryInferno
- SynRe (optional)

You can add baseline results manually or implement wrappers.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict
import logging


# ==================== Baseline Result Templates ====================

# Example baseline results (you can replace with actual values)
BASELINE_RESULTS = {
    'modbus': {
        'FieldHunter': {'accuracy': 0.5551, 'f1': 0.5484, 'perfect_match': 0.1536},
        'Netplier': {'accuracy': 0.4637, 'f1': 0.4247, 'perfect_match': 0.1361},
        'BinaryInferno': {'accuracy': 0.6901, 'f1': 0.7149, 'perfect_match': 0.1809},
        'Netzob': {'accuracy': 0.3767, 'f1': 0.3976, 'perfect_match': 0.5000}
    },
    'dnp3': {
        'FieldHunter': {'accuracy': 0.4052, 'f1': 0.4374, 'perfect_match': 0.0830},
        'Netplier': {'accuracy': 0.4801, 'f1': 0.4714, 'perfect_match': 0.0922},
        'BinaryInferno': {'accuracy': 0.3683, 'f1': 0.3398, 'perfect_match': 0.0369},
        'Netzob': {'accuracy': 0.4479, 'f1': 0.2470, 'perfect_match': 0.0000}
    },
    's7comm': {
        'FieldHunter': {'accuracy': 0.7844, 'f1': 0.6579, 'perfect_match': 0.3213},
        'Netplier': {'accuracy': 0.8550, 'f1': 0.6920, 'perfect_match': 0.3936},
        'BinaryInferno': {'accuracy': 0.7845, 'f1': 0.6063, 'perfect_match': 0.3085},
        'Netzob': {'accuracy': 0.6604, 'f1': 0.2373, 'perfect_match': 0.0000}
    },
    'dhcp': {
        'FieldHunter': {'accuracy': 0.9498, 'f1': 0.4503, 'perfect_match': 0.1502},
        'Netplier': {'accuracy': 0.9530, 'f1': 0.3464, 'perfect_match': 0.0082},
        'BinaryInferno': {'accuracy': 0.8730, 'f1': 0.1737, 'perfect_match': 0.0159},
        'Netzob': {'accuracy': 0.9690, 'f1': 0.2500, 'perfect_match': 0.0000}
    },
    'dns': {
        'FieldHunter': {'accuracy': 0.7866, 'f1': 0.6196, 'perfect_match': 0.5441},
        'Netplier': {'accuracy': 0.8253, 'f1': 0.5523, 'perfect_match': 0.3330},
        'BinaryInferno': {'accuracy': 0.8382, 'f1': 0.6546, 'perfect_match': 0.5956},
        'Netzob': {'accuracy': 0.7054, 'f1': 0.2500, 'perfect_match': 0.0000}
    },
    'smb': {
        'FieldHunter': {'accuracy': 0.8078, 'f1': 0.4455, 'perfect_match': 0.1447},
        'Netplier': {'accuracy': 0.8552, 'f1': 0.4575, 'perfect_match': 0.0765},
        'BinaryInferno': {'accuracy': 0.7648, 'f1': 0.3409, 'perfect_match': 0.1350},
        'Netzob': {'accuracy': 0.8253, 'f1': 0.2116, 'perfect_match': 0.0000}
    },
    'smb2': {
        'FieldHunter': {'accuracy': 0.8576, 'f1': 0.4005, 'perfect_match': 0.0198},
        'Netplier': {'accuracy': 0.8839, 'f1': 0.3887, 'perfect_match': 0.0743},
        'BinaryInferno': {'accuracy': 0.8098, 'f1': 0.2842, 'perfect_match': 0.0022},
        'Netzob': {'accuracy': 0.9020, 'f1': 0.2418, 'perfect_match': 0.0000}
    }
}


# ==================== Integration Functions ====================

def integrate_baseline_results(neupre_results_path: str,
                               baseline_data: Dict = BASELINE_RESULTS,
                               output_path: str = None) -> pd.DataFrame:
    """
    Integrate baseline results with NeuPRE results
    
    Args:
        neupre_results_path: Path to NeuPRE results JSON
        baseline_data: Dictionary containing baseline results
        output_path: Path to save integrated results
    
    Returns:
        DataFrame with integrated results
    """
    # Load NeuPRE results
    with open(neupre_results_path, 'r') as f:
        neupre_results = json.load(f)
    
    # Build integrated table
    table_rows = []
    
    for protocol in sorted(neupre_results.keys()):
        protocol_upper = protocol.upper()
        neupre_metrics = neupre_results[protocol]['metrics']
        
        row = {
            'Protocol': protocol_upper,
            'NeuPRE_Acc': neupre_metrics['accuracy'],
            'NeuPRE_F1': neupre_metrics['f1'],
            'NeuPRE_Perf': neupre_metrics['perfect_match']
        }
        
        # Add baseline results if available
        if protocol in baseline_data:
            for method in ['FieldHunter', 'Netplier', 'BinaryInferno', 'Netzob']:
                if method in baseline_data[protocol]:
                    baseline_metrics = baseline_data[protocol][method]
                    row[f'{method}_Acc'] = baseline_metrics['accuracy']
                    row[f'{method}_F1'] = baseline_metrics['f1']
                    row[f'{method}_Perf'] = baseline_metrics['perfect_match']
                else:
                    # Fill with placeholders
                    row[f'{method}_Acc'] = np.nan
                    row[f'{method}_F1'] = np.nan
                    row[f'{method}_Perf'] = np.nan
        else:
            # No baseline data for this protocol
            for method in ['FieldHunter', 'Netplier', 'BinaryInferno', 'Netzob']:
                row[f'{method}_Acc'] = np.nan
                row[f'{method}_F1'] = np.nan
                row[f'{method}_Perf'] = np.nan
        
        table_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_rows)
    
    # Calculate averages (ignoring NaN)
    avg_row = {'Protocol': 'Average'}
    for col in df.columns:
        if col != 'Protocol':
            avg_row[col] = df[col].mean()
    
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        logging.info(f"Integrated results saved to: {output_path}")
    
    return df


def format_table_for_publication(df: pd.DataFrame, 
                                 output_dir: str = './experiments/exp2_results'):
    """
    Format table for publication (CSV, LaTeX, Markdown)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Formatted CSV
    formatted_df = df.copy()
    for col in df.columns:
        if col != 'Protocol':
            formatted_df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
    
    csv_path = os.path.join(output_dir, 'table3_complete.csv')
    formatted_df.to_csv(csv_path, index=False)
    print(f"✓ CSV saved to: {csv_path}")
    
    # 2. LaTeX table
    latex_path = os.path.join(output_dir, 'table3_complete.tex')
    with open(latex_path, 'w') as f:
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison with State-of-the-Art Protocol Format Extraction Methods}\n")
        f.write("\\label{tab:format_extraction_complete}\n")
        f.write("\\scriptsize\n")
        f.write("\\begin{tabular}{l|rrr|rrr|rrr|rrr|rrr}\n")
        f.write("\\toprule\n")
        f.write("\\multirow{2}{*}{Protocol} & \\multicolumn{3}{c|}{NeuPRE} & \\multicolumn{3}{c|}{FieldHunter} & \\multicolumn{3}{c|}{Netplier} & \\multicolumn{3}{c|}{BinaryInferno} & \\multicolumn{3}{c}{Netzob} \\\\\n")
        f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13} \\cmidrule(lr){14-16}\n")
        f.write("& Acc. & F1 & Perf. & Acc. & F1 & Perf. & Acc. & F1 & Perf. & Acc. & F1 & Perf. & Acc. & F1 & Perf. \\\\\n")
        f.write("\\midrule\n")
        
        for idx, row in formatted_df.iterrows():
            protocol = row['Protocol']
            if protocol == 'Average':
                f.write("\\midrule\n")
            
            line = f"{protocol} & "
            line += f"{row['NeuPRE_Acc']} & {row['NeuPRE_F1']} & {row['NeuPRE_Perf']} & "
            line += f"{row['FieldHunter_Acc']} & {row['FieldHunter_F1']} & {row['FieldHunter_Perf']} & "
            line += f"{row['Netplier_Acc']} & {row['Netplier_F1']} & {row['Netplier_Perf']} & "
            line += f"{row['BinaryInferno_Acc']} & {row['BinaryInferno_F1']} & {row['BinaryInferno_Perf']} & "
            line += f"{row['Netzob_Acc']} & {row['Netzob_F1']} & {row['Netzob_Perf']} \\\\\n"
            
            f.write(line)
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    
    print(f"✓ LaTeX table saved to: {latex_path}")
    
    # 3. Markdown table
    md_path = os.path.join(output_dir, 'table3_complete.md')
    formatted_df.to_markdown(md_path, index=False)
    print(f"✓ Markdown table saved to: {md_path}")
    
    # 4. Print to console
    print("\n" + "="*150)
    print("Table 3: Comparison with State-of-the-Art Protocol Format Extraction Methods")
    print("="*150)
    print(formatted_df.to_string(index=False))
    print("="*150)


def highlight_best_results(df: pd.DataFrame, output_path: str = None):
    """
    Create a version with best results highlighted
    """
    result_df = df.copy()
    
    # For each protocol (excluding Average)
    protocols = result_df[result_df['Protocol'] != 'Average']
    
    metrics = ['Acc', 'F1', 'Perf']
    methods = ['NeuPRE', 'FieldHunter', 'Netplier', 'BinaryInferno', 'Netzob']
    
    # Add columns for best method
    for metric in metrics:
        best_method_col = []
        improvement_col = []
        
        for idx, row in result_df.iterrows():
            if row['Protocol'] == 'Average':
                best_method_col.append('-')
                improvement_col.append('-')
                continue
            
            # Find best value for this metric
            values = {}
            for method in methods:
                col_name = f'{method}_{metric}'
                if col_name in row and pd.notna(row[col_name]):
                    values[method] = row[col_name]
            
            if values:
                best_method = max(values, key=values.get)
                best_value = values[best_method]
                
                # Calculate improvement of NeuPRE over second best
                if 'NeuPRE' in values:
                    other_values = [v for k, v in values.items() if k != 'NeuPRE']
                    if other_values:
                        second_best = max(other_values)
                        improvement = ((values['NeuPRE'] - second_best) / second_best * 100) if second_best > 0 else 0
                        improvement_col.append(f"+{improvement:.1f}%")
                    else:
                        improvement_col.append('-')
                else:
                    improvement_col.append('-')
                
                best_method_col.append(best_method)
            else:
                best_method_col.append('-')
                improvement_col.append('-')
        
        result_df[f'Best_{metric}'] = best_method_col
        result_df[f'NeuPRE_Improvement_{metric}'] = improvement_col
    
    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"✓ Highlighted results saved to: {output_path}")
    
    return result_df


# ==================== Main Integration Script ====================

def main():
    """
    Main function to integrate baseline results
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrate baseline results with NeuPRE')
    parser.add_argument('--neupre-results', required=True,
                       help='Path to NeuPRE detailed_results.json')
    parser.add_argument('--output-dir', default='./experiments/exp2_results',
                       help='Output directory')
    parser.add_argument('--baseline-json', default=None,
                       help='Path to custom baseline results JSON (optional)')
    
    args = parser.parse_args()
    
    # Load custom baseline data if provided
    if args.baseline_json:
        with open(args.baseline_json, 'r') as f:
            baseline_data = json.load(f)
    else:
        baseline_data = BASELINE_RESULTS
        print("Using default baseline results (example data)")
    
    # Integrate results
    print("Integrating baseline results...")
    df = integrate_baseline_results(
        neupre_results_path=args.neupre_results,
        baseline_data=baseline_data
    )
    
    # Format for publication
    format_table_for_publication(df, output_dir=args.output_dir)
    
    # Highlight best results
    highlight_path = os.path.join(args.output_dir, 'table3_highlighted.csv')
    highlight_best_results(df, output_path=highlight_path)
    
    print(f"\n✓ Integration complete! Check {args.output_dir} for results.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()