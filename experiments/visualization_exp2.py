"""
Visualization Suite for Experiment 2
Generates publication-quality figures for the paper
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, List


# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Color palette
COLORS = {
    'NeuPRE': '#2E86AB',      # Blue
    'FieldHunter': '#A23B72',  # Purple
    'Netplier': '#F18F01',     # Orange
    'BinaryInferno': '#C73E1D', # Red
    'Netzob': '#6A994E'        # Green
}


def plot_metric_comparison(df: pd.DataFrame, 
                          metric: str,
                          output_path: str,
                          title: str = None):
    """
    Plot comparison of a specific metric across all methods
    
    Args:
        df: DataFrame with results
        metric: 'Acc', 'F1', or 'Perf'
        output_path: Path to save figure
        title: Custom title (optional)
    """
    # Exclude Average row
    plot_df = df[df['Protocol'] != 'Average'].copy()
    
    # Extract metric columns
    methods = ['NeuPRE', 'FieldHunter', 'Netplier', 'BinaryInferno', 'Netzob']
    data = []
    
    for method in methods:
        col_name = f'{method}_{metric}'
        if col_name in plot_df.columns:
            values = plot_df[col_name].values
            data.append(values)
        else:
            data.append(np.zeros(len(plot_df)))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(plot_df))
    width = 0.15
    
    for i, (method, values) in enumerate(zip(methods, data)):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, values, width, label=method, 
                      color=COLORS[method], alpha=0.8)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=7)
    
    # Formatting
    ax.set_xlabel('Protocol', fontweight='bold')
    ax.set_ylabel(f'{metric} Score', fontweight='bold')
    ax.set_title(title or f'{metric} Score Comparison Across Protocols', 
                fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['Protocol'], rotation=45, ha='right')
    ax.legend(loc='upper left', ncol=5, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ {metric} comparison saved to: {output_path}")


def plot_heatmap(df: pd.DataFrame, 
                metric: str,
                output_path: str):
    """
    Create heatmap showing performance across protocols and methods
    """
    # Exclude Average row
    plot_df = df[df['Protocol'] != 'Average'].copy()
    
    # Extract metric columns
    methods = ['NeuPRE', 'FieldHunter', 'Netplier', 'BinaryInferno', 'Netzob']
    heatmap_data = []
    
    for method in methods:
        col_name = f'{method}_{metric}'
        if col_name in plot_df.columns:
            heatmap_data.append(plot_df[col_name].values)
        else:
            heatmap_data.append(np.zeros(len(plot_df)))
    
    heatmap_data = np.array(heatmap_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(plot_df['Protocol'])
    ax.set_yticklabels(methods)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add values in cells
    for i in range(len(methods)):
        for j in range(len(plot_df)):
            value = heatmap_data[i, j]
            text = ax.text(j, i, f'{value:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{metric} Score', rotation=270, labelpad=20)
    
    ax.set_title(f'{metric} Score Heatmap', fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ {metric} heatmap saved to: {output_path}")


def plot_radar_chart(df: pd.DataFrame, 
                    protocol: str,
                    output_path: str):
    """
    Create radar chart comparing all methods on a single protocol
    """
    # Get data for specific protocol
    protocol_row = df[df['Protocol'] == protocol.upper()]
    
    if protocol_row.empty:
        print(f"Warning: Protocol {protocol} not found")
        return
    
    protocol_row = protocol_row.iloc[0]
    
    metrics = ['Acc', 'F1', 'Perf']
    methods = ['NeuPRE', 'FieldHunter', 'Netplier', 'BinaryInferno', 'Netzob']
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Plot each method
    for method in methods:
        values = []
        for metric in metrics:
            col_name = f'{method}_{metric}'
            if col_name in protocol_row and pd.notna(protocol_row[col_name]):
                values.append(protocol_row[col_name])
            else:
                values.append(0)
        
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method, 
               color=COLORS[method], alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=COLORS[method])
    
    # Fix axis to go in the right order
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    ax.set_title(f'{protocol.upper()} Protocol - Method Comparison', 
                fontweight='bold', pad=20, y=1.08)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Radar chart for {protocol} saved to: {output_path}")


def plot_average_comparison(df: pd.DataFrame, output_path: str):
    """
    Create grouped bar chart showing average performance
    """
    # Extract average row
    avg_row = df[df['Protocol'] == 'Average']
    
    if avg_row.empty:
        print("Warning: No average row found")
        return
    
    avg_row = avg_row.iloc[0]
    
    metrics = ['Acc', 'F1', 'Perf']
    methods = ['NeuPRE', 'FieldHunter', 'Netplier', 'BinaryInferno', 'Netzob']
    
    # Prepare data
    data = {method: [] for method in methods}
    
    for metric in metrics:
        for method in methods:
            col_name = f'{method}_{metric}'
            if col_name in avg_row and pd.notna(avg_row[col_name]):
                data[method].append(avg_row[col_name])
            else:
                data[method].append(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.15
    
    for i, method in enumerate(methods):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, data[method], width, label=method,
                     color=COLORS[method], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Average Performance Across All Protocols', 
                fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'F1 Score', 'Perfect Match'])
    ax.legend(loc='upper left', ncol=5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Average comparison saved to: {output_path}")


def plot_improvement_analysis(df: pd.DataFrame, output_path: str):
    """
    Show NeuPRE's improvement over best baseline for each protocol
    """
    # Exclude Average row
    plot_df = df[df['Protocol'] != 'Average'].copy()
    
    methods = ['FieldHunter', 'Netplier', 'BinaryInferno', 'Netzob']
    metrics = ['Acc', 'F1', 'Perf']
    
    improvements = {metric: [] for metric in metrics}
    protocols = plot_df['Protocol'].tolist()
    
    for idx, row in plot_df.iterrows():
        for metric in metrics:
            neupre_value = row[f'NeuPRE_{metric}']
            
            # Find best baseline
            baseline_values = []
            for method in methods:
                col_name = f'{method}_{metric}'
                if col_name in row and pd.notna(row[col_name]):
                    baseline_values.append(row[col_name])
            
            if baseline_values and pd.notna(neupre_value):
                best_baseline = max(baseline_values)
                if best_baseline > 0:
                    improvement = ((neupre_value - best_baseline) / best_baseline) * 100
                else:
                    improvement = 0
            else:
                improvement = 0
            
            improvements[metric].append(improvement)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(protocols))
    width = 0.25
    
    colors_metrics = {'Acc': '#1f77b4', 'F1': '#ff7f0e', 'Perf': '#2ca02c'}
    
    for i, metric in enumerate(metrics):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, improvements[metric], width, 
                     label=metric, color=colors_metrics[metric], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 1:  # Only label significant improvements
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Protocol', fontweight='bold')
    ax.set_ylabel('Improvement over Best Baseline (%)', fontweight='bold')
    ax.set_title('NeuPRE Performance Improvement', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(protocols, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Improvement analysis saved to: {output_path}")


def plot_protocol_type_analysis(df: pd.DataFrame, 
                                protocol_types: Dict[str, str],
                                output_path: str):
    """
    Analyze performance by protocol type (ICS, Network, etc.)
    """
    # Exclude Average row
    plot_df = df[df['Protocol'] != 'Average'].copy()
    
    # Add type column
    plot_df['Type'] = plot_df['Protocol'].map(
        lambda p: protocol_types.get(p.lower(), 'Other')
    )
    
    # Group by type
    type_groups = plot_df.groupby('Type')
    
    methods = ['NeuPRE', 'FieldHunter', 'Netplier', 'BinaryInferno', 'Netzob']
    metrics = ['Acc', 'F1', 'Perf']
    
    # Calculate averages by type
    type_averages = {}
    for type_name, group in type_groups:
        type_averages[type_name] = {}
        for method in methods:
            type_averages[type_name][method] = {}
            for metric in metrics:
                col_name = f'{method}_{metric}'
                if col_name in group.columns:
                    avg = group[col_name].mean()
                    type_averages[type_name][method][metric] = avg
    
    # Create subplot for each metric
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        types = list(type_averages.keys())
        x = np.arange(len(types))
        width = 0.15
        
        for i, method in enumerate(methods):
            values = [type_averages[t][method].get(metric, 0) for t in types]
            offset = (i - 2) * width
            ax.bar(x + offset, values, width, label=method,
                  color=COLORS[method], alpha=0.8)
        
        ax.set_ylabel(f'{metric} Score', fontweight='bold')
        ax.set_title(f'{metric} by Protocol Type', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(types)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Protocol type analysis saved to: {output_path}")


# ==================== Main Visualization Function ====================

def generate_all_figures(csv_path: str, 
                        output_dir: str,
                        protocol_types: Dict[str, str] = None):
    """
    Generate all visualization figures for Experiment 2
    
    Args:
        csv_path: Path to integrated results CSV
        output_dir: Directory to save figures
        protocol_types: Mapping of protocol names to types
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Convert string values to float
    for col in df.columns:
        if col != 'Protocol':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualization figures...")
    
    # 1. Metric comparisons
    for metric in ['Acc', 'F1', 'Perf']:
        plot_metric_comparison(
            df, metric,
            os.path.join(output_dir, f'fig_exp2_{metric.lower()}_comparison.png')
        )
    
    # 2. Heatmaps
    for metric in ['F1']:  # F1 is usually the most important
        plot_heatmap(
            df, metric,
            os.path.join(output_dir, f'fig_exp2_{metric.lower()}_heatmap.png')
        )
    
    # 3. Radar charts for selected protocols
    important_protocols = ['modbus', 'dnp3', 's7comm', 'dhcp', 'dns']
    for protocol in important_protocols:
        if protocol.upper() in df['Protocol'].values:
            plot_radar_chart(
                df, protocol,
                os.path.join(output_dir, f'fig_exp2_radar_{protocol}.png')
            )
    
    # 4. Average comparison
    plot_average_comparison(
        df,
        os.path.join(output_dir, 'fig_exp2_average_comparison.png')
    )
    
    # 5. Improvement analysis
    plot_improvement_analysis(
        df,
        os.path.join(output_dir, 'fig_exp2_improvement_analysis.png')
    )
    
    # 6. Protocol type analysis (if types provided)
    if protocol_types:
        plot_protocol_type_analysis(
            df, protocol_types,
            os.path.join(output_dir, 'fig_exp2_type_analysis.png')
        )
    
    print(f"\n✓ All figures generated in: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Experiment 2 visualizations')
    parser.add_argument('--csv', required=True,
                       help='Path to integrated results CSV')
    parser.add_argument('--output-dir', default='./experiments/exp2_figures',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Define protocol types
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
    
    generate_all_figures(args.csv, args.output_dir, protocol_types)