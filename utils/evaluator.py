"""
Evaluation Metrics and Comparison Framework

Implements the three core experiments for comparing NeuPRE with DYNpre:
1. State Coverage Efficiency
2. Field Boundary Accuracy
3. Complex Constraint Inference
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
import json
import os


@dataclass
class SegmentationMetrics:
    """Metrics for field segmentation accuracy"""
    precision: float
    recall: float
    f1_score: float
    perfect_match_rate: float
    avg_boundary_error: float


@dataclass
class StateCoverageMetrics:
    """Metrics for state coverage efficiency"""
    total_messages_sent: int
    unique_states_discovered: int
    messages_to_target_coverage: Optional[int]
    coverage_efficiency: float  # states / messages


@dataclass
class ConstraintInferenceMetrics:
    """Metrics for constraint inference"""
    total_constraints: int
    correctly_inferred: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float


class NeuPREEvaluator:
    """
    Main evaluation class for comparing NeuPRE with baselines.
    """

    def __init__(self, output_dir: str = './evaluation_results'):
        """
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        logging.info(f"NeuPREEvaluator initialized. Output: {output_dir}")

    def evaluate_segmentation_accuracy(self,
                                       predicted_boundaries: List[List[int]],
                                       ground_truth_boundaries: List[List[int]]) -> SegmentationMetrics:
        """
        Evaluate field boundary detection accuracy (Experiment 2).

        Args:
            predicted_boundaries: List of predicted boundaries for each message
            ground_truth_boundaries: List of ground truth boundaries

        Returns:
            Segmentation metrics
        """
        assert len(predicted_boundaries) == len(ground_truth_boundaries)

        total_tp = 0
        total_fp = 0
        total_fn = 0
        perfect_matches = 0
        total_boundary_errors = []

        for pred, gt in zip(predicted_boundaries, ground_truth_boundaries):
            pred_set = set(pred)
            gt_set = set(gt)

            tp = len(pred_set & gt_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            if pred_set == gt_set:
                perfect_matches += 1

            # Calculate boundary errors
            for p in pred:
                min_error = min([abs(p - g) for g in gt]) if gt else float('inf')
                total_boundary_errors.append(min_error)

        # Overall metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        perfect_rate = perfect_matches / len(predicted_boundaries)
        avg_error = np.mean(total_boundary_errors) if total_boundary_errors else 0

        metrics = SegmentationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            perfect_match_rate=perfect_rate,
            avg_boundary_error=avg_error
        )

        logging.info(f"Segmentation Metrics:")
        logging.info(f"  Precision: {precision:.4f}")
        logging.info(f"  Recall: {recall:.4f}")
        logging.info(f"  F1-Score: {f1:.4f}")
        logging.info(f"  Perfect Match Rate: {perfect_rate:.4f}")
        logging.info(f"  Avg Boundary Error: {avg_error:.2f} bytes")

        return metrics

    def evaluate_state_coverage(self,
                               messages_sent_history: List[int],
                               states_discovered_history: List[int],
                               target_coverage: Optional[int] = None) -> StateCoverageMetrics:
        """
        Evaluate state coverage efficiency (Experiment 1).

        Args:
            messages_sent_history: Cumulative message count over time
            states_discovered_history: Cumulative unique states over time
            target_coverage: Target number of states to reach (optional)

        Returns:
            State coverage metrics
        """
        total_messages = messages_sent_history[-1] if messages_sent_history else 0
        total_states = states_discovered_history[-1] if states_discovered_history else 0

        # Find messages needed to reach target coverage
        messages_to_target = None
        if target_coverage is not None:
            for i, states in enumerate(states_discovered_history):
                if states >= target_coverage:
                    messages_to_target = messages_sent_history[i]
                    break

        efficiency = total_states / total_messages if total_messages > 0 else 0

        metrics = StateCoverageMetrics(
            total_messages_sent=total_messages,
            unique_states_discovered=total_states,
            messages_to_target_coverage=messages_to_target,
            coverage_efficiency=efficiency
        )

        logging.info(f"State Coverage Metrics:")
        logging.info(f"  Total Messages: {total_messages}")
        logging.info(f"  Unique States: {total_states}")
        logging.info(f"  Efficiency: {efficiency:.4f} states/message")
        if messages_to_target:
            logging.info(f"  Messages to target: {messages_to_target}")

        return metrics

    def evaluate_constraint_inference(self,
                                     inferred_constraints: List[Dict],
                                     ground_truth_constraints: List[Dict]) -> ConstraintInferenceMetrics:
        """
        Evaluate constraint inference accuracy (Experiment 3).

        Args:
            inferred_constraints: List of inferred constraint dictionaries
            ground_truth_constraints: List of ground truth constraints

        Returns:
            Constraint inference metrics
        """
        # Convert to comparable format
        inferred_set = set()
        for c in inferred_constraints:
            key = (c.get('field_index'), c.get('constraint_type'))
            inferred_set.add(key)

        gt_set = set()
        for c in ground_truth_constraints:
            key = (c.get('field_index'), c.get('constraint_type'))
            gt_set.add(key)

        # Compute metrics
        tp = len(inferred_set & gt_set)
        fp = len(inferred_set - gt_set)
        fn = len(gt_set - inferred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics = ConstraintInferenceMetrics(
            total_constraints=len(gt_set),
            correctly_inferred=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1
        )

        logging.info(f"Constraint Inference Metrics:")
        logging.info(f"  Precision: {precision:.4f}")
        logging.info(f"  Recall: {recall:.4f}")
        logging.info(f"  F1-Score: {f1:.4f}")
        logging.info(f"  Correctly Inferred: {tp}/{len(gt_set)}")

        return metrics

    def compare_methods(self,
                       neupre_metrics: Dict,
                       dynpre_metrics: Dict,
                       metric_type: str) -> Dict:
        """
        Compare NeuPRE with DYNpre for a specific metric type.

        Args:
            neupre_metrics: Metrics from NeuPRE
            dynpre_metrics: Metrics from DYNpre
            metric_type: Type of metric ('segmentation', 'coverage', 'constraint')

        Returns:
            Comparison results
        """
        comparison = {
            'metric_type': metric_type,
            'neupre': neupre_metrics,
            'dynpre': dynpre_metrics,
            'improvement': {}
        }

        if metric_type == 'segmentation':
            comparison['improvement']['f1_score'] = \
                (neupre_metrics.f1_score - dynpre_metrics.f1_score) / dynpre_metrics.f1_score * 100 \
                if dynpre_metrics.f1_score > 0 else 0

            comparison['improvement']['perfect_match_rate'] = \
                (neupre_metrics.perfect_match_rate - dynpre_metrics.perfect_match_rate) / \
                dynpre_metrics.perfect_match_rate * 100 \
                if dynpre_metrics.perfect_match_rate > 0 else 0

        elif metric_type == 'coverage':
            if dynpre_metrics.messages_to_target_coverage and \
               neupre_metrics.messages_to_target_coverage:
                reduction = (dynpre_metrics.messages_to_target_coverage -
                           neupre_metrics.messages_to_target_coverage) / \
                           dynpre_metrics.messages_to_target_coverage * 100
                comparison['improvement']['message_reduction'] = reduction

            comparison['improvement']['efficiency'] = \
                (neupre_metrics.coverage_efficiency - dynpre_metrics.coverage_efficiency) / \
                dynpre_metrics.coverage_efficiency * 100 \
                if dynpre_metrics.coverage_efficiency > 0 else 0

        elif metric_type == 'constraint':
            comparison['improvement']['f1_score'] = \
                (neupre_metrics.f1_score - dynpre_metrics.f1_score) / dynpre_metrics.f1_score * 100 \
                if dynpre_metrics.f1_score > 0 else 0

        return comparison

    def plot_state_coverage_curve(self,
                                 neupre_history: Tuple[List[int], List[int]],
                                 dynpre_history: Tuple[List[int], List[int]],
                                 filename: str = 'state_coverage.png'):
        """
        Plot state coverage efficiency curves (Experiment 1).

        Args:
            neupre_history: (messages, states) for NeuPRE
            dynpre_history: (messages, states) for DYNpre
            filename: Output filename
        """
        plt.figure(figsize=(10, 6))

        neupre_msgs, neupre_states = neupre_history
        dynpre_msgs, dynpre_states = dynpre_history

        plt.plot(neupre_msgs, neupre_states, 'b-', linewidth=2, label='NeuPRE', marker='o', markersize=4)
        plt.plot(dynpre_msgs, dynpre_states, 'r--', linewidth=2, label='DYNpre', marker='s', markersize=4)

        plt.xlabel('Number of Messages Sent', fontsize=12)
        plt.ylabel('Unique States Discovered', fontsize=12)
        plt.title('State Coverage Efficiency Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()

        logging.info(f"State coverage curve saved to {filepath}")

    def plot_segmentation_comparison(self,
                                    neupre_metrics: SegmentationMetrics,
                                    dynpre_metrics: SegmentationMetrics,
                                    filename: str = 'segmentation_comparison.png'):
        """
        Plot segmentation accuracy comparison (Experiment 2).

        Args:
            neupre_metrics: Segmentation metrics for NeuPRE
            dynpre_metrics: Segmentation metrics for DYNpre
            filename: Output filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart for precision, recall, F1
        metrics = ['Precision', 'Recall', 'F1-Score']
        neupre_values = [neupre_metrics.precision, neupre_metrics.recall, neupre_metrics.f1_score]
        dynpre_values = [dynpre_metrics.precision, dynpre_metrics.recall, dynpre_metrics.f1_score]

        x = np.arange(len(metrics))
        width = 0.35

        axes[0].bar(x - width/2, neupre_values, width, label='NeuPRE', color='blue', alpha=0.7)
        axes[0].bar(x + width/2, dynpre_values, width, label='DYNpre', color='red', alpha=0.7)
        axes[0].set_ylabel('Score', fontsize=11)
        axes[0].set_title('Segmentation Accuracy Metrics', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics)
        axes[0].legend()
        axes[0].set_ylim([0, 1.1])
        axes[0].grid(True, alpha=0.3, axis='y')

        # Perfect match rate
        methods = ['NeuPRE', 'DYNpre']
        perfect_rates = [neupre_metrics.perfect_match_rate, dynpre_metrics.perfect_match_rate]

        axes[1].bar(methods, perfect_rates, color=['blue', 'red'], alpha=0.7)
        axes[1].set_ylabel('Perfect Match Rate', fontsize=11)
        axes[1].set_title('Perfect Segmentation Rate', fontsize=12, fontweight='bold')
        axes[1].set_ylim([0, 1.1])
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()

        logging.info(f"Segmentation comparison saved to {filepath}")

    def plot_constraint_inference(self,
                                 neupre_metrics: ConstraintInferenceMetrics,
                                 dynpre_metrics: ConstraintInferenceMetrics,
                                 filename: str = 'constraint_inference.png'):
        """
        Plot constraint inference comparison (Experiment 3).

        Args:
            neupre_metrics: Constraint metrics for NeuPRE
            dynpre_metrics: Constraint metrics for DYNpre
            filename: Output filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Precision, Recall, F1
        metrics = ['Precision', 'Recall', 'F1-Score']
        neupre_values = [neupre_metrics.precision, neupre_metrics.recall, neupre_metrics.f1_score]
        dynpre_values = [dynpre_metrics.precision, dynpre_metrics.recall, dynpre_metrics.f1_score]

        x = np.arange(len(metrics))
        width = 0.35

        axes[0].bar(x - width/2, neupre_values, width, label='NeuPRE', color='blue', alpha=0.7)
        axes[0].bar(x + width/2, dynpre_values, width, label='DYNpre', color='red', alpha=0.7)
        axes[0].set_ylabel('Score', fontsize=11)
        axes[0].set_title('Constraint Inference Accuracy', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics)
        axes[0].legend()
        axes[0].set_ylim([0, 1.1])
        axes[0].grid(True, alpha=0.3, axis='y')

        # Confusion matrix style
        categories = ['Correctly\nInferred', 'False\nPositives', 'False\nNegatives']
        neupre_counts = [neupre_metrics.correctly_inferred,
                        neupre_metrics.false_positives,
                        neupre_metrics.false_negatives]
        dynpre_counts = [dynpre_metrics.correctly_inferred,
                        dynpre_metrics.false_positives,
                        dynpre_metrics.false_negatives]

        x = np.arange(len(categories))
        axes[1].bar(x - width/2, neupre_counts, width, label='NeuPRE', color='blue', alpha=0.7)
        axes[1].bar(x + width/2, dynpre_counts, width, label='DYNpre', color='red', alpha=0.7)
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].set_title('Constraint Inference Breakdown', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(categories)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()

        logging.info(f"Constraint inference comparison saved to {filepath}")

    def generate_report(self,
                       all_comparisons: List[Dict],
                       filename: str = 'evaluation_report.txt'):
        """
        Generate comprehensive evaluation report.

        Args:
            all_comparisons: List of comparison results
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NeuPRE vs DYNpre Evaluation Report\n")
            f.write("=" * 80 + "\n\n")

            for comp in all_comparisons:
                metric_type = comp['metric_type']
                f.write(f"## {metric_type.upper()} METRICS\n\n")

                f.write("NeuPRE Results:\n")
                for key, value in comp['neupre'].__dict__.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

                f.write("DYNpre Results:\n")
                for key, value in comp['dynpre'].__dict__.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

                f.write("Improvement:\n")
                for key, value in comp['improvement'].items():
                    f.write(f"  {key}: {value:+.2f}%\n")
                f.write("\n")
                f.write("-" * 80 + "\n\n")

            f.write("=" * 80 + "\n")

        logging.info(f"Evaluation report saved to {filepath}")

    def save_metrics_json(self, metrics: Dict, filename: str = 'metrics.json'):
        """Save metrics to JSON file"""
        filepath = os.path.join(self.output_dir, filename)

        # Convert dataclass to dict
        def convert(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return obj

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=convert)

        logging.info(f"Metrics saved to {filepath}")
