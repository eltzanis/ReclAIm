import os
import json
import logging
from typing import Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from smolagents import Tool


class ModelPerformanceComparisonTool(Tool):
    name = "model_performance_comparison"
    description = """
    This tool compares model performance between training test results and inference results
    to detect performance degradation over time or on new datasets. It identifies metrics that
    have declined and suggests fine-tuning when performance drops significantly.
    """

    inputs = {
        "training_metrics_path": {
            "type": "string",
            "description": "Path to test_metrics.json file from training evaluation"
        },
        "inference_metrics_path": {
            "type": "string",
            "description": "Path to metrics.json file from inference evaluation"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory where comparison results will be saved"
        },
        "decline_threshold": {
            "type": "number",
            "description": "Threshold percentage for significant performance decline (default: 5%)",
            "required": False,
            "nullable": True
        },
        "class_names": {
            "type": "array",
            "description": "List of class names for better reporting (optional)",
            "required": False,
            "nullable": True
        }
    }

    output_type = "object"

    def forward(
        self,
        training_metrics_path: str,
        inference_metrics_path: str,
        output_dir: str,
        decline_threshold: Optional[float] = 5.0,
        class_names: Optional[List[str]] = None
    ):
        """
        Compare training test metrics with inference metrics to detect performance degradation.

        Args:
            training_metrics_path: Path to test_metrics.json from training
            inference_metrics_path: Path to metrics.json from inference
            output_dir: Directory to save comparison results
            decline_threshold: Percentage threshold for significant decline (default: 5%)
            class_names: List of class names for better reporting

        Returns:
            Dictionary with comparison results and recommendations
        """
        try:
            # Set up logging
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "performance_comparison.log")

            # Get logger and clear existing handlers
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # Create formatter and handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            logging.info("Starting model performance comparison...")
            logging.info(f"Training metrics: {training_metrics_path}")
            logging.info(f"Inference metrics: {inference_metrics_path}")
            logging.info(f"Decline threshold: {decline_threshold}%")

            # Load metrics files
            if not os.path.exists(training_metrics_path):
                raise FileNotFoundError(f"Training metrics file not found: {training_metrics_path}")

            if not os.path.exists(inference_metrics_path):
                raise FileNotFoundError(f"Inference metrics file not found: {inference_metrics_path}")

            with open(training_metrics_path, 'r') as f:
                training_metrics = json.load(f)

            with open(inference_metrics_path, 'r') as f:
                inference_metrics = json.load(f)

            logging.info("Successfully loaded both metrics files")

            # Initialize comparison results
            comparison_results = {
                'global_metrics': {},
                'per_class_metrics': {},
                'summary': {
                    'total_metrics_compared': 0,
                    'metrics_declined': 0,
                    'significant_declines': 0,
                    'max_decline_percentage': 0.0,
                    'overall_performance_status': 'GOOD'
                },
                'recommendations': []
            }

            # Compare global metrics
            global_metric_mappings = {
                'accuracy': 'accuracy',
                'precision_macro': 'precision_macro',
                'recall_macro': 'recall_macro',
                'f1_macro': 'f1_macro'
            }

            logging.info("Comparing global metrics...")

            for training_key, inference_key in global_metric_mappings.items():
                if training_key in training_metrics and inference_key in inference_metrics:
                    training_value = training_metrics[training_key]
                    inference_value = inference_metrics[inference_key]

                    # Calculate percentage change
                    if training_value > 0:
                        percentage_change = ((inference_value - training_value) / training_value) * 100
                        absolute_decline = abs(percentage_change) if percentage_change < 0 else 0
                    else:
                        percentage_change = 0
                        absolute_decline = 0

                    comparison_results['global_metrics'][training_key] = {
                        'training_value': training_value,
                        'inference_value': inference_value,
                        'percentage_change': percentage_change,
                        'absolute_decline': absolute_decline,
                        'is_significant_decline': absolute_decline > decline_threshold,
                        'status': 'DECLINED' if percentage_change < 0 else 'IMPROVED' if percentage_change > 0 else 'UNCHANGED'
                    }

                    comparison_results['summary']['total_metrics_compared'] += 1

                    if percentage_change < 0:
                        comparison_results['summary']['metrics_declined'] += 1

                        if absolute_decline > decline_threshold:
                            comparison_results['summary']['significant_declines'] += 1

                    # Track maximum decline
                    if absolute_decline > comparison_results['summary']['max_decline_percentage']:
                        comparison_results['summary']['max_decline_percentage'] = absolute_decline

                    logging.info(f"{training_key}: {training_value:.4f} → {inference_value:.4f} "
                               f"({percentage_change:+.2f}%)")

            # Determine number of classes from training metrics
            if 'precision_per_class' in training_metrics:
                num_classes = len(training_metrics['precision_per_class'])
            elif 'recall_per_class' in training_metrics:
                num_classes = len(training_metrics['recall_per_class'])
            elif 'f1_per_class' in training_metrics:
                num_classes = len(training_metrics['f1_per_class'])
            else:
                # Try to infer from inference metrics
                class_indices = set()
                for key in inference_metrics.keys():
                    if key.startswith(('precision_class_', 'recall_class_', 'f1_class_')):
                        try:
                            class_idx = int(key.split('_')[-1])
                            class_indices.add(class_idx)
                        except ValueError:
                            continue
                num_classes = len(class_indices) if class_indices else 0

            logging.info(f"Detected {num_classes} classes")

            # Compare per-class metrics
            if num_classes > 0:
                logging.info("Comparing per-class metrics...")

                per_class_mappings = {
                    'precision': 'precision_per_class',
                    'recall': 'recall_per_class',
                    'f1': 'f1_per_class'
                }

                # First, detect what naming convention the inference metrics use
                # Check for both formats: precision_class_0 (generic) and precision_benign (named)
                inference_naming_format = None  # 'generic' or 'named'
                detected_class_names = []

                # Try to detect named format first (e.g., precision_benign, precision_malignant)
                for metric_type in ['precision', 'recall', 'f1']:
                    for key in inference_metrics.keys():
                        if key.startswith(f"{metric_type}_") and not key.endswith(('_macro', '_weighted', '_per_class')):
                            # Extract the class name from the key
                            class_name_from_key = key[len(metric_type)+1:]  # Remove "precision_" prefix
                            if class_name_from_key not in ['macro', 'weighted'] and not class_name_from_key.startswith('class_'):
                                # This is a named format (e.g., "benign", "malignant")
                                inference_naming_format = 'named'
                                if class_name_from_key not in detected_class_names:
                                    detected_class_names.append(class_name_from_key)

                if not inference_naming_format:
                    # Check for generic format (e.g., precision_class_0)
                    for metric_type in ['precision', 'recall', 'f1']:
                        for key in inference_metrics.keys():
                            if key.startswith(f"{metric_type}_class_"):
                                inference_naming_format = 'generic'
                                break

                logging.info(f"Detected inference naming format: {inference_naming_format}")
                if detected_class_names:
                    logging.info(f"Detected class names: {detected_class_names}")

                for metric_name, training_key in per_class_mappings.items():
                    if training_key in training_metrics:
                        comparison_results['per_class_metrics'][metric_name] = {}

                        for class_idx in range(num_classes):
                            if class_idx >= len(training_metrics[training_key]):
                                continue

                            training_value = training_metrics[training_key][class_idx]

                            # Determine the inference key based on the detected format
                            inference_key = None
                            class_name = None

                            if inference_naming_format == 'named' and class_idx < len(detected_class_names):
                                # Use named format
                                class_name = detected_class_names[class_idx]
                                inference_key = f"{metric_name}_{class_name}"
                            elif inference_naming_format == 'generic':
                                # Use generic format
                                if class_names and class_idx < len(class_names):
                                    class_name = class_names[class_idx]
                                else:
                                    class_name = f"class_{class_idx}"
                                inference_key = f"{metric_name}_class_{class_idx}"
                            else:
                                # Fallback: try both formats
                                if class_names and class_idx < len(class_names):
                                    class_name = class_names[class_idx]
                                    inference_key = f"{metric_name}_{class_name}"
                                    if inference_key not in inference_metrics:
                                        inference_key = f"{metric_name}_class_{class_idx}"
                                else:
                                    class_name = f"class_{class_idx}"
                                    inference_key = f"{metric_name}_class_{class_idx}"

                            # Check if the inference key exists
                            if inference_key and inference_key in inference_metrics:
                                inference_value = inference_metrics[inference_key]

                                # Calculate percentage change
                                if training_value > 0:
                                    percentage_change = ((inference_value - training_value) / training_value) * 100
                                    absolute_decline = abs(percentage_change) if percentage_change < 0 else 0
                                else:
                                    percentage_change = 0
                                    absolute_decline = 0

                                comparison_results['per_class_metrics'][metric_name][class_name] = {
                                    'training_value': training_value,
                                    'inference_value': inference_value,
                                    'percentage_change': percentage_change,
                                    'absolute_decline': absolute_decline,
                                    'is_significant_decline': absolute_decline > decline_threshold,
                                    'status': 'DECLINED' if percentage_change < 0 else 'IMPROVED' if percentage_change > 0 else 'UNCHANGED'
                                }

                                comparison_results['summary']['total_metrics_compared'] += 1

                                if percentage_change < 0:
                                    comparison_results['summary']['metrics_declined'] += 1

                                    if absolute_decline > decline_threshold:
                                        comparison_results['summary']['significant_declines'] += 1

                                # Track maximum decline
                                if absolute_decline > comparison_results['summary']['max_decline_percentage']:
                                    comparison_results['summary']['max_decline_percentage'] = absolute_decline

                                logging.info(f"{metric_name} {class_name}: {training_value:.4f} → {inference_value:.4f} "
                                           f"({percentage_change:+.2f}%)")
                            else:
                                logging.warning(f"Could not find inference metric for {metric_name} class {class_idx} (tried key: {inference_key})")

            # Determine overall performance status and generate recommendations
            if comparison_results['summary']['significant_declines'] > 0:
                comparison_results['summary']['overall_performance_status'] = 'SIGNIFICANT_DECLINE'

                comparison_results['recommendations'].append({
                    'type': 'FINE_TUNING_REQUIRED',
                    'priority': 'HIGH',
                    'description': f"Model performance has significantly declined on {comparison_results['summary']['significant_declines']} metrics. Fine-tuning with new data is recommended.",
                    'details': {
                        'declined_metrics': comparison_results['summary']['significant_declines'],
                        'max_decline': comparison_results['summary']['max_decline_percentage'],
                        'threshold': decline_threshold
                    }
                })

                # Identify most problematic classes
                problematic_classes = []
                for metric_name, classes in comparison_results['per_class_metrics'].items():
                    for class_name, results in classes.items():
                        if results['is_significant_decline']:
                            problematic_classes.append({
                                'class': class_name,
                                'metric': metric_name,
                                'decline': results['absolute_decline']
                            })

                if problematic_classes:
                    # Sort by decline percentage
                    problematic_classes.sort(key=lambda x: x['decline'], reverse=True)

                    comparison_results['recommendations'].append({
                        'type': 'FOCUS_ON_CLASSES',
                        'priority': 'MEDIUM',
                        'description': f"Focus fine-tuning efforts on classes with the most significant performance declines.",
                        'details': {
                            'most_problematic_classes': problematic_classes[:3]  # Top 3 most problematic
                        }
                    })

            elif comparison_results['summary']['metrics_declined'] > 0:
                comparison_results['summary']['overall_performance_status'] = 'MINOR_DECLINE'

                comparison_results['recommendations'].append({
                    'type': 'MONITORING_RECOMMENDED',
                    'priority': 'LOW',
                    'description': f"Some metrics have declined but not significantly. Continue monitoring performance.",
                    'details': {
                        'declined_metrics': comparison_results['summary']['metrics_declined'],
                        'max_decline': comparison_results['summary']['max_decline_percentage']
                    }
                })

            else:
                comparison_results['recommendations'].append({
                    'type': 'PERFORMANCE_MAINTAINED',
                    'priority': 'INFO',
                    'description': "Model performance has been maintained or improved. No immediate action required.",
                    'details': {}
                })

            # Log summary
            logging.info("\n" + "="*60)
            logging.info("PERFORMANCE COMPARISON SUMMARY")
            logging.info("="*60)
            logging.info(f"Total metrics compared: {comparison_results['summary']['total_metrics_compared']}")
            logging.info(f"Metrics declined: {comparison_results['summary']['metrics_declined']}")
            logging.info(f"Significant declines (>{decline_threshold}%): {comparison_results['summary']['significant_declines']}")
            logging.info(f"Maximum decline: {comparison_results['summary']['max_decline_percentage']:.2f}%")
            logging.info(f"Overall status: {comparison_results['summary']['overall_performance_status']}")

            logging.info("\nRECOMMENDATIONS:")
            for i, rec in enumerate(comparison_results['recommendations'], 1):
                logging.info(f"{i}. [{rec['priority']}] {rec['description']}")

            # Save detailed comparison report
            report_path = os.path.join(output_dir, "performance_comparison_report.json")
            with open(report_path, 'w') as f:
                json.dump(comparison_results, f, indent=4)

            # Create a summary CSV for easy viewing
            summary_data = []

            # Add global metrics
            for metric_name, results in comparison_results['global_metrics'].items():
                summary_data.append({
                    'metric_type': 'global',
                    'metric_name': metric_name,
                    'class': 'N/A',
                    'training_value': results['training_value'],
                    'inference_value': results['inference_value'],
                    'percentage_change': results['percentage_change'],
                    'absolute_decline': results['absolute_decline'],
                    'is_significant_decline': results['is_significant_decline'],
                    'status': results['status']
                })

            # Add per-class metrics
            for metric_type, classes in comparison_results['per_class_metrics'].items():
                for class_name, results in classes.items():
                    summary_data.append({
                        'metric_type': 'per_class',
                        'metric_name': metric_type,
                        'class': class_name,
                        'training_value': results['training_value'],
                        'inference_value': results['inference_value'],
                        'percentage_change': results['percentage_change'],
                        'absolute_decline': results['absolute_decline'],
                        'is_significant_decline': results['is_significant_decline'],
                        'status': results['status']
                    })

            summary_df = pd.DataFrame(summary_data)
            summary_csv_path = os.path.join(output_dir, "performance_comparison_summary.csv")
            summary_df.to_csv(summary_csv_path, index=False)

            # Create visualization
            self._create_comparison_plots(comparison_results, output_dir, class_names)

            # Return results
            return {
                "status": "success",
                "overall_performance_status": comparison_results['summary']['overall_performance_status'],
                "total_metrics_compared": comparison_results['summary']['total_metrics_compared'],
                "metrics_declined": comparison_results['summary']['metrics_declined'],
                "significant_declines": comparison_results['summary']['significant_declines'],
                "max_decline_percentage": comparison_results['summary']['max_decline_percentage'],
                "fine_tuning_recommended": comparison_results['summary']['significant_declines'] > 0,
                "recommendations": comparison_results['recommendations'],
                "detailed_report_path": report_path,
                "summary_csv_path": summary_csv_path,
                "log_file": log_file,
                "output_dir": output_dir
            }

        except Exception as e:
            logging.error(f"Error during performance comparison: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error_message": str(e),
                "training_metrics_path": training_metrics_path,
                "inference_metrics_path": inference_metrics_path,
                "output_dir": output_dir
            }

    def _create_comparison_plots(self, comparison_results, output_dir, class_names=None):
        """Create visualization plots for the performance comparison."""
        try:
            # Plot 1: Global metrics comparison
            global_metrics = comparison_results['global_metrics']
            if global_metrics:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Bar plot of metric values
                metrics = list(global_metrics.keys())
                training_values = [global_metrics[m]['training_value'] for m in metrics]
                inference_values = [global_metrics[m]['inference_value'] for m in metrics]

                x = np.arange(len(metrics))
                width = 0.35

                ax1.bar(x - width/2, training_values, width, label='Training Test', alpha=0.8)
                ax1.bar(x + width/2, inference_values, width, label='Inference', alpha=0.8)
                ax1.set_xlabel('Metrics')
                ax1.set_ylabel('Value')
                ax1.set_title('Metrics Comparison')
                ax1.set_xticks(x)
                ax1.set_xticklabels(metrics, rotation=45)
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Percentage change plot
                percentage_changes = [global_metrics[m]['percentage_change'] for m in metrics]
                colors = ['red' if pc < 0 else 'green' if pc > 0 else 'gray' for pc in percentage_changes]

                ax2.bar(metrics, percentage_changes, color=colors, alpha=0.7)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax2.axhline(y=-5, color='black', linestyle='--', alpha=0.7, label='Decline Threshold')
                ax2.set_xlabel('Metrics')
                ax2.set_ylabel('Percentage Change (%)')
                ax2.set_title('Performance Change (%)')
                ax2.tick_params(axis='x', rotation=45)
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()

                # Save with explicit path and error handling
                output_path = os.path.join(output_dir, "global_metrics_comparison.png")
                logging.info(f"Saving global metrics plot to: {output_path}")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logging.info(f"Global metrics plot saved successfully")
                plt.close()

            # Plot 2: Per-class metrics heatmap
            per_class_metrics = comparison_results['per_class_metrics']
            if per_class_metrics:
                # Create a heatmap of percentage changes
                metric_types = list(per_class_metrics.keys())
                if metric_types:
                    # Get all class names
                    all_classes = set()
                    for metric_type in metric_types:
                        all_classes.update(per_class_metrics[metric_type].keys())
                    all_classes = sorted(list(all_classes))

                    # Create matrix of percentage changes
                    change_matrix = np.zeros((len(metric_types), len(all_classes)))

                    for i, metric_type in enumerate(metric_types):
                        for j, class_name in enumerate(all_classes):
                            if class_name in per_class_metrics[metric_type]:
                                change_matrix[i, j] = per_class_metrics[metric_type][class_name]['percentage_change']

                    # Create heatmap
                    fig, ax = plt.subplots(figsize=(max(10, len(all_classes)), max(6, len(metric_types))))

                    # Use BoundaryNorm to emphasize the -20% to +20% range
                    # This gives more color variation where it matters most
                    from matplotlib.colors import BoundaryNorm
                    import matplotlib.colors as mcolors

                    # Create boundaries that allocate more color levels to the critical range
                    # Symmetric range: -100 to +100
                    # We want 70% of colors for -20 to +20, and 30% for the outer ranges
                    boundaries = np.concatenate([
                        np.linspace(-100, -20, 16),    # 15 intervals from -100 to -20 (15% of colors)
                        np.linspace(-20, 0, 36)[1:],   # 35 intervals from -20 to 0 (35% of colors)
                        np.linspace(0, 20, 36)[1:],    # 35 intervals from 0 to 20 (35% of colors)
                        np.linspace(20, 100, 16)[1:]   # 15 intervals from 20 to 100 (15% of colors)
                    ])

                    # Get the colormap and create normalization
                    cmap = plt.cm.RdBu
                    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

                    # colormap: RdBu (Red=negative, Blue=positive)
                    # Alternative options: 'coolwarm', 'seismic', 'bwr'
                    im = ax.imshow(change_matrix, cmap=cmap, aspect='auto', norm=norm)

                    # Set ticks and labels with larger fonts
                    ax.set_xticks(np.arange(len(all_classes)))
                    ax.set_yticks(np.arange(len(metric_types)))
                    ax.set_xticklabels(all_classes, rotation=45, ha='right', fontsize=18)
                    ax.set_yticklabels(metric_types, fontsize=18)

                    # Add text annotations 
                    for i in range(len(metric_types)):
                        for j in range(len(all_classes)):
                            value = change_matrix[i, j]
                            label = f'{value:.1f}%'

                            # Choose text color for readability based on value
                            # White text on dark colors (strong negatives/positives), black text on light colors
                            if value < -20:
                                color = "white"
                            elif value > 20:
                                color = "white"
                            else:
                                color = "black"

                            text = ax.text(j, i, label,
                                         ha="center", va="center", color=color, fontsize=16, fontweight='bold')

                    ax.set_title('Per-Class Performance Change (%)', fontsize=22, fontweight='bold')
                    ax.set_xlabel('Classes', fontsize=20)
                    ax.set_ylabel('Metrics', fontsize=20)

                    
                    # BoundaryNorm automatically handles the colorbar rendering correctly
                    cbar = plt.colorbar(im, ax=ax, spacing='proportional')
                    cbar.set_label('Percentage Change (%)', rotation=270, labelpad=25, fontsize=18)

                    # Set explicit colorbar ticks: minimum, zero (center), and maximum
                    tick_values = [-100, 0, 100]
                    cbar.set_ticks(tick_values)
                    cbar.ax.tick_params(labelsize=16)

                    plt.tight_layout()

                    # Save with explicit path and error handling
                    output_path = os.path.join(output_dir, "per_class_metrics_heatmap.png")
                    logging.info(f"Saving heatmap to: {output_path}")
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    logging.info(f"Heatmap saved successfully")
                    plt.close()

        except Exception as e:
            logging.warning(f"Error creating comparison plots: {str(e)}")
