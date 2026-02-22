import os
import time
import logging
import json
from typing import Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

from smolagents import Tool

# MONAI imports
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50


class MedMNIST3DInferenceDataset(Dataset):
    """Dataset class for 3D medical image inference."""

    def __init__(self, images, labels=None, filenames=None):
        """
        Args:
            images: numpy array of shape (N, 28, 28, 28) or (N, 1, 28, 28, 28)
            labels: optional numpy array of ground truth labels
            filenames: optional list of filenames/identifiers
        """
        self.images = images

        # Ensure images have channel dimension: (N, C, D, H, W)
        if len(self.images.shape) == 4:
            self.images = self.images[:, np.newaxis, :, :, :]

        self.has_labels = labels is not None
        if self.has_labels:
            if len(labels.shape) > 1:
                self.labels = labels.flatten()
            else:
                self.labels = labels
        else:
            self.labels = None

        # Generate filenames if not provided
        if filenames is None:
            self.filenames = [f"image_{i:05d}" for i in range(len(self.images))]
        else:
            self.filenames = filenames

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)

        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        image = torch.from_numpy(image)

        if self.has_labels:
            label = int(self.labels[idx])
            return image, label, self.filenames[idx]
        else:
            return image, self.filenames[idx]


class PyTorch3DResNetInferenceTool(Tool):
    """Inference tool for 3D ResNet models using MONAI."""

    name = "pytorch_3dresnet_inference"
    description = """
    This tool uses a trained PyTorch 3D ResNet model to perform inference on 3D medical images.
    Supports MedMNIST format and can calculate performance metrics if ground truth labels are provided.
    """

    inputs = {
        "data_path": {
            "type": "string",
            "description": "Path to .npz file containing images (and optionally labels)"
        },
        "model_path": {
            "type": "string",
            "description": "Path to the trained model file (.pt format)"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory where prediction results will be saved"
        },
        "config_path": {
            "type": "string",
            "description": "Path to model configuration JSON file (optional)",
            "required": False,
            "nullable": True
        },
        "has_labels": {
            "type": "boolean",
            "description": "Whether the data file contains ground truth labels",
            "required": False,
            "nullable": True
        },
        "class_names": {
            "type": "array",
            "description": "List of class names corresponding to model output indices (optional)",
            "required": False,
            "nullable": True
        },
        "num_classes": {
            "type": "integer",
            "description": "Number of classes (needed if not provided in config file)",
            "required": False,
            "nullable": True
        },
        "model_variant": {
            "type": "string",
            "description": "3D ResNet variant: resnet10, resnet18, resnet34, or resnet50",
            "required": False,
            "nullable": True
        },
        "batch_size": {
            "type": "integer",
            "description": "Batch size for inference",
            "required": False,
            "nullable": True
        },
        "data_key_images": {
            "type": "string",
            "description": "Key for images in .npz file (default: 'images' or 'test_images')",
            "required": False,
            "nullable": True
        },
        "data_key_labels": {
            "type": "string",
            "description": "Key for labels in .npz file (default: 'labels' or 'test_labels')",
            "required": False,
            "nullable": True
        },
        "save_predictions_per_sample": {
            "type": "boolean",
            "description": "Whether to save detailed predictions for each sample",
            "required": False,
            "nullable": True
        }
    }

    output_type = "object"

    def forward(
        self,
        data_path: str,
        model_path: str,
        output_dir: str,
        config_path: Optional[str] = None,
        has_labels: Optional[bool] = None,
        class_names: Optional[List[str]] = None,
        num_classes: Optional[int] = None,
        model_variant: Optional[str] = None,
        batch_size: Optional[int] = 16,
        data_key_images: Optional[str] = None,
        data_key_labels: Optional[str] = None,
        save_predictions_per_sample: Optional[bool] = True
    ):
        """
        Run inference using a trained 3D ResNet model on 3D medical images.

        Returns:
            Dictionary with inference results and file paths
        """
        try:
            # Set up logging
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "inference.log")

            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            # Load model configuration
            model_config = {}

            # If config_path not provided, try to auto-detect from model directory
            if config_path is None:
                # Convert model_path to absolute path for reliable directory detection
                abs_model_path = os.path.abspath(model_path)
                model_dir = os.path.dirname(abs_model_path)
                auto_config_path = os.path.join(model_dir, 'model_config.json')

                logging.info(f"Looking for config file at: {auto_config_path}")

                if os.path.exists(auto_config_path):
                    config_path = auto_config_path
                    logging.info(f"Auto-detected config file: {config_path}")
                else:
                    logging.warning(f"Config file not found at: {auto_config_path}")

            if config_path and os.path.exists(config_path):
                logging.info(f"Loading model configuration from {config_path}")
                try:
                    with open(config_path, 'r') as f:
                        model_config = json.load(f)
                    logging.info(f"Model configuration loaded successfully")
                    logging.info(f"Config contains: pretrained={model_config.get('pretrained', 'not set')}")
                except Exception as e:
                    logging.warning(f"Error loading model configuration: {str(e)}")
            else:
                logging.warning("No model configuration file found. Using default parameters.")

            # Set parameters
            if model_variant is None:
                model_variant = model_config.get('model_variant', 'resnet18')

            if num_classes is None:
                num_classes = model_config.get('num_classes')
                if num_classes is None:
                    if class_names:
                        num_classes = len(class_names)
                    else:
                        raise ValueError("Number of classes must be provided")

            logging.info(f"Starting 3D ResNet inference:")
            logging.info(f"Model path: {model_path}")
            logging.info(f"Model variant: {model_variant}")
            logging.info(f"Number of classes: {num_classes}")
            logging.info(f"Batch size: {batch_size}")
            logging.info(f"Data path: {data_path}")

            # Check device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {device}")

            # Load data
            logging.info(f"Loading data from {data_path}")
            data = np.load(data_path)

            # Try to automatically detect data keys
            available_keys = list(data.keys())
            logging.info(f"Available keys in data file: {available_keys}")

            # Find images - validate provided key or auto-detect
            if data_key_images is not None:
                # If a key is explicitly provided, validate it exists
                if data_key_images not in available_keys:
                    logging.warning(f"Provided data_key_images '{data_key_images}' not found in file. "
                                   f"Available keys: {available_keys}. Falling back to auto-detection.")
                    data_key_images = None  # Trigger auto-detection

            if data_key_images is None:
                # Auto-detect the image key
                if 'test_images' in available_keys:
                    data_key_images = 'test_images'
                elif 'images' in available_keys:
                    data_key_images = 'images'
                elif 'val_images' in available_keys:
                    data_key_images = 'val_images'
                else:
                    raise ValueError(f"Could not find images in data file. Available keys: {available_keys}")

            logging.info(f"Using data key for images: '{data_key_images}'")
            images = data[data_key_images]
            logging.info(f"Loaded images with shape: {images.shape}")

            # Find labels if they exist - validate provided key or auto-detect
            labels = None
            if has_labels is None:
                # Try to auto-detect
                has_labels = any(key in available_keys for key in ['test_labels', 'labels', 'val_labels'])

            if has_labels:
                if data_key_labels is not None:
                    # If a key is explicitly provided, validate it exists
                    if data_key_labels not in available_keys:
                        logging.warning(f"Provided data_key_labels '{data_key_labels}' not found in file. "
                                       f"Available keys: {available_keys}. Falling back to auto-detection.")
                        data_key_labels = None  # Trigger auto-detection

                if data_key_labels is None:
                    # Auto-detect the labels key
                    if 'test_labels' in available_keys:
                        data_key_labels = 'test_labels'
                    elif 'labels' in available_keys:
                        data_key_labels = 'labels'
                    elif 'val_labels' in available_keys:
                        data_key_labels = 'val_labels'
                    else:
                        logging.warning("has_labels=True but could not find labels in file")
                        has_labels = False

                if data_key_labels:
                    logging.info(f"Using data key for labels: '{data_key_labels}'")
                    labels = data[data_key_labels]
                    logging.info(f"Loaded labels with shape: {labels.shape}")

            logging.info(f"Has ground truth: {has_labels}")

            # Create model
            logging.info(f"Creating {model_variant} model architecture...")

            model_dict = {
                'resnet10': resnet10,
                'resnet18': resnet18,
                'resnet34': resnet34,
                'resnet50': resnet50
            }

            if model_variant not in model_dict:
                logging.warning(f"Invalid model variant: {model_variant}. Using resnet18 instead.")
                model_variant = 'resnet18'

            model_fn = model_dict[model_variant]

            # Check if the original model was trained with pretrained weights
            was_pretrained = model_config.get('pretrained', False)

            # Build model kwargs
            model_kwargs = {
                'spatial_dims': 3,
                'n_input_channels': 1,
                'num_classes': num_classes,
                'pretrained': False  # Never use pretrained for inference, we'll load checkpoint
            }

            # If the model was trained with pretrained MedicalNet weights,
            # we need to use the same architecture parameters
            if was_pretrained:
                model_kwargs['feed_forward'] = False
                logging.info("Model was trained with pretrained weights, using feed_forward=False")

                # ResNet50 with pretrained weights requires additional parameters
                if model_variant == 'resnet50':
                    model_kwargs['shortcut_type'] = 'B'
                    model_kwargs['bias_downsample'] = False
                    logging.info("ResNet50 pretrained: using shortcut_type='B' and bias_downsample=False")

            model = model_fn(**model_kwargs)

            # If the model was trained with pretrained weights and feed_forward=False,
            # the final FC layer might be None, so we need to recreate it
            if was_pretrained and model.fc is None:
                # Determine in_features based on model variant
                if model_variant == 'resnet50':
                    in_features = 2048
                elif model_variant in ['resnet34', 'resnet18']:
                    in_features = 512
                elif model_variant == 'resnet10':
                    in_features = 512
                else:
                    in_features = 512  # default fallback

                model.fc = nn.Linear(in_features, num_classes)
                logging.info(f"Recreated final FC layer: {in_features} -> {num_classes} classes")

            # Load trained weights
            logging.info(f"Loading model weights from {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)

                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logging.info(f"Model loaded from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")
                    if 'accuracy' in checkpoint:
                        logging.info(f"Checkpoint accuracy: {checkpoint['accuracy']:.2f}%")
                else:
                    model.load_state_dict(checkpoint)
                    logging.info("Model loaded directly from state dict")
            except Exception as e:
                logging.error(f"Error loading model weights: {str(e)}")
                return {
                    "status": "error",
                    "error_message": f"Failed to load model weights: {str(e)}",
                    "model_path": model_path
                }

            model = model.to(device)
            model.eval()

            # Create dataset
            dataset = MedMNIST3DInferenceDataset(images, labels=labels)

            # Create data loader
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False
            )

            # Perform inference
            all_predictions = []
            all_filenames = []
            all_true_labels = []
            all_pred_labels = []
            all_probabilities = []

            start_time = time.time()

            logging.info("Starting inference...")
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(data_loader):
                    if has_labels:
                        images_batch, labels_batch, filenames_batch = batch_data
                    else:
                        images_batch, filenames_batch = batch_data
                        labels_batch = None

                    images_batch = images_batch.to(device)

                    # Forward pass
                    outputs = model(images_batch)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)

                    # Get predictions
                    _, predicted = torch.max(outputs, 1)

                    # Convert to CPU and numpy
                    pred_labels = predicted.cpu().numpy()
                    probs = probabilities.cpu().numpy()

                    # Store results
                    all_filenames.extend(filenames_batch)
                    all_pred_labels.extend(pred_labels)
                    all_probabilities.extend(probs)

                    if has_labels:
                        all_true_labels.extend(labels_batch.cpu().numpy())

                    # Log progress
                    if (batch_idx + 1) % 10 == 0:
                        logging.info(f"Processed {(batch_idx + 1) * batch_size}/{len(dataset)} images")

            processing_time = time.time() - start_time
            logging.info(f"Inference completed in {processing_time:.2f} seconds")
            logging.info(f"Average time per image: {processing_time/len(dataset):.4f} seconds")

            # Create predictions dataframe
            if save_predictions_per_sample:
                predictions_data = []
                for i in range(len(all_filenames)):
                    filename = all_filenames[i]
                    pred_class = int(all_pred_labels[i])
                    probs = all_probabilities[i]

                    result = {
                        'filename': filename,
                        'predicted_class': pred_class,
                        'confidence': float(probs[pred_class])
                    }

                    # Add true label if available
                    if has_labels:
                        true_label = int(all_true_labels[i])
                        result['true_label'] = true_label
                        result['correct'] = pred_class == true_label

                    # Add class name if available (as supplementary info only)
                    if class_names and pred_class < len(class_names):
                        result['predicted_class_name'] = class_names[pred_class]
                        if has_labels and true_label < len(class_names):
                            result['true_class_name'] = class_names[true_label]

                    # Add probabilities for each class (always use numeric labels)
                    for class_idx in range(len(probs)):
                        result[f'prob_class_{class_idx}'] = float(probs[class_idx])

                    predictions_data.append(result)

                predictions_df = pd.DataFrame(predictions_data)

                # Save predictions
                csv_path = os.path.join(output_dir, "predictions.csv")
                predictions_df.to_csv(csv_path, index=False)
                logging.info(f"Saved predictions to {csv_path}")
            else:
                csv_path = None

            # Calculate metrics if ground truth is available
            metrics = {}
            metrics_path = None
            cm_path = None
            roc_path = None

            if has_labels:
                logging.info("Calculating performance metrics...")

                unique_classes = sorted(list(set(all_true_labels)))

                # Basic metrics
                metrics['accuracy'] = accuracy_score(all_true_labels, all_pred_labels)
                metrics['total_images'] = len(all_true_labels)
                metrics['correct_predictions'] = sum(1 for i in range(len(all_true_labels))
                                                   if all_true_labels[i] == all_pred_labels[i])

                try:
                    # Multi-class metrics
                    metrics['precision_macro'] = precision_score(all_true_labels, all_pred_labels,
                                                               average='macro', zero_division=0)
                    metrics['recall_macro'] = recall_score(all_true_labels, all_pred_labels,
                                                         average='macro', zero_division=0)
                    metrics['f1_macro'] = f1_score(all_true_labels, all_pred_labels,
                                                  average='macro', zero_division=0)

                    metrics['precision_weighted'] = precision_score(all_true_labels, all_pred_labels,
                                                                   average='weighted', zero_division=0)
                    metrics['recall_weighted'] = recall_score(all_true_labels, all_pred_labels,
                                                            average='weighted', zero_division=0)
                    metrics['f1_weighted'] = f1_score(all_true_labels, all_pred_labels,
                                                     average='weighted', zero_division=0)

                    # Per-class metrics
                    class_report = classification_report(all_true_labels, all_pred_labels,
                                                       output_dict=True, zero_division=0)

                    for class_idx in unique_classes:
                        class_key = str(class_idx)
                        if class_key in class_report:
                            class_metrics = class_report[class_key]
                            # Always use numeric class labels for metric keys
                            class_label = f"class_{class_idx}"

                            metrics[f'precision_{class_label}'] = class_metrics['precision']
                            metrics[f'recall_{class_label}'] = class_metrics['recall']
                            metrics[f'f1_{class_label}'] = class_metrics['f1-score']
                            metrics[f'support_{class_label}'] = class_metrics['support']

                            # Sensitivity = recall
                            metrics[f'sensitivity_{class_label}'] = class_metrics['recall']

                            # Specificity
                            y_true_binary = [1 if y == class_idx else 0 for y in all_true_labels]
                            y_pred_binary = [1 if y == class_idx else 0 for y in all_pred_labels]
                            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
                            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                            metrics[f'specificity_{class_label}'] = specificity

                    # Calculate AUC
                    all_probs_array = np.array(all_probabilities)

                    if len(unique_classes) == 2:
                        # Binary classification
                        class_idx = max(unique_classes)
                        y_score = all_probs_array[:, class_idx]

                        metrics['auc'] = roc_auc_score(all_true_labels, y_score)
                        fpr, tpr, _ = roc_curve(all_true_labels, y_score)

                        # Plot ROC curve
                        plt.figure(figsize=(8, 8))
                        plt.plot(fpr, tpr, color='darkorange', lw=2,
                                label=f'ROC curve (area = {metrics["auc"]:.3f})')
                        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('Receiver Operating Characteristic')
                        plt.legend(loc="lower right")
                        plt.grid(True, alpha=0.3)

                        roc_path = os.path.join(output_dir, "roc_curve.png")
                        plt.savefig(roc_path)
                        plt.close()

                    elif len(unique_classes) > 2:
                        # Multi-class - calculate one-vs-rest AUC
                        for class_idx in unique_classes:
                            # Always use numeric class labels for metric keys
                            class_label = f"class_{class_idx}"

                            y_true_binary = [1 if y == class_idx else 0 for y in all_true_labels]
                            y_score = all_probs_array[:, class_idx]

                            try:
                                class_auc = roc_auc_score(y_true_binary, y_score)
                                metrics[f'auc_{class_label}'] = class_auc
                            except Exception as e:
                                logging.warning(f"Could not calculate AUC for {class_label}: {str(e)}")

                        # Overall multi-class AUC
                        try:
                            metrics['auc_macro'] = roc_auc_score(all_true_labels, all_probs_array,
                                                                multi_class='ovr', average='macro')
                        except Exception as e:
                            logging.warning(f"Could not calculate macro AUC: {str(e)}")

                except Exception as e:
                    logging.warning(f"Error calculating some metrics: {str(e)}")

                # Log all metrics
                logging.info("Performance metrics:")
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, float):
                        logging.info(f"  {metric_name}: {metric_value:.4f}")
                    else:
                        logging.info(f"  {metric_name}: {metric_value}")

                # Create confusion matrix
                cm = confusion_matrix(all_true_labels, all_pred_labels, labels=unique_classes)

                # Always use numeric class labels for confusion matrix
                labels = [str(i) for i in unique_classes]

                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                cm_path = os.path.join(output_dir, "confusion_matrix.png")
                plt.savefig(cm_path)
                plt.close()

                # Save metrics
                metrics_path = os.path.join(output_dir, "metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)

                # Metrics summary CSV
                metrics_summary = []
                for metric_name, metric_value in metrics.items():
                    metrics_summary.append({
                        'metric': metric_name,
                        'value': metric_value
                    })
                pd.DataFrame(metrics_summary).to_csv(
                    os.path.join(output_dir, "metrics_summary.csv"), index=False
                )

            # Return results
            return {
                "status": "success",
                "predictions_path": csv_path,
                "log_file": log_file,
                "metrics": metrics if metrics else None,
                "metrics_path": metrics_path,
                "confusion_matrix_path": cm_path,
                "roc_curve_path": roc_path,
                "data_path": data_path,
                "model_path": model_path,
                "output_dir": output_dir,
                "num_images_processed": len(all_filenames),
                "has_ground_truth": has_labels,
                "processing_time_seconds": processing_time,
                "avg_time_per_image_seconds": processing_time / len(dataset),
                "model_type": "3d_resnet",
                "model_variant": model_variant,
                "num_classes": num_classes
            }

        except Exception as e:
            logging.error(f"Error during inference: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error_message": str(e),
                "data_path": data_path,
                "model_path": model_path,
                "output_dir": output_dir
            }
