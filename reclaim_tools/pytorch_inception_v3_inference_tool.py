import os
import json
import logging
import time
from typing import Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from smolagents import Tool


class MedicalImageDataset(Dataset):
    """Dataset class for medical images that works with various CNN architectures.

    Supports ResNet, VGG16, Inception V3, and EfficientNetV2 models with their
    respective image size requirements.
    """

    def __init__(self, image_dir, labels_file=None, transform=None, is_test=False, image_size=None):
        """
        Initialize the dataset.

        Args:
            image_dir (str): Directory containing the images.
            labels_file (str, optional): Path to CSV file with image names and labels.
            transform (callable, optional): Transform to apply to the images.
            is_test (bool, optional): Whether this is a test dataset without labels.
            image_size (tuple, optional): Size for fallback images when loading fails.
                                         Default sizes by architecture:
                                         - ResNet: (224, 224)
                                         - VGG16: (224, 224)
                                         - Inception V3: (299, 299)
                                         - EfficientNetV2-S: (384, 384)
                                         - EfficientNetV2-M/L: (480, 480)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

        # Default image sizes based on architecture
        if image_size is None:
            # Default to ResNet/VGG size as it's most common
            self.image_size = (224, 224)
        else:
            self.image_size = image_size

        # Log the image size being used
        if hasattr(logging, 'info'):
            logging.info(f"MedicalImageDataset initialized with image_size: {self.image_size}")

        if not is_test and labels_file is not None:
            self.labels_df = pd.read_csv(labels_file)

            # Handle both naming conventions (filename and image_name)
            if 'filename' in self.labels_df.columns:
                self.filename_col = 'filename'
            elif 'image_name' in self.labels_df.columns:
                self.filename_col = 'image_name'
            else:
                raise ValueError("CSV file must contain either 'filename' or 'image_name' column")

            # Ensure filenames don't have directory paths
            self.labels_df[self.filename_col] = self.labels_df[self.filename_col].apply(
                lambda x: os.path.basename(x) if isinstance(x, str) else x
            )

            # Validate labels are numeric
            if 'label' in self.labels_df.columns:
                try:
                    self.labels_df['label'] = pd.to_numeric(self.labels_df['label'], errors='coerce')
                    if self.labels_df['label'].isna().any():
                        raise ValueError("Some labels could not be converted to numeric values")
                except Exception as e:
                    raise ValueError(f"Error processing labels: {e}")

        elif is_test:
            # For test set without labels, just list all images in the directory
            self.image_files = [f for f in os.listdir(image_dir)
                            if os.path.isfile(os.path.join(image_dir, f)) and
                            f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'))]
            if hasattr(logging, 'info'):
                logging.info(f"Found {len(self.image_files)} test images")

    def __len__(self):
        if self.is_test:
            return len(self.image_files)
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.is_test:
            img_name = self.image_files[idx]
            img_path = os.path.join(self.image_dir, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
                # Optionally verify image is not corrupted
                image.verify()
                # Re-open after verify (verify closes the file)
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                if hasattr(logging, 'warning'):
                    logging.warning(f"Error opening image {img_path}: {e}")
                # Create a placeholder image with the specified size
                image = Image.new('RGB', self.image_size, color='black')

            if self.transform:
                image = self.transform(image)

            return image, img_name  # Return filename for prediction output
        else:
            img_name = self.labels_df.iloc[idx][self.filename_col]
            img_path = os.path.join(self.image_dir, img_name)

            try:
                image = Image.open(img_path).convert('RGB')
                # Optionally verify image is not corrupted
                image.verify()
                # Re-open after verify (verify closes the file)
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                # If image can't be opened, create a black image as placeholder
                if hasattr(logging, 'warning'):
                    logging.warning(f"Error opening image {img_path}: {e}")
                image = Image.new('RGB', self.image_size, color='black')

            label = int(self.labels_df.iloc[idx]['label'])

            if self.transform:
                image = self.transform(image)

            return image, label

    def get_class_distribution(self):
        """Get the distribution of classes in the dataset (only for labeled datasets)."""
        if self.is_test or not hasattr(self, 'labels_df'):
            return None

        class_counts = self.labels_df['label'].value_counts().sort_index()
        return class_counts.to_dict()

    def get_sample_weights(self):
        """Calculate sample weights for balanced training (only for labeled datasets)."""
        if self.is_test or not hasattr(self, 'labels_df'):
            return None

        class_counts = self.labels_df['label'].value_counts()
        class_weights = 1.0 / class_counts
        sample_weights = self.labels_df['label'].map(class_weights).values

        return torch.from_numpy(sample_weights).float()


class PyTorchInceptionV3InferenceTool(Tool):
    name = "pytorch_inception_v3_inference"
    description = """
    This tool uses a trained PyTorch Inception V3 model to perform inference on new images.
    It can also calculate performance metrics if ground truth labels are provided.
    """

    inputs = {
        "image_dir": {
            "type": "string",
            "description": "Directory containing images for inference"
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
        "ground_truth_file": {
            "type": "string",
            "description": "CSV file with image filenames and ground truth labels for evaluation (optional)",
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
        "batch_size": {
            "type": "integer",
            "description": "Batch size for inference",
            "required": False,
            "nullable": True
        },
        "case_column": {
            "type": "string",
            "description": "Name of the column in ground truth file that contains case/file identifiers",
            "required": False,
            "nullable": True
        },
        "label_column": {
            "type": "string",
            "description": "Name of the column in ground truth file that contains the labels",
            "required": False,
            "nullable": True
        },
        "file_extension": {
            "type": "string",
            "description": "File extension to add to case IDs if they don't already have one",
            "required": False,
            "nullable": True
        },
        "aux_logits": {
            "type": "boolean",
            "description": "Whether the model was trained with auxiliary logits (Inception specific)",
            "required": False,
            "nullable": True
        }
    }

    output_type = "object"

    def forward(
        self,
        image_dir: str,
        model_path: str,
        output_dir: str,
        config_path: Optional[str] = None,
        ground_truth_file: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        num_classes: Optional[int] = None,
        batch_size: Optional[int] = 32,
        case_column: Optional[str] = "case",
        label_column: Optional[str] = "label",
        file_extension: Optional[str] = ".png",
        aux_logits: Optional[bool] = True
    ):
        """
        Run inference using a trained PyTorch Inception V3 model on new images.

        Args:
            image_dir: Directory containing images for inference
            model_path: Path to the trained model file (.pt format)
            output_dir: Directory to save prediction outputs
            config_path: Path to model configuration JSON file (optional)
            ground_truth_file: CSV file with filenames and ground truth labels (optional)
            class_names: List of class names corresponding to model output indices
            num_classes: Number of classes (needed if not in config file)
            batch_size: Batch size for inference
            case_column: Name of the column in ground truth file with case/file IDs
            label_column: Name of the column in ground truth file with labels
            file_extension: File extension to add to case IDs if needed
            aux_logits: Whether the model was trained with auxiliary logits

        Returns:
            Dictionary with inference results and file paths
        """
        try:
            # Set up logging
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "inference.log")

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

            # Load model configuration if provided
            model_config = {}
            if config_path and os.path.exists(config_path):
                logging.info(f"Loading model configuration from {config_path}")
                try:
                    with open(config_path, 'r') as f:
                        model_config = json.load(f)
                    logging.info(f"Model configuration loaded: {model_config}")
                except Exception as e:
                    logging.warning(f"Error loading model configuration: {str(e)}. Will use provided parameters.")

            # Set parameters, with priority to explicit parameters over config values
            model_type = 'inception_v3'  # Fixed for this tool

            if num_classes is None:
                num_classes = model_config.get('num_classes')
                if num_classes is None:
                    if class_names:
                        num_classes = len(class_names)
                    else:
                        raise ValueError("Number of classes must be provided either directly, in config file, or through class_names")

            # Inception V3 requires 299x299 images
            image_size = model_config.get('image_size', 299)
            inception_image_size = (image_size, image_size)

            # Get aux_logits from config if not provided
            if aux_logits is None:
                aux_logits = model_config.get('aux_logits', True)

            # Log inference settings
            logging.info(f"Starting Inception V3 inference with settings:")
            logging.info(f"Model path: {model_path}")
            logging.info(f"Model type: {model_type}")
            logging.info(f"Number of classes: {num_classes}")
            logging.info(f"Batch size: {batch_size}")
            logging.info(f"Image size: {image_size}x{image_size}")
            logging.info(f"Aux logits: {aux_logits}")
            logging.info(f"Image directory: {image_dir}")
            logging.info(f"Has ground truth: {ground_truth_file is not None}")

            # Check if CUDA is available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {device}")

            # Define image transformation (Inception V3 requires 299x299 input)
            transform = transforms.Compose([
                transforms.Resize(inception_image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # Create model
            logging.info(f"Creating Inception V3 model architecture...")

            # Create Inception V3 model
            model = models.inception_v3(pretrained=False, aux_logits=aux_logits)

            # Modify the classifiers for our number of classes
            # Main classifier
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

            # Auxiliary classifier (if used)
            if aux_logits:
                in_features_aux = model.AuxLogits.fc.in_features
                model.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)

            # Load trained weights
            logging.info(f"Loading model weights from {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)

                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logging.info(f"Model loaded from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")
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

            # Move model to device and set to evaluation mode
            model = model.to(device)
            model.eval()

            # Create dataset for inference
            if ground_truth_file and os.path.exists(ground_truth_file):
                # If ground truth is provided, use MedicalImageDataset with labels
                logging.info(f"Using ground truth file: {ground_truth_file}")

                # Read and prepare ground truth file
                gt_df = pd.read_csv(ground_truth_file)

                # Create a temporary CSV with standardized column names
                temp_labels_file = os.path.join(output_dir, "temp_labels.csv")

                # Standardize column names
                if case_column in gt_df.columns:
                    gt_df['filename'] = gt_df[case_column]
                if label_column in gt_df.columns:
                    gt_df['label'] = gt_df[label_column]

                # Ensure filenames have proper extensions
                gt_df['filename'] = gt_df['filename'].apply(
                    lambda x: x if any(str(x).lower().endswith(ext) for ext in
                                     ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'])
                             else f"{x}{file_extension}"
                )

                # Save temporary labels file
                gt_df[['filename', 'label']].to_csv(temp_labels_file, index=False)

                dataset = MedicalImageDataset(
                    image_dir=image_dir,
                    labels_file=temp_labels_file,
                    transform=transform,
                    is_test=False,
                    image_size=inception_image_size
                )
                has_labels = True
            else:
                # No ground truth, use MedicalImageDataset in test mode
                logging.info("No ground truth provided, running inference only")
                dataset = MedicalImageDataset(
                    image_dir=image_dir,
                    transform=transform,
                    is_test=True,
                    image_size=inception_image_size
                )
                has_labels = False

            # Create data loader
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False
            )

            # Process images and collect predictions
            all_predictions = []
            all_filenames = []
            all_true_labels = []
            all_pred_labels = []
            all_probabilities = []

            start_time = time.time()

            with torch.no_grad():
                for batch_idx, batch_data in enumerate(data_loader):
                    if has_labels:
                        images, labels = batch_data
                        # Get filenames for this batch
                        batch_filenames = []
                        for i in range(batch_idx * batch_size,
                                     min((batch_idx + 1) * batch_size, len(dataset))):
                            batch_filenames.append(dataset.labels_df.iloc[i]['filename'])
                    else:
                        images, batch_filenames = batch_data
                        labels = None

                    # Move images to device
                    images = images.to(device)

                    try:
                        # Forward pass - need to handle the aux_logits
                        outputs = model(images)

                        # Inception V3 in eval mode returns a tuple if aux_logits=True
                        if isinstance(outputs, tuple):
                            # During evaluation, we only use the main output (index 0)
                            outputs = outputs[0]

                        probabilities = torch.nn.functional.softmax(outputs, dim=1)

                        # Get predictions
                        _, predicted = torch.max(outputs, 1)

                        # Convert to CPU and numpy
                        pred_labels = predicted.cpu().numpy()
                        probs = probabilities.cpu().numpy()

                        # Store results
                        all_filenames.extend(batch_filenames)
                        all_pred_labels.extend(pred_labels)
                        all_probabilities.extend(probs)

                        if has_labels:
                            all_true_labels.extend(labels.cpu().numpy())

                    except Exception as e:
                        logging.error(f"Error during inference for batch {batch_idx}: {str(e)}")
                        # Skip this batch
                        continue

                    # Log progress
                    if (batch_idx + 1) % 10 == 0:
                        logging.info(f"Processed {(batch_idx + 1) * batch_size}/{len(dataset)} images")

            processing_time = time.time() - start_time
            logging.info(f"Inference completed in {processing_time:.2f} seconds")

            # Create predictions dataframe
            predictions_data = []
            for i in range(len(all_filenames)):
                filename = all_filenames[i]
                pred_class = int(all_pred_labels[i])
                probs = all_probabilities[i]

                # Extract case ID from filename
                case_id = os.path.splitext(filename)[0]

                result = {
                    'filename': filename,
                    'case_id': case_id,
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

            # Save predictions to CSV
            csv_path = os.path.join(output_dir, "predictions.csv")
            predictions_df.to_csv(csv_path, index=False)
            logging.info(f"Saved predictions to {csv_path}")

            # Calculate metrics if ground truth is available
            metrics = {}
            metrics_path = None
            cm_path = None
            roc_path = None

            if has_labels:
                logging.info("Calculating performance metrics...")

                # Get unique classes in sorted order
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

                    # Per-class metrics
                    class_report = classification_report(all_true_labels, all_pred_labels,
                                                       output_dict=True, zero_division=0)

                    # Extract per-class metrics
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

                            # Calculate sensitivity and specificity for each class
                            # Sensitivity = recall
                            metrics[f'sensitivity_{class_label}'] = class_metrics['recall']

                            # Specificity = TN / (TN + FP)
                            y_true_binary = [1 if y == class_idx else 0 for y in all_true_labels]
                            y_pred_binary = [1 if y == class_idx else 0 for y in all_pred_labels]
                            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
                            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                            metrics[f'specificity_{class_label}'] = specificity

                    # Calculate AUC and ROC curves if we have probability outputs
                    if len(unique_classes) == 2:  # Binary classification
                        # Extract probabilities for positive class
                        class_idx = max(unique_classes)  # Assuming 1 is positive class in binary case
                        y_score = [probs[class_idx] for probs in all_probabilities]

                        # Calculate ROC curve and AUC
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

                    elif len(unique_classes) > 2:  # Multi-class
                        # For multi-class, calculate one-vs-rest AUC for each class
                        for class_idx in unique_classes:
                            # Always use numeric class labels for metric keys
                            class_label = f"class_{class_idx}"

                            # Create binary labels for this class
                            y_true_binary = [1 if y == class_idx else 0 for y in all_true_labels]
                            y_score = [probs[class_idx] for probs in all_probabilities]

                            try:
                                # Calculate AUC
                                class_auc = roc_auc_score(y_true_binary, y_score)
                                metrics[f'auc_{class_label}'] = class_auc
                            except Exception as e:
                                logging.warning(f"Could not calculate AUC for {class_label}: {str(e)}")

                except Exception as e:
                    logging.warning(f"Error calculating some metrics: {str(e)}")

                # Log all metrics
                logging.info("Performance metrics:")
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, float):
                        logging.info(f"{metric_name}: {metric_value:.4f}")
                    else:
                        logging.info(f"{metric_name}: {metric_value}")

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

                # Save metrics to JSON
                metrics_path = os.path.join(output_dir, "metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)

                # Create a readable metrics summary as CSV
                metrics_summary = []
                for metric_name, metric_value in metrics.items():
                    metrics_summary.append({
                        'metric': metric_name,
                        'value': metric_value
                    })
                pd.DataFrame(metrics_summary).to_csv(
                    os.path.join(output_dir, "metrics_summary.csv"), index=False
                )

                # Clean up temporary files
                if 'temp_labels_file' in locals() and os.path.exists(temp_labels_file):
                    os.remove(temp_labels_file)

            # Return results
            return {
                "status": "success",
                "predictions_path": csv_path,
                "log_file": log_file,
                "metrics": metrics if metrics else None,
                "metrics_path": metrics_path,
                "confusion_matrix_path": cm_path,
                "roc_curve_path": roc_path,
                "image_dir": image_dir,
                "model_path": model_path,
                "output_dir": output_dir,
                "num_images_processed": len(all_filenames),
                "has_ground_truth": has_labels,
                "processing_time_seconds": processing_time,
                "model_type": "inception_v3",
                "aux_logits_used": aux_logits
            }

        except Exception as e:
            logging.error(f"Error during inference: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error_message": str(e),
                "image_dir": image_dir,
                "model_path": model_path,
                "output_dir": output_dir
            }
