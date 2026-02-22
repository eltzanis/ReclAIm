import os
import json
import logging
import time
from typing import Optional, List, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from smolagents import Tool


class TransformSubset(Dataset):
    """Wrapper to apply transforms to a Subset dataset.

    This is needed because torch.utils.data.Subset doesn't allow
    overriding transforms, so we need to wrap it to apply validation
    transforms separately from training transforms.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        # Expose targets for compatibility with class distribution code
        if hasattr(subset, 'dataset'):
            if hasattr(subset.dataset, 'targets'):
                self.targets = [subset.dataset.targets[i] for i in subset.indices]
            elif hasattr(subset.dataset, 'labels'):
                self.targets = [subset.dataset.labels[i] for i in subset.indices]

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


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


class ResampledDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, resampled_indices, resampled_labels):
        self.original_dataset = original_dataset
        self.resampled_indices = resampled_indices
        self.resampled_labels = resampled_labels

    def __len__(self):
        return len(self.resampled_indices)

    def __getitem__(self, idx):
        original_idx = self.resampled_indices[idx]
        # Get the original sample but use the resampled label
        if hasattr(self.original_dataset, '__getitem__'):
            sample, _ = self.original_dataset[original_idx]
            return sample, self.resampled_labels[idx]
        else:
            return self.original_dataset[original_idx]


class PyTorchInceptionV3TrainingTool(Tool):
    name = "pytorch_inception_v3_training"
    description = """
    This tool trains an Inception V3 model using PyTorch for medical image classification.
    It can train from scratch or fine-tune a pre-trained model, and includes validation metrics
    with configurable data augmentation and comprehensive class imbalance handling for both
    binary and multiclass classification problems.
    """

    inputs = {
        "data_dir": {
            "type": "string",
            "description": "Directory containing dataset with training and validation folders"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory where the trained model and results will be saved"
        },
        "num_classes": {
            "type": "integer",
            "description": "Number of classes for classification"
        },
        "num_epochs": {
            "type": "integer",
            "description": "Number of training epochs",
            "required": False,
            "nullable": True
        },
        "batch_size": {
            "type": "integer",
            "description": "Batch size for training",
            "required": False,
            "nullable": True
        },
        "pretrained": {
            "type": "boolean",
            "description": "Whether to use pretrained weights",
            "required": False,
            "nullable": True
        },
        "early_stopping": {
            "type": "boolean",
            "description": "Whether to use early stopping",
            "required": False,
            "nullable": True
        },
        "patience": {
            "type": "integer",
            "description": "Number of epochs to wait before early stopping",
            "required": False,
            "nullable": True
        },
        "aux_logits": {
            "type": "boolean",
            "description": "Whether to use auxiliary logits during training (Inception specific)",
            "required": False,
            "nullable": True
        },
        "augmentation_level": {
            "type": "string",
            "description": "Level of augmentation: 'none', 'basic', 'moderate', 'heavy', or 'custom'",
            "required": False,
            "nullable": True
        },
        "custom_augmentations": {
            "type": "object",
            "description": "Custom augmentation settings when augmentation_level is 'custom'",
            "required": False,
            "nullable": True
        },
        # Class imbalance handling parameters
        "handle_class_imbalance": {
            "type": "boolean",
            "description": "Whether to apply class imbalance handling techniques",
            "required": False,
            "nullable": True
        },
        "imbalance_strategy": {
            "type": "string",
            "description": "Strategy for handling class imbalance: 'weighted_loss', 'focal_loss', 'oversampling', 'undersampling'",
            "required": False,
            "nullable": True
        },
        "focal_loss_alpha": {
            "type": "number",
            "description": "Alpha parameter for focal loss (class weighting factor)",
            "required": False,
            "nullable": True
        },
        "focal_loss_gamma": {
            "type": "number",
            "description": "Gamma parameter for focal loss (focusing parameter)",
            "required": False,
            "nullable": True
        },
        "sampling_strategy": {
            "type": "string",
            "description": "Sampling strategy for over/undersampling: 'auto', 'minority', 'majority', 'not_minority', 'not_majority', or dict",
            "required": False,
            "nullable": True
        },
        "class_weights": {
            "type": "object",
            "description": "Custom class weights as dictionary {class_id: weight} or 'balanced' for automatic calculation",
            "required": False,
            "nullable": True
        },
        "evaluation_metric": {
            "type": "string",
            "description": "Primary metric for model selection: 'accuracy', 'f1_macro', 'f1_weighted', 'balanced_accuracy', 'auc_roc'",
            "required": False,
            "nullable": True
        }
    }

    output_type = "object"

    def forward(
        self,
        data_dir: str,
        output_dir: str,
        num_classes: int,
        num_epochs: Optional[int] = 10,
        batch_size: Optional[int] = 16,
        pretrained: Optional[bool] = True,
        early_stopping: Optional[bool] = True,
        patience: Optional[int] = 5,
        aux_logits: Optional[bool] = True,
        augmentation_level: Optional[str] = "basic",
        custom_augmentations: Optional[dict] = None,
        # Class imbalance parameters
        handle_class_imbalance: Optional[bool] = False,
        imbalance_strategy: Optional[str] = "weighted_loss",
        focal_loss_alpha: Optional[float] = 1.0,
        focal_loss_gamma: Optional[float] = 2.0,
        sampling_strategy: Optional[str] = "auto",
        class_weights: Optional[Union[dict, str]] = None,
        evaluation_metric: Optional[str] = "accuracy"
    ):
        """
        Train an Inception V3 model for medical image classification with class imbalance handling.

        Args:
            data_dir: Directory containing dataset with train and val subdirectories
            output_dir: Directory to save model and results
            num_classes: Number of classes for classification
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            pretrained: Whether to use pretrained weights
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait before early stopping
            aux_logits: Whether to use auxiliary logits during training (Inception specific)
            augmentation_level: Level of augmentation ('none', 'basic', 'moderate', 'heavy', 'custom')
            custom_augmentations: Dictionary with custom augmentation settings
            handle_class_imbalance: Whether to apply class imbalance handling
            imbalance_strategy: Strategy for handling imbalance
            focal_loss_alpha: Alpha parameter for focal loss
            focal_loss_gamma: Gamma parameter for focal loss
            sampling_strategy: Strategy for sampling techniques
            class_weights: Custom class weights or 'balanced'
            evaluation_metric: Primary metric for model selection

        Returns:
            Dictionary with training results and model paths
        """
        try:
            # Define the image size for Inception V3
            inception_image_size = (299, 299)

            # Set up logging
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "training.log")

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

            # Log configuration
            logging.info(f"Starting Inception V3 training with configuration:")
            logging.info(f"Number of classes: {num_classes}")
            logging.info(f"Pretrained: {pretrained}")
            logging.info(f"Early stopping: {early_stopping}")
            logging.info(f"Auxiliary logits: {aux_logits}")
            logging.info(f"Augmentation level: {augmentation_level}")
            logging.info(f"Handle class imbalance: {handle_class_imbalance}")
            if handle_class_imbalance:
                logging.info(f"Imbalance strategy: {imbalance_strategy}")
                logging.info(f"Evaluation metric: {evaluation_metric}")
            if early_stopping:
                logging.info(f"Patience: {patience}")
            logging.info(f"Data directory: {data_dir}")

            # Check if CUDA is available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {device}")

            # Define Focal Loss class
            class FocalLoss(nn.Module):
                def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
                    super(FocalLoss, self).__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                    self.reduction = reduction

                def forward(self, inputs, targets):
                    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                    pt = torch.exp(-ce_loss)
                    focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

                    if self.reduction == 'mean':
                        return focal_loss.mean()
                    elif self.reduction == 'sum':
                        return focal_loss.sum()
                    else:
                        return focal_loss

            # Define augmentation transforms (keeping original implementation)
            def get_augmentation_transforms(level, img_size, custom_settings=None):
                """Get augmentation transforms based on the specified level."""

                if level == "none":
                    return transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

                elif level == "basic":
                    return transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(degrees=10),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

                elif level == "moderate":
                    return transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(degrees=15),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

                elif level == "heavy":
                    return transforms.Compose([
                        transforms.Resize(int(img_size[0] * 1.1)),
                        transforms.RandomCrop(img_size),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.3),
                        transforms.RandomRotation(degrees=20),
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
                        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                        transforms.RandomGrayscale(p=0.1),
                        transforms.ToTensor(),
                        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

                elif level == "custom" and custom_settings:
                    aug_list = [transforms.Resize(img_size)]

                    if custom_settings.get('horizontal_flip', False):
                        aug_list.append(transforms.RandomHorizontalFlip(p=custom_settings.get('horizontal_flip_p', 0.5)))

                    if custom_settings.get('vertical_flip', False):
                        aug_list.append(transforms.RandomVerticalFlip(p=custom_settings.get('vertical_flip_p', 0.5)))

                    if custom_settings.get('rotation', False):
                        aug_list.append(transforms.RandomRotation(degrees=custom_settings.get('rotation_degrees', 15)))

                    if custom_settings.get('color_jitter', False):
                        aug_list.append(transforms.ColorJitter(
                            brightness=custom_settings.get('brightness', 0.2),
                            contrast=custom_settings.get('contrast', 0.2),
                            saturation=custom_settings.get('saturation', 0.2),
                            hue=custom_settings.get('hue', 0.1)
                        ))

                    if custom_settings.get('random_crop', False):
                        aug_list.insert(0, transforms.Resize(int(img_size[0] * 1.1)))
                        aug_list.append(transforms.RandomCrop(img_size))

                    if custom_settings.get('affine', False):
                        aug_list.append(transforms.RandomAffine(
                            degrees=custom_settings.get('affine_degrees', 0),
                            translate=custom_settings.get('affine_translate', (0.1, 0.1)),
                            scale=custom_settings.get('affine_scale', (0.9, 1.1)),
                            shear=custom_settings.get('affine_shear', 0)
                        ))

                    if custom_settings.get('perspective', False):
                        aug_list.append(transforms.RandomPerspective(
                            distortion_scale=custom_settings.get('perspective_distortion', 0.2),
                            p=custom_settings.get('perspective_p', 0.5)
                        ))

                    if custom_settings.get('grayscale', False):
                        aug_list.append(transforms.RandomGrayscale(p=custom_settings.get('grayscale_p', 0.1)))

                    if custom_settings.get('gaussian_blur', False):
                        aug_list.append(transforms.GaussianBlur(
                            kernel_size=custom_settings.get('blur_kernel', 3),
                            sigma=custom_settings.get('blur_sigma', (0.1, 2.0))
                        ))

                    aug_list.extend([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

                    if custom_settings.get('random_erasing', False):
                        aug_list.append(transforms.RandomErasing(
                            p=custom_settings.get('erasing_p', 0.1),
                            scale=custom_settings.get('erasing_scale', (0.02, 0.1))
                        ))

                    return transforms.Compose(aug_list)

                else:
                    logging.warning(f"Unknown augmentation level: {level}. Using 'basic'.")
                    return get_augmentation_transforms("basic", img_size)

            # Get training and validation transforms
            train_transform = get_augmentation_transforms(augmentation_level, inception_image_size, custom_augmentations)
            val_transform = transforms.Compose([
                transforms.Resize(inception_image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # Setup directories
            train_dir = os.path.join(data_dir, 'train')
            val_dir = os.path.join(data_dir, 'val')
            test_dir = os.path.join(data_dir, 'test')

            if not os.path.exists(train_dir):
                raise ValueError(f"Training directory not found: {train_dir}")

            # Check if validation directory exists
            use_train_val_split = not os.path.exists(val_dir)

            # Check if data directory has a labels.csv file
            train_labels_file = os.path.join(data_dir, 'labels.csv')
            use_csv_labels = os.path.exists(train_labels_file)

            # Create datasets
            if use_csv_labels:
                logging.info(f"Using labels from CSV file: {train_labels_file}")
                if use_train_val_split:
                    # Create base dataset without transforms for splitting
                    base_dataset = MedicalImageDataset(
                        image_dir=train_dir,
                        labels_file=train_labels_file,
                        transform=None,
                        image_size=inception_image_size
                    )
                    # Split into train and val
                    train_size = int(0.8 * len(base_dataset))
                    val_size = len(base_dataset) - train_size
                    train_subset, val_subset = torch.utils.data.random_split(
                        base_dataset, [train_size, val_size]
                    )
                    # Apply appropriate transforms to each subset
                    train_dataset = TransformSubset(train_subset, transform=train_transform)
                    val_dataset = TransformSubset(val_subset, transform=val_transform)
                else:
                    train_dataset = MedicalImageDataset(
                        image_dir=train_dir,
                        labels_file=train_labels_file,
                        transform=train_transform,
                        image_size=inception_image_size
                    )
                    val_labels_file = os.path.join(data_dir, 'val_labels.csv')
                    if not os.path.exists(val_labels_file):
                        val_labels_file = train_labels_file

                    val_dataset = MedicalImageDataset(
                        image_dir=val_dir,
                        labels_file=val_labels_file,
                        transform=val_transform,
                        image_size=inception_image_size
                    )
            else:
                logging.info("Using directory structure for class labels")
                if use_train_val_split:
                    # Create base dataset without transforms for splitting
                    base_dataset = datasets.ImageFolder(
                        root=train_dir,
                        transform=None
                    )
                    # Split into train and val
                    train_size = int(0.8 * len(base_dataset))
                    val_size = len(base_dataset) - train_size
                    train_subset, val_subset = torch.utils.data.random_split(
                        base_dataset, [train_size, val_size]
                    )
                    # Apply appropriate transforms to each subset
                    train_dataset = TransformSubset(train_subset, transform=train_transform)
                    val_dataset = TransformSubset(val_subset, transform=val_transform)
                else:
                    train_dataset = datasets.ImageFolder(
                        root=train_dir,
                        transform=train_transform
                    )
                    val_dataset = datasets.ImageFolder(
                        root=val_dir,
                        transform=val_transform
                    )

            logging.info(f"Training dataset size: {len(train_dataset)}")
            logging.info(f"Validation dataset size: {len(val_dataset)}")

            # Get class distribution and handle class imbalance
            def get_class_distribution(dataset):
                """Get class distribution from dataset."""
                if hasattr(dataset, 'targets'):
                    return np.bincount(dataset.targets)
                elif hasattr(dataset, 'dataset'):
                    base_dataset = dataset.dataset
                    if hasattr(base_dataset, 'targets'):
                        indices = dataset.indices
                        targets = np.array(base_dataset.targets)[indices]
                        return np.bincount(targets)
                    elif hasattr(base_dataset, 'labels'):
                        indices = dataset.indices
                        targets = np.array(base_dataset.labels)[indices]
                        return np.bincount(targets)
                elif hasattr(dataset, 'labels'):
                    return np.bincount(dataset.labels)
                return None

            # Analyze class distribution
            train_class_dist = get_class_distribution(train_dataset)
            if train_class_dist is not None:
                logging.info(f"Training class distribution: {train_class_dist}")

                # Calculate imbalance ratio
                max_class_count = np.max(train_class_dist)
                min_class_count = np.min(train_class_dist)
                imbalance_ratio = max_class_count / min_class_count
                logging.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")

                # Recommend enabling imbalance handling if ratio > 2
                if not handle_class_imbalance and imbalance_ratio > 2.0:
                    logging.warning(f"High class imbalance detected (ratio: {imbalance_ratio:.2f}). "
                                  "Consider enabling handle_class_imbalance=True")

            # Initialize variables for sampling
            original_train_dataset = train_dataset
            sampled_train_dataset = None

            # Apply class imbalance handling if enabled
            if handle_class_imbalance and train_class_dist is not None:
                logging.info(f"Applying class imbalance handling with strategy: {imbalance_strategy}")

                # Extract features and labels for sampling techniques
                if imbalance_strategy in ['oversampling', 'undersampling']:
                    # Get all training samples
                    train_samples = []
                    train_labels = []

                    if hasattr(train_dataset, 'samples'):
                        train_samples = [sample[0] for sample in train_dataset.samples]
                        train_labels = [sample[1] for sample in train_dataset.samples]
                    elif hasattr(train_dataset, 'dataset'):
                        base_dataset = train_dataset.dataset
                        indices = train_dataset.indices
                        if hasattr(base_dataset, 'samples'):
                            train_samples = [base_dataset.samples[i][0] for i in indices]
                            train_labels = [base_dataset.samples[i][1] for i in indices]
                        elif hasattr(base_dataset, 'image_paths'):
                            train_samples = [base_dataset.image_paths[i] for i in indices]
                            train_labels = [base_dataset.labels[i] for i in indices]

                    # Create a simplified feature representation
                    X_train = np.arange(len(train_samples)).reshape(-1, 1)
                    y_train = np.array(train_labels)

                    # Apply sampling strategy
                    if imbalance_strategy == 'oversampling':
                        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
                        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

                    elif imbalance_strategy == 'undersampling':
                        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
                        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

                    # Create new dataset with resampled indices
                    resampled_indices = X_resampled.flatten()
                    sampled_train_dataset = ResampledDataset(train_dataset, resampled_indices, y_resampled)
                    train_dataset = sampled_train_dataset

                    # Log resampling results
                    resampled_dist = np.bincount(y_resampled)
                    logging.info(f"After resampling - Class distribution: {resampled_dist}")
                    logging.info(f"Dataset size changed from {len(original_train_dataset)} to {len(train_dataset)}")

            # Calculate class weights for loss function
            class_weights_tensor = None
            if handle_class_imbalance and imbalance_strategy in ['weighted_loss', 'focal_loss']:
                if class_weights == 'balanced' or class_weights is None:
                    # Calculate balanced class weights
                    class_weights_array = compute_class_weight(
                        'balanced',
                        classes=np.arange(num_classes),
                        y=train_labels if 'train_labels' in locals() else [
                            train_dataset.targets[i] if hasattr(train_dataset, 'targets')
                            else train_dataset[i][1] for i in range(len(train_dataset))
                        ]
                    )
                    class_weights_tensor = torch.FloatTensor(class_weights_array).to(device)
                    logging.info(f"Calculated balanced class weights: {class_weights_array}")
                elif isinstance(class_weights, dict):
                    weights_array = np.ones(num_classes)
                    for class_id, weight in class_weights.items():
                        weights_array[class_id] = weight
                    class_weights_tensor = torch.FloatTensor(weights_array).to(device)
                    logging.info(f"Using provided class weights: {weights_array}")

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=True
            )

            # Create Inception V3 model
            logging.info("Creating Inception V3 model...")

            if pretrained:
                model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=aux_logits)
                logging.info("Using pretrained weights from ImageNet")
            else:
                model = models.inception_v3(weights=None, aux_logits=aux_logits)
                logging.info("Initializing model with random weights")

            # Modify the classifier for our number of classes
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

            # Replace the auxiliary classifier if aux_logits is enabled
            if aux_logits:
                in_features_aux = model.AuxLogits.fc.in_features
                model.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)

            model = model.to(device)

            # Setup loss function based on imbalance strategy
            if handle_class_imbalance and imbalance_strategy == 'focal_loss':
                criterion = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
                logging.info(f"Using Focal Loss with alpha={focal_loss_alpha}, gamma={focal_loss_gamma}")
            elif handle_class_imbalance and imbalance_strategy == 'weighted_loss' and class_weights_tensor is not None:
                criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
                logging.info("Using weighted CrossEntropyLoss")
            else:
                criterion = nn.CrossEntropyLoss()
                logging.info("Using standard CrossEntropyLoss")

            # Optimizer and scheduler
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=2, factor=0.5
            )

            # Define evaluation metric function
            def calculate_evaluation_metric(y_true, y_pred, y_pred_proba=None, metric=evaluation_metric):
                """Calculate the specified evaluation metric."""
                if metric == 'accuracy':
                    return accuracy_score(y_true, y_pred)
                elif metric == 'f1_macro':
                    return f1_score(y_true, y_pred, average='macro', zero_division=0)
                elif metric == 'f1_weighted':
                    return f1_score(y_true, y_pred, average='weighted', zero_division=0)
                elif metric == 'balanced_accuracy':
                    return balanced_accuracy_score(y_true, y_pred)
                elif metric == 'auc_roc' and y_pred_proba is not None:
                    if num_classes == 2:
                        return roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:
                        return roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                else:
                    return accuracy_score(y_true, y_pred)

            # Track best model based on selected metric
            best_val_loss = float('inf')
            best_val_metric = 0.0
            best_val_acc = 0.0
            best_epoch = 0
            patience_counter = 0

            # Training history
            history = {
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': [],
                'val_metric': [],
                'learning_rates': []
            }

            # Training loop
            logging.info("Starting training...")
            start_time = time.time()

            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()

                    # Forward pass - Inception v3 specific handling
                    if aux_logits and model.training:
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        # During training, the auxiliary classifier's loss is weighted by 0.3
                        loss = loss1 + 0.3 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                epoch_train_loss = train_loss / train_total
                epoch_train_acc = 100 * train_correct / train_total

                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                val_preds = []
                val_labels = []
                val_probs = []

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        outputs = model(inputs)

                        # Handle the case where model returns a tuple
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]

                        loss = criterion(outputs, labels)

                        val_loss += loss.item() * inputs.size(0)
                        probs = torch.softmax(outputs, dim=1)
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                        val_preds.extend(predicted.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())
                        val_probs.extend(probs.cpu().numpy())

                epoch_val_loss = val_loss / val_total
                epoch_val_acc = 100 * val_correct / val_total

                # Calculate evaluation metric
                val_probs_array = np.array(val_probs)
                epoch_val_metric = calculate_evaluation_metric(
                    val_labels, val_preds, val_probs_array, evaluation_metric
                )

                # Update learning rate
                scheduler.step(epoch_val_loss)
                current_lr = optimizer.param_groups[0]['lr']

                # Save history
                history['train_loss'].append(epoch_train_loss)
                history['val_loss'].append(epoch_val_loss)
                history['train_acc'].append(epoch_train_acc)
                history['val_acc'].append(epoch_val_acc)
                history['val_metric'].append(epoch_val_metric)
                history['learning_rates'].append(current_lr)

                # Log progress
                logging.info(f"Epoch {epoch+1}/{num_epochs} - "
                           f"Train Loss: {epoch_train_loss:.4f}, "
                           f"Train Acc: {epoch_train_acc:.2f}%, "
                           f"Val Loss: {epoch_val_loss:.4f}, "
                           f"Val Acc: {epoch_val_acc:.2f}%, "
                           f"Val {evaluation_metric}: {epoch_val_metric:.4f}, "
                           f"LR: {current_lr:.6f}")

                # Check for improvement based on selected metric
                improved = False
                if evaluation_metric in ['accuracy', 'f1_macro', 'f1_weighted', 'balanced_accuracy', 'auc_roc']:
                    if epoch_val_metric > best_val_metric:
                        improved = True
                        best_val_metric = epoch_val_metric
                        best_val_acc = epoch_val_acc
                        best_val_loss = epoch_val_loss
                else:
                    if epoch_val_acc > best_val_acc:
                        improved = True
                        best_val_acc = epoch_val_acc
                        best_val_metric = epoch_val_metric
                        best_val_loss = epoch_val_loss

                if improved:
                    best_epoch = epoch
                    patience_counter = 0

                    # Save best model
                    best_model_path = os.path.join(output_dir, "best_model.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                        'accuracy': best_val_acc,
                        'metric': best_val_metric,
                        'metric_name': evaluation_metric
                    }, best_model_path)
                    logging.info(f"Saved best model with {evaluation_metric}: {best_val_metric:.4f}")
                else:
                    patience_counter += 1

                # Early stopping
                if early_stopping and patience_counter >= patience:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

            # Calculate training time
            total_time = time.time() - start_time
            logging.info(f"Training completed in {total_time:.2f} seconds")

            # Save final model
            final_model_path = os.path.join(output_dir, "final_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_val_loss,
                'accuracy': epoch_val_acc,
                'metric': epoch_val_metric,
                'metric_name': evaluation_metric
            }, final_model_path)

            # Test evaluation
            test_acc = None
            test_metrics = {}
            metrics_path = None
            cm_path = None

            if os.path.exists(test_dir):
                logging.info("Evaluating model on test set...")

                # Load best model for testing
                if os.path.exists(best_model_path):
                    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logging.info(f"Loaded best model from epoch {checkpoint['epoch']} for testing")

                # Create test dataset
                if use_csv_labels:
                    test_labels_file = os.path.join(data_dir, 'test_labels.csv')
                    if not os.path.exists(test_labels_file):
                        test_labels_file = train_labels_file

                    test_dataset = MedicalImageDataset(
                        image_dir=test_dir,
                        labels_file=test_labels_file,
                        transform=val_transform,
                        image_size=inception_image_size
                    )
                else:
                    test_dataset = datasets.ImageFolder(
                        root=test_dir,
                        transform=val_transform
                    )

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4
                )

                # Evaluate on test set
                model.eval()
                test_loss = 0.0
                test_correct = 0
                test_total = 0
                all_preds = []
                all_labels = []
                all_probs = []

                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        outputs = model(inputs)

                        # Handle the case where model returns a tuple
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]

                        loss = criterion(outputs, labels)

                        test_loss += loss.item() * inputs.size(0)
                        probs = torch.softmax(outputs, dim=1)
                        _, predicted = torch.max(outputs, 1)
                        test_total += labels.size(0)
                        test_correct += (predicted == labels).sum().item()

                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())

                test_acc = 100 * test_correct / test_total
                logging.info(f"Test accuracy: {test_acc:.2f}%")

                # Calculate comprehensive test metrics
                all_probs_array = np.array(all_probs)
                test_metrics = {
                    'accuracy': accuracy_score(all_labels, all_preds),
                    'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
                    'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
                    'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
                    'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
                    'precision_weighted': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
                    'recall_weighted': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
                    'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
                }

                # Add AUC-ROC if applicable
                try:
                    if num_classes == 2:
                        test_metrics['auc_roc'] = roc_auc_score(all_labels, all_probs_array[:, 1])
                    else:
                        test_metrics['auc_roc'] = roc_auc_score(all_labels, all_probs_array, multi_class='ovr', average='macro')
                except:
                    logging.warning("Could not calculate AUC-ROC score")

                # Add per-class metrics
                if num_classes > 1:
                    test_metrics['precision_per_class'] = precision_score(all_labels, all_preds, average=None, zero_division=0).tolist()
                    test_metrics['recall_per_class'] = recall_score(all_labels, all_preds, average=None, zero_division=0).tolist()
                    test_metrics['f1_per_class'] = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()

                # Save test metrics
                metrics_path = os.path.join(output_dir, "test_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(test_metrics, f, indent=4)

                # Generate confusion matrix
                cm = confusion_matrix(all_labels, all_preds)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                cm_path = os.path.join(output_dir, "confusion_matrix.png")
                plt.savefig(cm_path)
                plt.close()

                # Log detailed test results
                logging.info("Test Results:")
                for metric, value in test_metrics.items():
                    if isinstance(value, (int, float)):
                        logging.info(f"  {metric}: {value:.4f}")

            # Save training history
            history_path = os.path.join(output_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)

            # Create comprehensive plots
            plt.figure(figsize=(20, 10))

            plt.subplot(2, 4, 1)
            plt.plot(history['train_loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss Curves')

            plt.subplot(2, 4, 2)
            plt.plot(history['train_acc'], label='Training Accuracy')
            plt.plot(history['val_acc'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.title('Accuracy Curves')

            plt.subplot(2, 4, 3)
            plt.plot(history['val_metric'], label=f'Validation {evaluation_metric}')
            plt.xlabel('Epoch')
            plt.ylabel(f'{evaluation_metric}')
            plt.legend()
            plt.title(f'{evaluation_metric} Curve')

            plt.subplot(2, 4, 4)
            plt.plot(history['learning_rates'])
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.yscale('log')

            # Class distribution plots
            if train_class_dist is not None:
                plt.subplot(2, 4, 5)
                plt.bar(range(len(train_class_dist)), train_class_dist)
                plt.xlabel('Class')
                plt.ylabel('Count')
                plt.title('Original Class Distribution')

                if sampled_train_dataset is not None:
                    plt.subplot(2, 4, 6)
                    resampled_dist = np.bincount(y_resampled)
                    plt.bar(range(len(resampled_dist)), resampled_dist)
                    plt.xlabel('Class')
                    plt.ylabel('Count')
                    plt.title('After Resampling')

            # Test metrics visualization
            if test_metrics:
                plt.subplot(2, 4, 7)
                metrics_to_plot = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted']
                metrics_values = [test_metrics.get(m, 0) for m in metrics_to_plot]
                plt.bar(metrics_to_plot, metrics_values)
                plt.ylabel('Score')
                plt.title('Test Metrics')
                plt.xticks(rotation=45)

            plt.tight_layout()
            plots_path = os.path.join(output_dir, "training_plots.png")
            plt.savefig(plots_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Save comprehensive configuration
            config = {
                'model_type': 'inception_v3',
                'num_classes': num_classes,
                'image_size': 299,
                'pretrained': pretrained,
                'early_stopping': early_stopping,
                'patience': patience if early_stopping else None,
                'best_epoch': best_epoch,
                'best_accuracy': best_val_acc,
                'best_metric_value': best_val_metric,
                'evaluation_metric': evaluation_metric,
                'training_epochs': epoch + 1,
                'early_stopped': early_stopping and patience_counter >= patience,
                'batch_size': batch_size,
                'aux_logits': aux_logits,
                'augmentation_level': augmentation_level,
                'custom_augmentations': custom_augmentations if augmentation_level == 'custom' else None,
                # Class imbalance handling configuration
                'handle_class_imbalance': handle_class_imbalance,
                'imbalance_strategy': imbalance_strategy if handle_class_imbalance else None,
                'focal_loss_alpha': focal_loss_alpha if imbalance_strategy == 'focal_loss' else None,
                'focal_loss_gamma': focal_loss_gamma if imbalance_strategy == 'focal_loss' else None,
                'sampling_strategy': sampling_strategy if imbalance_strategy in ['oversampling', 'undersampling'] else None,
                'class_weights': class_weights if imbalance_strategy == 'weighted_loss' else None,
                'original_dataset_size': len(original_train_dataset),
                'final_dataset_size': len(train_dataset),
                'class_distribution_original': train_class_dist.tolist() if train_class_dist is not None else None,
                'class_distribution_final': np.bincount(y_resampled).tolist() if 'y_resampled' in locals() else None,
                'imbalance_ratio': float(imbalance_ratio) if 'imbalance_ratio' in locals() else None
            }

            config_path = os.path.join(output_dir, "model_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

            # Return comprehensive results
            return {
                "status": "success",
                "best_model_path": best_model_path,
                "final_model_path": final_model_path,
                "config_path": config_path,
                "plots_path": plots_path,
                "history_path": history_path,
                "test_metrics_path": metrics_path,
                "confusion_matrix_path": cm_path,
                "best_accuracy": best_val_acc,
                "best_metric_value": best_val_metric,
                "evaluation_metric": evaluation_metric,
                "test_accuracy": test_acc,
                "test_metrics": test_metrics,
                "training_time_seconds": total_time,
                "epochs_completed": epoch + 1,
                "early_stopped": early_stopping and patience_counter >= patience,
                "aux_logits_used": aux_logits,
                "augmentation_level": augmentation_level,
                # Class imbalance results
                "class_imbalance_handled": handle_class_imbalance,
                "imbalance_strategy_used": imbalance_strategy if handle_class_imbalance else None,
                "original_dataset_size": len(original_train_dataset),
                "final_dataset_size": len(train_dataset),
                "imbalance_ratio": float(imbalance_ratio) if 'imbalance_ratio' in locals() else None,
                "class_distribution_original": train_class_dist.tolist() if train_class_dist is not None else None,
                "class_distribution_final": np.bincount(y_resampled).tolist() if 'y_resampled' in locals() else None
            }

        except Exception as e:
            logging.error(f"Error during training: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error_message": str(e),
                "data_dir": data_dir,
                "output_dir": output_dir
            }
