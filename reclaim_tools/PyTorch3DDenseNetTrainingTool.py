import os
import time
import logging
import json
from typing import Optional, Union, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, balanced_accuracy_score,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from smolagents import Tool

# MONAI imports
from monai.networks.nets import DenseNet
from monai.transforms import (
    Compose, RandRotate90, RandFlip, RandGaussianNoise,
    RandAdjustContrast, RandGaussianSmooth, RandScaleIntensity,
    RandShiftIntensity, Rand3DElastic
)


class MedMNIST3DDataset(Dataset):
    """Dataset class for MedMNIST 3D numpy arrays."""

    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: numpy array of shape (N, 28, 28, 28) or (N, 1, 28, 28, 28)
            labels: numpy array of shape (N,) or (N, 1)
            transform: MONAI transforms to apply
        """
        self.images = images

        # Ensure images have channel dimension: (N, C, D, H, W)
        if len(self.images.shape) == 4:
            self.images = self.images[:, np.newaxis, :, :, :]  # Add channel dim

        # Handle labels
        if len(labels.shape) > 1:
            self.labels = labels.flatten()
        else:
            self.labels = labels

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)
        label = int(self.labels[idx])

        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        # Apply transforms if provided
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Convert to tensor
            image = torch.from_numpy(image)

        return image, label


class ResampledDataset(Dataset):
    """Wrapper dataset for resampled indices (for handling class imbalance)."""

    def __init__(self, base_dataset, resampled_indices, resampled_labels):
        self.base_dataset = base_dataset
        self.resampled_indices = resampled_indices
        self.resampled_labels = resampled_labels

    def __len__(self):
        return len(self.resampled_indices)

    def __getitem__(self, idx):
        original_idx = self.resampled_indices[idx]
        image, _ = self.base_dataset[original_idx]
        label = self.resampled_labels[idx]
        return image, label


class PyTorch3DDenseNetTrainingTool(Tool):
    """Training tool for 3D DenseNet models using MONAI for medical image classification."""

    name = "pytorch_3ddensenet_training"
    description = """
    This tool trains a 3D DenseNet model using PyTorch and MONAI for 3D medical image classification.
    Supports both standard DenseNet architectures and custom configurations optimized for small volumes.
    Supports DenseNet121, DenseNet169, DenseNet201, and DenseNet264 variants with configurable 3D augmentation.
    Includes comprehensive class imbalance handling for both binary and multiclass problems.
    Use config_type='custom' for small 28x28x28 volumes or 'standard' for full-size medical images.
    """

    inputs = {
        "data_path": {
            "type": "string",
            "description": "Path to .npz file containing MedMNIST data (or directory with train.npz, val.npz, test.npz)"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory where the trained model and results will be saved"
        },
        "num_classes": {
            "type": "integer",
            "description": "Number of classes for classification"
        },
        "model_variant": {
            "type": "string",
            "description": "3D DenseNet variant: densenet121, densenet169, densenet201, or densenet264",
            "required": False,
            "nullable": True
        },
        "num_epochs": {
            "type": "integer",
            "description": "Number of training epochs",
            "required": False,
            "nullable": True
        },
        "batch_size": {
            "type": "integer",
            "description": "Batch size for training (recommend 4-8 for 3D data)",
            "required": False,
            "nullable": True
        },
        "pretrained": {
            "type": "boolean",
            "description": "Whether to use pretrained weights (MedicalNet if available)",
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
        "handle_class_imbalance": {
            "type": "boolean",
            "description": "Whether to apply class imbalance handling techniques",
            "required": False,
            "nullable": True
        },
        "imbalance_strategy": {
            "type": "string",
            "description": "Strategy: 'weighted_loss', 'focal_loss', 'oversampling', 'undersampling'",
            "required": False,
            "nullable": True
        },
        "focal_loss_alpha": {
            "type": "number",
            "description": "Alpha parameter for focal loss",
            "required": False,
            "nullable": True
        },
        "focal_loss_gamma": {
            "type": "number",
            "description": "Gamma parameter for focal loss",
            "required": False,
            "nullable": True
        },
        "sampling_strategy": {
            "type": "string",
            "description": "Sampling strategy for over/undersampling",
            "required": False,
            "nullable": True
        },
        "class_weights": {
            "type": "object",
            "description": "Custom class weights or 'balanced'",
            "required": False,
            "nullable": True
        },
        "evaluation_metric": {
            "type": "string",
            "description": "Primary metric: 'accuracy', 'f1_macro', 'f1_weighted', 'balanced_accuracy', 'auc_roc'",
            "required": False,
            "nullable": True
        },
        "learning_rate": {
            "type": "number",
            "description": "Initial learning rate",
            "required": False,
            "nullable": True
        },
        "validation_split": {
            "type": "number",
            "description": "Fraction of training data to use for validation if no val set provided",
            "required": False,
            "nullable": True
        },
        "config_type": {
            "type": "string",
            "description": "Configuration type: 'custom' (optimized for small 28x28x28 volumes) or 'standard' (full DenseNet architecture)",
            "required": False,
            "nullable": True
        }
    }

    output_type = "object"

    def forward(
        self,
        data_path: str,
        output_dir: str,
        num_classes: int,
        model_variant: Optional[str] = "densenet121",
        num_epochs: Optional[int] = 50,
        batch_size: Optional[int] = 8,
        pretrained: Optional[bool] = False,
        early_stopping: Optional[bool] = True,
        patience: Optional[int] = 10,
        augmentation_level: Optional[str] = "basic",
        custom_augmentations: Optional[dict] = None,
        handle_class_imbalance: Optional[bool] = False,
        imbalance_strategy: Optional[str] = "weighted_loss",
        focal_loss_alpha: Optional[float] = 1.0,
        focal_loss_gamma: Optional[float] = 2.0,
        sampling_strategy: Optional[str] = "auto",
        class_weights: Optional[Union[dict, str]] = None,
        evaluation_metric: Optional[str] = "accuracy",
        learning_rate: Optional[float] = 1e-3,
        validation_split: Optional[float] = 0.2,
        config_type: Optional[str] = "custom"
    ):
        """
        Train a 3D DenseNet model for medical image classification.

        Returns:
            Dictionary with training results and model paths
        """
        try:
            # Handle None values for optional parameters (set defaults)
            if learning_rate is None:
                learning_rate = 1e-3
            if validation_split is None:
                validation_split = 0.2

            # Set up logging
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "training.log")

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

            logging.info(f"Starting 3D DenseNet training with configuration:")
            logging.info(f"Model variant: {model_variant}")
            logging.info(f"Configuration type: {config_type}")
            logging.info(f"Number of classes: {num_classes}")
            logging.info(f"Batch size: {batch_size}")
            logging.info(f"Augmentation level: {augmentation_level}")
            logging.info(f"Handle class imbalance: {handle_class_imbalance}")

            # Check device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {device}")

            # Define Focal Loss
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

            # Load data
            logging.info(f"Loading data from {data_path}")

            # Check if data_path is a file or directory
            if os.path.isfile(data_path):
                # Single .npz file - split into train/val
                data = np.load(data_path)

                # MedMNIST format: 'train_images', 'train_labels', 'val_images', 'val_labels', 'test_images', 'test_labels'
                if 'train_images' in data:
                    train_images = data['train_images']
                    train_labels = data['train_labels']

                    if 'val_images' in data:
                        val_images = data['val_images']
                        val_labels = data['val_labels']
                        use_train_val_split = False
                    else:
                        use_train_val_split = True

                    if 'test_images' in data:
                        test_images = data['test_images']
                        test_labels = data['test_labels']
                        has_test = True
                    else:
                        has_test = False
                else:
                    # Generic format - assume 'images' and 'labels'
                    all_images = data['images']
                    all_labels = data['labels']

                    # Split into train/val
                    n_samples = len(all_images)
                    n_val = int(n_samples * validation_split)
                    n_train = n_samples - n_val

                    indices = np.random.permutation(n_samples)
                    train_images = all_images[indices[:n_train]]
                    train_labels = all_labels[indices[:n_train]]
                    val_images = all_images[indices[n_train:]]
                    val_labels = all_labels[indices[n_train:]]
                    use_train_val_split = False
                    has_test = False
            else:
                # Directory with separate files
                train_data = np.load(os.path.join(data_path, 'train.npz'))
                train_images = train_data['images']
                train_labels = train_data['labels']

                if os.path.exists(os.path.join(data_path, 'val.npz')):
                    val_data = np.load(os.path.join(data_path, 'val.npz'))
                    val_images = val_data['images']
                    val_labels = val_data['labels']
                    use_train_val_split = False
                else:
                    use_train_val_split = True

                if os.path.exists(os.path.join(data_path, 'test.npz')):
                    test_data = np.load(os.path.join(data_path, 'test.npz'))
                    test_images = test_data['images']
                    test_labels = test_data['labels']
                    has_test = True
                else:
                    has_test = False

            logging.info(f"Loaded training data: {train_images.shape}")

            # Define 3D augmentation transforms
            def get_3d_augmentation_transforms(level, custom_settings=None):
                """Get 3D augmentation transforms based on the specified level."""

                if level == "none":
                    return None

                elif level == "basic":
                    return Compose([
                        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                        RandFlip(prob=0.5, spatial_axis=0),
                        RandFlip(prob=0.5, spatial_axis=1),
                        RandFlip(prob=0.5, spatial_axis=2),
                    ])

                elif level == "moderate":
                    return Compose([
                        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                        RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                        RandFlip(prob=0.5, spatial_axis=0),
                        RandFlip(prob=0.5, spatial_axis=1),
                        RandFlip(prob=0.5, spatial_axis=2),
                        RandGaussianNoise(prob=0.3, mean=0.0, std=0.1),
                        RandAdjustContrast(prob=0.3, gamma=(0.8, 1.2)),
                        RandScaleIntensity(factors=0.2, prob=0.3),
                    ])

                elif level == "heavy":
                    return Compose([
                        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                        RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                        RandRotate90(prob=0.5, spatial_axes=(0, 2)),
                        RandFlip(prob=0.5, spatial_axis=0),
                        RandFlip(prob=0.5, spatial_axis=1),
                        RandFlip(prob=0.5, spatial_axis=2),
                        RandGaussianNoise(prob=0.3, mean=0.0, std=0.1),
                        RandAdjustContrast(prob=0.3, gamma=(0.7, 1.3)),
                        RandScaleIntensity(factors=0.3, prob=0.3),
                        RandShiftIntensity(offsets=0.1, prob=0.3),
                        RandGaussianSmooth(prob=0.2),
                        Rand3DElastic(prob=0.2, sigma_range=(5, 7), magnitude_range=(50, 150)),
                    ])

                elif level == "custom" and custom_settings:
                    transforms = []

                    if custom_settings.get('rotate90', True):
                        transforms.append(RandRotate90(prob=custom_settings.get('rotate90_prob', 0.5)))

                    if custom_settings.get('flip', True):
                        for axis in range(3):
                            transforms.append(RandFlip(prob=custom_settings.get('flip_prob', 0.5), spatial_axis=axis))

                    if custom_settings.get('gaussian_noise', False):
                        transforms.append(RandGaussianNoise(
                            prob=custom_settings.get('noise_prob', 0.3),
                            mean=custom_settings.get('noise_mean', 0.0),
                            std=custom_settings.get('noise_std', 0.1)
                        ))

                    if custom_settings.get('adjust_contrast', False):
                        transforms.append(RandAdjustContrast(
                            prob=custom_settings.get('contrast_prob', 0.3),
                            gamma=custom_settings.get('contrast_gamma', (0.8, 1.2))
                        ))

                    if custom_settings.get('scale_intensity', False):
                        transforms.append(RandScaleIntensity(
                            factors=custom_settings.get('scale_factors', 0.2),
                            prob=custom_settings.get('scale_prob', 0.3)
                        ))

                    if custom_settings.get('elastic', False):
                        transforms.append(Rand3DElastic(
                            prob=custom_settings.get('elastic_prob', 0.2),
                            sigma_range=custom_settings.get('elastic_sigma', (5, 7)),
                            magnitude_range=custom_settings.get('elastic_magnitude', (50, 150))
                        ))

                    return Compose(transforms) if transforms else None

                else:
                    logging.warning(f"Unknown augmentation level: {level}. Using 'basic'.")
                    return get_3d_augmentation_transforms("basic")

            # Get transforms
            train_transform = get_3d_augmentation_transforms(augmentation_level, custom_augmentations)
            val_transform = None  # No augmentation for validation

            # Create datasets
            if use_train_val_split:
                # Split training data - first create base dataset without transforms
                base_dataset = MedMNIST3DDataset(train_images, train_labels, transform=None)
                train_size = int(len(base_dataset) * (1 - validation_split))
                val_size = len(base_dataset) - train_size

                # Split indices
                train_indices, val_indices = torch.utils.data.random_split(
                    range(len(base_dataset)), [train_size, val_size]
                )

                # Create separate datasets with appropriate transforms
                train_dataset = MedMNIST3DDataset(
                    train_images[train_indices.indices],
                    train_labels[train_indices.indices],
                    transform=train_transform
                )
                val_dataset = MedMNIST3DDataset(
                    train_images[val_indices.indices],
                    train_labels[val_indices.indices],
                    transform=val_transform
                )
            else:
                train_dataset = MedMNIST3DDataset(train_images, train_labels, transform=train_transform)
                val_dataset = MedMNIST3DDataset(val_images, val_labels, transform=val_transform)

            logging.info(f"Training dataset size: {len(train_dataset)}")
            logging.info(f"Validation dataset size: {len(val_dataset)}")

            # Get class distribution
            if hasattr(train_dataset, 'labels'):
                train_labels_list = train_dataset.labels
            else:
                train_labels_list = train_labels.flatten()

            train_class_dist = np.bincount(train_labels_list, minlength=num_classes)
            logging.info(f"Training class distribution: {train_class_dist}")

            if len(train_class_dist) > 0:
                max_class_count = np.max(train_class_dist)
                min_class_count = np.min(train_class_dist[train_class_dist > 0])
                imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else 1
                logging.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")

                if not handle_class_imbalance and imbalance_ratio > 2.0:
                    logging.warning(f"High class imbalance detected (ratio: {imbalance_ratio:.2f}). "
                                  "Consider enabling handle_class_imbalance=True")

            # Handle class imbalance with sampling
            original_train_dataset = train_dataset
            y_resampled = None

            if handle_class_imbalance and imbalance_strategy in ['oversampling', 'undersampling']:
                logging.info(f"Applying {imbalance_strategy}...")

                X_train = np.arange(len(train_labels_list)).reshape(-1, 1)
                y_train = np.array(train_labels_list)

                if imbalance_strategy == 'oversampling':
                    sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
                    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                elif imbalance_strategy == 'undersampling':
                    sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
                    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

                resampled_indices = X_resampled.flatten()
                train_dataset = ResampledDataset(original_train_dataset, resampled_indices, y_resampled)

                resampled_dist = np.bincount(y_resampled, minlength=num_classes)
                logging.info(f"After resampling - Class distribution: {resampled_dist}")
                logging.info(f"Dataset size changed from {len(original_train_dataset)} to {len(train_dataset)}")

            # Calculate class weights for loss
            class_weights_tensor = None
            if handle_class_imbalance and imbalance_strategy in ['weighted_loss', 'focal_loss']:
                if class_weights == 'balanced' or class_weights is None:
                    class_weights_array = compute_class_weight(
                        'balanced',
                        classes=np.arange(num_classes),
                        y=train_labels_list
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
                pin_memory=True if torch.cuda.is_available() else False
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False
            )

            # Create model
            logging.info(f"Creating {model_variant} model...")

            # Validate config_type
            if config_type not in ['custom', 'standard']:
                logging.warning(f"Invalid config_type: {config_type}. Using 'custom' instead.")
                config_type = 'custom'

            # Standard DenseNet configurations (full architecture)
            standard_configs = {
                'densenet121': {
                    'init_features': 64,
                    'growth_rate': 32,
                    'block_config': (6, 12, 24, 16),
                    'bn_size': 4,
                    'dropout_prob': 0.0
                },
                'densenet169': {
                    'init_features': 64,
                    'growth_rate': 32,
                    'block_config': (6, 12, 32, 32),
                    'bn_size': 4,
                    'dropout_prob': 0.0
                },
                'densenet201': {
                    'init_features': 64,
                    'growth_rate': 32,
                    'block_config': (6, 12, 48, 32),
                    'bn_size': 4,
                    'dropout_prob': 0.0
                },
                'densenet264': {
                    'init_features': 64,
                    'growth_rate': 32,
                    'block_config': (6, 12, 64, 48),
                    'bn_size': 4,
                    'dropout_prob': 0.0
                }
            }

            # Custom configurations for small 28x28x28 volumes
            # Reduced depth to prevent spatial dimensions from becoming too small
            custom_configs = {
                'densenet121': {
                    'init_features': 32,  # Reduced from 64
                    'growth_rate': 16,     # Reduced from 32
                    'block_config': (3, 6, 12),  # Reduced from (6, 12, 24, 16)
                    'bn_size': 4,
                    'dropout_prob': 0.0
                },
                'densenet169': {
                    'init_features': 32,
                    'growth_rate': 16,
                    'block_config': (3, 6, 12, 12),  # Reduced from (6, 12, 32, 32)
                    'bn_size': 4,
                    'dropout_prob': 0.0
                },
                'densenet201': {
                    'init_features': 32,
                    'growth_rate': 16,
                    'block_config': (3, 6, 12, 16),  # Reduced from (6, 12, 48, 32)
                    'bn_size': 4,
                    'dropout_prob': 0.0
                },
                'densenet264': {
                    'init_features': 32,
                    'growth_rate': 16,
                    'block_config': (3, 6, 16, 20),  # Reduced from (6, 12, 64, 48)
                    'bn_size': 4,
                    'dropout_prob': 0.0
                }
            }

            # Select configuration based on config_type
            model_configs = standard_configs if config_type == 'standard' else custom_configs

            if model_variant not in model_configs:
                logging.warning(f"Invalid model variant: {model_variant}. Using densenet121 instead.")
                model_variant = 'densenet121'

            config = model_configs[model_variant]
            logging.info(f"Using {config_type} configuration: {config}")

            # MONAI DenseNet - supports both standard and custom configurations
            model = DenseNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=num_classes,
                init_features=config['init_features'],
                growth_rate=config['growth_rate'],
                block_config=config['block_config'],
                bn_size=config['bn_size'],
                dropout_prob=config['dropout_prob']
            )

            if pretrained:
                logging.warning(
                    "Pretrained weights are not currently supported for DenseNet. "
                    "MONAI's DenseNet does not have pretrained weight infrastructure like MedicalNet for ResNet. "
                    f"Training will proceed from random initialization using {config_type} configuration."
                )

            model = model.to(device)
            logging.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

            # Setup loss function
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
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=1e-6
            )

            # Evaluation metric function
            def calculate_evaluation_metric(y_true, y_pred, y_pred_proba=None, metric=evaluation_metric):
                if metric == 'accuracy':
                    return accuracy_score(y_true, y_pred)
                elif metric == 'f1_macro':
                    return f1_score(y_true, y_pred, average='macro', zero_division=0)
                elif metric == 'f1_weighted':
                    return f1_score(y_true, y_pred, average='weighted', zero_division=0)
                elif metric == 'balanced_accuracy':
                    return balanced_accuracy_score(y_true, y_pred)
                elif metric == 'auc_roc' and y_pred_proba is not None:
                    try:
                        if num_classes == 2:
                            return roc_auc_score(y_true, y_pred_proba[:, 1])
                        else:
                            return roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                    except:
                        return accuracy_score(y_true, y_pred)
                else:
                    return accuracy_score(y_true, y_pred)

            # Training tracking
            best_val_loss = float('inf')
            best_val_metric = 0.0
            best_val_acc = 0.0
            patience_counter = 0
            best_epoch = 0

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
                val_labels_list = []
                val_probs = []

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item() * inputs.size(0)
                        probs = torch.softmax(outputs, dim=1)
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                        val_preds.extend(predicted.cpu().numpy())
                        val_labels_list.extend(labels.cpu().numpy())
                        val_probs.extend(probs.cpu().numpy())

                epoch_val_loss = val_loss / val_total
                epoch_val_acc = 100 * val_correct / val_total

                # Calculate evaluation metric
                val_probs_array = np.array(val_probs)
                epoch_val_metric = calculate_evaluation_metric(
                    val_labels_list, val_preds, val_probs_array, evaluation_metric
                )

                # Update learning rate
                scheduler.step()
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

                # Check for improvement
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

            if has_test:
                logging.info("Evaluating model on test set...")

                # Load best model
                if os.path.exists(best_model_path):
                    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logging.info(f"Loaded best model from epoch {checkpoint['epoch']}")

                test_dataset = MedMNIST3DDataset(test_images, test_labels, transform=None)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4
                )

                model.eval()
                test_correct = 0
                test_total = 0
                all_preds = []
                all_labels = []
                all_probs = []

                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        outputs = model(inputs)
                        probs = torch.softmax(outputs, dim=1)
                        _, predicted = torch.max(outputs, 1)
                        test_total += labels.size(0)
                        test_correct += (predicted == labels).sum().item()

                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())

                test_acc = 100 * test_correct / test_total
                logging.info(f"Test accuracy: {test_acc:.2f}%")

                # Calculate comprehensive metrics
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

                # Add AUC if applicable
                try:
                    if num_classes == 2:
                        test_metrics['auc_roc'] = roc_auc_score(all_labels, all_probs_array[:, 1])
                    else:
                        test_metrics['auc_roc'] = roc_auc_score(all_labels, all_probs_array, multi_class='ovr', average='macro')
                except:
                    logging.warning("Could not calculate AUC-ROC score")

                # Per-class metrics
                if num_classes > 1:
                    test_metrics['precision_per_class'] = precision_score(all_labels, all_preds, average=None, zero_division=0).tolist()
                    test_metrics['recall_per_class'] = recall_score(all_labels, all_preds, average=None, zero_division=0).tolist()
                    test_metrics['f1_per_class'] = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()

                # Save metrics
                metrics_path = os.path.join(output_dir, "test_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(test_metrics, f, indent=4)

                # Confusion matrix
                cm = confusion_matrix(all_labels, all_preds)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix - Test Set')
                cm_path = os.path.join(output_dir, "confusion_matrix.png")
                plt.savefig(cm_path)
                plt.close()

                logging.info("Test Results:")
                for metric, value in test_metrics.items():
                    if isinstance(value, (int, float)):
                        logging.info(f"  {metric}: {value:.4f}")

            # Save training history
            history_path = os.path.join(output_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)

            # Create plots
            plt.figure(figsize=(20, 10))

            plt.subplot(2, 3, 1)
            plt.plot(history['train_loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss Curves')

            plt.subplot(2, 3, 2)
            plt.plot(history['train_acc'], label='Training Accuracy')
            plt.plot(history['val_acc'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.title('Accuracy Curves')

            plt.subplot(2, 3, 3)
            plt.plot(history['val_metric'], label=f'Validation {evaluation_metric}')
            plt.xlabel('Epoch')
            plt.ylabel(f'{evaluation_metric}')
            plt.legend()
            plt.title(f'{evaluation_metric} Curve')

            plt.subplot(2, 3, 4)
            plt.plot(history['learning_rates'])
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.yscale('log')

            plt.subplot(2, 3, 5)
            plt.bar(range(len(train_class_dist)), train_class_dist)
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.title('Training Class Distribution')

            if test_metrics:
                plt.subplot(2, 3, 6)
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

            # Save configuration
            config_save = {
                'model_type': '3d_densenet',
                'model_variant': model_variant,
                'config_type': config_type,
                'num_classes': num_classes,
                'input_shape': list(train_images.shape[1:]),
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
                'learning_rate': learning_rate,
                'augmentation_level': augmentation_level,
                'handle_class_imbalance': handle_class_imbalance,
                'imbalance_strategy': imbalance_strategy if handle_class_imbalance else None,
                'original_dataset_size': len(original_train_dataset),
                'final_dataset_size': len(train_dataset),
                'class_distribution': train_class_dist.tolist(),
                'imbalance_ratio': float(imbalance_ratio) if 'imbalance_ratio' in locals() else None
            }

            config_path = os.path.join(output_dir, "model_config.json")
            with open(config_path, 'w') as f:
                json.dump(config_save, f, indent=4)

            # Return results
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
                "model_variant": model_variant,
                "augmentation_level": augmentation_level,
                "class_imbalance_handled": handle_class_imbalance,
                "imbalance_strategy_used": imbalance_strategy if handle_class_imbalance else None,
                "original_dataset_size": len(original_train_dataset),
                "final_dataset_size": len(train_dataset)
            }

        except Exception as e:
            logging.error(f"Error during training: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error_message": str(e),
                "data_path": data_path,
                "output_dir": output_dir
            }
