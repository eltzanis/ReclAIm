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


class PyTorchModelFineTuningTool(Tool):
    name = "pytorch_model_fine_tuning"
    description = """
    This tool fine-tunes an already trained PyTorch model with new data. It supports various
    fine-tuning strategies including layer freezing, differential learning rates, and
    catastrophic forgetting prevention techniques. Includes comprehensive class imbalance
    handling for both binary and multiclass classification problems.
    """

    inputs = {
        "pretrained_model_path": {
            "type": "string",
            "description": "Path to the already trained model file (.pt format)"
        },
        "config_path": {
            "type": "string",
            "description": "Path to original model configuration JSON file"
        },
        "new_data_dir": {
            "type": "string",
            "description": "Directory containing new training and validation data"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory where fine-tuned model and results will be saved"
        },
        "fine_tuning_strategy": {
            "type": "string",
            "description": "Strategy: 'full', 'partial', 'head_only', 'gradual_unfreezing'",
            "required": False,
            "nullable": True
        },
        "freeze_layers": {
            "type": "integer",
            "description": "Number of layers to freeze from the beginning (for partial strategy)",
            "required": False,
            "nullable": True
        },
        "fine_tune_learning_rate": {
            "type": "number",
            "description": "Learning rate for fine-tuning (default: 1e-5)",
            "required": False,
            "nullable": True
        },
        "backbone_learning_rate": {
            "type": "number",
            "description": "Lower learning rate for frozen/backbone layers (differential LR)",
            "required": False,
            "nullable": True
        },
        "num_epochs": {
            "type": "integer",
            "description": "Number of fine-tuning epochs",
            "required": False,
            "nullable": True
        },
        "batch_size": {
            "type": "integer",
            "description": "Batch size for fine-tuning",
            "required": False,
            "nullable": True
        },
        "reset_optimizer": {
            "type": "boolean",
            "description": "Whether to reset optimizer state (vs continuing from checkpoint)",
            "required": False,
            "nullable": True
        },
        "warmup_epochs": {
            "type": "integer",
            "description": "Number of warmup epochs for gradual unfreezing",
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
            "description": "Early stopping patience",
            "required": False,
            "nullable": True
        },
        "augmentation_level": {
            "type": "string",
            "description": "Data augmentation level: 'none', 'basic', 'moderate', 'heavy'",
            "required": False,
            "nullable": True
        },
        "weight_decay": {
            "type": "number",
            "description": "Weight decay for regularization",
            "required": False,
            "nullable": True
        },
        "use_cosine_annealing": {
            "type": "boolean",
            "description": "Whether to use cosine annealing LR scheduler",
            "required": False,
            "nullable": True
        },
        "add_new_classes": {
            "type": "boolean",
            "description": "Whether new data contains additional classes",
            "required": False,
            "nullable": True
        },
        "new_num_classes": {
            "type": "integer",
            "description": "Total number of classes after adding new ones",
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
        },
        "preserve_original_performance": {
            "type": "boolean",
            "description": "Whether to monitor and preserve performance on original classes during fine-tuning",
            "required": False,
            "nullable": True
        },
        "catastrophic_forgetting_weight": {
            "type": "number",
            "description": "Weight for catastrophic forgetting prevention loss (0.0-1.0)",
            "required": False,
            "nullable": True
        }
    }

    output_type = "object"

    def forward(
        self,
        pretrained_model_path: str,
        config_path: str,
        new_data_dir: str,
        output_dir: str,
        fine_tuning_strategy: Optional[str] = "partial",
        freeze_layers: Optional[int] = None,
        fine_tune_learning_rate: Optional[float] = 1e-5,
        backbone_learning_rate: Optional[float] = 1e-6,
        num_epochs: Optional[int] = 10,
        batch_size: Optional[int] = 16,
        reset_optimizer: Optional[bool] = True,
        warmup_epochs: Optional[int] = 3,
        early_stopping: Optional[bool] = True,
        patience: Optional[int] = 5,
        augmentation_level: Optional[str] = "basic",
        weight_decay: Optional[float] = 1e-5,
        use_cosine_annealing: Optional[bool] = True,
        add_new_classes: Optional[bool] = False,
        new_num_classes: Optional[int] = None,
        # Class imbalance parameters
        handle_class_imbalance: Optional[bool] = False,
        imbalance_strategy: Optional[str] = "weighted_loss",
        focal_loss_alpha: Optional[float] = 1.0,
        focal_loss_gamma: Optional[float] = 2.0,
        sampling_strategy: Optional[str] = "auto",
        class_weights: Optional[Union[dict, str]] = None,
        evaluation_metric: Optional[str] = "accuracy",
        preserve_original_performance: Optional[bool] = False,
        catastrophic_forgetting_weight: Optional[float] = 0.1
    ):
        """
        Fine-tune a pre-trained model with new data including class imbalance handling.
        """
        try:
            # Handle None values for optional parameters (set defaults)
            if fine_tune_learning_rate is None:
                fine_tune_learning_rate = 1e-5
            if backbone_learning_rate is None:
                backbone_learning_rate = 1e-6

            # Set up logging
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "fine_tuning.log")

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

            logging.info("Starting model fine-tuning with class imbalance handling...")
            logging.info(f"Pretrained model: {pretrained_model_path}")
            logging.info(f"Fine-tuning strategy: {fine_tuning_strategy}")
            logging.info(f"Learning rate: {fine_tune_learning_rate}")
            logging.info(f"Handle class imbalance: {handle_class_imbalance}")
            if handle_class_imbalance:
                logging.info(f"Imbalance strategy: {imbalance_strategy}")
                logging.info(f"Evaluation metric: {evaluation_metric}")
            logging.info(f"New data directory: {new_data_dir}")

            # Load original model configuration
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Model config file not found: {config_path}")

            with open(config_path, 'r') as f:
                original_config = json.load(f)

            model_type = original_config['model_type']
            original_num_classes = original_config['num_classes']
            image_size = original_config.get('image_size', 224)

            # Determine the actual model type to use
            # For EfficientNet, use model_variant if available, otherwise fall back to model_type
            if model_type == 'efficientnet_v2' and 'model_variant' in original_config:
                resolved_model_type = original_config['model_variant']
                logging.info(f"Using EfficientNet variant: {resolved_model_type}")
            else:
                resolved_model_type = model_type

            # Determine final number of classes
            if add_new_classes:
                if new_num_classes is None:
                    raise ValueError("new_num_classes must be specified when add_new_classes=True")
                final_num_classes = new_num_classes
                logging.info(f"Expanding from {original_num_classes} to {final_num_classes} classes")
            else:
                final_num_classes = original_num_classes
                logging.info(f"Fine-tuning with same number of classes: {final_num_classes}")

            # Check device
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

            # Load pretrained model
            logging.info("Loading pretrained model...")
            model = self._load_pretrained_model(
                resolved_model_type, original_config, pretrained_model_path, original_num_classes,
                final_num_classes, device, add_new_classes
            )

            # Store original model weights for catastrophic forgetting prevention
            original_model_state = None
            if preserve_original_performance:
                original_model_state = {name: param.clone().detach()
                                     for name, param in model.named_parameters()}
                logging.info("Stored original model weights for catastrophic forgetting prevention")

            # Apply fine-tuning strategy
            logging.info(f"Applying fine-tuning strategy: {fine_tuning_strategy}")
            if freeze_layers is None:
                freeze_layers = self._get_default_freeze_layers(resolved_model_type, fine_tuning_strategy)

            self._apply_fine_tuning_strategy(
                model, fine_tuning_strategy, freeze_layers, resolved_model_type
            )

            # Setup data transforms
            train_transform, val_transform = self._get_transforms(
                augmentation_level, image_size, resolved_model_type
            )

            # Setup datasets and dataloaders
            train_loader, val_loader, class_distribution = self._setup_dataloaders_with_imbalance(
                new_data_dir, train_transform, val_transform, batch_size,
                handle_class_imbalance, imbalance_strategy, sampling_strategy, final_num_classes
            )

            logging.info(f"Training dataset size: {len(train_loader.dataset)}")
            logging.info(f"Validation dataset size: {len(val_loader.dataset)}")

            # Handle class imbalance
            class_weights_tensor = None
            if handle_class_imbalance and class_distribution is not None:
                # Log class distribution
                logging.info(f"Class distribution: {class_distribution}")

                # Calculate imbalance ratio
                max_class_count = np.max(class_distribution)
                min_class_count = np.min(class_distribution)
                imbalance_ratio = max_class_count / min_class_count
                logging.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")

                # Calculate class weights if needed
                if imbalance_strategy in ['weighted_loss', 'focal_loss']:
                    if class_weights == 'balanced' or class_weights is None:
                        # Get labels from training data
                        train_labels = []
                        if hasattr(train_loader.dataset, 'targets'):
                            train_labels = train_loader.dataset.targets
                        elif hasattr(train_loader.dataset, 'dataset'):
                            # For resampled datasets or split datasets
                            if hasattr(train_loader.dataset.dataset, 'targets'):
                                indices = getattr(train_loader.dataset, 'indices', None)
                                if indices is not None:
                                    train_labels = np.array(train_loader.dataset.dataset.targets)[indices]
                                else:
                                    train_labels = train_loader.dataset.dataset.targets
                            elif hasattr(train_loader.dataset, 'resampled_labels'):
                                train_labels = train_loader.dataset.resampled_labels

                        if len(train_labels) > 0:
                            class_weights_array = compute_class_weight(
                                'balanced',
                                classes=np.arange(final_num_classes),
                                y=train_labels
                            )
                            class_weights_tensor = torch.FloatTensor(class_weights_array).to(device)
                            logging.info(f"Calculated balanced class weights: {class_weights_array}")
                    elif isinstance(class_weights, dict):
                        weights_array = np.ones(final_num_classes)
                        for class_id, weight in class_weights.items():
                            weights_array[class_id] = weight
                        class_weights_tensor = torch.FloatTensor(weights_array).to(device)
                        logging.info(f"Using provided class weights: {weights_array}")

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

            # Setup optimizer with differential learning rates
            optimizer = self._setup_optimizer(
                model, fine_tuning_strategy, fine_tune_learning_rate,
                backbone_learning_rate, weight_decay, reset_optimizer,
                pretrained_model_path if not reset_optimizer else None
            )

            # Setup scheduler
            if use_cosine_annealing:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_epochs, eta_min=1e-7
                )
            else:
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
                    if final_num_classes == 2:
                        return roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:
                        return roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                else:
                    return accuracy_score(y_true, y_pred)

            # Training variables
            best_val_acc = 0.0
            best_val_loss = float('inf')
            best_val_metric = 0.0
            patience_counter = 0
            best_epoch = 0

            # Training history
            history = {
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': [],
                'val_metric': [],
                'learning_rates': [],
                'frozen_layers': [],
                'catastrophic_forgetting_loss': [] if preserve_original_performance else None
            }

            # Fine-tuning loop
            logging.info("Starting fine-tuning...")
            start_time = time.time()

            for epoch in range(num_epochs):
                # Gradual unfreezing
                if fine_tuning_strategy == "gradual_unfreezing" and epoch < warmup_epochs:
                    self._gradual_unfreeze(model, epoch, warmup_epochs, resolved_model_type)

                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                cf_loss_epoch = 0.0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                    # Main classification loss
                    main_loss = criterion(outputs, labels)

                    # Catastrophic forgetting prevention loss
                    cf_loss = 0.0
                    if preserve_original_performance and original_model_state is not None:
                        for name, param in model.named_parameters():
                            if name in original_model_state and param.requires_grad:
                                cf_loss += torch.norm(param - original_model_state[name])
                        cf_loss *= catastrophic_forgetting_weight

                    # Total loss
                    total_loss = main_loss + cf_loss

                    # Backward pass
                    total_loss.backward()
                    optimizer.step()

                    # Statistics
                    train_loss += main_loss.item() * inputs.size(0)
                    cf_loss_epoch += cf_loss if isinstance(cf_loss, (int, float)) else cf_loss.item()
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                epoch_train_loss = train_loss / train_total
                epoch_train_acc = 100 * train_correct / train_total
                epoch_cf_loss = cf_loss_epoch / len(train_loader)

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
                if use_cosine_annealing:
                    scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    scheduler.step(epoch_val_loss)
                    current_lr = optimizer.param_groups[0]['lr']

                # Log frozen layers info
                frozen_layers_count = sum(1 for param in model.parameters() if not param.requires_grad)
                total_params = sum(1 for param in model.parameters())

                # Save history
                history['train_loss'].append(epoch_train_loss)
                history['val_loss'].append(epoch_val_loss)
                history['train_acc'].append(epoch_train_acc)
                history['val_acc'].append(epoch_val_acc)
                history['val_metric'].append(epoch_val_metric)
                history['learning_rates'].append(current_lr)
                history['frozen_layers'].append(frozen_layers_count)
                if preserve_original_performance:
                    history['catastrophic_forgetting_loss'].append(epoch_cf_loss)

                # Log progress
                log_message = (f"Epoch {epoch+1}/{num_epochs} - "
                             f"Train Loss: {epoch_train_loss:.4f}, "
                             f"Train Acc: {epoch_train_acc:.2f}%, "
                             f"Val Loss: {epoch_val_loss:.4f}, "
                             f"Val Acc: {epoch_val_acc:.2f}%, "
                             f"Val {evaluation_metric}: {epoch_val_metric:.4f}, "
                             f"LR: {current_lr:.6f}, "
                             f"Frozen: {frozen_layers_count}/{total_params}")

                if preserve_original_performance:
                    log_message += f", CF Loss: {epoch_cf_loss:.4f}"

                logging.info(log_message)

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
                    best_model_path = os.path.join(output_dir, "best_finetuned_model.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                        'accuracy': best_val_acc,
                        'metric': best_val_metric,
                        'metric_name': evaluation_metric,
                        'original_config': original_config,
                        'fine_tuning_config': {
                            'strategy': fine_tuning_strategy,
                            'freeze_layers': freeze_layers,
                            'learning_rate': fine_tune_learning_rate,
                            'add_new_classes': add_new_classes,
                            'final_num_classes': final_num_classes,
                            'handle_class_imbalance': handle_class_imbalance,
                            'imbalance_strategy': imbalance_strategy if handle_class_imbalance else None,
                            'evaluation_metric': evaluation_metric
                        }
                    }, best_model_path)
                    logging.info(f"Saved best fine-tuned model with {evaluation_metric}: {best_val_metric:.4f}")
                else:
                    patience_counter += 1

                # Early stopping
                if early_stopping and patience_counter >= patience:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

            # Calculate training time
            total_time = time.time() - start_time
            logging.info(f"Fine-tuning completed in {total_time:.2f} seconds")

            # Save final model
            final_model_path = os.path.join(output_dir, "final_finetuned_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_val_loss,
                'accuracy': epoch_val_acc,
                'metric': epoch_val_metric,
                'metric_name': evaluation_metric,
                'original_config': original_config,
                'fine_tuning_config': {
                    'strategy': fine_tuning_strategy,
                    'freeze_layers': freeze_layers,
                    'learning_rate': fine_tune_learning_rate,
                    'add_new_classes': add_new_classes,
                    'final_num_classes': final_num_classes,
                    'handle_class_imbalance': handle_class_imbalance,
                    'imbalance_strategy': imbalance_strategy if handle_class_imbalance else None,
                    'evaluation_metric': evaluation_metric
                }
            }, final_model_path)

            # Evaluate on test set if available
            test_acc = None
            test_metrics = {}
            metrics_path = None
            cm_path = None

            test_dir = os.path.join(new_data_dir, 'test')
            if os.path.exists(test_dir):
                logging.info("Evaluating fine-tuned model on test set...")
                test_acc, test_metrics, metrics_path, cm_path = self._evaluate_on_test_set_with_imbalance(
                    model, test_dir, val_transform, batch_size, device,
                    criterion, final_num_classes, output_dir, evaluation_metric
                )

            # Save training history and plots
            history_path = os.path.join(output_dir, "fine_tuning_history.json")
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)

            plots_path = self._create_fine_tuning_plots_with_imbalance(
                history, output_dir, handle_class_imbalance, preserve_original_performance
            )

            # Save comprehensive fine-tuning configuration
            fine_tune_config = {
                'original_model_path': pretrained_model_path,
                'original_config': original_config,
                'fine_tuning_strategy': fine_tuning_strategy,
                'freeze_layers': freeze_layers,
                'fine_tune_learning_rate': fine_tune_learning_rate,
                'backbone_learning_rate': backbone_learning_rate,
                'num_epochs': num_epochs,
                'best_epoch': best_epoch,
                'best_accuracy': best_val_acc,
                'best_metric_value': best_val_metric,
                'evaluation_metric': evaluation_metric,
                'epochs_completed': epoch + 1,
                'early_stopped': early_stopping and patience_counter >= patience,
                'add_new_classes': add_new_classes,
                'final_num_classes': final_num_classes,
                'training_time_seconds': total_time,
                # Class imbalance configuration
                'handle_class_imbalance': handle_class_imbalance,
                'imbalance_strategy': imbalance_strategy if handle_class_imbalance else None,
                'focal_loss_alpha': focal_loss_alpha if imbalance_strategy == 'focal_loss' else None,
                'focal_loss_gamma': focal_loss_gamma if imbalance_strategy == 'focal_loss' else None,
                'sampling_strategy': sampling_strategy if imbalance_strategy in ['oversampling', 'undersampling'] else None,
                'class_weights': class_weights if imbalance_strategy == 'weighted_loss' else None,
                'class_distribution': class_distribution.tolist() if class_distribution is not None else None,
                'imbalance_ratio': float(imbalance_ratio) if 'imbalance_ratio' in locals() else None,
                # Catastrophic forgetting prevention
                'preserve_original_performance': preserve_original_performance,
                'catastrophic_forgetting_weight': catastrophic_forgetting_weight if preserve_original_performance else None
            }

            config_save_path = os.path.join(output_dir, "fine_tuning_config.json")
            with open(config_save_path, 'w') as f:
                json.dump(fine_tune_config, f, indent=4)

            return {
                "status": "success",
                "best_model_path": best_model_path,
                "final_model_path": final_model_path,
                "config_path": config_save_path,
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
                "fine_tuning_strategy": fine_tuning_strategy,
                "original_num_classes": original_num_classes,
                "final_num_classes": final_num_classes,
                "classes_added": add_new_classes,
                # Class imbalance results
                "class_imbalance_handled": handle_class_imbalance,
                "imbalance_strategy_used": imbalance_strategy if handle_class_imbalance else None,
                "class_distribution": class_distribution.tolist() if class_distribution is not None else None,
                "imbalance_ratio": float(imbalance_ratio) if 'imbalance_ratio' in locals() else None,
                "catastrophic_forgetting_prevention": preserve_original_performance
            }

        except Exception as e:
            logging.error(f"Error during fine-tuning: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error_message": str(e),
                "pretrained_model_path": pretrained_model_path,
                "new_data_dir": new_data_dir,
                "output_dir": output_dir
            }

    def _setup_dataloaders_with_imbalance(self, new_data_dir, train_transform, val_transform,
                                        batch_size, handle_class_imbalance, imbalance_strategy,
                                        sampling_strategy, num_classes):
        """Setup training and validation dataloaders with imbalance handling."""
        train_dir = os.path.join(new_data_dir, 'train')
        val_dir = os.path.join(new_data_dir, 'val')

        if not os.path.exists(train_dir):
            raise ValueError(f"Training directory not found: {train_dir}")

        # Check if using CSV labels or directory structure
        train_labels_file = os.path.join(new_data_dir, 'labels.csv')
        use_csv_labels = os.path.exists(train_labels_file)

        if use_csv_labels:
            if os.path.exists(val_dir):
                train_dataset = MedicalImageDataset(
                    image_dir=train_dir,
                    labels_file=train_labels_file,
                    transform=train_transform
                )
                val_labels_file = os.path.join(new_data_dir, 'val_labels.csv')
                if not os.path.exists(val_labels_file):
                    val_labels_file = train_labels_file

                val_dataset = MedicalImageDataset(
                    image_dir=val_dir,
                    labels_file=val_labels_file,
                    transform=val_transform
                )
            else:
                # Create base dataset without transforms for splitting
                base_dataset = MedicalImageDataset(
                    image_dir=train_dir,
                    labels_file=train_labels_file,
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
            if os.path.exists(val_dir):
                train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
                val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
            else:
                # Create base dataset without transforms for splitting
                base_dataset = datasets.ImageFolder(root=train_dir, transform=None)
                # Split into train and val
                train_size = int(0.8 * len(base_dataset))
                val_size = len(base_dataset) - train_size
                train_subset, val_subset = torch.utils.data.random_split(
                    base_dataset, [train_size, val_size]
                )
                # Apply appropriate transforms to each subset
                train_dataset = TransformSubset(train_subset, transform=train_transform)
                val_dataset = TransformSubset(val_subset, transform=val_transform)

        # Get class distribution
        def get_class_distribution(dataset):
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

        class_distribution = get_class_distribution(train_dataset)
        original_train_dataset = train_dataset

        # Apply sampling if needed
        if handle_class_imbalance and imbalance_strategy in ['oversampling', 'undersampling'] and class_distribution is not None:
            logging.info(f"Applying {imbalance_strategy} with strategy: {sampling_strategy}")

            # Get samples and labels
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

            # Create simplified features for sampling
            X_train = np.arange(len(train_samples)).reshape(-1, 1)
            y_train = np.array(train_labels)

            # Apply sampling
            if imbalance_strategy == 'oversampling':
                sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
            else:
                sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)

            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            resampled_indices = X_resampled.flatten()
            train_dataset = ResampledDataset(original_train_dataset, resampled_indices, y_resampled)
            class_distribution = np.bincount(y_resampled)

            logging.info(f"Dataset size changed from {len(original_train_dataset)} to {len(train_dataset)}")
            logging.info(f"New class distribution: {class_distribution}")

        # Create dataloaders
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

        return train_loader, val_loader, class_distribution

    def _evaluate_on_test_set_with_imbalance(self, model, test_dir, transform, batch_size,
                                           device, criterion, num_classes, output_dir, evaluation_metric):
        """Evaluate the fine-tuned model on test set with comprehensive metrics."""
        # Check if using CSV labels or directory structure
        test_labels_file = os.path.join(os.path.dirname(test_dir), 'test_labels.csv')
        use_csv_labels = os.path.exists(test_labels_file)

        if use_csv_labels:
            test_dataset = MedicalImageDataset(
                image_dir=test_dir,
                labels_file=test_labels_file,
                transform=transform
            )
        else:
            test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )

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

        # Save metrics
        metrics_path = os.path.join(output_dir, "test_metrics_finetuned.json")
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)

        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - Fine-tuned Model')
        cm_path = os.path.join(output_dir, "confusion_matrix_finetuned.png")
        plt.savefig(cm_path)
        plt.close()

        # Log detailed results
        logging.info("Fine-tuned Model Test Results:")
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                logging.info(f"  {metric}: {value:.4f}")

        return test_acc, test_metrics, metrics_path, cm_path

    def _create_fine_tuning_plots_with_imbalance(self, history, output_dir, handle_class_imbalance,
                                               preserve_original_performance):
        """Create comprehensive visualization plots for fine-tuning progress."""
        # Determine number of subplots needed
        n_plots = 4  # Base plots: loss, accuracy, LR, frozen layers
        if handle_class_imbalance:
            n_plots += 1  # Add evaluation metric plot
        if preserve_original_performance:
            n_plots += 1  # Add catastrophic forgetting loss plot

        # Calculate subplot layout
        n_rows = 2 if n_plots <= 4 else 3
        n_cols = 3 if n_plots > 4 else 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        plot_idx = 0

        # Loss curves
        axes[plot_idx].plot(history['train_loss'], label='Training Loss', color='blue')
        axes[plot_idx].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].set_title('Fine-tuning Loss Curves')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        # Accuracy curves
        axes[plot_idx].plot(history['train_acc'], label='Training Accuracy', color='blue')
        axes[plot_idx].plot(history['val_acc'], label='Validation Accuracy', color='red')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Accuracy (%)')
        axes[plot_idx].set_title('Fine-tuning Accuracy Curves')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        # Evaluation metric curve (if using imbalance handling)
        if handle_class_imbalance and 'val_metric' in history:
            axes[plot_idx].plot(history['val_metric'], label='Validation Metric', color='green')
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('Metric Value')
            axes[plot_idx].set_title('Evaluation Metric Curve')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # Learning rate schedule
        axes[plot_idx].plot(history['learning_rates'], color='purple')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Learning Rate')
        axes[plot_idx].set_title('Learning Rate Schedule')
        axes[plot_idx].set_yscale('log')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        # Frozen layers over time
        axes[plot_idx].plot(history['frozen_layers'], color='orange')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Number of Frozen Parameters')
        axes[plot_idx].set_title('Frozen Parameters Over Time')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        # Catastrophic forgetting loss (if enabled)
        if preserve_original_performance and history['catastrophic_forgetting_loss']:
            axes[plot_idx].plot(history['catastrophic_forgetting_loss'], color='red')
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('CF Loss')
            axes[plot_idx].set_title('Catastrophic Forgetting Prevention Loss')
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plots_path = os.path.join(output_dir, "fine_tuning_plots.png")
        plt.savefig(plots_path, dpi=300, bbox_inches='tight')
        plt.close()

        return plots_path

    def _load_pretrained_model(self, model_type, config, model_path, original_num_classes,
                             final_num_classes, device, add_new_classes):
        """Load the pretrained model and modify if needed for new classes."""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Check for auxiliary logits in Inception v3
        has_aux_logits = any('AuxLogits' in key for key in state_dict.keys())

        # Create model architecture
        if model_type == 'resnet18':
            model = models.resnet18(weights=None)
        elif model_type == 'resnet34':
            model = models.resnet34(weights=None)
        elif model_type == 'resnet50':
            model = models.resnet50(weights=None)
        elif model_type == 'resnet101':
            model = models.resnet101(weights=None)
        elif model_type == 'resnet152':
            model = models.resnet152(weights=None)
        elif model_type == 'vgg16':
            model = models.vgg16(weights=None)
        elif model_type == 'vgg16_bn':
            model = models.vgg16_bn(weights=None)
        elif model_type == 'inception_v3':
            model = models.inception_v3(weights=None, aux_logits=has_aux_logits)
        elif model_type == 'efficientnet_v2_s':
            model = models.efficientnet_v2_s(weights=None)
        elif model_type == 'efficientnet_v2_m':
            model = models.efficientnet_v2_m(weights=None)
        elif model_type == 'efficientnet_v2_l':
            model = models.efficientnet_v2_l(weights=None)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # First, load with original number of classes
        if 'resnet' in model_type:
            model.fc = nn.Linear(model.fc.in_features, original_num_classes)
        elif 'vgg' in model_type:
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, original_num_classes)
        elif 'inception' in model_type:
            model.fc = nn.Linear(model.fc.in_features, original_num_classes)
            if has_aux_logits:
                model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, original_num_classes)
        elif 'efficientnet' in model_type:
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, original_num_classes)

        # Load the pretrained weights
        model.load_state_dict(state_dict)

        # If adding new classes, modify the final layer
        if add_new_classes:
            logging.info(f"Expanding classifier from {original_num_classes} to {final_num_classes} classes")

            if 'resnet' in model_type:
                # Save old weights and biases
                old_weight = model.fc.weight.data.clone()
                old_bias = model.fc.bias.data.clone()

                # Create new layer
                model.fc = nn.Linear(model.fc.in_features, final_num_classes)

                # Initialize new layer
                nn.init.xavier_uniform_(model.fc.weight)
                nn.init.zeros_(model.fc.bias)

                # Copy old weights
                model.fc.weight.data[:original_num_classes] = old_weight
                model.fc.bias.data[:original_num_classes] = old_bias

            elif 'vgg' in model_type:
                old_weight = model.classifier[-1].weight.data.clone()
                old_bias = model.classifier[-1].bias.data.clone()

                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, final_num_classes)
                nn.init.xavier_uniform_(model.classifier[-1].weight)
                nn.init.zeros_(model.classifier[-1].bias)

                model.classifier[-1].weight.data[:original_num_classes] = old_weight
                model.classifier[-1].bias.data[:original_num_classes] = old_bias

            elif 'inception' in model_type:
                old_weight = model.fc.weight.data.clone()
                old_bias = model.fc.bias.data.clone()

                model.fc = nn.Linear(model.fc.in_features, final_num_classes)
                nn.init.xavier_uniform_(model.fc.weight)
                nn.init.zeros_(model.fc.bias)

                model.fc.weight.data[:original_num_classes] = old_weight
                model.fc.bias.data[:original_num_classes] = old_bias

                if has_aux_logits:
                    old_aux_weight = model.AuxLogits.fc.weight.data.clone()
                    old_aux_bias = model.AuxLogits.fc.bias.data.clone()

                    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, final_num_classes)
                    nn.init.xavier_uniform_(model.AuxLogits.fc.weight)
                    nn.init.zeros_(model.AuxLogits.fc.bias)

                    model.AuxLogits.fc.weight.data[:original_num_classes] = old_aux_weight
                    model.AuxLogits.fc.bias.data[:original_num_classes] = old_aux_bias

            elif 'efficientnet' in model_type:
                old_weight = model.classifier[-1].weight.data.clone()
                old_bias = model.classifier[-1].bias.data.clone()

                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, final_num_classes)
                nn.init.xavier_uniform_(model.classifier[-1].weight)
                nn.init.zeros_(model.classifier[-1].bias)

                model.classifier[-1].weight.data[:original_num_classes] = old_weight
                model.classifier[-1].bias.data[:original_num_classes] = old_bias

        model = model.to(device)
        return model

    def _get_default_freeze_layers(self, model_type, strategy):
        """Get default number of layers to freeze based on model type and strategy."""
        if strategy == "head_only":
            return -1  # Freeze all except head
        elif strategy == "full":
            return 0   # Freeze nothing
        elif strategy in ["partial", "gradual_unfreezing"]:
            # Conservative defaults - freeze early layers
            if 'resnet' in model_type:
                return 2  # Freeze first 2 residual blocks
            elif 'vgg' in model_type:
                return 10  # Freeze first 10 layers
            elif 'inception' in model_type:
                return 5   # Freeze first 5 mixed layers
            elif 'efficientnet' in model_type:
                return 3   # Freeze first 3 stages
        return 0

    def _apply_fine_tuning_strategy(self, model, strategy, freeze_layers, model_type):
        """Apply the fine-tuning strategy by freezing appropriate layers."""
        if strategy == "full":
            # Don't freeze anything
            for param in model.parameters():
                param.requires_grad = True
            logging.info("Full fine-tuning: All layers trainable")

        elif strategy == "head_only":
            # Freeze all layers except the final classifier
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze classifier
            if 'resnet' in model_type:
                for param in model.fc.parameters():
                    param.requires_grad = True
            elif 'vgg' in model_type:
                for param in model.classifier.parameters():
                    param.requires_grad = True
            elif 'inception' in model_type:
                for param in model.fc.parameters():
                    param.requires_grad = True
                if hasattr(model, 'AuxLogits'):
                    for param in model.AuxLogits.parameters():
                        param.requires_grad = True
            elif 'efficientnet' in model_type:
                for param in model.classifier.parameters():
                    param.requires_grad = True

            logging.info("Head-only fine-tuning: Only classifier layers trainable")

        elif strategy in ["partial", "gradual_unfreezing"]:
            # Freeze specified number of early layers
            if 'resnet' in model_type:
                self._freeze_resnet_layers(model, freeze_layers)
            elif 'vgg' in model_type:
                self._freeze_vgg_layers(model, freeze_layers)
            elif 'inception' in model_type:
                self._freeze_inception_layers(model, freeze_layers)
            elif 'efficientnet' in model_type:
                self._freeze_efficientnet_layers(model, freeze_layers)

            frozen_count = sum(1 for param in model.parameters() if not param.requires_grad)
            total_count = sum(1 for param in model.parameters())
            logging.info(f"Partial fine-tuning: Frozen {frozen_count}/{total_count} parameters")

    def _freeze_resnet_layers(self, model, freeze_layers):
        """Freeze specified number of ResNet layers."""
        layers_to_freeze = []

        if freeze_layers >= 1:
            layers_to_freeze.extend([model.conv1, model.bn1])
        if freeze_layers >= 2:
            layers_to_freeze.append(model.layer1)
        if freeze_layers >= 3:
            layers_to_freeze.append(model.layer2)
        if freeze_layers >= 4:
            layers_to_freeze.append(model.layer3)
        if freeze_layers >= 5:
            layers_to_freeze.append(model.layer4)

        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    def _freeze_vgg_layers(self, model, freeze_layers):
        """Freeze specified number of VGG layers."""
        feature_layers = list(model.features.children())
        for i in range(min(freeze_layers, len(feature_layers))):
            for param in feature_layers[i].parameters():
                param.requires_grad = False

    def _freeze_inception_layers(self, model, freeze_layers):
        """Freeze specified number of Inception layers."""
        layers = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3']
        mixed_layers = ['Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e']

        all_layers = layers + mixed_layers

        for i in range(min(freeze_layers, len(all_layers))):
            layer_name = all_layers[i]
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                for param in layer.parameters():
                    param.requires_grad = False

    def _freeze_efficientnet_layers(self, model, freeze_layers):
        """Freeze specified number of EfficientNet stages."""
        if hasattr(model, 'features'):
            feature_layers = list(model.features.children())
            for i in range(min(freeze_layers, len(feature_layers))):
                for param in feature_layers[i].parameters():
                    param.requires_grad = False

    def _gradual_unfreeze(self, model, current_epoch, warmup_epochs, model_type):
        """Gradually unfreeze layers during warmup epochs."""
        unfreeze_step = current_epoch + 1
        total_steps = warmup_epochs

        if 'resnet' in model_type:
            layers = [model.layer1, model.layer2, model.layer3, model.layer4]
        elif 'vgg' in model_type:
            layers = list(model.features.children())
        elif 'inception' in model_type:
            # Get all named children that are mixed layers
            layers = [getattr(model, name) for name, _ in model.named_children()
                     if name.startswith('Mixed_')]
        elif 'efficientnet' in model_type:
            layers = list(model.features.children())
        else:
            return

        # Calculate how many layers to unfreeze
        layers_to_unfreeze = int((unfreeze_step / total_steps) * len(layers))

        # Unfreeze layers from the end (deeper layers first)
        for i in range(max(0, len(layers) - layers_to_unfreeze), len(layers)):
            for param in layers[i].parameters():
                param.requires_grad = True

        logging.info(f"Gradual unfreezing: Unfroze {layers_to_unfreeze}/{len(layers)} layer groups")

    def _get_transforms(self, augmentation_level, image_size, model_type):
        """Get training and validation transforms."""
        # Determine actual image size
        if 'inception' in model_type and image_size == 224:
            image_size = 299

        target_size = (image_size, image_size)

        # Validation transform (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Training transform with augmentation
        if augmentation_level == "none":
            train_transform = val_transform
        elif augmentation_level == "basic":
            train_transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif augmentation_level == "moderate":
            train_transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif augmentation_level == "heavy":
            train_transform = transforms.Compose([
                transforms.Resize(int(target_size[0] * 1.1)),
                transforms.RandomCrop(target_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.RandomGrayscale(p=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
            ])
        else:
            train_transform = val_transform

        return train_transform, val_transform

    def _setup_optimizer(self, model, strategy, fine_tune_lr, backbone_lr, weight_decay,
                        reset_optimizer, checkpoint_path=None):
        """Setup optimizer with differential learning rates."""

        if strategy == "head_only":
            # Only optimize classifier parameters
            if hasattr(model, 'fc'):  # ResNet, Inception
                optimizer = optim.AdamW(model.fc.parameters(), lr=fine_tune_lr, weight_decay=weight_decay)
            elif hasattr(model, 'classifier'):  # VGG, EfficientNet
                optimizer = optim.AdamW(model.classifier.parameters(), lr=fine_tune_lr, weight_decay=weight_decay)
            else:
                # Fallback: only trainable parameters
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                optimizer = optim.AdamW(trainable_params, lr=fine_tune_lr, weight_decay=weight_decay)

        elif strategy in ["partial", "gradual_unfreezing"] and backbone_lr is not None:
            # Differential learning rates
            backbone_params = []
            head_params = []

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'fc' in name or 'classifier' in name:
                        head_params.append(param)
                    else:
                        backbone_params.append(param)

            param_groups = []
            if backbone_params:
                param_groups.append({'params': backbone_params, 'lr': backbone_lr})
            if head_params:
                param_groups.append({'params': head_params, 'lr': fine_tune_lr})

            optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)

        else:
            # Single learning rate for all trainable parameters
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(trainable_params, lr=fine_tune_lr, weight_decay=weight_decay)

        # Load optimizer state if not resetting
        if not reset_optimizer and checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logging.info("Loaded optimizer state from checkpoint")
            except Exception as e:
                logging.warning(f"Could not load optimizer state: {e}")

        return optimizer
