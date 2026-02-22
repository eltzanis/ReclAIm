import os
import time
import logging
import json
from typing import Optional, Union
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
    confusion_matrix, balanced_accuracy_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from smolagents import Tool

# MONAI imports
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50, DenseNet
from monai.transforms import (
    Compose, RandRotate90, RandFlip, RandGaussianNoise,
    RandAdjustContrast, RandScaleIntensity
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
            self.images = self.images[:, np.newaxis, :, :, :]

        # Handle labels
        if len(labels.shape) > 1:
            self.labels = labels.flatten()
        else:
            self.labels = labels

        self.transform = transform
        self.targets = self.labels  

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
            image = torch.from_numpy(image)

        return image, label


class ResampledDataset(Dataset):
    """Wrapper dataset for resampled indices (for handling class imbalance)."""

    def __init__(self, base_dataset, resampled_indices, resampled_labels):
        self.base_dataset = base_dataset
        self.resampled_indices = resampled_indices
        self.resampled_labels = resampled_labels
        self.targets = resampled_labels  

    def __len__(self):
        return len(self.resampled_indices)

    def __getitem__(self, idx):
        original_idx = self.resampled_indices[idx]
        image, _ = self.base_dataset[original_idx]
        label = self.resampled_labels[idx]
        return image, label


class PyTorch3DModelFineTuningTool(Tool):
    """Fine-tuning tool for 3D ResNet and DenseNet models using MONAI."""

    name = "pytorch_3d_model_fine_tuning"
    description = """
    This tool fine-tunes pre-trained 3D ResNet or DenseNet models with new data.
    Supports layer freezing, differential learning rates, class imbalance handling,
    and catastrophic forgetting prevention for medical imaging applications.
    """

    inputs = {
        "pretrained_model_path": {
            "type": "string",
            "description": "Path to the trained model file (.pt format)"
        },
        "config_path": {
            "type": "string",
            "description": "Path to original model configuration JSON file"
        },
        "new_data_path": {
            "type": "string",
            "description": "Path to .npz file containing new training data"
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
            "description": "Number of layer groups to freeze from the beginning",
            "required": False,
            "nullable": True
        },
        "fine_tune_learning_rate": {
            "type": "number",
            "description": "Learning rate for fine-tuning",
            "required": False,
            "nullable": True
        },
        "backbone_learning_rate": {
            "type": "number",
            "description": "Lower learning rate for frozen/backbone layers",
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
            "description": "Whether to reset optimizer state",
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
            "description": "Augmentation level: 'none', 'basic', 'moderate', 'heavy'",
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
        "handle_class_imbalance": {
            "type": "boolean",
            "description": "Whether to apply class imbalance handling",
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
        "preserve_original_performance": {
            "type": "boolean",
            "description": "Whether to prevent catastrophic forgetting",
            "required": False,
            "nullable": True
        },
        "catastrophic_forgetting_weight": {
            "type": "number",
            "description": "Weight for catastrophic forgetting prevention loss",
            "required": False,
            "nullable": True
        },
        "validation_split": {
            "type": "number",
            "description": "Fraction of data to use for validation if no val set provided",
            "required": False,
            "nullable": True
        }
    }

    output_type = "object"

    def forward(
        self,
        pretrained_model_path: str,
        config_path: str,
        new_data_path: str,
        output_dir: str,
        fine_tuning_strategy: Optional[str] = "partial",
        freeze_layers: Optional[int] = None,
        fine_tune_learning_rate: Optional[float] = 1e-5,
        backbone_learning_rate: Optional[float] = 1e-6,
        num_epochs: Optional[int] = 10,
        batch_size: Optional[int] = 8,
        reset_optimizer: Optional[bool] = True,
        warmup_epochs: Optional[int] = 3,
        early_stopping: Optional[bool] = True,
        patience: Optional[int] = 5,
        augmentation_level: Optional[str] = "basic",
        weight_decay: Optional[float] = 1e-5,
        use_cosine_annealing: Optional[bool] = True,
        add_new_classes: Optional[bool] = False,
        new_num_classes: Optional[int] = None,
        handle_class_imbalance: Optional[bool] = False,
        imbalance_strategy: Optional[str] = "weighted_loss",
        focal_loss_alpha: Optional[float] = 1.0,
        focal_loss_gamma: Optional[float] = 2.0,
        sampling_strategy: Optional[str] = "auto",
        class_weights: Optional[Union[dict, str]] = None,
        evaluation_metric: Optional[str] = "f1_macro",
        preserve_original_performance: Optional[bool] = False,
        catastrophic_forgetting_weight: Optional[float] = 0.1,
        validation_split: Optional[float] = 0.2
    ):
        """
        Fine-tune a pre-trained 3D model with new data.

        Returns:
            Dictionary with fine-tuning results and model paths
        """
        try:
            # Handle None values for optional parameters (set defaults)
            if fine_tune_learning_rate is None:
                fine_tune_learning_rate = 1e-5
            if backbone_learning_rate is None:
                backbone_learning_rate = 1e-6
            if validation_split is None:
                validation_split = 0.2

            # Set up logging
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "fine_tuning.log")

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

            logging.info("Starting 3D model fine-tuning...")
            logging.info(f"Pretrained model: {pretrained_model_path}")
            logging.info(f"Fine-tuning strategy: {fine_tuning_strategy}")
            logging.info(f"Learning rate: {fine_tune_learning_rate}")
            logging.info(f"Handle class imbalance: {handle_class_imbalance}")

            # Load original model configuration
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Model config file not found: {config_path}")

            with open(config_path, 'r') as f:
                original_config = json.load(f)

            model_type = original_config['model_type']
            original_num_classes = original_config['num_classes']
            model_variant = original_config.get('model_variant', model_type)

            logging.info(f"Original model: {model_type} ({model_variant})")
            logging.info(f"Original classes: {original_num_classes}")

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

            # Load pretrained model
            logging.info("Loading pretrained model...")
            model = self._load_pretrained_model(
                model_variant, original_config, pretrained_model_path,
                original_num_classes, final_num_classes, device, add_new_classes
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
                freeze_layers = self._get_default_freeze_layers(model_type, fine_tuning_strategy)

            self._apply_fine_tuning_strategy(
                model, fine_tuning_strategy, freeze_layers, model_type
            )

            # Setup data transforms
            train_transform, val_transform = self._get_3d_transforms(augmentation_level)

            # Load data and setup dataloaders
            train_loader, val_loader, class_distribution = self._setup_dataloaders_with_imbalance(
                new_data_path, train_transform, val_transform, batch_size,
                handle_class_imbalance, imbalance_strategy, sampling_strategy,
                final_num_classes, validation_split
            )

            logging.info(f"Training dataset size: {len(train_loader.dataset)}")
            logging.info(f"Validation dataset size: {len(val_loader.dataset)}")

            # Handle class imbalance
            class_weights_tensor = None
            if handle_class_imbalance and class_distribution is not None:
                logging.info(f"Class distribution: {class_distribution}")

                max_class_count = np.max(class_distribution)
                min_class_count = np.min(class_distribution[class_distribution > 0])
                imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else 1
                logging.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")

                if imbalance_strategy in ['weighted_loss', 'focal_loss']:
                    # Get train labels for weight calculation
                    # Handle both direct datasets and wrapped datasets (Subset from random_split)
                    train_labels = []
                    dataset = train_loader.dataset

                    if hasattr(dataset, 'targets'):
                        train_labels = dataset.targets
                    elif hasattr(dataset, 'labels'):
                        train_labels = dataset.labels
                    elif hasattr(dataset, 'dataset'):
                        # Handle Subset from random_split
                        base_dataset = dataset.dataset
                        if hasattr(base_dataset, 'targets'):
                            indices = dataset.indices
                            train_labels = np.array(base_dataset.targets)[indices]
                        elif hasattr(base_dataset, 'labels'):
                            indices = dataset.indices
                            train_labels = np.array(base_dataset.labels)[indices]

                    if len(train_labels) > 0:
                        if class_weights == 'balanced' or class_weights is None:
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
                        if final_num_classes == 2:
                            return roc_auc_score(y_true, y_pred_proba[:, 1])
                        else:
                            return roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                    except:
                        return accuracy_score(y_true, y_pred)
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
                    self._gradual_unfreeze(model, epoch, warmup_epochs, model_type)

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

            # Evaluate on test set if available in data
            test_acc, test_metrics, metrics_path, cm_path = self._evaluate_on_test_set(
                model, new_data_path, device, criterion, final_num_classes,
                output_dir, evaluation_metric
            )

            # Save training history
            history_path = os.path.join(output_dir, "fine_tuning_history.json")
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)

            # Create plots
            plots_path = self._create_fine_tuning_plots(
                history, output_dir, handle_class_imbalance, preserve_original_performance
            )

            # Save fine-tuning configuration
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
                'handle_class_imbalance': handle_class_imbalance,
                'imbalance_strategy': imbalance_strategy if handle_class_imbalance else None,
                'class_distribution': class_distribution.tolist() if class_distribution is not None else None,
                'imbalance_ratio': float(imbalance_ratio) if 'imbalance_ratio' in locals() else None,
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
                "new_data_path": new_data_path,
                "output_dir": output_dir
            }

    def _load_pretrained_model(self, model_variant, config, model_path,
                             original_num_classes, final_num_classes, device, add_new_classes):
        """Load the pretrained 3D model and modify if needed for new classes."""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model_type = config.get('model_type', 'unknown')

        # Create model architecture
        if 'resnet' in model_type.lower():
            # ResNet models
            resnet_models = {
                'resnet10': resnet10,
                'resnet18': resnet18,
                'resnet34': resnet34,
                'resnet50': resnet50
            }

            if model_variant not in resnet_models:
                logging.warning(f"Unknown ResNet variant: {model_variant}. Using resnet18.")
                model_variant = 'resnet18'

            model_fn = resnet_models[model_variant]
            model = model_fn(
                spatial_dims=3,
                n_input_channels=1,
                num_classes=original_num_classes,
                pretrained=False
            )

        elif 'densenet' in model_type.lower():
            # Get config_type from original config (standard or custom)
            config_type = config.get('config_type', 'custom')

            # Standard DenseNet configurations (full architecture)
            standard_configs = {
                'densenet121': {'init_features': 64, 'growth_rate': 32, 'block_config': (6, 12, 24, 16), 'bn_size': 4, 'dropout_prob': 0.0},
                'densenet169': {'init_features': 64, 'growth_rate': 32, 'block_config': (6, 12, 32, 32), 'bn_size': 4, 'dropout_prob': 0.0},
                'densenet201': {'init_features': 64, 'growth_rate': 32, 'block_config': (6, 12, 48, 32), 'bn_size': 4, 'dropout_prob': 0.0},
                'densenet264': {'init_features': 64, 'growth_rate': 32, 'block_config': (6, 12, 64, 48), 'bn_size': 4, 'dropout_prob': 0.0}
            }

            # Custom configurations for small volumes
            custom_configs = {
                'densenet121': {'init_features': 32, 'growth_rate': 16, 'block_config': (3, 6, 12), 'bn_size': 4, 'dropout_prob': 0.0},
                'densenet169': {'init_features': 32, 'growth_rate': 16, 'block_config': (3, 6, 12, 12), 'bn_size': 4, 'dropout_prob': 0.0},
                'densenet201': {'init_features': 32, 'growth_rate': 16, 'block_config': (3, 6, 12, 16), 'bn_size': 4, 'dropout_prob': 0.0},
                'densenet264': {'init_features': 32, 'growth_rate': 16, 'block_config': (3, 6, 16, 20), 'bn_size': 4, 'dropout_prob': 0.0}
            }

            # Select configuration based on config_type
            densenet_configs = standard_configs if config_type == 'standard' else custom_configs
            logging.info(f"Using {config_type} DenseNet configuration")

            if model_variant not in densenet_configs:
                logging.warning(f"Unknown DenseNet variant: {model_variant}. Using densenet121.")
                model_variant = 'densenet121'

            dn_config = densenet_configs[model_variant]
            model = DenseNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=original_num_classes,
                init_features=dn_config['init_features'],
                growth_rate=dn_config['growth_rate'],
                block_config=dn_config['block_config'],
                bn_size=dn_config['bn_size'],
                dropout_prob=dn_config['dropout_prob']
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Load the pretrained weights
        model.load_state_dict(state_dict)
        logging.info(f"Loaded pretrained weights from {model_path}")

        # If adding new classes, modify the final layer
        if add_new_classes:
            logging.info(f"Expanding classifier from {original_num_classes} to {final_num_classes} classes")

            if 'resnet' in model_type.lower():
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

            elif 'densenet' in model_type.lower():
                # DenseNet uses class_layers.out
                old_weight = model.class_layers.out.weight.data.clone()
                old_bias = model.class_layers.out.bias.data.clone()

                # Create new layer
                in_features = model.class_layers.out.in_features
                model.class_layers.out = nn.Linear(in_features, final_num_classes)

                # Initialize new layer
                nn.init.xavier_uniform_(model.class_layers.out.weight)
                nn.init.zeros_(model.class_layers.out.bias)

                # Copy old weights
                model.class_layers.out.weight.data[:original_num_classes] = old_weight
                model.class_layers.out.bias.data[:original_num_classes] = old_bias

        model = model.to(device)
        return model

    def _get_default_freeze_layers(self, model_type, strategy):
        """Get default number of layers to freeze based on model type and strategy."""
        if strategy == "head_only":
            return -1  # Freeze all except head
        elif strategy == "full":
            return 0   # Freeze nothing
        elif strategy in ["partial", "gradual_unfreezing"]:
            # Conservative defaults
            if 'resnet' in model_type.lower():
                return 2  # Freeze first 2 residual blocks
            elif 'densenet' in model_type.lower():
                return 2  # Freeze first 2 dense blocks
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
            if hasattr(model, 'fc'):  # ResNet
                for param in model.fc.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'class_layers'):  # DenseNet
                for param in model.class_layers.parameters():
                    param.requires_grad = True

            logging.info("Head-only fine-tuning: Only classifier layers trainable")

        elif strategy in ["partial", "gradual_unfreezing"]:
            # Freeze specified number of early layers
            if 'resnet' in model_type.lower():
                self._freeze_resnet_layers(model, freeze_layers)
            elif 'densenet' in model_type.lower():
                self._freeze_densenet_layers(model, freeze_layers)

            frozen_count = sum(1 for param in model.parameters() if not param.requires_grad)
            total_count = sum(1 for param in model.parameters())
            logging.info(f"Partial fine-tuning: Frozen {frozen_count}/{total_count} parameters")

    def _freeze_resnet_layers(self, model, freeze_layers):
        """Freeze specified number of ResNet layers."""
        layers_to_freeze = []

        if freeze_layers >= 1:
            if hasattr(model, 'conv1'):
                layers_to_freeze.append(model.conv1)
            if hasattr(model, 'bn1'):
                layers_to_freeze.append(model.bn1)
        if freeze_layers >= 2:
            if hasattr(model, 'layer1'):
                layers_to_freeze.append(model.layer1)
        if freeze_layers >= 3:
            if hasattr(model, 'layer2'):
                layers_to_freeze.append(model.layer2)
        if freeze_layers >= 4:
            if hasattr(model, 'layer3'):
                layers_to_freeze.append(model.layer3)
        if freeze_layers >= 5:
            if hasattr(model, 'layer4'):
                layers_to_freeze.append(model.layer4)

        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    def _freeze_densenet_layers(self, model, freeze_layers):
        """Freeze specified number of DenseNet layers."""
        if hasattr(model, 'features'):
            feature_layers = list(model.features.children())
            for i in range(min(freeze_layers, len(feature_layers))):
                for param in feature_layers[i].parameters():
                    param.requires_grad = False

    def _gradual_unfreeze(self, model, current_epoch, warmup_epochs, model_type):
        """Gradually unfreeze layers during warmup epochs."""
        unfreeze_step = current_epoch + 1
        total_steps = warmup_epochs

        if 'resnet' in model_type.lower():
            layers = []
            if hasattr(model, 'layer1'):
                layers.append(model.layer1)
            if hasattr(model, 'layer2'):
                layers.append(model.layer2)
            if hasattr(model, 'layer3'):
                layers.append(model.layer3)
            if hasattr(model, 'layer4'):
                layers.append(model.layer4)
        elif 'densenet' in model_type.lower():
            if hasattr(model, 'features'):
                layers = list(model.features.children())
            else:
                return
        else:
            return

        # Calculate how many layers to unfreeze
        layers_to_unfreeze = int((unfreeze_step / total_steps) * len(layers))

        # Unfreeze layers from the end (deeper layers first)
        for i in range(max(0, len(layers) - layers_to_unfreeze), len(layers)):
            for param in layers[i].parameters():
                param.requires_grad = True

        logging.info(f"Gradual unfreezing: Unfroze {layers_to_unfreeze}/{len(layers)} layer groups")

    def _get_3d_transforms(self, augmentation_level):
        """Get 3D training and validation transforms."""
        # Validation transform (no augmentation)
        val_transform = None

        # Training transform with augmentation
        if augmentation_level == "none":
            train_transform = None
        elif augmentation_level == "basic":
            train_transform = Compose([
                RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                RandFlip(prob=0.5, spatial_axis=0),
                RandFlip(prob=0.5, spatial_axis=1),
                RandFlip(prob=0.5, spatial_axis=2),
            ])
        elif augmentation_level == "moderate":
            train_transform = Compose([
                RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                RandFlip(prob=0.5, spatial_axis=0),
                RandFlip(prob=0.5, spatial_axis=1),
                RandFlip(prob=0.5, spatial_axis=2),
                RandGaussianNoise(prob=0.3, mean=0.0, std=0.1),
                RandAdjustContrast(prob=0.3, gamma=(0.8, 1.2)),
                RandScaleIntensity(factors=0.2, prob=0.3),
            ])
        elif augmentation_level == "heavy":
            train_transform = Compose([
                RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                RandRotate90(prob=0.5, spatial_axes=(0, 2)),
                RandFlip(prob=0.5, spatial_axis=0),
                RandFlip(prob=0.5, spatial_axis=1),
                RandFlip(prob=0.5, spatial_axis=2),
                RandGaussianNoise(prob=0.3, mean=0.0, std=0.1),
                RandAdjustContrast(prob=0.3, gamma=(0.7, 1.3)),
                RandScaleIntensity(factors=0.3, prob=0.3),
            ])
        else:
            train_transform = None

        return train_transform, val_transform

    def _setup_dataloaders_with_imbalance(self, data_path, train_transform, val_transform,
                                        batch_size, handle_class_imbalance, imbalance_strategy,
                                        sampling_strategy, num_classes, validation_split):
        """Setup training and validation dataloaders with imbalance handling."""
        # Load data
        data = np.load(data_path)

        # Check for train/val split
        if 'train_images' in data and 'train_labels' in data:
            train_images = data['train_images']
            train_labels = data['train_labels']

            if 'val_images' in data and 'val_labels' in data:
                val_images = data['val_images']
                val_labels = data['val_labels']
                use_split = False
            else:
                use_split = True
        elif 'images' in data and 'labels' in data:
            # Single dataset - will split
            train_images = data['images']
            train_labels = data['labels']
            use_split = True
        else:
            raise ValueError("Data file must contain either (train_images, train_labels) or (images, labels)")

        # Create validation dataset
        if use_split:
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

        # Get class distribution
        def get_class_distribution(dataset):
            if hasattr(dataset, 'targets'):
                return np.bincount(dataset.targets, minlength=num_classes)
            elif hasattr(dataset, 'labels'):
                return np.bincount(dataset.labels, minlength=num_classes)
            elif hasattr(dataset, 'dataset'):
                base_dataset = dataset.dataset
                if hasattr(base_dataset, 'targets'):
                    indices = dataset.indices
                    targets = np.array(base_dataset.targets)[indices]
                    return np.bincount(targets, minlength=num_classes)
            return None

        class_distribution = get_class_distribution(train_dataset)
        original_train_dataset = train_dataset

        # Apply sampling if needed
        if handle_class_imbalance and imbalance_strategy in ['oversampling', 'undersampling'] and class_distribution is not None:
            logging.info(f"Applying {imbalance_strategy} with strategy: {sampling_strategy}")

            # Get train labels
            if hasattr(train_dataset, 'targets'):
                train_labels_list = train_dataset.targets
            elif hasattr(train_dataset, 'labels'):
                train_labels_list = train_dataset.labels
            elif hasattr(train_dataset, 'dataset'):
                base_dataset = train_dataset.dataset
                indices = train_dataset.indices
                if hasattr(base_dataset, 'targets'):
                    train_labels_list = np.array(base_dataset.targets)[indices]
                elif hasattr(base_dataset, 'labels'):
                    train_labels_list = np.array(base_dataset.labels)[indices]

            # Create simplified features for sampling
            X_train = np.arange(len(train_labels_list)).reshape(-1, 1)
            y_train = np.array(train_labels_list)

            # Apply sampling
            if imbalance_strategy == 'oversampling':
                sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
            else:
                sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)

            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            resampled_indices = X_resampled.flatten()
            train_dataset = ResampledDataset(original_train_dataset, resampled_indices, y_resampled)
            class_distribution = np.bincount(y_resampled, minlength=num_classes)

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

    def _setup_optimizer(self, model, strategy, fine_tune_lr, backbone_lr, weight_decay,
                        reset_optimizer, checkpoint_path=None):
        """Setup optimizer with differential learning rates."""

        if strategy == "head_only":
            # Only optimize classifier parameters
            if hasattr(model, 'fc'):  # ResNet
                optimizer = optim.AdamW(model.fc.parameters(), lr=fine_tune_lr, weight_decay=weight_decay)
            elif hasattr(model, 'class_layers'):  # DenseNet
                optimizer = optim.AdamW(model.class_layers.parameters(), lr=fine_tune_lr, weight_decay=weight_decay)
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
                    if 'fc' in name or 'class_layers' in name:
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

    def _evaluate_on_test_set(self, model, data_path, device, criterion, num_classes,
                            output_dir, evaluation_metric):
        """Evaluate the fine-tuned model on test set if available."""
        # Check if test data exists in the file
        try:
            data = np.load(data_path)

            if 'test_images' not in data or 'test_labels' not in data:
                logging.info("No test set found in data file")
                return None, {}, None, None

            test_images = data['test_images']
            test_labels = data['test_labels']

            logging.info("Evaluating fine-tuned model on test set...")

            test_dataset = MedMNIST3DDataset(test_images, test_labels, transform=None)
            test_loader = DataLoader(
                test_dataset,
                batch_size=16,
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

            logging.info("Fine-tuned Model Test Results:")
            for metric, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    logging.info(f"  {metric}: {value:.4f}")

            return test_acc, test_metrics, metrics_path, cm_path

        except Exception as e:
            logging.warning(f"Could not evaluate on test set: {e}")
            return None, {}, None, None

    def _create_fine_tuning_plots(self, history, output_dir, handle_class_imbalance,
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
