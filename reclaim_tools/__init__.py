"""
Reclaim Tools - PyTorch Medical Image Classification Tools
===========================================================

A collection of tools for training, inference, fine-tuning, and comparing
deep learning models for medical image classification.

Supported Architectures:
2D Models:
- ResNet (18, 34, 50, 101, 152)
- VGG16 (with and without batch normalization)
- Inception V3
- EfficientNetV2 (S, M, L)

3D Models:
- 3D ResNet (10, 18, 34, 50, 101, 152, 200)
- 3D DenseNet (121, 169, 201, 264)

Available Tools:
- Training tools for each architecture
- Inference tools for each architecture
- Fine-tuning tools for 2D and 3D models
- Model performance comparison tool
"""

# 2D Training Tools
from .pytorch_resnet_training_tool import PyTorchResNetTrainingTool
from .pytorch_vgg16_training_tool import PyTorchVGG16TrainingTool
from .pytorch_inception_v3_training_tool import PyTorchInceptionV3TrainingTool
from .pytorch_efficientnetv2_training_tool import PyTorchEfficientNetV2TrainingTool

# 2D Inference Tools
from .pytorch_resnet_inference_tool import PyTorchResNetInferenceTool
from .pytorch_vgg16_inference_tool import PyTorchVGG16InferenceTool
from .pytorch_inception_v3_inference_tool import PyTorchInceptionV3InferenceTool
from .pytorch_efficientnetv2_inference_tool import PyTorchEfficientNetV2InferenceTool

# 3D Training Tools
from .PyTorch3DResNetTrainingTool import PyTorch3DResNetTrainingTool
from .PyTorch3DDenseNetTrainingTool import PyTorch3DDenseNetTrainingTool

# 3D Inference Tools
from .PyTorch3DResNetInferenceTool import PyTorch3DResNetInferenceTool
from .PyTorch3DDenseNetInferenceTool import PyTorch3DDenseNetInferenceTool

# Fine-tuning Tools
from .pytorch_model_fine_tuning_tool import PyTorchModelFineTuningTool
from .PyTorch3DModelFineTuningTool import PyTorch3DModelFineTuningTool

# Comparison Tool
from .model_performance_comparison_tool import ModelPerformanceComparisonTool

__all__ = [
    # 2D Training Tools
    'PyTorchResNetTrainingTool',
    'PyTorchVGG16TrainingTool',
    'PyTorchInceptionV3TrainingTool',
    'PyTorchEfficientNetV2TrainingTool',

    # 2D Inference Tools
    'PyTorchResNetInferenceTool',
    'PyTorchVGG16InferenceTool',
    'PyTorchInceptionV3InferenceTool',
    'PyTorchEfficientNetV2InferenceTool',

    # 3D Training Tools
    'PyTorch3DResNetTrainingTool',
    'PyTorch3DDenseNetTrainingTool',

    # 3D Inference Tools
    'PyTorch3DResNetInferenceTool',
    'PyTorch3DDenseNetInferenceTool',

    # Fine-tuning Tools
    'PyTorchModelFineTuningTool',
    'PyTorch3DModelFineTuningTool',

    # Comparison Tool
    'ModelPerformanceComparisonTool',
]

__version__ = '1.0.0'
