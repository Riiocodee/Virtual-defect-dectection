"""
CNN Model Architectures for Manufacturing Defect Detection

This module implements:
- Custom CNN architecture
- Pre-trained models (ResNet50, EfficientNet, MobileNet)
- Vision Transformer (ViT) implementation
- Model factory for easy model selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights, MobileNet_V2_Weights
import timm
from typing import Dict, Optional, Tuple


class CustomCNN(nn.Module):
    """Custom CNN architecture for defect detection"""
    
    def __init__(self, num_classes: int = 2, input_channels: int = 3):
        super(CustomCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNet50DefectDetector(nn.Module):
    """ResNet50-based defect detector"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(ResNet50DefectDetector, self).__init__()
        
        # Load pre-trained ResNet50
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace final classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class EfficientNetDefectDetector(nn.Module):
    """EfficientNet-B0 based defect detector"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(EfficientNetDefectDetector, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        if pretrained:
            self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace final classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MobileNetDefectDetector(nn.Module):
    """MobileNetV2 based defect detector"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(MobileNetDefectDetector, self).__init__()
        
        # Load pre-trained MobileNetV2
        if pretrained:
            self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.mobilenet_v2(weights=None)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-15]:
            param.requires_grad = False
        
        # Replace final classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class VisionTransformerDefectDetector(nn.Module):
    """Vision Transformer (ViT) based defect detector"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, model_name: str = "vit_base_patch16_224"):
        super(VisionTransformerDefectDetector, self).__init__()
        
        # Load pre-trained ViT from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # Add dropout for better regularization
        if hasattr(self.backbone, 'head'):
            if isinstance(self.backbone.head, nn.Linear):
                self.backbone.head = nn.Sequential(
                    nn.Dropout(0.5),
                    self.backbone.head
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ModelFactory:
    """Factory class for creating models"""
    
    @staticmethod
    def create_model(model_type: str, 
                    num_classes: int = 2, 
                    pretrained: bool = True,
                    **kwargs) -> nn.Module:
        """
        Create a model based on the specified type
        
        Args:
            model_type: Type of model to create
            num_classes: Number of output classes
            pretrained: Whether to use pre-trained weights
            **kwargs: Additional model-specific arguments
            
        Returns:
            PyTorch model
        """
        model_type = model_type.lower()
        
        if model_type == "custom_cnn":
            return CustomCNN(num_classes=num_classes, **kwargs)
        
        elif model_type == "resnet50":
            return ResNet50DefectDetector(num_classes=num_classes, pretrained=pretrained)
        
        elif model_type == "efficientnet":
            return EfficientNetDefectDetector(num_classes=num_classes, pretrained=pretrained)
        
        elif model_type == "mobilenet":
            return MobileNetDefectDetector(num_classes=num_classes, pretrained=pretrained)
        
        elif model_type == "vit":
            return VisionTransformerDefectDetector(num_classes=num_classes, pretrained=pretrained, **kwargs)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """Get dictionary of available models and their descriptions"""
        return {
            "custom_cnn": "Custom CNN architecture designed for defect detection",
            "resnet50": "ResNet50 pre-trained on ImageNet",
            "efficientnet": "EfficientNet-B0 pre-trained on ImageNet",
            "mobilenet": "MobileNetV2 pre-trained on ImageNet (lightweight)",
            "vit": "Vision Transformer (ViT) pre-trained on ImageNet"
        }
    
    @staticmethod
    def get_model_info(model_type: str) -> Dict:
        """Get information about a specific model"""
        models_info = {
            "custom_cnn": {
                "parameters": "2.5M",
                "input_size": "224x224",
                "speed": "Fast",
                "accuracy": "Medium",
                "memory_usage": "Low"
            },
            "resnet50": {
                "parameters": "25.6M",
                "input_size": "224x224",
                "speed": "Medium",
                "accuracy": "High",
                "memory_usage": "Medium"
            },
            "efficientnet": {
                "parameters": "5.3M",
                "input_size": "224x224",
                "speed": "Fast",
                "accuracy": "High",
                "memory_usage": "Low"
            },
            "mobilenet": {
                "parameters": "3.5M",
                "input_size": "224x224",
                "speed": "Very Fast",
                "accuracy": "Medium",
                "memory_usage": "Very Low"
            },
            "vit": {
                "parameters": "86M",
                "input_size": "224x224",
                "speed": "Slow",
                "accuracy": "Very High",
                "memory_usage": "High"
            }
        }
        
        return models_info.get(model_type.lower(), {})


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def test_model(model: nn.Module, input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)):
    """
    Test model with dummy input
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch, channels, height, width)
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Model test successful!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    # Test different models
    print("Testing CNN Models for Defect Detection\n")
    
    available_models = ModelFactory.get_available_models()
    
    for model_name, description in available_models.items():
        print(f"\n{'='*50}")
        print(f"Testing: {model_name.upper()}")
        print(f"Description: {description}")
        print(f"{'='*50}")
        
        try:
            # Create model
            model = ModelFactory.create_model(model_name, num_classes=2)
            
            # Test model
            test_model(model)
            
            # Get model info
            info = ModelFactory.get_model_info(model_name)
            if info:
                print(f"Model Info: {info}")
            
        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
    
    print("\n" + "="*50)
    print("Model testing completed!")
    print("="*50)
