"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation

This module provides:
- Grad-CAM visualization for CNN interpretability
- Support for different model architectures
- Heatmap generation and overlay
- Multiple target layer support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt


class GradCAM:
    """
    Grad-CAM implementation for CNN interpretability
    
    Reference: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    """
    
    def __init__(self, model: nn.Module, target_layer: str, use_cuda: bool = True):
        """
        Initialize Grad-CAM
        
        Args:
            model: PyTorch model
            target_layer: Name of target layer for visualization
            use_cuda: Whether to use CUDA
        """
        self.model = model
        self.target_layer = target_layer
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        if self.use_cuda:
            self.model = model.cuda()
        
        # Hook for gradients and features
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer and register hooks
        target_module = self._find_target_layer()
        if target_module is not None:
            target_module.register_forward_hook(forward_hook)
            target_module.register_backward_hook(backward_hook)
        else:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model")
    
    def _find_target_layer(self) -> Optional[nn.Module]:
        """Find the target layer in the model"""
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                return module
        return None
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Class Activation Map
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            class_idx: Target class index. If None, uses predicted class
            
        Returns:
            Class activation map
        """
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1)  # (1, H, W)
        cam = F.relu(cam)  # Apply ReLU
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Convert to numpy
        cam = cam.squeeze().cpu().numpy()
        
        return cam
    
    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Make GradCAM callable"""
        return self.generate_cam(input_tensor, class_idx)


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ implementation for better visualization
    
    Reference: Chattopadhyay et al. "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"
    """
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM++ visualization"""
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Grad-CAM++ computation
        # Calculate alpha values
        grads_power_2 = torch.pow(gradients, 2)
        grads_power_3 = torch.pow(gradients, 3)
        
        # Sum over spatial dimensions
        sum_grads = torch.sum(activations * grads_power_2, dim=(2, 3), keepdim=True)
        eps = 1e-7
        alpha = grads_power_2 / (2.0 * grads_power_3 + sum_grads * grads_power_2 + eps)
        
        # Calculate weights
        weights = torch.sum(alpha * torch.relu(gradients), dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1)  # (1, H, W)
        cam = F.relu(cam)  # Apply ReLU
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Convert to numpy
        cam = cam.squeeze().cpu().numpy()
        
        return cam


def visualize_cam(cam: np.ndarray, original_image: np.ndarray, 
                 alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Visualize CAM by overlaying on original image
    
    Args:
        cam: Class activation map
        original_image: Original image (H, W, C) in RGB format
        alpha: Transparency factor for overlay
        colormap: OpenCV colormap
        
    Returns:
        Overlayed image
    """
    # Resize CAM to match original image size
    h, w = original_image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay on original image
    overlayed = original_image * (1 - alpha) + heatmap * alpha
    overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)
    
    return overlayed


def create_multi_layer_cam(model: nn.Module, input_tensor: torch.Tensor,
                          target_layers: List[str], class_idx: Optional[int = None],
                          use_cuda: bool = True) -> Dict[str, np.ndarray]:
    """
    Create CAM visualizations for multiple layers
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor
        target_layers: List of target layer names
        class_idx: Target class index
        use_cuda: Whether to use CUDA
        
    Returns:
        Dictionary mapping layer names to CAM arrays
    """
    cams = {}
    
    for layer_name in target_layers:
        try:
            grad_cam = GradCAM(model, layer_name, use_cuda)
            cam = grad_cam.generate_cam(input_tensor, class_idx)
            cams[layer_name] = cam
        except Exception as e:
            print(f"Warning: Could not generate CAM for layer {layer_name}: {str(e)}")
    
    return cams


def get_default_target_layers(model_type: str) -> List[str]:
    """
    Get default target layers for different model types
    
    Args:
        model_type: Type of model
        
    Returns:
        List of default target layer names
    """
    layer_mapping = {
        "resnet50": ["layer4", "layer3", "layer2"],
        "efficientnet": ["features.8", "features.7", "features.6"],
        "mobilenet": ["features.16", "features.14", "features.12"],
        "custom_cnn": ["features.6", "features.4", "features.2"],
        "vit": ["blocks.11", "blocks.10", "blocks.9"]  # For Vision Transformers
    }
    
    return layer_mapping.get(model_type.lower(), ["layer4"])


def visualize_multiple_cams(cams: Dict[str, np.ndarray], original_image: np.ndarray,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize multiple CAMs side by side
    
    Args:
        cams: Dictionary of CAM arrays
        original_image: Original image
        save_path: Path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    num_cams = len(cams)
    if num_cams == 0:
        return None
    
    # Create subplot grid
    cols = min(3, num_cams + 1)  # +1 for original image
    rows = (num_cams + 1) // cols + (1 if (num_cams + 1) % cols else 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    if num_cams == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten()
    
    # Original image
    axes_flat[0].imshow(original_image)
    axes_flat[0].set_title("Original Image")
    axes_flat[0].axis('off')
    
    # CAM visualizations
    for i, (layer_name, cam) in enumerate(cams.items()):
        if i + 1 < len(axes_flat):
            overlayed = visualize_cam(cam, original_image)
            axes_flat[i + 1].imshow(overlayed)
            axes_flat[i + 1].set_title(f"Layer: {layer_name}")
            axes_flat[i + 1].axis('off')
    
    # Hide unused subplots
    for i in range(len(cams) + 1, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_cam_regions(cam: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Analyze CAM regions to extract insights
    
    Args:
        cam: Class activation map
        threshold: Threshold for considering regions important
        
    Returns:
        Analysis results
    """
    # Find important regions
    important_mask = cam > threshold
    
    # Find connected components
    num_regions, labels, stats, centroids = cv2.connectedComponentsWithStats(
        important_mask.astype(np.uint8), connectivity=8
    )
    
    # Analyze regions
    regions = []
    for i in range(1, num_regions):  # Skip background (0)
        region_info = {
            "region_id": i,
            "area": stats[i, cv2.CC_STAT_AREA],
            "centroid": centroids[i],
            "bbox": (
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT]
            ),
            "max_activation": cam[labels == i].max()
        }
        regions.append(region_info)
    
    # Sort by area
    regions.sort(key=lambda x: x["area"], reverse=True)
    
    analysis = {
        "num_regions": len(regions),
        "total_important_area": np.sum(important_mask),
        "important_percentage": np.sum(important_mask) / cam.size * 100,
        "max_activation_value": cam.max(),
        "mean_activation_value": cam.mean(),
        "regions": regions
    }
    
    return analysis


if __name__ == "__main__":
    # Example usage
    import torch
    from torchvision import models
    
    # Load a pretrained model
    model = models.resnet50(pretrained=True)
    model.eval()
    
    # Create dummy input
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # Create Grad-CAM
    grad_cam = GradCAM(model, "layer4")
    
    # Generate CAM
    cam = grad_cam.generate_cam(input_tensor)
    
    print(f"Generated CAM with shape: {cam.shape}")
    print(f"CAM value range: [{cam.min():.3f}, {cam.max():.3f}]")
    
    # Create dummy original image
    original_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Visualize
    overlayed = visualize_cam(cam, original_image)
    
    # Save result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(overlayed)
    plt.title("Grad-CAM")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("grad_cam_example.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Grad-CAM example completed!")
