"""
Data Loading and Preprocessing Module for Manufacturing Defect Detection

This module handles:
- Custom PyTorch Dataset implementation
- Data augmentation and preprocessing
- Data loading with proper batching
- Support for both training and inference
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DefectDataset(Dataset):
    """Custom PyTorch Dataset for manufacturing defect detection"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = "train",
                 transform: Optional[transforms.Compose] = None,
                 metadata_file: Optional[str] = None):
        """
        Initialize dataset
        
        Args:
            data_dir: Root directory of the dataset
            split: Dataset split ("train", "val", "test")
            transform: Image transformations
            metadata_file: Path to metadata CSV file
        """
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / split
        self.transform = transform
        self.classes = {"defective": 1, "non_defective": 0}
        self.class_names = list(self.classes.keys())
        
        # Load metadata if provided, otherwise scan directories
        if metadata_file and Path(metadata_file).exists():
            self.metadata = pd.read_csv(metadata_file)
            self.metadata = self.metadata[self.metadata["split"] == split]
        else:
            self.metadata = self._scan_directories()
        
        print(f"Loaded {len(self.metadata)} samples for {split} split")
    
    def _scan_directories(self) -> pd.DataFrame:
        """Scan directories to create metadata"""
        metadata = []
        
        for class_name in self.class_names:
            class_dir = self.split_dir / class_name
            for file_path in class_dir.glob("*"):
                if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    metadata.append({
                        "filename": file_path.name,
                        "filepath": str(file_path),
                        "label": class_name,
                        "label_id": self.classes[class_name]
                    })
        
        return pd.DataFrame(metadata)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item from dataset
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (image, label)
        """
        # Get metadata
        row = self.metadata.iloc[idx]
        image_path = row["filepath"]
        label = row["label_id"]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


class AlbumentationsDataset(Dataset):
    """Dataset using Albumentations for advanced augmentations"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = "train",
                 transform: Optional[A.Compose] = None,
                 metadata_file: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / split
        self.transform = transform
        self.classes = {"defective": 1, "non_defective": 0}
        self.class_names = list(self.classes.keys())
        
        # Load metadata
        if metadata_file and Path(metadata_file).exists():
            self.metadata = pd.read_csv(metadata_file)
            self.metadata = self.metadata[self.metadata["split"] == split]
        else:
            self.metadata = self._scan_directories()
        
        print(f"Loaded {len(self.metadata)} samples for {split} split")
    
    def _scan_directories(self) -> pd.DataFrame:
        """Scan directories to create metadata"""
        metadata = []
        
        for class_name in self.class_names:
            class_dir = self.split_dir / class_name
            for file_path in class_dir.glob("*"):
                if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    metadata.append({
                        "filename": file_path.name,
                        "filepath": str(file_path),
                        "label": class_name,
                        "label_id": self.classes[class_name]
                    })
        
        return pd.DataFrame(metadata)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.metadata.iloc[idx]
        image_path = row["filepath"]
        label = row["label_id"]
        
        # Load image with OpenCV (BGR) and convert to RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image=image)["image"]
        
        return image, label


def get_train_transforms(image_size: int = 224, use_albumentations: bool = True) -> transforms.Compose:
    """
    Get training transformations with augmentation
    
    Args:
        image_size: Target image size
        use_albumentations: Whether to use Albumentations for augmentations
        
    Returns:
        Transformation pipeline
    """
    if use_albumentations:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_val_transforms(image_size: int = 224, use_albumentations: bool = True) -> transforms.Compose:
    """
    Get validation/test transformations (no augmentation)
    
    Args:
        image_size: Target image size
        use_albumentations: Whether to use Albumentations
        
    Returns:
        Transformation pipeline
    """
    if use_albumentations:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(data_dir: str,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       image_size: int = 224,
                       use_albumentations: bool = True,
                       metadata_file: Optional[str] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        data_dir: Root directory of the dataset
        batch_size: Batch size for data loading
        num_workers: Number of worker processes
        image_size: Target image size
        use_albumentations: Whether to use Albumentations
        metadata_file: Path to metadata CSV file
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get transformations
    train_transform = get_train_transforms(image_size, use_albumentations)
    val_transform = get_val_transforms(image_size, use_albumentations)
    
    # Create datasets
    if use_albumentations:
        train_dataset = AlbumentationsDataset(
            data_dir, "train", train_transform, metadata_file
        )
        val_dataset = AlbumentationsDataset(
            data_dir, "val", val_transform, metadata_file
        )
        test_dataset = AlbumentationsDataset(
            data_dir, "test", val_transform, metadata_file
        )
    else:
        train_dataset = DefectDataset(
            data_dir, "train", train_transform, metadata_file
        )
        val_dataset = DefectDataset(
            data_dir, "val", val_transform, metadata_file
        )
        test_dataset = DefectDataset(
            data_dir, "test", val_transform, metadata_file
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class InferenceDataset(Dataset):
    """Dataset for inference on single images"""
    
    def __init__(self, image_paths: List[str], transform: Optional[transforms.Compose] = None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, image_path


def create_inference_loader(image_paths: List[str],
                           batch_size: int = 1,
                           image_size: int = 224,
                           use_albumentations: bool = True) -> DataLoader:
    """
    Create data loader for inference
    
    Args:
        image_paths: List of image file paths
        batch_size: Batch size
        image_size: Target image size
        use_albumentations: Whether to use Albumentations
        
    Returns:
        DataLoader for inference
    """
    # Get validation transforms (no augmentation)
    transform = get_val_transforms(image_size, use_albumentations)
    
    # Create dataset
    dataset = InferenceDataset(image_paths, transform)
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return loader


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir="data",
        batch_size=4,
        image_size=224
    )
    
    # Test data loading
    print("Testing data loading...")
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels: {labels}")
        
        # Display first image
        plt.figure(figsize=(10, 2))
        for i in range(min(4, len(images))):
            plt.subplot(1, 4, i+1)
            
            # Denormalize image for display
            img = images[i].permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            plt.imshow(img)
            plt.title(f"Label: {labels[i].item()}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("data_sample.png")
        plt.show()
        
        break
    
    print("Data loading test completed!")
