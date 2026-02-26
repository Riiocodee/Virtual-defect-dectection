"""
Dataset Management Module for Manufacturing Defect Detection

This module handles:
- Data collection and organization
- Dataset splitting (train/validation/test)
- Data loading and preprocessing
- Label management
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


class DatasetManager:
    """Manage manufacturing defect detection dataset"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.test_dir = self.data_dir / "test"
        
        # Class labels
        self.classes = {"defective": 1, "non_defective": 0}
        self.class_names = list(self.classes.keys())
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directory structure"""
        for dir_path in [self.raw_dir, self.processed_dir, self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create class subdirectories
            for class_name in self.class_names:
                (dir_path / class_name).mkdir(exist_ok=True)
    
    def organize_raw_data(self, source_dir: str, copy_files: bool = True):
        """
        Organize raw data from source directory into class folders
        
        Args:
            source_dir: Path to directory containing images
            copy_files: If True, copy files; if False, move files
        """
        source_path = Path(source_dir)
        
        # Look for common naming patterns
        defect_patterns = ['defect', 'defective', 'fault', 'error', 'bad']
        no_defect_patterns = ['good', 'ok', 'normal', 'perfect', 'non_defective']
        
        for file_path in source_path.glob("*"):
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                file_name = file_path.name.lower()
                
                # Determine class based on filename
                if any(pattern in file_name for pattern in defect_patterns):
                    target_class = "defective"
                elif any(pattern in file_name for pattern in no_defect_patterns):
                    target_class = "non_defective"
                else:
                    print(f"Could not determine class for {file_path.name}. Skipping.")
                    continue
                
                target_path = self.raw_dir / target_class / file_path.name
                
                if copy_files:
                    shutil.copy2(file_path, target_path)
                else:
                    shutil.move(str(file_path), str(target_path))
        
        print(f"Organized data into {self.raw_dir}")
    
    def create_sample_dataset(self, samples_per_class: int = 50):
        """
        Create a sample dataset with synthetic images for testing
        
        Args:
            samples_per_class: Number of samples to generate per class
        """
        print("Creating sample dataset...")
        
        for class_name in self.class_names:
            class_dir = self.raw_dir / class_name
            
            for i in range(samples_per_class):
                # Create synthetic image
                if class_name == "defective":
                    # Create image with defect (random noise pattern)
                    img = self._create_defective_image()
                else:
                    # Create clean image
                    img = self._create_clean_image()
                
                # Save image
                img_path = class_dir / f"{class_name}_{i:04d}.png"
                img.save(img_path)
        
        print(f"Created {samples_per_class} samples per class in {self.raw_dir}")
    
    def _create_defective_image(self, size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """Create a synthetic defective image"""
        img = np.random.randint(50, 150, (*size, 3), dtype=np.uint8)
        
        # Add defect patterns
        num_defects = random.randint(1, 3)
        for _ in range(num_defects):
            x, y = random.randint(0, size[0]-20), random.randint(0, size[1]-20)
            w, h = random.randint(5, 20), random.randint(5, 20)
            
            # Create defect (dark spot or bright spot)
            if random.random() > 0.5:
                img[y:y+h, x:x+w] = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8)
            else:
                img[y:y+h, x:x+w] = np.random.randint(200, 255, (h, w, 3), dtype=np.uint8)
        
        return Image.fromarray(img)
    
    def _create_clean_image(self, size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """Create a synthetic clean image"""
        # Create uniform background
        img = np.full((*size, 3), 128, dtype=np.uint8)
        
        # Add some texture
        noise = np.random.normal(0, 10, (*size, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img)
    
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
        """
        Split dataset into train, validation, and test sets
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        random.seed(seed)
        
        for class_name in self.class_names:
            class_raw_dir = self.raw_dir / class_name
            image_files = list(class_raw_dir.glob("*"))
            
            # Shuffle files
            random.shuffle(image_files)
            
            # Calculate split indices
            n_files = len(image_files)
            n_train = int(n_files * train_ratio)
            n_val = int(n_files * val_ratio)
            
            train_files = image_files[:n_train]
            val_files = image_files[n_train:n_train + n_val]
            test_files = image_files[n_train + n_val:]
            
            # Copy files to respective directories
            self._copy_files_to_split(train_files, self.train_dir / class_name)
            self._copy_files_to_split(val_files, self.val_dir / class_name)
            self._copy_files_to_split(test_files, self.test_dir / class_name)
            
            print(f"{class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    def _copy_files_to_split(self, files: List[Path], target_dir: Path):
        """Copy files to target directory"""
        for file_path in files:
            target_path = target_dir / file_path.name
            shutil.copy2(file_path, target_path)
    
    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
        stats = {}
        
        for split_name, split_dir in [("train", self.train_dir), 
                                      ("val", self.val_dir), 
                                      ("test", self.test_dir)]:
            split_stats = {}
            total_files = 0
            
            for class_name in self.class_names:
                class_dir = split_dir / class_name
                n_files = len(list(class_dir.glob("*")))
                split_stats[class_name] = n_files
                total_files += n_files
            
            split_stats["total"] = total_files
            stats[split_name] = split_stats
        
        return stats
    
    def create_metadata_csv(self, output_path: str = "data/metadata.csv"):
        """Create CSV file with dataset metadata"""
        metadata = []
        
        for split_name, split_dir in [("train", self.train_dir), 
                                      ("val", self.val_dir), 
                                      ("test", self.test_dir)]:
            for class_name in self.class_names:
                class_dir = split_dir / class_name
                for file_path in class_dir.glob("*"):
                    metadata.append({
                        "filename": file_path.name,
                        "filepath": str(file_path.relative_to(self.data_dir)),
                        "label": class_name,
                        "label_id": self.classes[class_name],
                        "split": split_name
                    })
        
        df = pd.DataFrame(metadata)
        df.to_csv(output_path, index=False)
        print(f"Metadata saved to {output_path}")
        
        return df


if __name__ == "__main__":
    # Example usage
    dataset_manager = DatasetManager()
    
    # Create sample dataset
    dataset_manager.create_sample_dataset(samples_per_class=100)
    
    # Split dataset
    dataset_manager.split_dataset()
    
    # Get statistics
    stats = dataset_manager.get_dataset_stats()
    print("\nDataset Statistics:")
    for split, split_stats in stats.items():
        print(f"{split}: {split_stats}")
    
    # Create metadata
    metadata_df = dataset_manager.create_metadata_csv()
    print(f"\nTotal samples: {len(metadata_df)}")
