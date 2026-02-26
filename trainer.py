"""
Training Module for Manufacturing Defect Detection

This module handles:
- Model training with proper loss functions and optimizers
- Learning rate scheduling
- Early stopping
- Training/validation loops
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
            model: Current model
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class Trainer:
    """Training class for defect detection models"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 device: str = "auto",
                 save_dir: str = "models"):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            device: Device to use ("auto", "cpu", "cuda")
            save_dir: Directory to save models and logs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": []
        }
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def setup_training(self,
                      learning_rate: float = 0.001,
                      weight_decay: float = 1e-4,
                      scheduler_type: str = "plateau",
                      scheduler_params: Optional[Dict] = None,
                      class_weights: Optional[torch.Tensor] = None):
        """
        Setup training components
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            scheduler_type: Type of learning rate scheduler
            scheduler_params: Parameters for scheduler
            class_weights: Class weights for loss function
        """
        # Loss function
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler_params = scheduler_params or {}
        
        if scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_params.get('factor', 0.5),
                patience=scheduler_params.get('patience', 5)
            )
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_params.get('T_max', 50),
                eta_min=scheduler_params.get('eta_min', 1e-6)
            )
        elif scheduler_type == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=scheduler_params.get('step_size', 20),
                gamma=scheduler_params.get('gamma', 0.1)
            )
        else:
            self.scheduler = None
        
        print(f"Training setup completed:")
        print(f"  Loss: {type(self.criterion).__name__}")
        print(f"  Optimizer: {type(self.optimizer).__name__}")
        print(f"  Scheduler: {type(self.scheduler).__name__ if self.scheduler else 'None'}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            current_acc = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self) -> Tuple[float, float]:
        """
        Validate for one epoch
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
            
            for data, targets in progress_bar:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                current_acc = 100. * correct / total
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self,
              num_epochs: int = 100,
              early_stopping_patience: int = 15,
              save_best: bool = True,
              save_frequency: int = 10) -> Dict:
        """
        Train the model
        
        Args:
            num_epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save the best model
            save_frequency: Frequency of saving checkpoints
            
        Returns:
            Training history
        """
        print(f"Starting training for {num_epochs} epochs...")
        
        # Early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Best model tracking
        best_val_acc = 0.0
        best_epoch = 0
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate_epoch()
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rates"].append(current_lr)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                self.save_model("best_model.pth", epoch, val_acc)
                print(f"New best model saved! (Val Acc: {val_acc:.2f}%)")
            
            # Regular checkpoint saving
            if (epoch + 1) % save_frequency == 0:
                self.save_model(f"checkpoint_epoch_{epoch + 1}.pth", epoch, val_acc)
            
            # Early stopping
            if early_stopping(val_loss, self.model):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch + 1})")
                break
        
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch + 1})")
        
        # Save training history
        self.save_training_history()
        
        # Plot training curves
        self.plot_training_curves()
        
        return self.history
    
    def test(self) -> Dict:
        """
        Test the model on test set
        
        Returns:
            Test results dictionary
        """
        if self.test_loader is None:
            print("No test loader provided. Skipping testing.")
            return {}
        
        print("Testing model...")
        
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(self.test_loader, desc="Testing"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_loss = running_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        
        results = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "predictions": all_predictions,
            "targets": all_targets
        }
        
        print(f"Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.2f}%")
        
        return results
    
    def save_model(self, filename: str, epoch: int, val_acc: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_model(self, filename: str, load_optimizer: bool = True):
        """Load model checkpoint"""
        checkpoint_path = self.save_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"Model loaded from {filename}")
        print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Validation accuracy: {checkpoint.get('val_acc', 'Unknown'):.2f}%")
    
    def save_training_history(self):
        """Save training history to JSON file"""
        history_path = self.save_dir / "training_history.json"
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training history saved to {history_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history["train_loss"], label='Train Loss')
        axes[0, 0].plot(self.history["val_loss"], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.history["train_acc"], label='Train Accuracy')
        axes[0, 1].plot(self.history["val_acc"], label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate curve
        axes[1, 0].plot(self.history["learning_rates"])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Combined plot
        axes[1, 1].plot(self.history["train_loss"], label='Train Loss', alpha=0.7)
        axes[1, 1].plot(self.history["val_loss"], label='Val Loss', alpha=0.7)
        axes[1, 1].set_title('Loss Progression')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training curves saved to {self.save_dir / 'training_curves.png'}")


if __name__ == "__main__":
    # Example usage
    from src.models.cnn_models import ModelFactory
    from src.data.data_loader import create_data_loaders
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir="data",
        batch_size=32,
        image_size=224
    )
    
    # Create model
    model = ModelFactory.create_model("resnet50", num_classes=2)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        save_dir="models"
    )
    
    # Setup training
    trainer.setup_training(
        learning_rate=0.001,
        scheduler_type="plateau",
        scheduler_params={"patience": 5, "factor": 0.5}
    )
    
    # Train model
    history = trainer.train(num_epochs=50, early_stopping_patience=15)
    
    # Test model
    test_results = trainer.test()
    
    print("Training and testing completed!")
