"""
Main entry point for the Manufacturing Defect Detection System

This script provides a command-line interface for:
- Dataset creation and management
- Model training
- Evaluation
- Deployment
- Model optimization
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.dataset_manager import DatasetManager
from models.cnn_models import ModelFactory
from data.data_loader import create_data_loaders
from training.trainer import Trainer
from evaluation.metrics import evaluate_and_save_results
# Optional imports - handle gracefully if not available
try:
    from utils.model_optimization import ModelOptimizer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("Warning: Model optimization features not available (ONNX compatibility issue).")


def create_dataset(args):
    """Create and prepare dataset"""
    print("🔧 Creating dataset...")
    
    dm = DatasetManager(data_dir=args.data_dir)
    
    if args.create_sample:
        dm.create_sample_dataset(samples_per_class=args.samples_per_class)
        print(f"✅ Created sample dataset with {args.samples_per_class} samples per class")
    
    if args.organize_data and args.source_dir:
        dm.organize_raw_data(args.source_dir, copy_files=True)
        print(f"✅ Organized data from {args.source_dir}")
    
    # Split dataset
    dm.split_dataset(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Create metadata
    metadata_df = dm.create_metadata_csv()
    
    # Print statistics
    stats = dm.get_dataset_stats()
    print("\n📊 Dataset Statistics:")
    for split, split_stats in stats.items():
        print(f"  {split}: {split_stats}")
    
    print(f"\n✅ Dataset preparation completed!")
    return True


def train_model(args):
    """Train the model"""
    print("🚀 Starting model training...")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    # Create model
    model = ModelFactory.create_model(
        args.model_type,
        num_classes=2,
        pretrained=args.pretrained
    )
    
    print(f"✅ Created {args.model_type} model")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        save_dir=args.save_dir
    )
    
    # Setup training
    trainer.setup_training(
        learning_rate=args.learning_rate,
        scheduler_type=args.scheduler,
        weight_decay=args.weight_decay
    )
    
    # Train model
    history = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_best=True,
        save_frequency=args.save_frequency
    )
    
    # Test model
    if test_loader:
        test_results = trainer.test()
        print(f"🎯 Test Accuracy: {test_results['test_accuracy']:.2f}%")
    
    print(f"✅ Training completed! Model saved to {args.save_dir}")
    return True


def evaluate_model(args):
    """Evaluate the model"""
    print("📊 Starting model evaluation...")
    
    # Create data loader
    _, _, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    # Load model
    model = ModelFactory.create_model(args.model_type, num_classes=2)
    
    # Load checkpoint
    checkpoint_path = Path(args.save_dir) / "best_model.pth"
    if checkpoint_path.exists():
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from {checkpoint_path}")
    else:
        print(f"❌ Model checkpoint not found at {checkpoint_path}")
        return False
    
    # Evaluate
    results = evaluate_and_save_results(
        model=model,
        data_loader=test_loader,
        save_dir=args.save_dir / "evaluation",
        device=args.device,
        class_names=["non_defective", "defective"]
    )
    
    print(f"✅ Evaluation completed! Results saved to {args.save_dir / 'evaluation'}")
    return True


def optimize_model(args):
    """Optimize the model"""
    print("⚡ Starting model optimization...")
    
    # Load model
    model = ModelFactory.create_model(args.model_type, num_classes=2)
    
    # Load checkpoint
    checkpoint_path = Path(args.save_dir) / "best_model.pth"
    if checkpoint_path.exists():
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from {checkpoint_path}")
    else:
        print(f"❌ Model checkpoint not found at {checkpoint_path}")
        return False
    
    # Create optimizer
    optimizer = ModelOptimizer(model, f"{args.model_type}_defect_detector")
    
    # Compare optimizations
    input_shape = (1, 3, args.image_size, args.image_size)
    results = optimizer.compare_models(input_shape)
    
    # Print summary
    optimizer.print_comparison_summary(results)
    
    print(f"✅ Optimization completed! Results saved to {args.save_dir}")
    return True


def run_deployment(args):
    """Run the deployment app"""
    print("🌐 Starting deployment...")
    
    # Check if model exists
    checkpoint_path = Path(args.save_dir) / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"❌ Model checkpoint not found at {checkpoint_path}")
        print("Please train a model first using: python main.py train")
        return False
    
    # Run Streamlit app
    import subprocess
    app_path = Path(__file__).parent / "src" / "deployment" / "streamlit_app.py"
    
    print(f"🚀 Launching Streamlit app...")
    subprocess.run(["streamlit", "run", str(app_path)])
    
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Manufacturing Defect Detection System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Dataset arguments
    dataset_parser = subparsers.add_parser('dataset', help='Create and manage dataset')
    dataset_parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    dataset_parser.add_argument('--create-sample', action='store_true', help='Create sample dataset')
    dataset_parser.add_argument('--samples-per-class', type=int, default=100, help='Samples per class for sample dataset')
    dataset_parser.add_argument('--organize-data', action='store_true', help='Organize raw data')
    dataset_parser.add_argument('--source-dir', type=str, help='Source directory for raw data')
    dataset_parser.add_argument('--train-ratio', type=float, default=0.7, help='Training data ratio')
    dataset_parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation data ratio')
    dataset_parser.add_argument('--test-ratio', type=float, default=0.15, help='Test data ratio')
    
    # Training arguments
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    train_parser.add_argument('--model-type', type=str, default='resnet50', 
                             choices=['custom_cnn', 'resnet50', 'efficientnet', 'mobilenet', 'vit'],
                             help='Model architecture')
    train_parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--image-size', type=int, default=224, help='Image size')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    train_parser.add_argument('--scheduler', type=str, default='plateau', 
                              choices=['plateau', 'cosine', 'step'], help='Learning rate scheduler')
    train_parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    train_parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    train_parser.add_argument('--save-dir', type=str, default='models', help='Save directory')
    train_parser.add_argument('--save-frequency', type=int, default=10, help='Checkpoint save frequency')
    
    # Evaluation arguments
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    eval_parser.add_argument('--model-type', type=str, default='resnet50', help='Model architecture')
    eval_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    eval_parser.add_argument('--image-size', type=int, default=224, help='Image size')
    eval_parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    eval_parser.add_argument('--save-dir', type=str, default='models', help='Save directory')
    
    # Optimization arguments
    opt_parser = subparsers.add_parser('optimize', help='Optimize the model')
    opt_parser.add_argument('--model-type', type=str, default='resnet50', help='Model architecture')
    opt_parser.add_argument('--image-size', type=int, default=224, help='Image size')
    opt_parser.add_argument('--save-dir', type=str, default='models', help='Save directory')
    
    # Deployment arguments
    deploy_parser = subparsers.add_parser('deploy', help='Run deployment app')
    deploy_parser.add_argument('--save-dir', type=str, default='models', help='Save directory')
    
    args = parser.parse_args()
    
    if args.command == 'dataset':
        create_dataset(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'optimize':
        optimize_model(args)
    elif args.command == 'deploy':
        run_deployment(args)
    else:
        parser.print_help()
        print("\n🔧 Quick Start Guide:")
        print("1. Create sample dataset: python main.py dataset --create-sample")
        print("2. Train model: python main.py train --model-type resnet50")
        print("3. Evaluate model: python main.py evaluate")
        print("4. Run deployment: python main.py deploy")


if __name__ == "__main__":
    main()
