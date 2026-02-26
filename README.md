#  Manufacturing Defect Detection System

A comprehensive AI-powered computer vision system for detecting manufacturing defects in product images using deep learning.

##  Overview

This project implements an automated inspection system that identifies defects in product images using state-of-the-art CNN and Vision Transformer models. The system provides end-to-end capabilities from data collection to deployment with real-time inference and explainability features.

##  Features

- **Multiple Model Architectures**: Custom CNN, ResNet50, EfficientNet, MobileNet, Vision Transformer
- **Advanced Data Processing**: Comprehensive preprocessing and augmentation pipeline
- **Robust Training**: Early stopping, learning rate scheduling, class imbalance handling
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix, ROC curves
- **Real-time Deployment**: Streamlit-based web interface for live inference
- **Explainability**: Grad-CAM visualizations for model interpretability
- **Model Optimization**: Quantization, ONNX conversion, and performance benchmarking
- **Batch Processing**: Efficient processing of multiple images
- **Interactive Analytics**: Detailed performance dashboards and visualizations

##  Project Structure

```
Virtual_defect_detection/
├── data/                          # Data directories
│   ├── raw/                      # Raw dataset
│   ├── processed/                # Processed data
│   ├── train/                    # Training set
│   ├── val/                      # Validation set
│   └── test/                     # Test set
├── models/                        # Saved models and checkpoints
├── src/                          # Source code
│   ├── data/                     # Data handling modules
│   │   ├── dataset_manager.py    # Dataset collection and management
│   │   └── data_loader.py        # Data loading and preprocessing
│   ├── models/                   # Model architectures
│   │   └── cnn_models.py         # CNN and ViT implementations
│   ├── training/                 # Training modules
│   │   └── trainer.py            # Training loop and utilities
│   ├── evaluation/               # Evaluation modules
│   │   └── metrics.py            # Metrics and visualizations
│   ├── deployment/               # Deployment modules
│   │   └── streamlit_app.py      # Web interface
│   └── utils/                    # Utility modules
│       ├── grad_cam.py           # Grad-CAM visualization
│       └── model_optimization.py # Model optimization
├── notebooks/                    # Jupyter notebooks for experimentation
├── logs/                         # Training logs
├── checkpoints/                  # Model checkpoints
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

##  Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Virtual_defect_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

```bash
# Create sample dataset (for testing)
python -c "from src.data.dataset_manager import DatasetManager; dm = DatasetManager(); dm.create_sample_dataset(samples_per_class=100); dm.split_dataset()"

# Or organize your own data
python -c "from src.data.dataset_manager import DatasetManager; dm = DatasetManager(); dm.organize_raw_data('path/to/your/images')"
```

### 3. Training

```bash
# Train a model
python -c "
from src.models.cnn_models import ModelFactory
from src.data.data_loader import create_data_loaders
from src.training.trainer import Trainer

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders('data', batch_size=32)

# Create model
model = ModelFactory.create_model('resnet50', num_classes=2)

# Train
trainer = Trainer(model, train_loader, val_loader, test_loader)
trainer.setup_training(learning_rate=0.001, scheduler_type='plateau')
trainer.train(num_epochs=50)
"
```

### 4. Deployment

```bash
# Run the Streamlit app
streamlit run src/deployment/streamlit_app.py
```

##  Model Performance

The system supports multiple model architectures with different trade-offs:

| Model | Parameters | Speed | Accuracy | Memory Usage |
|-------|------------|-------|----------|--------------|
| Custom CNN | 2.5M | Fast | Medium | Low |
| ResNet50 | 25.6M | Medium | High | Medium |
| EfficientNet | 5.3M | Fast | High | Low |
| MobileNet | 3.5M | Very Fast | Medium | Very Low |
| Vision Transformer | 86M | Slow | Very High | High |

##  Configuration

### Training Parameters

```python
# Example training configuration
training_config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 100,
    'early_stopping_patience': 15,
    'scheduler_type': 'plateau',
    'weight_decay': 1e-4,
    'image_size': 224
}
```

### Data Augmentation

The system uses comprehensive data augmentation including:
- Horizontal and vertical flipping
- Random rotation (±15°)
- Brightness and contrast adjustments
- Random zoom and shift
- Gaussian noise and blur

### Model Optimization

Available optimization techniques:
- **Dynamic Quantization**: Reduces model size and improves CPU inference
- **Static Quantization**: Better performance with calibration data
- **ONNX Conversion**: Cross-platform deployment
- **Model Pruning**: Reduce parameter count

##  Evaluation Metrics

The system provides comprehensive evaluation:

- **Basic Metrics**: Accuracy, Precision, Recall, F1-Score
- **Advanced Metrics**: ROC-AUC, Precision-Recall AUC
- **Per-Class Analysis**: Detailed performance per class
- **Visualizations**: Confusion matrix, ROC curves, training curves
- **Interactive Dashboards**: Plotly-based analytics

##  Usage Examples

### Single Image Prediction

```python
from src.deployment.streamlit_app import DefectDetectionApp
from PIL import Image

# Load the app
app = DefectDetectionApp()

# Predict on an image
image = Image.open('test_image.jpg')
result = app.predict_single_image(image)

print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Processing

```python
# Process multiple images
import glob
from src.data.data_loader import create_inference_loader

image_paths = glob.glob('test_images/*.jpg')
loader = create_inference_loader(image_paths, batch_size=4)

# Process in batches
for batch, paths in loader:
    # Run inference
    predictions = model(batch)
    # Process results
    pass
```

### Model Evaluation

```python
from src.evaluation.metrics import evaluate_and_save_results

# Comprehensive evaluation
results = evaluate_and_save_results(
    model, test_loader, 
    save_dir="evaluation_results",
    class_names=["non_defective", "defective"]
)
```

##  Grad-CAM Visualization

The system includes Grad-CAM for model interpretability:

```python
from src.utils.grad_cam import GradCAM, visualize_cam

# Create Grad-CAM
grad_cam = GradCAM(model, target_layer='layer4')

# Generate visualization
cam = grad_cam.generate_cam(input_tensor, class_idx=1)
overlayed = visualize_cam(cam, original_image)
```

##  Model Optimization

```python
from src.utils.model_optimization import ModelOptimizer

# Create optimizer
optimizer = ModelOptimizer(model, "defect_detector")

# Compare optimizations
results = optimizer.compare_models(input_shape=(1, 3, 224, 224))

# Print summary
optimizer.print_comparison_summary(results)
```

##  Deployment Options

### Streamlit Web App
- Interactive interface
- Real-time inference
- Batch processing
- Visual analytics

### Flask API (Optional)
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Handle image upload and prediction
    pass
```

### Edge Deployment
- Model quantization for mobile devices
- ONNX Runtime for cross-platform
- TensorFlow Lite support (optional)

##  API Reference

### DatasetManager
```python
from src.data.dataset_manager import DatasetManager

dm = DatasetManager(data_dir="data")
dm.create_sample_dataset(samples_per_class=100)
dm.split_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
```

### ModelFactory
```python
from src.models.cnn_models import ModelFactory

model = ModelFactory.create_model(
    model_type="resnet50",
    num_classes=2,
    pretrained=True
)
```

### Trainer
```python
from src.training.trainer import Trainer

trainer = Trainer(model, train_loader, val_loader)
trainer.setup_training(learning_rate=0.001)
history = trainer.train(num_epochs=50)
```

##  Testing

```bash
# Run unit tests
python -m pytest tests/

# Test model loading
python -c "from src.models.cnn_models import ModelFactory; ModelFactory.create_model('resnet50')"

# Test data loading
python -c "from src.data.data_loader import create_data_loaders; create_data_loaders('data')"
```

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Streamlit](https://streamlit.io/) - Web app framework
- [Albumentations](https://albumentations.ai/) - Data augmentation
- [Grad-CAM](https://arxiv.org/abs/1610.02391) - Explainability technique

##  Support

For questions and support:
- Create an issue on GitHub
- Check the [Wiki](https://github.com/your-repo/wiki) for detailed guides
- Review the [Examples](notebooks/) directory for code samples

##  Version History

- **v1.0.0** - Initial release with core functionality
- **v1.1.0** - Added Vision Transformer support
- **v1.2.0** - Enhanced optimization and deployment features

---

**Built with ❤️ for manufacturing quality control**
