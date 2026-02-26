"""
Evaluation Metrics Module for Manufacturing Defect Detection

This module implements:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- ROC curve and AUC
- Classification report
- Per-class performance analysis
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ClassificationMetrics:
    """Comprehensive classification metrics for defect detection"""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize metrics calculator
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names or ["non_defective", "defective"]
        self.num_classes = len(self.class_names)
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate basic classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        return metrics
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            normalize: bool = False, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the matrix
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        cm = self.calculate_confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                   xticklabels=self.class_names, yticklabels=self.class_names)
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        # Add text annotations for each cell
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j + 0.5, i + 0.5, format(cm[i, j], fmt),
                       ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, 
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_scores: Predicted probabilities
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])  # Assuming binary classification
        auc_score = roc_auc_score(y_true, y_scores[:, 1])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_scores: Predicted probabilities
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        precision, recall, _ = precision_recall_curve(y_true, y_scores[:, 1])
        avg_precision = average_precision_score(y_true, y_scores[:, 1])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, color='blue', lw=2, 
                label=f'PR Curve (AP = {avg_precision:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report string
        """
        return classification_report(y_true, y_pred, target_names=self.class_names)
    
    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Calculate detailed per-class metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            DataFrame with per-class metrics
        """
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)
        support = np.bincount(y_true)
        
        metrics_df = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        }, index=self.class_names)
        
        # Add macro and weighted averages
        macro_precision = precision_score(y_true, y_pred, average='macro')
        macro_recall = recall_score(y_true, y_pred, average='macro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        weighted_precision = precision_score(y_true, y_pred, average='weighted')
        weighted_recall = recall_score(y_true, y_pred, average='weighted')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        
        metrics_df.loc['Macro Avg'] = [macro_precision, macro_recall, macro_f1, support.sum()]
        metrics_df.loc['Weighted Avg'] = [weighted_precision, weighted_recall, weighted_f1, support.sum()]
        
        return metrics_df
    
    def create_interactive_dashboard(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_scores: np.ndarray) -> go.Figure:
        """
        Create interactive dashboard with all metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Predicted probabilities
            
        Returns:
            Plotly figure
        """
        # Calculate metrics
        metrics = self.calculate_basic_metrics(y_true, y_pred)
        cm = self.calculate_confusion_matrix(y_true, y_pred)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confusion Matrix', 'ROC Curve', 'Metrics Summary', 'Class Distribution'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Confusion Matrix
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=self.class_names,
                y=self.class_names,
                colorscale='Blues',
                showscale=True,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 12}
            ),
            row=1, col=1
        )
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])
        auc_score = roc_auc_score(y_true, y_scores[:, 1])
        
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC (AUC = {auc_score:.3f})',
                line=dict(color='blue', width=2)
            ),
            row=1, col=2
        )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=2
        )
        
        # Metrics Summary
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [metrics['accuracy'], metrics['precision'], 
                        metrics['recall'], metrics['f1_score']]
        
        fig.add_trace(
            go.Bar(
                x=metric_names,
                y=metric_values,
                text=[f'{v:.3f}' for v in metric_values],
                textposition='auto',
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            ),
            row=2, col=1
        )
        
        # Class Distribution
        unique, counts = np.unique(y_true, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        fig.add_trace(
            go.Bar(
                x=[self.class_names[i] for i in unique],
                y=counts,
                text=counts,
                textposition='auto',
                marker_color='#17becf'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Model Evaluation Dashboard',
            height=800,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Predicted", row=1, col=1)
        fig.update_yaxes(title_text="True", row=1, col=1)
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)
        fig.update_xaxes(title_text="Metrics", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        fig.update_xaxes(title_text="Class", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        return fig
    
    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader,
                      device: str = "cpu") -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            model: PyTorch model
            data_loader: Data loader for evaluation
            device: Device to run evaluation on
            
        Returns:
            Dictionary with all evaluation results
        """
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(device), targets.to(device)
                
                outputs = model(data)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        y_scores = np.array(all_probabilities)
        
        # Calculate all metrics
        results = {
            'basic_metrics': self.calculate_basic_metrics(y_true, y_pred),
            'confusion_matrix': self.calculate_confusion_matrix(y_true, y_pred),
            'classification_report': self.generate_classification_report(y_true, y_pred),
            'per_class_metrics': self.calculate_per_class_metrics(y_true, y_pred),
            'predictions': y_pred,
            'targets': y_true,
            'probabilities': y_scores
        }
        
        # Calculate ROC AUC for binary classification
        if self.num_classes == 2:
            results['roc_auc'] = roc_auc_score(y_true, y_scores[:, 1])
        
        return results


def evaluate_and_save_results(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                             save_dir: str = "evaluation_results", 
                             device: str = "cpu",
                             class_names: List[str] = None) -> Dict:
    """
    Evaluate model and save comprehensive results
    
    Args:
        model: PyTorch model to evaluate
        data_loader: Data loader for evaluation
        save_dir: Directory to save results
        device: Device to run evaluation on
        class_names: List of class names
        
    Returns:
        Evaluation results dictionary
    """
    from pathlib import Path
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize metrics calculator
    metrics_calculator = ClassificationMetrics(class_names)
    
    # Evaluate model
    results = metrics_calculator.evaluate_model(model, data_loader, device)
    
    # Save basic metrics
    metrics_df = pd.DataFrame([results['basic_metrics']])
    metrics_df.to_csv(save_dir / "basic_metrics.csv", index=False)
    
    # Save per-class metrics
    results['per_class_metrics'].to_csv(save_dir / "per_class_metrics.csv")
    
    # Save classification report
    with open(save_dir / "classification_report.txt", 'w') as f:
        f.write(results['classification_report'])
    
    # Save confusion matrix plot
    fig_cm = metrics_calculator.plot_confusion_matrix(
        results['targets'], results['predictions'], 
        normalize=True, save_path=save_dir / "confusion_matrix.png"
    )
    
    # Save ROC curve (for binary classification)
    if len(class_names) == 2:
        fig_roc = metrics_calculator.plot_roc_curve(
            results['targets'], results['probabilities'],
            save_path=save_dir / "roc_curve.png"
        )
        
        fig_pr = metrics_calculator.plot_precision_recall_curve(
            results['targets'], results['probabilities'],
            save_path=save_dir / "precision_recall_curve.png"
        )
    
    # Save interactive dashboard
    dashboard_fig = metrics_calculator.create_interactive_dashboard(
        results['targets'], results['predictions'], results['probabilities']
    )
    dashboard_fig.write_html(save_dir / "interactive_dashboard.html")
    
    # Save raw predictions and targets
    results_df = pd.DataFrame({
        'true_label': results['targets'],
        'predicted_label': results['predictions'],
        'confidence_non_defective': results['probabilities'][:, 0],
        'confidence_defective': results['probabilities'][:, 1]
    })
    results_df.to_csv(save_dir / "predictions.csv", index=False)
    
    print(f"Evaluation results saved to {save_dir}")
    print(f"Basic metrics: {results['basic_metrics']}")
    
    return results


if __name__ == "__main__":
    # Example usage
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create dummy data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic predictions and targets
    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)
    y_scores = np.random.rand(n_samples, 2)
    y_scores = y_scores / y_scores.sum(axis=1, keepdims=True)  # Normalize
    y_pred = np.argmax(y_scores, axis=1)
    
    # Initialize metrics calculator
    metrics = ClassificationMetrics(["non_defective", "defective"])
    
    # Calculate metrics
    basic_metrics = metrics.calculate_basic_metrics(y_true, y_pred)
    print("Basic Metrics:")
    for metric, value in basic_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Generate classification report
    print("\nClassification Report:")
    print(metrics.generate_classification_report(y_true, y_pred))
    
    # Create plots
    fig_cm = metrics.plot_confusion_matrix(y_true, y_pred, normalize=True)
    plt.show()
    
    fig_roc = metrics.plot_roc_curve(y_true, y_scores)
    plt.show()
    
    # Create interactive dashboard
    dashboard = metrics.create_interactive_dashboard(y_true, y_pred, y_scores)
    dashboard.show()
    
    print("Metrics evaluation completed!")
