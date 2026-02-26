"""
Model Optimization Module for Manufacturing Defect Detection

This module provides:
- Model quantization for faster inference
- ONNX model conversion (optional)
- TensorRT optimization (optional)
- Model compression techniques
- Performance benchmarking
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import logging
from tqdm import tqdm

# Optional imports - handle gracefully if not available
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX not available. Model export features will be disabled.")

try:
    import onnxruntime as ort
    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False
    print("Warning: ONNX Runtime not available. ONNX inference will be disabled.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Model optimization utilities"""
    
    def __init__(self, model: nn.Module, model_name: str = "model"):
        """
        Initialize model optimizer
        
        Args:
            model: PyTorch model to optimize
            model_name: Name for saving optimized models
        """
        self.model = model
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Optimization results
        self.optimization_results = {}
    
    def benchmark_model(self, model: nn.Module, input_shape: Tuple[int, ...], 
                       num_runs: int = 100, warmup_runs: int = 10) -> Dict:
        """
        Benchmark model performance
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Benchmark results
        """
        model.eval()
        model.to(self.device)
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)
        
        # Benchmark runs
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        
        start_time = time.time()
        with torch.no_grad():
            for _ in tqdm(range(num_runs), desc="Benchmarking"):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = num_runs / total_time
        
        # Memory usage
        if self.device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        # Model size
        model_size = self._get_model_size(model)
        
        results = {
            "avg_inference_time_ms": avg_time * 1000,
            "throughput_fps": throughput,
            "model_size_mb": model_size,
            "memory_allocated_gb": memory_allocated,
            "memory_reserved_gb": memory_reserved,
            "device": str(self.device)
        }
        
        return results
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def quantize_model_dynamic(self, model: nn.Module) -> nn.Module:
        """
        Apply dynamic quantization to model
        
        Args:
            model: Model to quantize
            
        Returns:
            Quantized model
        """
        logger.info("Applying dynamic quantization...")
        
        # Quantize only linear layers
        quantized_model = quantization.quantize_dynamic(
            model.cpu(),  # Move to CPU for quantization
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        logger.info("Dynamic quantization completed")
        return quantized_model
    
    def quantize_model_static(self, model: nn.Module, 
                             calibration_loader: torch.utils.data.DataLoader) -> nn.Module:
        """
        Apply static quantization to model
        
        Args:
            model: Model to quantize
            calibration_loader: Data loader for calibration
            
        Returns:
            Quantized model
        """
        logger.info("Applying static quantization...")
        
        # Prepare model for quantization
        model.eval()
        model.qconfig = quantization.get_default_qconfig('fbgemm')
        quantization.prepare(model, inplace=True)
        
        # Calibrate with sample data
        logger.info("Calibrating model...")
        with torch.no_grad():
            for data, _ in tqdm(calibration_loader, desc="Calibration"):
                model(data)
                break  # Only need a few batches for calibration
        
        # Convert to quantized model
        quantized_model = quantization.convert(model, inplace=False)
        
        logger.info("Static quantization completed")
        return quantized_model
    
    def convert_to_onnx(self, model: nn.Module, input_shape: Tuple[int, ...],
                       onnx_path: str, opset_version: int = 11) -> str:
        """
        Convert PyTorch model to ONNX format
        
        Args:
            model: PyTorch model to convert
            input_shape: Input tensor shape
            onnx_path: Path to save ONNX model
            opset_version: ONNX opset version
            
        Returns:
            Path to saved ONNX model
        """
        logger.info("Converting model to ONNX...")
        
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"ONNX model saved to {onnx_path}")
        return onnx_path
    
    def optimize_onnx_model(self, onnx_path: str, optimized_path: str) -> str:
        """
        Optimize ONNX model for better performance
        
        Args:
            onnx_path: Path to original ONNX model
            optimized_path: Path to save optimized model
            
        Returns:
            Path to optimized ONNX model
        """
        logger.info("Optimizing ONNX model...")
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Apply optimizations (basic)
        from onnx import optimizer
        
        passes = [
            'eliminate_unused_initializer',
            'fuse_add_bias_into_conv',
            'fuse_bn_into_conv',
            'fuse_consecutive_concats',
            'fuse_consecutive_reduce_unsqueeze',
            'fuse_consecutive_squeezes',
            'fuse_consecutive_transposes',
            'fuse_matmul_add_bias_into_gemm',
            'fuse_pad_into_conv',
            'fuse_transpose_into_gemm',
            'eliminate_nop_transpose',
            'eliminate_nop_pad',
            'eliminate_identity',
            'eliminate_deadend'
        ]
        
        optimized_model = optimizer.optimize(onnx_model, passes)
        
        # Save optimized model
        onnx.save(optimized_model, optimized_path)
        
        logger.info(f"Optimized ONNX model saved to {optimized_path}")
        return optimized_path
    
    def benchmark_onnx_model(self, onnx_path: str, input_shape: Tuple[int, ...],
                           num_runs: int = 100) -> Dict:
        """
        Benchmark ONNX model performance
        
        Args:
            onnx_path: Path to ONNX model
            input_shape: Input tensor shape
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking ONNX model: {onnx_path}")
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path)
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup runs
        for _ in range(10):
            _ = session.run(None, {input_name: dummy_input})
        
        # Benchmark runs
        start_time = time.time()
        for _ in tqdm(range(num_runs), desc="Benchmarking ONNX"):
            _ = session.run(None, {input_name: dummy_input})
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = num_runs / total_time
        
        # Model size
        model_size = Path(onnx_path).stat().st_size / 1024 / 1024  # MB
        
        results = {
            "avg_inference_time_ms": avg_time * 1000,
            "throughput_fps": throughput,
            "model_size_mb": model_size,
            "format": "ONNX"
        }
        
        return results
    
    def prune_model(self, model: nn.Module, pruning_ratio: float = 0.2) -> nn.Module:
        """
        Apply model pruning to reduce size
        
        Args:
            model: Model to prune
            pruning_ratio: Fraction of parameters to prune
            
        Returns:
            Pruned model
        """
        logger.info(f"Applying model pruning with ratio {pruning_ratio}")
        
        # Note: This is a basic implementation
        # For more advanced pruning, consider using torch.nn.utils.prune
        
        import torch.nn.utils.prune as prune
        
        # Global pruning
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio
        )
        
        # Remove pruning reparameterization to make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        logger.info("Model pruning completed")
        return model
    
    def compare_models(self, input_shape: Tuple[int, ...], 
                      calibration_loader: Optional[torch.utils.data.DataLoader] = None) -> Dict:
        """
        Compare original model with optimized versions
        
        Args:
            input_shape: Input tensor shape
            calibration_loader: Data loader for static quantization calibration
            
        Returns:
            Comparison results
        """
        logger.info("Comparing model optimizations...")
        
        results = {}
        
        # Benchmark original model
        logger.info("Benchmarking original model...")
        original_results = self.benchmark_model(self.model, input_shape)
        results["original"] = original_results
        
        # Dynamic quantization
        logger.info("Testing dynamic quantization...")
        try:
            quantized_dynamic = self.quantize_model_dynamic(self.model)
            dynamic_results = self.benchmark_model(quantized_dynamic, input_shape)
            results["dynamic_quantized"] = dynamic_results
        except Exception as e:
            logger.warning(f"Dynamic quantization failed: {str(e)}")
        
        # Static quantization (if calibration loader provided)
        if calibration_loader:
            logger.info("Testing static quantization...")
            try:
                quantized_static = self.quantize_model_static(self.model, calibration_loader)
                static_results = self.benchmark_model(quantized_static, input_shape)
                results["static_quantized"] = static_results
            except Exception as e:
                logger.warning(f"Static quantization failed: {str(e)}")
        
        # ONNX conversion
        logger.info("Testing ONNX conversion...")
        try:
            onnx_path = f"models/{self.model_name}_original.onnx"
            self.convert_to_onnx(self.model, input_shape, onnx_path)
            
            # Benchmark ONNX
            onnx_results = self.benchmark_onnx_model(onnx_path, input_shape)
            results["onnx"] = onnx_results
            
            # Optimized ONNX
            optimized_onnx_path = f"models/{self.model_name}_optimized.onnx"
            self.optimize_onnx_model(onnx_path, optimized_onnx_path)
            
            optimized_onnx_results = self.benchmark_onnx_model(optimized_onnx_path, input_shape)
            results["onnx_optimized"] = optimized_onnx_results
            
        except Exception as e:
            logger.warning(f"ONNX conversion failed: {str(e)}")
        
        # Pruning
        logger.info("Testing model pruning...")
        try:
            pruned_model = self.prune_model(self.model, pruning_ratio=0.2)
            pruned_results = self.benchmark_model(pruned_model, input_shape)
            results["pruned"] = pruned_results
        except Exception as e:
            logger.warning(f"Model pruning failed: {str(e)}")
        
        # Save comparison results
        self._save_comparison_results(results)
        
        return results
    
    def _save_comparison_results(self, results: Dict):
        """Save comparison results to file"""
        results_path = f"models/{self.model_name}_optimization_comparison.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Comparison results saved to {results_path}")
    
    def print_comparison_summary(self, results: Dict):
        """Print comparison summary"""
        print("\n" + "="*80)
        print("MODEL OPTIMIZATION COMPARISON SUMMARY")
        print("="*80)
        
        # Headers
        print(f"{'Model Type':<20} {'Size (MB)':<12} {'Inference (ms)':<15} {'Throughput (FPS)':<15} {'Speedup':<10}")
        print("-" * 80)
        
        baseline_time = results.get("original", {}).get("avg_inference_time_ms", 1)
        
        for model_type, metrics in results.items():
            size_mb = metrics.get("model_size_mb", "N/A")
            inf_time = metrics.get("avg_inference_time_ms", "N/A")
            throughput = metrics.get("throughput_fps", "N/A")
            
            # Calculate speedup
            if isinstance(inf_time, (int, float)) and inf_time > 0:
                speedup = baseline_time / inf_time
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"
            
            print(f"{model_type:<20} {size_mb:<12} {inf_time:<15} {throughput:<15} {speedup_str:<10}")
        
        print("="*80)
    
    def export_optimized_model(self, optimization_type: str, output_path: str):
        """
        Export optimized model
        
        Args:
            optimization_type: Type of optimization to export
            output_path: Path to save the model
        """
        if optimization_type == "dynamic_quantized":
            quantized_model = self.quantize_model_dynamic(self.model)
            torch.save(quantized_model.state_dict(), output_path)
        elif optimization_type == "pruned":
            pruned_model = self.prune_model(self.model)
            torch.save(pruned_model.state_dict(), output_path)
        else:
            logger.warning(f"Unknown optimization type: {optimization_type}")


def create_optimization_report(results: Dict, save_path: str = "models/optimization_report.html"):
    """
    Create HTML report for optimization results
    
    Args:
        results: Optimization results dictionary
        save_path: Path to save HTML report
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Optimization Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .metric { font-weight: bold; color: #2c3e50; }
            .improvement { color: #27ae60; }
            .degradation { color: #e74c3c; }
        </style>
    </head>
    <body>
        <h1>Model Optimization Report</h1>
        
        <h2>Performance Comparison</h2>
        <table>
            <tr>
                <th>Model Type</th>
                <th>Size (MB)</th>
                <th>Inference Time (ms)</th>
                <th>Throughput (FPS)</th>
                <th>Speedup</th>
            </tr>
    """
    
    baseline_time = results.get("original", {}).get("avg_inference_time_ms", 1)
    
    for model_type, metrics in results.items():
        size_mb = metrics.get("model_size_mb", "N/A")
        inf_time = metrics.get("avg_inference_time_ms", "N/A")
        throughput = metrics.get("throughput_fps", "N/A")
        
        # Calculate speedup
        if isinstance(inf_time, (int, float)) and inf_time > 0:
            speedup = baseline_time / inf_time
            speedup_str = f"{speedup:.2f}x"
            speedup_class = "improvement" if speedup > 1 else "degradation"
        else:
            speedup_str = "N/A"
            speedup_class = ""
        
        html_content += f"""
            <tr>
                <td class="metric">{model_type}</td>
                <td>{size_mb}</td>
                <td>{inf_time}</td>
                <td>{throughput}</td>
                <td class="{speedup_class}">{speedup_str}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Recommendations</h2>
        <ul>
            <li>Use dynamic quantization for CPU deployment with minimal accuracy loss</li>
            <li>Use ONNX for cross-platform deployment</li>
            <li>Consider pruning for memory-constrained environments</li>
            <li>Test accuracy after optimization to ensure acceptable performance</li>
        </ul>
    </body>
    </html>
    """
    
    with open(save_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Optimization report saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    from src.models.cnn_models import ModelFactory
    
    # Create a sample model
    model = ModelFactory.create_model("resnet50", num_classes=2)
    
    # Create optimizer
    optimizer = ModelOptimizer(model, "resnet50_defect_detector")
    
    # Input shape for benchmarking
    input_shape = (1, 3, 224, 224)
    
    # Compare optimizations
    results = optimizer.compare_models(input_shape)
    
    # Print summary
    optimizer.print_comparison_summary(results)
    
    # Create report
    create_optimization_report(results)
    
    print("Model optimization example completed!")
