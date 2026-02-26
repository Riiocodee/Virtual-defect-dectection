"""
Streamlit Deployment App for Manufacturing Defect Detection

This module provides:
- Real-time defect detection interface
- Image upload and processing
- Model inference with confidence scores
- Grad-CAM visualization
- Batch processing capabilities
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time
import io
import base64
from typing import List, Tuple, Dict, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.cnn_models import ModelFactory
from data.data_loader import get_val_transforms
from utils.grad_cam import GradCAM, visualize_cam


class DefectDetectionApp:
    """Streamlit app for defect detection"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.load_model()
        self.setup_transforms()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Manufacturing Defect Detection",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
        if 'processed_images' not in st.session_state:
            st.session_state.processed_images = []
    
    def load_model(self):
        """Load the trained model"""
        try:
            model_path = "models/best_model.pth"
            
            if Path(model_path).exists():
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Determine model type (you might want to save this in the checkpoint)
                model_type = "resnet50"  # Default, should be saved in checkpoint
                
                # Create model
                self.model = ModelFactory.create_model(model_type, num_classes=2)
                
                # Load state dict
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                st.session_state.model_loaded = True
                st.success(f"✅ Model loaded successfully from {model_path}")
            else:
                st.warning(f"⚠️ Model file not found at {model_path}")
                st.session_state.model_loaded = False
                
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.session_state.model_loaded = False
    
    def setup_transforms(self):
        """Setup image transformations"""
        self.transform = get_val_transforms(image_size=224, use_albumentations=False)
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for inference"""
        # Apply transformations
        if self.transform:
            image_tensor = self.transform(image)
        else:
            # Basic preprocessing if no transform
            image_tensor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])(image)
        
        return image_tensor.unsqueeze(0)  # Add batch dimension
    
    def predict_single_image(self, image: Image.Image) -> Dict:
        """Make prediction on a single image"""
        if not st.session_state.model_loaded:
            return {"error": "Model not loaded"}
        
        try:
            # Preprocess
            input_tensor = self.preprocess_image(image)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(outputs, 1)
            
            # Convert to numpy
            pred_class = predicted.item()
            confidence_score = confidence.item()
            prob_defective = probabilities[0][1].item()
            prob_non_defective = probabilities[0][0].item()
            
            result = {
                "predicted_class": "defective" if pred_class == 1 else "non_defective",
                "confidence": confidence_score,
                "prob_defective": prob_defective,
                "prob_non_defective": prob_non_defective,
                "success": True
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}", "success": False}
    
    def generate_grad_cam(self, image: Image.Image, target_class: int) -> Optional[np.ndarray]:
        """Generate Grad-CAM visualization"""
        try:
            # Convert PIL to numpy
            img_np = np.array(image)
            
            # Preprocess for model
            input_tensor = self.preprocess_image(image)
            
            # Create GradCAM
            grad_cam = GradCAM(self.model, target_layer='layer4')  # Adjust layer name based on model
            
            # Generate CAM
            cam = grad_cam(input_tensor, class_idx=target_class)
            
            # Visualize
            visualization = visualize_cam(cam, img_np)
            
            return visualization
            
        except Exception as e:
            st.error(f"Grad-CAM generation failed: {str(e)}")
            return None
    
    def run(self):
        """Run the Streamlit app"""
        # Header
        st.title("🔍 Manufacturing Defect Detection System")
        st.markdown("---")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        if st.session_state.model_loaded:
            tab1, tab2, tab3 = st.tabs(["📸 Single Image", "📁 Batch Processing", "📊 Analytics"])
            
            with tab1:
                self.render_single_image_tab()
            
            with tab2:
                self.render_batch_processing_tab()
            
            with tab3:
                self.render_analytics_tab()
        else:
            st.info("Please load a model to start using the defect detection system.")
            self.render_model_loading_section()
    
    def render_sidebar(self):
        """Render sidebar with model information and settings"""
        st.sidebar.header("⚙️ Settings")
        
        # Model information
        if st.session_state.model_loaded:
            st.sidebar.success("✅ Model Loaded")
            
            # Model stats
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            st.sidebar.metric("Total Parameters", f"{total_params:,}")
            st.sidebar.metric("Trainable Parameters", f"{trainable_params:,}")
            
            # Confidence threshold
            self.confidence_threshold = st.sidebar.slider(
                "Confidence Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5,
                step=0.05
            )
            
            # Show Grad-CAM option
            self.show_grad_cam = st.sidebar.checkbox("Show Grad-CAM Visualization", value=True)
            
        else:
            st.sidebar.error("❌ Model Not Loaded")
    
    def render_single_image_tab(self):
        """Render single image detection tab"""
        st.header("📸 Single Image Detection")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a manufacturing product image for defect detection"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
                
                # Image info
                st.info(f"Image Size: {image.size}")
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                result = self.predict_single_image(image)
            
            if result.get("success", False):
                with col2:
                    self.render_prediction_result(result, image)
            else:
                st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")
    
    def render_prediction_result(self, result: Dict, image: Image.Image):
        """Render prediction result with visualizations"""
        st.subheader("Detection Result")
        
        # Prediction badge
        pred_class = result["predicted_class"]
        confidence = result["confidence"]
        
        if pred_class == "defective":
            st.error(f"🚨 **DEFECTIVE**")
            st.metric("Confidence", f"{confidence:.2%}")
        else:
            st.success(f"✅ **NON-DEFECTIVE**")
            st.metric("Confidence", f"{confidence:.2%}")
        
        # Probability bars
        st.subheader("Class Probabilities")
        
        prob_data = {
            "Non-Defective": result["prob_non_defective"],
            "Defective": result["prob_defective"]
        }
        
        fig = px.bar(
            x=list(prob_data.values()),
            y=list(prob_data.keys()),
            orientation='h',
            title="Prediction Confidence",
            color=list(prob_data.keys()),
            color_discrete_map={"Non-Defective": "green", "Defective": "red"}
        )
        
        fig.update_xaxes(range=[0, 1])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Grad-CAM visualization
        if self.show_grad_cam:
            st.subheader("🔥 Grad-CAM Visualization")
            
            target_class = 1 if pred_class == "defective" else 0
            
            with st.spinner("Generating Grad-CAM..."):
                cam_image = self.generate_grad_cam(image, target_class)
            
            if cam_image is not None:
                st.image(cam_image, caption="Grad-CAM Heatmap", use_column_width=True)
                st.info("Grad-CAM shows which regions of the image contributed most to the prediction.")
    
    def render_batch_processing_tab(self):
        """Render batch processing tab"""
        st.header("📁 Batch Processing")
        
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="Upload multiple images for batch defect detection"
        )
        
        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} images...")
            
            # Progress bar
            progress_bar = st.progress(0)
            
            results = []
            processed_images = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Process image
                image = Image.open(uploaded_file).convert("RGB")
                result = self.predict_single_image(image)
                
                if result.get("success", False):
                    result["filename"] = uploaded_file.name
                    results.append(result)
                    processed_images.append((uploaded_file.name, image, result))
            
            # Display results
            if results:
                st.success(f"Successfully processed {len(results)} images")
                
                # Summary statistics
                self.render_batch_summary(results)
                
                # Detailed results
                st.subheader("Detailed Results")
                
                for filename, image, result in processed_images:
                    with st.expander(f"📸 {filename}"):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.image(image, width=200)
                        
                        with col2:
                            pred_class = result["predicted_class"]
                            confidence = result["confidence"]
                            
                            if pred_class == "defective":
                                st.error(f"🚨 **DEFECTIVE** ({confidence:.2%})")
                            else:
                                st.success(f"✅ **NON-DEFECTIVE** ({confidence:.2%})")
                            
                            # Probabilities
                            st.write(f"Non-Defective: {result['prob_non_defective']:.2%}")
                            st.write(f"Defective: {result['prob_defective']:.2%}")
            else:
                st.error("No images were successfully processed.")
    
    def render_batch_summary(self, results: List[Dict]):
        """Render batch processing summary"""
        st.subheader("📊 Batch Summary")
        
        # Count predictions
        defective_count = sum(1 for r in results if r["predicted_class"] == "defective")
        non_defective_count = len(results) - defective_count
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Images", len(results))
        
        with col2:
            st.metric("Defective", defective_count)
        
        with col3:
            st.metric("Non-Defective", non_defective_count)
        
        with col4:
            defect_rate = defective_count / len(results) * 100
            st.metric("Defect Rate", f"{defect_rate:.1f}%")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig = px.pie(
                values=[defective_count, non_defective_count],
                names=["Defective", "Non-Defective"],
                title="Defect Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence distribution
            confidences = [r["confidence"] for r in results]
            fig = px.histogram(
                x=confidences,
                nbins=20,
                title="Confidence Distribution",
                labels={"x": "Confidence", "y": "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_analytics_tab(self):
        """Render analytics tab"""
        st.header("📊 Analytics Dashboard")
        
        if not st.session_state.predictions:
            st.info("No prediction data available. Process some images first to see analytics.")
            return
        
        # Convert session state predictions to DataFrame
        import pandas as pd
        
        df = pd.DataFrame(st.session_state.predictions)
        
        # Analytics charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction distribution
            pred_counts = df["predicted_class"].value_counts()
            fig = px.bar(
                x=pred_counts.index,
                y=pred_counts.values,
                title="Prediction Distribution",
                labels={"x": "Class", "y": "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence distribution by class
            fig = px.box(
                df, 
                x="predicted_class", 
                y="confidence",
                title="Confidence by Class"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed statistics
        st.subheader("Detailed Statistics")
        st.dataframe(df.describe())
    
    def render_model_loading_section(self):
        """Render model loading section"""
        st.header("📥 Load Model")
        
        model_path = st.text_input("Model Path", value="models/best_model.pth")
        
        if st.button("Load Model"):
            try:
                self.load_model()
                if st.session_state.model_loaded:
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")


def main():
    """Main function to run the Streamlit app"""
    app = DefectDetectionApp()
    app.run()


if __name__ == "__main__":
    main()
