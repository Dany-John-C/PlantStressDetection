import os
import yaml
import torch
from torchvision import transforms, models
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Import our custom explainability module
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.explainability import GradCAM, overlay_heatmap
from src.models.mobilenet import get_mobilenet_model

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease AI",
    page_icon="🌿",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Make the app look modern and clean */
[data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e6edf3;
    font-family: 'Inter', sans-serif;
}
.header {
    text-align: center;
    padding: 2em 0;
}
.header h1 {
    color: #3fb950;
    font-size: 3rem;
    font-weight: 800;
}
.header p {
    color: #8b949e;
    font-size: 1.1rem;
}
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px;
    margin-top: 20px;
}
.result-title {
    font-size: 1.5rem;
    color: #e6edf3;
    margin-bottom: 5px;
}
.confidence {
    font-size: 2rem;
    font-weight: 700;
    color: #3fb950;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>🌿 Disease Detective</h1>
    <p>Upload a leaf image and let our AI diagnose the plant automatically.</p>
</div>
""", unsafe_allow_html=True)

# ── Utility Functions ──────────────────────────────────────────────────────────
@st.cache_resource
def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

@st.cache_resource
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_class_mapping(config):
    # Load class mapping from dataset directory
    data_dir = config["dataset"]["original_data_dir"]
    if os.path.exists(data_dir):
        all_classes = sorted(os.listdir(data_dir))
        class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
        # Clean up names for display
        idx_to_class = {v: k.split('___')[1].replace('_',' ') if '___' in k else k for k, v in class_to_idx.items()}
        return idx_to_class, len(all_classes)
    return {}, 0

@st.cache_resource
def load_model(num_classes, device):
    model_path = "models/mobilenet_classifier.pth"
    if not os.path.exists(model_path):
        return None
        
    # Get model architecture
    model = get_mobilenet_model(num_classes)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

# Setup image transformation matching PyTorch pipeline
def get_transform(config):
    size = config["dataset"]["image_size"]
    return transforms.Compose([
        transforms.Resize((size[0], size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ── Main Flow ──────────────────────────────────────────────────────────────────

config = load_config()
device = get_device()
idx_to_class, num_classes = load_class_mapping(config)

if num_classes == 0:
    st.error("Dataset not found. Please ensure your dataset is located at the path specified in `config.yaml`.")
    st.stop()

model = load_model(num_classes, device)
if model is None:
    st.error("Model weights `models/mobilenet_classifier.pth` not found. Please run Training first.")
    st.stop()

# ── File Uploader ──
uploaded_file = st.file_uploader("Upload Leaf Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Create layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Image**")
        st.image(image, use_container_width=True)
    
    with st.spinner("Analyzing Leaf..."):
        # Preprocess
        transform = get_transform(config)
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Initialize Grad-CAM on final conv layer
        # MobileNet features[-1] is typically a BatchNorm, so we go slightly deeper or hook into 'features' 
        # For torch torchvision mobilenetv3, .features block holds the conv layers
        target_layer = model.features[-1] 
        grad_cam = GradCAM(model, target_layer)
        
        # Run inference + Explainability
        # Note: hook functions require parameters to have requires_grad=True
        input_tensor.requires_grad = True
        
        # Get heatmap
        cam, pred_idx, logits = grad_cam.generate(input_tensor)
        
        # Calculate Confidence %
        probabilities = torch.nn.functional.softmax(logits, dim=0)
        confidence = probabilities[pred_idx].item() * 100
        
        predicted_class_name = idx_to_class[pred_idx]
        
    # ── Display Results ──
    with col2:
        st.markdown("**AI Assessment:**")
        st.markdown(f'<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="result-title">Detected Disease:</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 1.8rem; font-weight: bold; margin-bottom: 20px;">{predicted_class_name.capitalize()}</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="result-title">Confidence:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence">{confidence:.2f}%</div>', unsafe_allow_html=True)
        st.markdown(f'</div>', unsafe_allow_html=True)
        
    # ── Display Visual Explanation ──
    st.markdown("---")
    st.markdown("### 🔬 How did the AI decide?")
    st.markdown("The heatmap below highlights the areas of the leaf that the Artificial Intelligence focused most heavily on to make its diagnosis. Red patches indicate high importance, while blue patches signify areas the AI ignored.")
    
    # Overlay heatmap onto original image
    img_array = np.array(image.resize(config["dataset"]["image_size"]))
    overlayed_img = overlay_heatmap(img_array, cam)
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
         st.image(overlayed_img, caption="Grad-CAM Activation Heatmap", use_container_width=True)
