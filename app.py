import streamlit as st
import torch
import clip
from PIL import Image

# Load the model
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

st.title("CLIP Image-Text Similarity")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
prompts = [
        "a delivery person leaving a package",
        "a stranger looking into windows",
        "normal street activity"
    ]

if uploaded_image is not None and prompts:
    # Process image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
    
    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
    
    # Calculate similarity with clamping
    st.write(f"Similarity score: {["Safe: Delivery", "Warning: Suspicious", "Safe: Normal"][best_match_idx]}")
