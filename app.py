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
text_input = st.text_input("Enter text to compare with the image")

if uploaded_image is not None and text_input:
    # Process image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize([text_input]).to(device)
    
    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
    
    # Calculate similarity with clamping
    similarity = (image_features @ text_features.T).item()
    similarity_normalized = max(0.0, min(1.0, (similarity + 1) / 2))  # Ensures value is between 0 and 1
    st.write(f"Similarity score: {similarity:.4f}")
    st.progress(similarity_normalized)