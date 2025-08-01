import streamlit as st
import torch
import clip
from PIL import Image

# Set up the app title and description
st.title("🏠 Security Scene Analyzer")
st.markdown("""
Upload an image of your property to analyze for potential security concerns.
The AI will detect if it shows normal activity, a delivery, or suspicious behavior.
""")

# Load the CLIP model (cached for performance)
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

def analyze_security_scene(image):
    prompts = [
        "a delivery person leaving a package",
        "a stranger looking into windows",
        "normal street activity"
    ]
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
    
    best_match_idx = (image_features @ text_features.T).argmax().item()
    return ["Safe: Delivery detected", "Warning: Suspicious activity", "Safe: Normal street activity"][best_match_idx]

# File uploader
uploaded_file = st.file_uploader("Upload a security camera image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Add a loading spinner while processing
    with st.spinner("Analyzing security scene..."):
        result = analyze_security_scene(image)
    
    # Display results with appropriate styling
    if "Warning" in result:
        st.error(f"🚨 {result}")
        st.markdown("""
        **Recommended actions:**
        - Review camera footage
        - Notify neighbors
        - Consider contacting authorities if suspicious activity continues
        """)
    else:
        st.success(f"✅ {result}")
        st.balloons()
    
    # Show confidence scores (optional)
    st.markdown("---")
    st.subheader("Detailed Analysis")
    with st.expander("Show confidence scores for all categories"):
        # Calculate and display scores for all prompts
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = torch.cat([clip.tokenize(p) for p in [
            "a delivery person leaving a package",
            "a stranger looking into windows",
            "normal street activity"
        ]]).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
        
        scores = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Delivery Confidence", f"{scores[0]*100:.1f}%")
            st.metric("Normal Activity Confidence", f"{scores[2]*100:.1f}%")
        with col2:
            st.metric("Suspicious Activity Confidence", f"{scores[1]*100:.1f}%", 
                      delta=f"{scores[1]*100 - 33:.1f}% vs random chance" if scores[1] > 0.33 else None)