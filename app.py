import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np

# Set up the app
st.set_page_config(page_title="Advanced Security Analyzer", layout="wide")
st.title("🛡️ Advanced Property Security Analyzer")
st.markdown("""
Upload images from your security cameras to detect potential threats, deliveries, 
or normal activity with comprehensive scenario analysis.
""")

# Load the CLIP model
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# Organized prompts with categories
PROMPT_CATEGORIES = [
    {
        "name": "Delivery",
        "prompts": [
            "a delivery person leaving a package at a doorstep",
            "UPS, FedEx, or Amazon delivery truck parked on street",
            "food delivery person on a scooter or bike",
            "mail carrier putting letters in mailbox",
            "courier signing for a package at the door",
            "grocery delivery person unloading bags",
            "flower delivery person holding bouquet",
            "furniture delivery truck with ramp down",
            "appliance delivery team moving large boxes",
            "uniformed delivery person scanning barcode",
            "package being placed in a parcel locker",
            "delivery drone hovering near a house",
            "pizza delivery person holding thermal bag",
            "moving van with workers carrying furniture",
            "laundry service picking up bags"
        ],
        "response": "✅ Safe: Delivery detected",
        "color": "green",
        "icon": "📦"
    },
    {
        "name": "Suspicious",
        "prompts": [
            "a person looking into car windows",
            "someone testing door handles on vehicles",
            "individual hiding their face from cameras",
            "person carrying burglary tools",
            "someone climbing over a fence",
            "individual wearing gloves in warm weather",
            "person looking into residential windows",
            "suspicious individual checking multiple houses",
            "someone running while carrying valuables",
            "person dressed in dark clothing at night",
            "individual peering into parked cars",
            "someone tampering with security cameras",
            "person carrying crowbar or pry tool",
            "individual wearing ski mask in summer",
            "someone casing the neighborhood slowly"
        ],
        "response": "🚨 Warning: Suspicious activity",
        "color": "red",
        "icon": "⚠️"
    },
    {
        "name": "Normal",
        "prompts": [
            "children playing in front yard",
            "neighbors walking their dogs",
            "people jogging on sidewalk",
            "resident taking out trash bins",
            "gardener mowing the lawn",
            "family unloading groceries from car",
            "kids riding bicycles on street",
            "couple pushing baby stroller",
            "person washing their car in driveway",
            "neighbors chatting over fence",
            "mail carrier making normal rounds",
            "utility worker reading meter",
            "street sweeper cleaning road",
            "newspaper being delivered at dawn",
            "school bus picking up children"
        ],
        "response": "✅ Safe: Normal activity",
        "color": "green",
        "icon": "🏡"
    },
    {
        "name": "Emergency",
        "prompts": [
            "smoke coming from house windows",
            "firefighters spraying water on building",
            "police cars with lights flashing",
            "ambulance parked in front of house",
            "medical personnel carrying stretcher",
            "car accident with deployed airbags",
            "downed power lines sparking",
            "flood water covering street",
            "tree fallen on house roof",
            "gas leak repair crew working"
        ],
        "response": "🚑 Emergency: Immediate action needed",
        "color": "red",
        "icon": "🚨"
    },
    {
        "name": "Animals",
        "prompts": [
            "coyote walking through yard",
            "raccoons going through trash",
            "stray dogs roaming street",
            "deer eating garden plants",
            "bear cub climbing tree",
            "skunk wandering at night",
            "possum playing dead",
            "birds nesting in eaves",
            "squirrels chewing on wires",
            "bats flying at dusk"
        ],
        "response": "🐾 Notice: Animal activity",
        "color": "orange",
        "icon": "🐾"
    }
]

def analyze_security_scene(image):
    # Prepare all prompts and track their categories
    all_prompts = []
    prompt_categories = []
    
    for category in PROMPT_CATEGORIES:
        all_prompts.extend(category["prompts"])
        prompt_categories.extend([category["name"]] * len(category["prompts"]))
    
    # Process image and text
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(p) for p in all_prompts]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        similarities = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]
    
    # Aggregate scores by category
    category_scores = {cat["name"]: 0 for cat in PROMPT_CATEGORIES}
    for i, category in enumerate(prompt_categories):
        category_scores[category] += similarities[i]
    
    # Normalize scores by number of prompts in each category
    for cat in PROMPT_CATEGORIES:
        category_scores[cat["name"]] /= len(cat["prompts"])
    
    # Get top category
    top_category_name = max(category_scores.items(), key=lambda x: x[1])[0]
    top_category = next(cat for cat in PROMPT_CATEGORIES if cat["name"] == top_category_name)
    
    return top_category, category_scores, similarities, all_prompts

# File uploader
uploaded_file = st.file_uploader("Upload a security camera image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Security Camera Feed", use_container_width=True)
        
        # Add a loading spinner while processing
        with st.spinner("Analyzing scene..."):
            result, category_scores, all_scores, all_prompts = analyze_security_scene(image)
        
        # Display main result with appropriate styling
        if result["color"] == "red":
            st.error(f"{result['icon']} {result['response']}", icon="⚠️")
        elif result["color"] == "orange":
            st.warning(f"{result['icon']} {result['response']}", icon="⚠️")
        else:
            st.success(f"{result['icon']} {result['response']}", icon="✅")
            
        # Show action recommendations
        st.markdown("---")
        st.subheader("Recommended Actions")
        
        if result["name"] == "Suspicious":
            st.markdown("""
            - Review full camera footage
            - Notify neighbors
            - Contact authorities if threat is immediate
            - Check all entry points
            """)
        elif result["name"] == "Emergency":
            st.markdown("""
            - Call emergency services if needed
            - Evacuate if necessary
            - Alert nearby residents
            - Avoid the danger area
            """)
        else:
            st.markdown("""
            - No immediate action required
            - Continue normal monitoring
            - Review periodically
            """)
    
    with col2:
        st.subheader("Detailed Analysis")
        
        # Show confidence meters for each category
        for category in PROMPT_CATEGORIES:
            score = category_scores[category["name"]]
            st.metric(
                label=f"{category['icon']} {category['name']}",
                value=f"{score*100:.1f}%",
                help=", ".join(category["prompts"][:3]))
            st.progress(min(1.0, score))  # Ensure progress doesn't exceed 1.0
        
        # Show top matching prompts
        st.markdown("---")
        st.subheader("Top Matching Scenarios")
        top_indices = np.argsort(all_scores)[-5:][::-1]  # Show top 5 matches
        
        for idx in top_indices:
            matching_category = next(cat for cat in PROMPT_CATEGORIES 
                                  if cat["name"] == prompt_categories[idx])
            st.markdown(
                f"{matching_category['icon']} `{all_scores[idx]*100:.1f}%` "
                f"{all_prompts[idx]}"
            )

# Add footer with security tips
st.markdown("---")
st.subheader("Security Tips")
st.markdown("""
- Regularly check your camera angles and clean lenses
- Ensure all entry points are well-lit at night
- Establish a neighborhood watch program
- Test your security system monthly
- Keep shrubs trimmed near windows and doors
""")