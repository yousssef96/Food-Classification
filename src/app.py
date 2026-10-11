import streamlit as st
from pathlib import Path
import torch
import pandas as pd
from PIL import Image

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# ---------- Config ----------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth"
EXAMPLES_DIR = BASE_DIR / "examples"
CLASS_NAMES = ["pizza", "steak", "sushi"]

st.set_page_config(page_title="FoodVision Mini", page_icon="🍕")


# ---------- Model and transforms (loaded once) ----------
@st.cache_resource
def load_model():
    model, transforms = create_effnetb2_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, transforms


# ---------- Predict function ----------
def predict(img, model, transforms) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img.
    Returns a dict of class probabilities and the time taken."""
    start_time = timer()

    # Transform the image and add a batch dimension
    x = transforms(img).unsqueeze(0)

    # Logits -> probabilities
    with torch.inference_mode():
        pred_probs = torch.softmax(model(x), dim=1)[0]

    pred_dict = {}
    for i in range(len(CLASS_NAMES)):
        pred_dict[CLASS_NAMES[i]] = float(pred_probs[i])

    pred_time = round(timer() - start_time, 5)
    return pred_dict, pred_time


# ---------- UI ----------
st.title("FoodVision Mini 🍕🥩🍣")
st.write("An EfficientNetB2 feature extractor that classifies food images as pizza, steak or sushi.")

model, transforms = load_model()

source = st.radio("Choose an image source", ["Upload an image", "Use an example"], horizontal=True)

image = None
if source == "Upload an image":
    uploaded = st.file_uploader("Upload a food photo", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
else:
    examples = sorted(EXAMPLES_DIR.glob("*.jpg"))
    if examples:
        choice = st.selectbox("Example image", examples, format_func=lambda p: p.name)
        image = Image.open(choice).convert("RGB")
    else:
        st.warning(f"No example images found in {EXAMPLES_DIR}")

if image is not None:
    st.image(image, caption="Input image", use_container_width=True)

    with st.spinner("Predicting..."):
        pred_dict, pred_time = predict(image, model, transforms)

    top_class = max(pred_dict, key=pred_dict.get)
    st.subheader(f"Prediction: {top_class} ({pred_dict[top_class]:.1%})")
    st.bar_chart(pd.Series(pred_dict))
    st.caption(f"Prediction time: {pred_time:.3f} s")