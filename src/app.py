import streamlit as st
from pathlib import Path
import torch
from PIL import Image

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# ---------- Config ----------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth"
EXAMPLES_DIR = BASE_DIR / "examples"
CLASS_NAMES = ["pizza", "steak", "sushi"]

# One colour and emoji per class, used in the result card and bars
CLASS_STYLE = {
    "pizza": {"emoji": "🍕", "color": "#E4572E"},
    "steak": {"emoji": "🥩", "color": "#9B2D3C"},
    "sushi": {"emoji": "🍣", "color": "#1F8A8A"},
}
LOW_CONFIDENCE = 0.60

st.set_page_config(page_title="FoodVision Mini", page_icon="🍕", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; max-width: 1100px; }

    .result-card {
        display: flex; align-items: center; gap: 1.25rem;
        padding: 1.25rem 1.5rem; margin-bottom: 1.25rem;
        border-radius: 14px; border: 1px solid rgba(128,128,128,0.25);
        border-left: 8px solid var(--accent);
    }
    .result-emoji { font-size: 3.5rem; line-height: 1; }
    .result-name { font-size: 2rem; font-weight: 700; line-height: 1.1; }
    .result-conf { font-size: 1.05rem; opacity: 0.75; margin-top: 0.25rem; }

    .bar-row { display: flex; justify-content: space-between; margin: 0.9rem 0 0.3rem 0; font-size: 1rem; }
    .bar-track { height: 10px; border-radius: 6px; background: rgba(128,128,128,0.2); overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 6px; }

    .time-note { margin-top: 1.25rem; font-size: 0.9rem; opacity: 0.6; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Model and transforms (loaded once) ----------
@st.cache_resource
def load_model():
    model, transforms = create_effnetb2_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, transforms


# ---------- Predict function ----------
def predict(img, model, transforms) -> Tuple[Dict, float]:
    """Returns a dict of class probabilities and the time taken."""
    start_time = timer()

    x = transforms(img).unsqueeze(0)

    with torch.inference_mode():
        pred_probs = torch.softmax(model(x), dim=1)[0]

    pred_dict = {}
    for i in range(len(CLASS_NAMES)):
        pred_dict[CLASS_NAMES[i]] = float(pred_probs[i])

    pred_time = round(timer() - start_time, 5)
    return pred_dict, pred_time


# ---------- Helpers ----------
def use_upload():
    st.session_state.active = "upload"


def show_result(pred_dict: Dict, pred_time: float):
    top = max(pred_dict, key=pred_dict.get)
    style = CLASS_STYLE[top]

    st.markdown(
        f"""
        <div class="result-card" style="--accent:{style['color']}">
            <div class="result-emoji">{style['emoji']}</div>
            <div>
                <div class="result-name">{top.capitalize()}</div>
                <div class="result-conf">{pred_dict[top]:.1%} confident</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if pred_dict[top] < LOW_CONFIDENCE:
        st.info("The model isn't sure about this one. It only knows pizza, steak and sushi.")

    # Probability bars, highest first
    bars = ""
    for name, prob in sorted(pred_dict.items(), key=lambda kv: kv[1], reverse=True):
        s = CLASS_STYLE[name]
        bars += f"""
        <div class="bar-row"><span>{s['emoji']} {name.capitalize()}</span><span>{prob:.1%}</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:{prob*100:.1f}%; background:{s['color']}"></div></div>
        """
    st.markdown(bars, unsafe_allow_html=True)
    st.markdown(f'<div class="time-note">Predicted in {pred_time * 1000:.0f} ms</div>', unsafe_allow_html=True)


# ---------- UI ----------
st.title("FoodVision Mini")
st.write("Upload a food photo and find out if it's pizza, steak or sushi.")

model, transforms = load_model()

if "active" not in st.session_state:
    st.session_state.active = None

left, right = st.columns([1, 1], gap="large")

with left:
    tab_upload, tab_examples = st.tabs(["Upload a photo", "Try an example"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Choose a JPG or PNG",
            type=["jpg", "jpeg", "png"],
            key="uploader",
            on_change=use_upload,
            label_visibility="collapsed",
        )

    with tab_examples:
        examples = sorted(EXAMPLES_DIR.glob("*.jpg"))
        if examples:
            cols = st.columns(len(examples))
            for col, path in zip(cols, examples):
                with col:
                    st.image(Image.open(path), use_container_width=True)
                    if st.button("Use this", key=f"ex_{path.name}", use_container_width=True):
                        st.session_state.active = str(path)
        else:
            st.warning(f"No example images found in {EXAMPLES_DIR}")

    # Work out which image is active
    image = None
    active = st.session_state.active
    if active == "upload" and uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
    elif active and active != "upload":
        image = Image.open(active).convert("RGB")

    if image is not None:
        st.image(image, use_container_width=True)

with right:
    if image is None:
        st.markdown("#### Your result will appear here")
        st.write("Add a photo on the left to get a prediction.")
    else:
        with st.spinner("Predicting..."):
            pred_dict, pred_time = predict(image, model, transforms)
        show_result(pred_dict, pred_time)