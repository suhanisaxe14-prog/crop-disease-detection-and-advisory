"""
AI-Based Crop Disease Detection and Advisory
Streamlit app: loads the model trained in the notebook and predicts disease from a leaf image.

Run:  streamlit run app.py
Needs the folder  model_output/  next to this file (copy it from the notebook output).
"""
import os
import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# ---------------------------------------------------------------- paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_output")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_disease_model.keras")
CLASSES_PATH = os.path.join(MODEL_DIR, "class_names.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")
ADVISORY_PATH = os.path.join(MODEL_DIR, "advisory.csv")

CONFIDENCE_THRESHOLD = 0.70     # below this, the app asks for a clearer image
MIN_LEAF_RATIO = 0.05           # quick first filter: at least 5% plant colours (the Not_a_leaf class does the main work)

st.set_page_config(page_title="Crop Disease Detection", page_icon="🌿", layout="centered")


# ---------------------------------------------------------------- loaders (cached)
@st.cache_resource(show_spinner="Loading model...")
def load_model_and_data():
    from tensorflow import keras
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False,
        custom_objects={"preprocess_input": preprocess_input},
    )
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    with open(CONFIG_PATH) as f:
        img_size = tuple(json.load(f)["img_size"])
    advisory = pd.read_csv(ADVISORY_PATH).set_index("class_name")
    return model, class_names, img_size, advisory


def pretty(name: str) -> str:
    """'Potato___Early_blight' -> 'Potato - Early blight'"""
    parts = [p for p in name.replace("__", "_").split("_") if p]
    crop = parts[0]
    rest = " ".join(parts[1:]).replace("bell", "").strip()
    if crop.lower() == "pepper":
        crop = "Pepper (bell)"
        rest = rest.replace("  ", " ")
    return f"{crop} - {rest.capitalize()}" if rest else crop


def leaf_ratio(image: Image.Image) -> float:
    """Fraction of pixels with green/yellow-green plant colours (simple leaf check).
    Used to reject images that are clearly not leaves (road signs, hands, objects...)."""
    small = ImageOps.exif_transpose(image).convert("RGB").resize((128, 128))
    hsv = np.asarray(small.convert("HSV"), dtype="float32")
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # PIL hue scale is 0-255; about 31 to 169 degrees = yellow-green to green-cyan
    mask = (h >= 22) & (h <= 120) & (sat >= 40) & (val >= 40)
    return float(mask.mean())


def predict(model, image: Image.Image, img_size):
    image = ImageOps.exif_transpose(image).convert("RGB").resize(img_size)
    arr = np.expand_dims(np.asarray(image, dtype="float32"), axis=0)   # preprocessing is inside the model
    return model.predict(arr, verbose=0)[0]


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About the project")
    st.write(
        "AI-based system that identifies common crop diseases from a leaf image "
        "and gives a general advisory."
    )
    st.markdown(
        "**Model:** MobileNetV2 (transfer learning)  \n"
        "**Dataset:** PlantVillage  \n"
        "**Framework:** TensorFlow / Keras + Streamlit"
    )
    st.markdown("**Team:**  Suhani Saxena, Urvashi Chavhan, Vaidika Pandey,")
    st.markdown("**Guide:** Prof. Shubham Sharma")
    st.divider()
    st.caption("Tips: use a clear, well-lit photo of a single leaf, close up, plain background.")

# ---------------------------------------------------------------- main page
st.title("🌿 AI-Based Crop Disease Detection and Advisory")
st.write("Upload or capture a leaf image to get the predicted disease, confidence score and general advice.")

# check files exist
missing = [p for p in (MODEL_PATH, CLASSES_PATH, CONFIG_PATH, ADVISORY_PATH) if not os.path.exists(p)]
if missing:
    st.error("Some model files are missing. Copy the `model_output` folder next to `app.py`.")
    st.code("\n".join(missing))
    st.stop()

try:
    model, class_names, img_size, advisory = load_model_and_data()
except Exception as e:
    st.error("Could not load the model.")
    st.exception(e)
    st.stop()

with st.expander("Supported crops and diseases"):
    for c in class_names:
        if c != "Not_a_leaf":
            st.write("•", pretty(c))

mode = st.radio("Choose input method", ["Upload image", "Use camera"], horizontal=True)
if mode == "Upload image":
    file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
else:
    file = st.camera_input("Take a photo of the leaf")

if file is not None:
    image = Image.open(file)
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(image, caption="Input image", width=320)

    # ---- step 1: is this even a leaf?
    if leaf_ratio(image) < MIN_LEAF_RATIO:
        st.error(
            "This image does not look like a plant leaf, so no prediction was made. "
            "Please upload a clear photo of a single crop leaf (tomato, potato or bell pepper)."
        )
        st.stop()

    # ---- step 2: predict the disease
    with st.spinner("Analyzing..."):
        probs = predict(model, image, img_size)

    top = int(np.argmax(probs))
    confidence = float(probs[top])
    predicted = class_names[top]

    # the model itself says this is not a supported leaf
    if predicted == "Not_a_leaf":
        st.error(
            "This does not look like a supported crop leaf (tomato, potato or bell pepper), "
            "so no disease prediction or advice is shown. Please upload a clear photo of a single real leaf."
        )
        st.stop()

    with col2:
        st.subheader("Prediction")
        st.markdown(f"### {pretty(predicted)}")
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.progress(min(max(confidence, 0.0), 1.0))

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(
            "The model is not confident about this image. Please try a clearer, closer photo of a "
            "single leaf. The image may also be a crop or disease that this app does not support."
        )
    else:
        # ---- advisory
        st.divider()
        st.subheader("Advisory")
        info = advisory.loc[predicted] if predicted in advisory.index else None
        if info is not None:
            if "healthy" in predicted.lower():
                st.success("The leaf appears healthy.")
            st.markdown("**Symptoms**")
            st.write(info["symptoms"])
            st.markdown("**Preventive practices**")
            st.write(info["prevention"])
            st.markdown("**Recommended next steps**")
            st.write(info["next_steps"])
        else:
            st.info("No advisory available for this class.")

    # ---- all probabilities
    st.divider()
    st.subheader("Prediction probabilities")
    chart_df = pd.DataFrame(
        {"Probability (%)": probs * 100},
        index=[pretty(c) if c != "Not_a_leaf" else "Not a leaf" for c in class_names],
    )
    st.bar_chart(chart_df)

    st.caption(
        "⚠️ This is general guidance from an AI model, not a replacement for professional agricultural "
        "diagnosis. Please consult a local agriculture expert before applying any treatment."
    )
else:
    st.info("Waiting for an image...")
