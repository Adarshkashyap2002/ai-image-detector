import torch
from PIL import Image
import streamlit as st
from transformers import CLIPProcessor, CLIPModel

st.title("AI Image Detector")

st.write("Upload an image and the system will estimate if it is AI-generated.")

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

def detect_ai(image):

    labels = [
        "a real photograph",
        "an AI generated image"
    ]

    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    outputs = model(**inputs)

    logits = outputs.logits_per_image
    probs = logits.softmax(dim=1)

    real_prob = probs[0][0].item()
    ai_prob = probs[0][1].item()

    return real_prob, ai_prob


uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png","webp"])

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    real, ai = detect_ai(image)

    st.write("Real probability:", round(real,3))
    st.write("AI probability:", round(ai,3))

    if ai > real:
        st.error("Prediction: AI Generated")
    else:
        st.success("Prediction: Real Image")