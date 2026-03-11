import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --------------------------------------------------
# Load CLIP model once
# --------------------------------------------------

print("Loading CLIP model...")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Model loaded successfully\n")


# --------------------------------------------------
# Function: Detect AI vs Real
# --------------------------------------------------

def detect_ai_image(image_path):

    image = Image.open(image_path).convert("RGB")

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

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    real_prob = probs[0][0].item()
    ai_prob = probs[0][1].item()

    return real_prob, ai_prob


# --------------------------------------------------
# Function: Scan Folder
# --------------------------------------------------

def analyze_folder(folder):

    if not os.path.exists(folder):
        print("Folder not found:", folder)
        return

    files = os.listdir(folder)

    if not files:
        print("No images found in folder.")
        return

    for file in files:

        path = os.path.join(folder, file)

        if path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):

            try:

                real, ai = detect_ai_image(path)

                print("-----------------------------------")
                print("Image:", file)
                print("Real probability:", round(real, 3))
                print("AI probability:", round(ai, 3))

                if ai > real:
                    print("Prediction: AI Generated")
                else:
                    print("Prediction: Real Image")

            except Exception as e:

                print("Skipping", file, "Error:", e)


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    folder_path = "data/images"

    analyze_folder(folder_path)