



import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import pandas as pd

# ======================
# 1. Paths
# ======================
MODEL_PATH = r"indian_cattle_breed_model.pth"
CLASSES_PATH = r"breed_classes.json"

# ======================
# 2. Load Classes & Model
# ======================
@st.cache_resource
def load_classes(path):
    if not os.path.exists(path):
        st.error(f"Classes file not found: {path}")
        return []
    with open(path, "r") as f:
        return json.load(f)

@st.cache_resource
def load_model(model_path, num_classes):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# Load files
classes = load_classes(CLASSES_PATH)
if len(classes) == 0:
    st.stop()  # Stop app if classes file not found

model, device = load_model(MODEL_PATH, len(classes))
if model is None:
    st.stop()  # Stop app if model file not found

# ======================
# 3. Image Transform
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ======================
# 4. Streamlit UI
# ======================
st.title("🐄 Indian Bovine Breed Recognition")
st.write("Upload an image of a cow/bull to identify its breed.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Transform and add batch dimension
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
        probs_list = probabilities.cpu().numpy()[0]

        # Top prediction
        top_idx = probs_list.argmax()
        top_breed = classes[top_idx]
        top_confidence = probs_list[top_idx] * 100
        st.success(f"Predicted Breed: **{top_breed}** ({top_confidence:.2f}%)")

        # Top 3 predictions
        top3_idx = probs_list.argsort()[-3:][::-1]  # indices of top 3
        top3_breeds = [classes[i] for i in top3_idx]
        top3_probs = [probs_list[i]*100 for i in top3_idx]

        top3_df = pd.DataFrame({
            "Breed": top3_breeds,
            "Confidence (%)": top3_probs
        })

        st.subheader("Top 3 Predicted Breeds")
        st.table(top3_df.style.format({"Confidence (%)": "{:.2f}"}))

        # Optional: show all breeds probabilities
        all_df = pd.DataFrame({
            "Breed": classes,
            "Probability (%)": probs_list * 100
        }).sort_values(by="Probability (%)", ascending=False)

        st.subheader("All Breed Probabilities")
        st.dataframe(all_df.style.format({"Probability (%)": "{:.2f}"}))
