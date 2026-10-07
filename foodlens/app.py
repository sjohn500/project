import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

st.set_page_config(page_title="FoodLens", page_icon="🍲", layout="centered")
st.title("🍲 FoodLens")
st.write("Snap a Nigerian dish → know what it is + health warnings.")
st.caption("v1 demo — trained on jollof rice & egusi soup")

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    ckpt = torch.load("models/foodlens_v2.pt", map_location=device, weights_only=False)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(ckpt["classes"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["classes"], device

model, classes, device = load_model()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

HEALTH = {
    "jollof_rice": {
        "name": "Jollof Rice",
        "emoji": "��",
        "warnings": [
            "High in carbohydrates — blood sugar spike",
            "Fried in oil — high fat content",
            "Spicy — may irritate ulcers",
        ],
        "avoid_if": ["Diabetic", "High blood pressure", "Ulcer"],
    },
    "egusi_soup": {
        "name": "Egusi Soup",
        "emoji": "🥣",
        "warnings": [
            "High in palm oil — cholesterol risk",
            "Rich and heavy — hard to digest",
            "Contains seeds — possible nut allergy",
        ],
        "avoid_if": ["Heart issues", "Nut allergy"],
    },
}

uploaded = st.file_uploader("Upload a food photo", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Uploaded", use_container_width=True)

    with col2:
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            idx = probs.argmax().item()
        pred_class = classes[idx]
        confidence = probs[idx].item()

        info = HEALTH.get(pred_class, {})
        st.subheader(f"{info.get('emoji','🍽️')} {info.get('name', pred_class)}")
        st.metric("Confidence", f"{confidence:.1%}")

        st.markdown("### ⚠️ If you eat too much:")
        for w in info.get("warnings", []):
            st.write(f"- {w}")

        st.markdown("### ❌ Avoid if:")
        for a in info.get("avoid_if", []):
            st.write(f"- {a}")
else:
    st.info("👆 Upload a photo to try it")

st.divider()
st.caption("Built with PyTorch + Streamlit | FoodLens v1")
