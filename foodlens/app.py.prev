import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="FoodLens — Nigerian Food Recognition",
    page_icon="🥣",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =========================================================
# STYLING — loaded from assets/styles.css
# =========================================================
CSS_PATH = Path(__file__).parent / "assets" / "styles.css"
st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="page-header">
    <h1>FoodLens</h1>
    <p class="sub">Nigerian food recognition &amp; nutritional awareness — v1</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# ABOUT
# =========================================================
st.markdown("### What this does")
st.markdown("""
<div class="info-card">
FoodLens identifies Nigerian dishes from a photograph and provides
general information on how each dish affects your body when consumed
regularly or in excess.
<br><br>
<b>How to use it:</b> upload a clear photo of your food below. The model
will identify the dish and display a summary of its key ingredients,
health considerations, and who should be cautious about it.
</div>
""", unsafe_allow_html=True)

st.markdown("### Currently supported")
st.markdown("""
- **Jollof Rice** — tomato-pepper rice base
- **Egusi Soup** — melon-seed soup with leafy greens

*Additional Nigerian dishes are in development.*
""")

# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    MODEL_PATH = Path(__file__).parent / "models" / "foodlens_v2.pt"
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
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

# =========================================================
# HEALTH DATABASE
# =========================================================
HEALTH = {
    "jollof_rice": {
        "name": "Jollof Rice",
        "summary": "A one-pot rice dish cooked in a tomato and pepper base, seasoned with spices and oil. Widely considered Nigeria's most iconic meal.",
        "warnings": [
            ("High in carbohydrates — may cause rapid blood sugar elevation", "high"),
            ("Prepared with oil — contributes to fat and calorie density", "medium"),
            ("Spicy — may aggravate ulcers or acid reflux", "medium"),
            ("Typically served in large portions — easy to overconsume", "medium"),
        ],
        "avoid_if": ["Diabetes", "Hypertension", "Ulcers / GERD"],
        "considerations": [
            "Pair with vegetables or protein to slow glucose absorption",
            "Best consumed earlier in the day for active individuals",
        ],
    },
    "egusi_soup": {
        "name": "Egusi Soup",
        "summary": "A thick soup made from ground melon seeds, leafy greens, palm oil, and protein (meat or fish). Commonly eaten with pounded yam or eba.",
        "warnings": [
            ("Very high in palm oil — associated with elevated cholesterol", "high"),
            ("Dense and slow to digest — may cause heaviness", "medium"),
            ("Contains melon seeds — potential allergen", "medium"),
            ("Often served with starchy swallows — high total calorie load", "medium"),
        ],
        "avoid_if": ["Cardiovascular conditions", "Nut / seed allergies", "Acid reflux"],
        "considerations": [
            "Consume in moderation if managing cholesterol",
            "Balanced choice for protein intake",
        ],
    },
}

# =========================================================
# UPLOAD
# =========================================================
st.markdown("### Upload a photo")
uploaded = st.file_uploader(
    "Choose a food photo",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.image(img, use_container_width=True)

    with col_res:
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            idx = probs.argmax().item()
        pred_class = classes[idx]
        confidence = probs[idx].item()

        info = HEALTH.get(pred_class, {})
        st.markdown(f"""
        <div class="result-panel">
            <div class="label">Identified dish</div>
            <div class="dish">{info.get('name', pred_class)}</div>
            <div class="summary">{info.get('summary','')}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Confidence**")
        st.progress(confidence)
        st.caption(f"{confidence:.1%}")

        if confidence < 0.70:
            st.info(
                "Low confidence — try a clearer, well-lit image with the "
                "food centered and unobstructed."
            )

    st.markdown("---")
    st.markdown("### Health considerations")

    for text, level in info.get("warnings", []):
        pill_class = f"pill-{level}"
        label = {"high": "High", "medium": "Medium"}.get(level, "")
        st.markdown(
            f'<div class="warn-row">'
            f'<span class="pill {pill_class}">{label}</span>'
            f'<span>{text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("#### Not recommended for")
        for a in info.get("avoid_if", []):
            st.markdown(f"- {a}")

    with col_b:
        st.markdown("#### Considerations")
        for c in info.get("considerations", []):
            st.markdown(f"- {c}")

else:
    st.caption("Upload an image above to see the identification and health notes.")

# =========================================================
# FOOTER
# =========================================================
st.divider()

st.markdown("""
<div class="meta">
<b>Model</b> · MobileNetV2 · 92.9% validation accuracy<br>
<b>Training data</b> · 80 curated images · 2 Nigerian dishes<br>
<b>Source</b> · <a href="https://github.com/sjohn500/project">github.com/sjohn500/project</a><br>
<b>Disclaimer</b> · FoodLens provides general information and is not a substitute
for professional medical advice.
</div>
""", unsafe_allow_html=True)
