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
    page_title="ChopWell — Eat Well, Live Fully",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =========================================================
# STYLING
# =========================================================
CSS_PATH = Path(__file__).parent / "assets" / "styles.css"
st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="page-header">
    <h1>🌿 ChopWell</h1>
    <p class="tagline">Eat well. Live fully.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# ABOUT
# =========================================================
st.markdown("""
<div class="intro-card">
<strong>Your body deserves mindful choices.</strong><br><br>
ChopWell helps you understand what's on your plate — and how it treats your
body. Upload a photo of your meal, and we'll identify the dish, describe how
your body responds to it, and share mindful guidance on enjoying it well.
</div>
""", unsafe_allow_html=True)

st.markdown("### Currently supported")
st.markdown("""
<span class="chip">🍚 Jollof Rice</span>
<span class="chip">🥣 Egusi Soup</span>
<span class="chip chip-muted">🍌 Fried Plantain — soon</span>
<span class="chip chip-muted">🍢 Suya — soon</span>
<span class="chip chip-muted">🍥 Pounded Yam — soon</span>
""", unsafe_allow_html=True)

# =========================================================
# MODEL
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
# KNOWLEDGE BASE
# =========================================================
DISHES = {
    "jollof_rice": {
        "name": "Jollof Rice",
        "summary": "A one-pot rice dish slow-cooked in a tomato and pepper base, "
                   "seasoned with spices, herbs, and oil. Nigeria's most iconic meal.",
        "effects": {
            "Energy": "Quick and high",
            "Blood sugar": "Significant rise",
            "Fullness": "Lasting",
        },
        "mindful_notes": [
            ("Enjoy in smaller portions if you're managing diabetes", "high"),
            ("Pair with vegetables or protein to slow glucose absorption", "info"),
            ("Consider skipping fried sides on the same plate", "medium"),
        ],
        "who_should_pause": [
            "Diabetes",
            "Hypertension",
            "Ulcers or acid reflux",
        ],
        "good_for": [
            "Active days and physical work",
            "Cold weather meals",
            "Post-workout recovery",
        ],
    },
    "egusi_soup": {
        "name": "Egusi Soup",
        "summary": "A thick, comforting soup made from ground melon seeds, "
                   "leafy greens, palm oil, and protein (meat or fish). Often "
                   "enjoyed with pounded yam or eba.",
        "effects": {
            "Energy": "Slow, sustained",
            "Blood sugar": "Moderate rise",
            "Fullness": "Very heavy",
        },
        "mindful_notes": [
            ("Very rich in palm oil — enjoy in moderation", "high"),
            ("Heavy meal — best eaten earlier in the day", "medium"),
            ("Contains melon seeds — note if you have allergies", "info"),
        ],
        "who_should_pause": [
            "Cardiovascular conditions",
            "Nut or seed allergies",
            "Acid reflux",
        ],
        "good_for": [
            "Protein needs",
            "Cold weather comfort",
            "Long-lasting fullness",
        ],
    },
}

# =========================================================
# UPLOAD
# =========================================================
st.markdown("### See how your meal treats your body")

uploaded = st.file_uploader(
    "Upload a photo of your meal",
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

        info = DISHES.get(pred_class, {})
        st.markdown(f"""
        <div class="result-panel">
            <div class="label">Your meal</div>
            <div class="dish">{info.get('name', pred_class)}</div>
            <div class="summary">{info.get('summary','')}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Recognition confidence**")
        st.progress(confidence)
        st.caption(f"{confidence:.1%}")

        if confidence < 0.70:
            st.info(
                "Low confidence — try a clearer, well-lit photo with the "
                "food centered and unobstructed."
            )

    # Body effects grid
    st.markdown("### How your body responds")
    effects = info.get("effects", {})
    if effects:
        cols = st.columns(len(effects), gap="small")
        for col, (label, value) in zip(cols, effects.items()):
            with col:
                st.markdown(f"""
                <div class="effect-card">
                    <div class="effect-label">{label}</div>
                    <div class="effect-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)

    # Mindful notes
    st.markdown("### Enjoy mindfully")
    for text, level in info.get("mindful_notes", []):
        pill_class = f"pill-{level}"
        label = {"high": "Note", "medium": "Note", "info": "Tip"}.get(level, "")
        st.markdown(
            f'<div class="warn-row">'
            f'<span class="pill {pill_class}">{label}</span>'
            f'<span>{text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Who should pause + good for
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("#### Pause and consider if you have")
        for a in info.get("who_should_pause", []):
            st.markdown(f"- {a}")

    with col_b:
        st.markdown("#### Especially good for")
        for g in info.get("good_for", []):
            st.markdown(f"- {g}")

else:
    st.caption("Upload a photo above to see how your meal treats your body.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="meta">
<b>Model</b> · MobileNetV2 · 92.9% validation accuracy<br>
<b>Trained on</b> · 80 curated images · 2 Nigerian dishes<br>
<b>Source</b> · <a href="https://github.com/sjohn500/project">github.com/sjohn500/project</a><br>
<b>Note</b> · ChopWell shares general guidance — it is not medical advice.
Speak with a healthcare professional for personal decisions.
</div>
""", unsafe_allow_html=True)
