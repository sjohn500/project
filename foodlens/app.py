import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import base64

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
<span class="chip chip-muted">🍌 Fried Plantain — coming soon</span>
<span class="chip chip-muted">🍢 Suya — coming soon</span>
<span class="chip chip-muted">🍥 Pounded Yam — coming soon</span>
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
        "summary": "A one-pot rice dish slow-cooked in a tomato and pepper base, seasoned with spices, herbs, and oil. Nigeria's most iconic meal.",
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
        "who_should_pause": ["Diabetes", "Hypertension", "Ulcers or acid reflux"],
        "good_for": ["Active days and physical work", "Cold weather meals", "Post-workout recovery"],
    },
    "egusi_soup": {
        "name": "Egusi Soup",
        "summary": "A thick, comforting soup made from ground melon seeds, leafy greens, palm oil, and protein (meat or fish). Often enjoyed with pounded yam or eba.",
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
        "who_should_pause": ["Cardiovascular conditions", "Nut or seed allergies", "Acid reflux"],
        "good_for": ["Protein needs", "Cold weather comfort", "Long-lasting fullness"],
    },
}

# =========================================================
# SAMPLE IMAGES
# =========================================================
SAMPLES_DIR = Path(__file__).parent / "samples"

def sample_to_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

if "use_sample" not in st.session_state:
    st.session_state["use_sample"] = None

st.markdown("### Try a sample")
col_s1, col_s2 = st.columns(2, gap="medium")

with col_s1:
    p = SAMPLES_DIR / "jollof.jpg"
    if p.exists():
        b64 = sample_to_base64(p)
        st.markdown(f"""
        <div class="sample-card">
            <img src="data:image/jpeg;base64,{b64}" />
            <div class="sample-label">Jollof Rice</div>
        </div>
        """, unsafe_allow_html=True)
    if st.button("Try Jollof", key="sample_jollof", use_container_width=True):
        st.session_state["use_sample"] = "jollof"

with col_s2:
    p = SAMPLES_DIR / "egusi.jpg"
    if p.exists():
        b64 = sample_to_base64(p)
        st.markdown(f"""
        <div class="sample-card">
            <img src="data:image/jpeg;base64,{b64}" />
            <div class="sample-label">Egusi Soup</div>
        </div>
        """, unsafe_allow_html=True)
    if st.button("Try Egusi", key="sample_egusi", use_container_width=True):
        st.session_state["use_sample"] = "egusi"

st.markdown("---")

# =========================================================
# UPLOAD
# =========================================================
st.markdown("### Or upload your own photo")
uploaded = st.file_uploader(
    "Choose a photo",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

# Determine image source
img = None
source = None

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    source = "uploaded"
    st.session_state["use_sample"] = None
elif st.session_state["use_sample"]:
    sample = st.session_state["use_sample"]
    p = SAMPLES_DIR / f"{sample}.jpg"
    if p.exists():
        img = Image.open(p).convert("RGB")
        source = f"sample_{sample}"

# =========================================================
# PREDICT + DISPLAY
# =========================================================
if img is not None:
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.image(img, use_container_width=True)

    with col_res:
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]

        top_k = min(3, len(classes))
        top_probs, top_idxs = torch.topk(probs, top_k)

        pred_class = classes[top_idxs[0].item()]
        confidence = top_probs[0].item()
        info = DISHES.get(pred_class, {})

        st.markdown(f"""
        <div class="result-panel">
            <div class="label">Your meal</div>
            <div class="dish">{info.get('name', pred_class)}</div>
            <div class="summary">{info.get('summary','')}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="pred-list">', unsafe_allow_html=True)
        for rank, (p, idx) in enumerate(zip(top_probs.tolist(), top_idxs.tolist()), 1):
            cls_name = classes[idx]
            disp = DISHES.get(cls_name, {}).get("name", cls_name)
            pct = p * 100
            fill_class = "" if rank == 1 else "secondary"
            st.markdown(f"""
            <div class="pred-row">
                <div class="pred-rank">#{rank}</div>
                <div class="pred-name">{disp}</div>
                <div class="pred-bar-bg">
                    <div class="pred-bar-fill {fill_class}" style="width: {pct:.1f}%"></div>
                </div>
                <div class="pred-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if confidence < 0.60:
            st.markdown("""
            <div class="low-conf-banner">
            ⚠️ <b>Low confidence.</b> The model isn't sure about this one.
            Try a clearer photo with the food centered and well-lit.
            </div>
            """, unsafe_allow_html=True)

    # Health info
    st.markdown("---")
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

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("#### Pause and consider if you have")
        for a in info.get("who_should_pause", []):
            st.markdown(f"- {a}")
    with col_b:
        st.markdown("#### Especially good for")
        for g in info.get("good_for", []):
            st.markdown(f"- {g}")

    # Feedback
    st.markdown("""
    <div class="feedback-block">
        <h4>Was this correct?</h4>
        <p>Your feedback helps improve the next version of ChopWell.</p>
    </div>
    """, unsafe_allow_html=True)

    col_fb1, col_fb2, _ = st.columns([1, 1, 4])
    with col_fb1:
        if st.button("👍 Yes", key=f"fb_yes_{source}", use_container_width=True):
            st.success("Thanks! Feedback recorded.")
    with col_fb2:
        if st.button("👎 No", key=f"fb_no_{source}", use_container_width=True):
            st.info("Thanks — we'll use this to improve.")

else:
    st.caption("Upload a photo above to see how your meal treats your body.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="meta">
<b>Model</b> · MobileNetV2 · 92.9% validation accuracy<br>
<b>Trained on</b> · 2 Nigerian dishes<br>
<b>Source</b> · <a href="https://github.com/sjohn500/project">github.com/sjohn500/project</a><br>
<b>Note</b> · ChopWell shares general guidance — it is not medical advice.
Speak with a healthcare professional for personal decisions.
</div>
""", unsafe_allow_html=True)
