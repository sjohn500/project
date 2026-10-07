# Portfolio — AI Engineering Journey

**Name:** Sarah Naomi John (sjohn500)
**Repo:** https://github.com/sjohn500/project
**Timeline:** 3 days of focused work
**Live app:** https://foodlens-sjohn500.streamlit.app

---

## 🎯 What I've Built

### 🌿 ChopWell — Nigerian Food AI (Days 2–3)

**Live app:** https://foodlens-sjohn500.streamlit.app

A web app that identifies Nigerian dishes from a photo and provides
health guidance on how each dish affects your body when eaten in excess.

**What it does:**
- User uploads a food photo
- A trained neural network identifies the dish
- The app shows health effects, mindful notes, and who should be cautious

**How I built it:**
1. Collected and curated a dataset of 80 food images (jollof rice, egusi soup)
2. Wrote a Python pipeline (`build_metadata.py`) to auto-track every image
3. Trained a MobileNetV2 classifier using transfer learning
4. Handled class imbalance with weighted loss
5. Built a Streamlit web app
6. Deployed to Streamlit Cloud (public URL)

**Results:**
- **92.9% validation accuracy**
- 2 classes · 80 images · 20 epochs · ~2 min training on CPU
- Model size: 8.8 MB

**Tech stack:**
Python · PyTorch · torchvision · MobileNetV2 · Streamlit · Pillow ·
Pandas · Git · Streamlit Cloud

**Why it matters:**
Nigerian food is missing from mainstream AI datasets. ChopWell
addresses that gap and provides real health awareness for a real
population — many Nigerians eat dishes that are fine in small
amounts but problematic in excess.

---

### 🏠 House Price EDA (Day 1)

Exploratory data analysis on Kaggle's House Prices dataset
(1460 houses × 81 features).

**What I did:**
- Analyzed target distribution (SalePrice is right-skewed)
- Identified missing values (19 columns)
- Computed correlations — top predictors: OverallQual (0.79),
  GrLivArea (0.71), GarageCars (0.64)
- Visualized scatter grids and distributions

**Location:** `week1-house-price/notebooks/eda.ipynb`

---

## �� Key Numbers

| Fact | Value |
|------|-------|
| Total commits | ~21 |
| Days of work | 3 |
| Dataset size | 80 images |
| Dishes recognized | 2 |
| Model | MobileNetV2 |
| **Validation accuracy** | **92.9%** |
| Model file size | 8.8 MB |

---

## 🛠️ Skills Demonstrated

### Machine Learning
- Supervised learning
- Transfer learning with pretrained models
- Handling class imbalance
- Train/validation split
- Data augmentation
- Model evaluation (loss, accuracy)
- Jupyter notebooks for experimentation

### Data Engineering
- Dataset curation from scratch
- Building metadata pipelines
- Image preprocessing (resize, normalize)
- Working with imbalanced data

### Software Engineering
- Git version control with conventional commits
- Project structure (`src/`, `data/`, `models/`, `notebooks/`)
- `.gitignore`, `requirements.txt`
- Modular Python (functions, scripts)
- Absolute paths for portability

### Deployment
- Streamlit for ML web apps
- Streamlit Cloud for hosting
- Model persistence (torch.save / torch.load)
- Handling model file paths across environments

### Product & Design
- Brand identity (name, tagline, palette)
- User-centered UX (self-explanatory interface)
- CSS theming
- Trust signals (accuracy badge, disclaimer)

---

## 💬 Interview-Ready Summary

> I built ChopWell — a Nigerian food classifier deployed as a live
> web app. It identifies dishes from photos and provides health
> guidance. I built my own dataset (80 images), trained a MobileNetV2
> model with transfer learning (92.9% validation accuracy), and
> deployed it via Streamlit Cloud. The whole pipeline is on GitHub.
>
> The project addresses a real gap: Nigerian food is missing from
> mainstream AI datasets. The app gives people accessible, culturally
> relevant health awareness.

---

## 🎤 The 60-Second Pitch

> "I'm building ChopWell — a web app where you snap a photo of your
> Nigerian food, and it tells you what the dish is and how it affects
> your body if you eat too much of it.
>
> Right now it recognizes jollof rice and egusi soup with 92.9%
> validation accuracy. I trained it on 80 photos I collected and
> cleaned myself.
>
> The app is live — you can try it at the URL in my repo.
>
> My next steps are adding more Nigerian dishes, improving accuracy,
> and integrating it into a food marketplace I'm building."

---

## 🗺️ Roadmap (What's Next)

**Week 2:**
- Add 3 more dishes — fried plantain, pounded yam, suya
- Retrain with 5 classes
- Improve accuracy to 95%+

**Week 3:**
- Add ingredient detection (from dish name)
- Add hot/cold classification
- Expand the health knowledge base

**Week 4:**
- Integrate into the food marketplace
- Add user history and personalization

---

## 🔗 Links

- **GitHub:** https://github.com/sjohn500/project
- **Live app:** https://foodlens-sjohn500.streamlit.app
- **ChopWell source:** `foodlens/` in this repo
- **House Price EDA:** `week1-house-price/notebooks/eda.ipynb`

---

## ✍️ Honest Notes

I started this journey three days ago with no practical ML experience.
Day 1 was environment setup and confusion. Day 3 ended with a live AI
web app that real people can use.

The hard parts weren't the model — they were the plumbing: environment
setup, dependencies, Kaggle API authentication, deploy paths, kernel
restarts. Those aren't "interesting" but they're what separates
someone who has read about ML from someone who has shipped something.

The model trains in 2 minutes. The value is in shipping.
