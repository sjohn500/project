# Day 4 — Polish + Reality Check

## What I did
- Rebuilt Python venv after data loss (photos + metadata gone)
- Verified trained models survived (v2: 92.86% val acc)
- Added 4 polish features to app.py:
  - Top-3 predictions with confidence bars
  - Low-confidence warning banner (<60%)
  - Feedback buttons (Yes/No)
  - Sample images on homepage (self-demo)
- Extracted and expanded CSS in assets/styles.css
- Created /samples/ with jollof.jpg and egusi.jpg
- Discovered model misclassifies samples (jollof → 49% jollof / 51% egusi)

## What I learned
- **Data loss is real.** Photos in a repo folder can disappear if not backed up properly. git clean -fdx, accidental rm, or path confusion can wipe untracked data.
- **Validation accuracy ≠ real-world accuracy.** My model hit 92.86% on validation, but was ~50/50 on new photos. Small + imbalanced data causes this.
- **Jupyter kernel state is ephemeral.** Models must be saved to disk immediately or they're lost.
- **Ship beats perfection.** Better to deploy a working v1 than polish a perfect but unreleased v2.

## What broke
- Sample jollof photo predicted as egusi (50.7% vs 49.3%)
- Root cause: 80 images is too few, 55:25 imbalance biases toward egusi

## What's next (Day 5)
- Re-download 25+ jollof + 25+ egusi photos (balanced)
- Retrain model with balanced data
- Replace sample photos with clear examples
- Redeploy with confident predictions
