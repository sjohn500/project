# Day 2 — Built my first dataset

## What I did
- Created foodlens project structure (data/raw, src, notebooks, models)
- Downloaded 25 jollof rice photos (iStock/Unsplash, legally free)
- Cleaned and renamed them to jollof_001.jpg ... jollof_025.jpg
- Wrote build_metadata.py — scans folders, generates metadata.csv
- Verified dataset in Jupyter with a 12-image grid preview

## What I learned
- Real ML datasets start messy and need cleaning
- Folder names = labels (the "ImageFolder" pattern)
- metadata.csv is the closest thing to a "database" at this scale
- A data pipeline script is reusable — will work for all future dishes

## Confusion cleared
- Training a model = showing examples + adjusting until predictions get good
- Untrained model = garbage output; trained model = good predictions

## Tomorrow's plan
- Add second dish (egusi soup or fried plantain) — another 25 photos
- Start training the first model: jollof vs egusi
