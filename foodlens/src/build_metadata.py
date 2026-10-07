"""
Scan foodlens/data/raw/ folders and generate metadata.csv.
Every image file in each dish folder becomes one row.
"""
import csv
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_CSV = Path(__file__).parent.parent / "data" / "metadata.csv"

rows = []
for label_dir in sorted(RAW_DIR.iterdir()):
    if not label_dir.is_dir():
        continue
    label = label_dir.name
    for img in sorted(label_dir.glob("*")):
        if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            rows.append({
                "filename": img.name,
                "label": label,
                "source": "istock",
                "angle": "",
                "lighting": "",
                "notes": "",
            })

with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["filename", "label", "source", "angle", "lighting", "notes"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT_CSV}")
