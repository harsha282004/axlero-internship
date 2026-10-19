import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(sys.argv[1]).expanduser()
OUT = Path("../data/eda")
OUT.mkdir(exist_ok=True)
inv = pd.read_csv("../data/inventory_outer_with_patients.csv")
spl = pd.read_csv("../data/splits.csv")
print(len(inv), "images,", inv.patient_id.nunique(), "patients")

# class distribution: original vs proposed split
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for a, (col, title) in zip(ax, [("orig_split", "Original split"),
                                ("split", "Proposed patient-level split")]):
    t = spl.groupby([col, "label"]).size().unstack().reindex(["train", "val", "test"])
    t.plot(kind="bar", ax=a, rot=0)
    a.set_title(title)
    a.set_xlabel("")
    for c in a.containers:
        a.bar_label(c)
plt.tight_layout()
plt.savefig(OUT / "class_distribution.png", dpi=150)
plt.close()

# image size distribution
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
inv.width.hist(ax=ax[0], bins=40)
ax[0].set_title("Width (px)")
inv.height.hist(ax=ax[1], bins=40)
ax[1].set_title("Height (px)")
(inv.width / inv.height).hist(ax=ax[2], bins=40)
ax[2].set_title("Aspect ratio (w/h)")
plt.tight_layout()
plt.savefig(OUT / "image_sizes.png", dpi=150)
plt.close()
print(inv[["width", "height"]].describe().round(0))
print(inv["mode"].value_counts())

# images per patient
inv.groupby("patient_id").size().clip(upper=15).hist(bins=15)
plt.title("Images per patient (capped at 15)")
plt.xlabel("images")
plt.ylabel("patients")
plt.savefig(OUT / "images_per_patient.png", dpi=150)
plt.close()

# sample images (4 per class)
fig, ax = plt.subplots(2, 4, figsize=(12, 6.5))
for r, lab in enumerate(["NORMAL", "PNEUMONIA"]):
    s = inv[inv.label == lab].sample(4, random_state=42)
    for c, (_, row) in enumerate(s.iterrows()):
        im = Image.open(ROOT / row.p).convert("L")
        ax[r][c].imshow(im, cmap="gray")
        ax[r][c].axis("off")
        ax[r][c].set_title(lab, fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "sample_images.png", dpi=120)
plt.close()
print("Figures saved to", OUT.resolve())
