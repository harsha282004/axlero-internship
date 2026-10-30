import json

def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}

def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": t.strip("\n").splitlines(True)}

cells = [
md("""# EDA: Chest X-Ray Images (Pneumonia)
Dataset: Kermany et al., Cell 2018 (CC BY 4.0). Provenance, licence and
data-quality findings are in `data/README.md`. Images are not stored in the repo;
this notebook uses the CSVs in `data/` and the figures in `data/eda/`."""),
code("""import pandas as pd
from IPython.display import Image, display

inv = pd.read_csv("../data/inventory_outer_with_patients.csv")
spl = pd.read_csv("../data/splits.csv")
print(len(inv), "images,", inv.patient_id.nunique(), "patients")"""),
md("## Class counts: original vs proposed patient-level split"),
code("""for col in ["orig_split", "split"]:
    print(col)
    print(spl.groupby([col, "label"]).size().unstack().reindex(["train", "val", "test"]), "\\n")"""),
code("""display(Image("../data/eda/class_distribution.png"))"""),
md("## Image size and colour mode, by class"),
code("""print(inv.groupby("label")[["width", "height"]].median())
print(pd.crosstab(inv.label, inv["mode"]))
print(inv[["width", "height"]].describe().round(0))"""),
code("""display(Image("../data/eda/image_sizes.png"))"""),
md("## Patient overlap in the original split"),
code("""n = spl.groupby("patient_id").orig_split.nunique()
leak = n[n > 1].index
lp = spl[spl.patient_id.isin(leak)].drop_duplicates("patient_id")
print("Patients in >1 original split, by label:")
print(lp.groupby("label").size())
print("Patients in >1 PROPOSED split:", (spl.groupby("patient_id").split.nunique() > 1).sum())"""),
code("""display(Image("../data/eda/images_per_patient.png"))"""),
md("## Representative images (4 per class, random_state=42)"),
code("""display(Image("../data/eda/sample_images.png"))"""),
md("""## Conclusions (observed in this dataset)
- 5,856 unique images from 2,790 patient IDs; 0 corrupt files; the zip also holds an
  identical nested copy, excluded from analysis.
- Classes are imbalanced, about 2.7:1 PNEUMONIA to NORMAL, and the proposed split
  keeps that ratio in train, val and test.
- Image sizes vary widely (width 384-2,916 px, height 127-2,713 px) and differ by
  class (median 1654x1323 px NORMAL vs 1160x776 px PNEUMONIA).
- 283 images (4.8%) are RGB, all PNEUMONIA; the rest are grayscale.
- A single height threshold classifies 84.8% of images correctly vs a 73.0%
  majority baseline (full dataset, no held-out set): size is a possible shortcut.
- The original split is not patient-level: 170 pneumonia patient IDs appear in both
  train and test. 94 normal IDs also appear in more than one split, but may be ID
  collisions between two filename series (unconfirmed).
- The proposed split has 0 patients in more than one split (`verify_splits.py` PASS)."""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
      "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("../notebooks/eda.ipynb", "w"), indent=1)
print("Wrote ../notebooks/eda.ipynb")
