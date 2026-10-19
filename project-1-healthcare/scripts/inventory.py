import hashlib, sys
from pathlib import Path
import pandas as pd
from PIL import Image

root = Path(sys.argv[1]).expanduser()
exts = {".jpg", ".jpeg", ".png"}
rows = []
for p in sorted(root.rglob("*")):
    if "__MACOSX" in p.parts or not p.is_file() or p.suffix.lower() not in exts:
        continue
    rel = p.relative_to(root)
    parts = [x.lower() for x in rel.parts]
    row = {
        "path": str(rel),
        "split": next((x for x in parts if x in ("train", "val", "test")), ""),
        "label": p.parent.name,
        "format": p.suffix.lower(),
        "width": None, "height": None, "mode": "",
        "sha256": "", "status": "ok",
    }
    try:
        row["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im.load()
            row["width"], row["height"], row["mode"] = im.width, im.height, im.mode
    except Exception as e:
        row["status"] = f"corrupt: {type(e).__name__}"
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("../data/inventory.csv", index=False)
print("Total images:", len(df))
print("Flagged as corrupt:", (df.status != "ok").sum())
print("Duplicate hashes (extra copies):", df.sha256.duplicated().sum())
print(df.groupby(["split", "label"]).size())
