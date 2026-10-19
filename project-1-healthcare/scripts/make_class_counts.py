import pandas as pd

s = pd.read_csv("../data/splits.csv")
rows = []
for kind, col in [("original", "orig_split"), ("proposed", "split")]:
    t = s.groupby([col, "label"]).size().unstack(fill_value=0)
    for sp in ["train", "val", "test"]:
        for lab in t.columns:
            rows.append({"split_type": kind, "split": sp, "label": lab,
                         "count": int(t.loc[sp, lab])})
out = pd.DataFrame(rows)
out.to_csv("../data/class_counts.csv", index=False)
print(out.pivot_table(index=["split_type", "split"], columns="label", values="count"))
