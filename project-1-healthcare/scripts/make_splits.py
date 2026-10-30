import numpy as np, pandas as pd

SEED = 42
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
COLS = ["NORMAL", "PNEUMONIA"]

df = pd.read_csv("../data/inventory_outer_with_patients.csv")
df = df[["p", "label", "subtype", "patient_id", "split"]].rename(
    columns={"p": "path", "split": "orig_split"})

g = df.groupby("patient_id").label.value_counts().unstack(fill_value=0)
for c in COLS:
    if c not in g.columns:
        g[c] = 0
g = g[COLS]
print("Patients with both labels:", ((g > 0).sum(axis=1) > 1).sum())

arr = g.values.astype(float)
tot = arr.sum(axis=0)
target = {s: tot * r for s, r in RATIOS.items()}
cur = {s: np.zeros(2) for s in RATIOS}

rng = np.random.RandomState(SEED)
order = sorted(rng.permutation(len(g)), key=lambda i: -arr[i].sum())

assign = {}
for i in order:
    best, best_cost = None, None
    for s in RATIOS:
        before = (((cur[s] - target[s]) / target[s]) ** 2).sum()
        after = (((cur[s] + arr[i] - target[s]) / target[s]) ** 2).sum()
        cost = after - before
        if best_cost is None or cost < best_cost:
            best, best_cost = s, cost
    assign[g.index[i]] = best
    cur[best] += arr[i]

df["split"] = df.patient_id.map(assign)
df[["path", "label", "subtype", "patient_id", "orig_split", "split"]].to_csv(
    "../data/splits.csv", index=False)

print(df.groupby(["split", "label"]).size().unstack())
print(df.split.value_counts(normalize=True).round(3))
