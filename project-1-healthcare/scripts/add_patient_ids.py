import re
import pandas as pd

df = pd.read_csv("../data/inventory.csv")
df["p"] = df.path.str.replace("\\", "/", regex=False)
# exclude the byte-identical nested copy (reported, not deleted)
df = df[~df.p.str.startswith("chest_xray/chest_xray/")].copy()

def parse(path):
    name = path.split("/")[-1]
    m = re.match(r"(person\d+)_(bacteria|virus)_", name, re.I)
    if m:
        return m.group(1).lower(), m.group(2).lower()
    m = re.search(r"(IM-\d+)", name, re.I)
    if m:
        return m.group(1).upper(), "normal"
    return None, None

df[["patient_id", "subtype"]] = df.p.apply(lambda x: pd.Series(parse(x)))
print("Images:", len(df), "| unparsed:", df.patient_id.isna().sum())
df.to_csv("../data/inventory_outer_with_patients.csv", index=False)
