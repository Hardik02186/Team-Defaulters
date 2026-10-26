import os, sys, pandas as pd, torch
from common import load, last_token_states

DATA = "C:\\Users\\Sid\\Desktop\\Study\\Team-Defaulters\\Data"
OUT = "results/acts"
TEXT_COL, LABEL_COL = "statement", "label"
DATASETS = ["cities", "neg_cities", "sp_en_trans", "larger_than"]   # start with these

os.makedirs(OUT, exist_ok=True)
tok, model = load()

for name in DATASETS:
    path = f"{DATA}/{name}.csv"
    if not os.path.exists(path):
        print("missing:", path); continue
    df = pd.read_csv(path)
    if TEXT_COL not in df or LABEL_COL not in df:
        print(name, "columns are:", list(df.columns)); continue
    print(name, len(df), "rows | label balance:", df[LABEL_COL].mean().round(2))
    X = last_token_states(tok, model, df[TEXT_COL].tolist())
    y = torch.tensor(df[LABEL_COL].values, dtype=torch.long)
    torch.save({"X": X, "y": y}, f"{OUT}/{name}.pt")
    print("saved", tuple(X.shape))