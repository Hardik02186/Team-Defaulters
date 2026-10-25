import glob, os, pandas as pd, torch
from common import load, last_token_states

# finds your TruthfulQA csv anywhere under the current folder (src/)
cands = [p for p in glob.glob("/data3/Hardik/LSS/data/TruthfulQA.csv", recursive=True) if not p.startswith("results")]
assert cands, "Couldn't find a TruthfulQA csv under src/. Put the path in PATH manually."
PATH = cands[0]
print("using:", PATH)

raw = pd.read_csv(PATH)
split = lambda v: [a.strip() for a in v.split(";") if a.strip()] if isinstance(v, str) else []
fix = lambda a: a if a.endswith((".", "!", "?")) else a + "."

rows = []
for qid, r in raw.iterrows():
    for a in split(r["Correct Answers"]):
        if not a.lower().startswith("i have no comment"):
            rows.append((qid, r["Question"], fix(a), 1))
    for a in split(r["Incorrect Answers"]):
        rows.append((qid, r["Question"], fix(a), 0))

df = pd.DataFrame(rows, columns=["qid", "question", "answer", "label"])
df["text"] = "Q: " + df.question + "\nA: " + df.answer
os.makedirs("results/acts", exist_ok=True)
df.to_csv("results/tqa_rows.csv", index=False)
print(len(df), "rows |", df.qid.nunique(), "questions | fraction correct:", df.label.mean().round(2))

tok, model = load()
X = last_token_states(tok, model, df.text.tolist(), batch_size=8)
torch.save({"X": X, "y": torch.tensor(df.label.values)}, "results/acts/tqa.pt")
print("saved", tuple(X.shape))