import torch, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import load, last_token_states

LAYER = 23
rng = np.random.RandomState(0)

# 1) 4-option items: 1 correct + 3 incorrect per question, shuffled
df = pd.read_csv("results/tqa_rows.csv")
items = []
for qid, g in df.groupby("qid"):
    c, w = g[g.label == 1].answer.tolist(), g[g.label == 0].answer.tolist()
    if len(c) < 1 or len(w) < 3: continue
    opts = [str(rng.choice(c))] + [str(x) for x in rng.choice(w, 3, replace=False)]
    order = rng.permutation(4)
    opts = [opts[i] for i in order]
    items.append({"qid": qid, "question": g.question.iloc[0], "opts": opts,
                  "correct": int(np.where(order == 0)[0][0])})
print(len(items), "items")

tok, model = load()
letters = "ABCD"
ids = [tok.encode(l, add_special_tokens=False) for l in letters]
assert all(len(i) == 1 for i in ids), ids
ids = [i[0] for i in ids]

def prompt(it):
    body = it["question"] + "\n" + "\n".join(f"{l}. {o}" for l, o in zip(letters, it["opts"]))
    msg = [{"role": "user", "content": body + "\n\nAnswer with the letter of the correct option only."}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)

# 2) the model's own pick + confidence (probability of its chosen letter)
probs, mass = [], []
with torch.no_grad():
    for i in tqdm(range(0, len(items), 8)):
        enc = tok([prompt(it) for it in items[i:i + 8]], return_tensors="pt", padding=True).to(model.device)
        full = torch.softmax(model(**enc).logits[:, -1, :].float(), -1)[:, ids]
        mass.append(full.sum(1).cpu().numpy())
        probs.append((full / full.sum(1, keepdim=True)).cpu().numpy())
probs, mass = np.concatenate(probs), np.concatenate(mass)
pick, conf = probs.argmax(1), probs.max(1)
right = np.array([p == it["correct"] for p, it in zip(pick, items)]); wrong = ~right
print(f"accuracy {right.mean():.3f} (chance 0.25) | picks per letter {np.bincount(pick, minlength=4)}"
      f" | mean prob mass on A-D {mass.mean():.3f}")
print("confidence quantiles (10/25/50/75/90%):", np.quantile(conf, [.1, .25, .5, .75, .9]).round(3))

# 3) belief: zero-shot probe (cities + neg_cities) on the CHOSEN option, same text format as before
texts = [f"Q: {it['question']}\nA: {it['opts'][p]}" for it, p in zip(items, pick)]
H = last_token_states(tok, model, texts, batch_size=8)[:, LAYER].float()
a, b = torch.load("results/acts/cities.pt"), torch.load("results/acts/neg_cities.pt")
X = torch.cat([a["X"], b["X"]])[:, LAYER].float(); y = torch.cat([a["y"], b["y"]]).numpy()
v0 = X[torch.from_numpy(y == 1)].mean(0) - X[torch.from_numpy(y == 0)].mean(0)
belief = (H @ v0).numpy()
pd.DataFrame({"conf": conf, "belief": belief, "wrong": wrong, "qid": [it["qid"] for it in items]}
             ).to_csv("results/mc_radar.csv", index=False)

# 4) AUROC for predicting "the model's pick is WRONG" (CV logistic for the combination)
z = lambda v: (v - v.mean()) / v.std()
cv = KFold(5, shuffle=True, random_state=0)
comb = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression()),
                         np.c_[conf, belief], wrong, cv=cv, method="decision_function")
print("\nAUROC for predicting 'pick is WRONG'")
for name, s in [("low confidence", -conf), ("low belief (probe)", -belief),
                ("confidence + belief (CV logistic)", comb), ("divergence (conf - belief)", z(conf) - z(belief))]:
    print(f"  {name:36s} {roc_auc_score(wrong, s):.3f}")

# 5) the radar's zone view (medians use no labels)
hc, hb = conf > np.median(conf), belief > np.median(belief)
print("\nzone                              |   n | % model wrong")
for name, m in [("high conf, LOW belief (alert)", hc & ~hb), ("high conf, high belief", hc & hb),
                ("low conf,  LOW belief", ~hc & ~hb), ("low conf,  high belief", ~hc & hb)]:
    print(f"{name:33s} | {m.sum():3d} | {100 * wrong[m].mean():.1f}")