import torch, numpy as np, pandas as pd
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import load, last_token_states

LAYER = 23
rng = np.random.RandomState(0)

# 1) same 4-option items as mc_radar.py (same seed, same order of random calls)
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

tok, model = load()
letters = "ABCD"
ids = [tok.encode(l, add_special_tokens=False)[0] for l in letters]

def prompt(it):
    body = it["question"] + "\n" + "\n".join(f"{l}. {o}" for l, o in zip(letters, it["opts"]))
    msg = [{"role": "user", "content": body + "\n\nAnswer with the letter of the correct option only."}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)

# 2) raw letter logits -> pick and unsaturated confidence (margin between top two letters)
LL = []
with torch.no_grad():
    for i in tqdm(range(0, len(items), 8)):
        enc = tok([prompt(it) for it in items[i:i + 8]], return_tensors="pt", padding=True).to(model.device)
        LL.append(model(**enc).logits[:, -1, ids].float().cpu().numpy())
LL = np.concatenate(LL)
pick = LL.argmax(1)
srt = np.sort(LL, 1); margin = srt[:, -1] - srt[:, -2]
p32 = torch.softmax(torch.from_numpy(LL), 1).numpy().max(1)
wrong = np.array([k != it["correct"] for k, it in zip(pick, items)])
print(f"accuracy {1 - wrong.mean():.3f} (earlier run: 0.632) | n={len(items)}"
      f" | float32 prob == 1.0 for {100 * (p32 >= 1.0).mean():.0f}% of items (ties)")

# 3) belief: zero-shot probe (cities + neg_cities) on the chosen option, same text format as before
texts = [f"Q: {it['question']}\nA: {it['opts'][k]}" for it, k in zip(items, pick)]
H = last_token_states(tok, model, texts, batch_size=8)[:, LAYER].float()
a, b = torch.load("results/acts/cities.pt"), torch.load("results/acts/neg_cities.pt")
X = torch.cat([a["X"], b["X"]])[:, LAYER].float(); y = torch.cat([a["y"], b["y"]]).numpy()
def mm(X, y):
    m = torch.from_numpy(y == 1); return X[m].mean(0) - X[~m].mean(0)
belief = (H @ mm(X, y)).numpy()
pd.DataFrame({"margin": margin, "belief": belief, "wrong": wrong,
              "qid": [it["qid"] for it in items]}).to_csv("results/mc_radar2.csv", index=False)

# 4) AUROC for predicting 'pick is WRONG'
auc = lambda s: roc_auc_score(wrong, s)
cv = KFold(5, shuffle=True, random_state=0)
def combo(*cols):
    return cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression()),
                             np.column_stack(cols), wrong, cv=cv, method="decision_function")
s_c, s_b, s_cb = -margin, -belief, combo(margin, belief)
print("\nAUROC for predicting 'pick is WRONG'")
for name, s in [("low confidence (letter-logit margin)", s_c), ("low belief (probe)", s_b),
                ("margin + belief (CV logistic)", s_cb)]:
    print(f"  {name:38s} {auc(s):.3f}")

# 5) bootstrap CI for the gain of (margin + belief) over margin alone
n, gains = len(wrong), []
for _ in range(1000):
    ix = rng.randint(0, n, n)
    if wrong[ix].min() == wrong[ix].max(): continue
    gains.append(roc_auc_score(wrong[ix], s_cb[ix]) - roc_auc_score(wrong[ix], s_c[ix]))
real_gain = auc(s_cb) - auc(s_c)
print(f"\ngain over margin alone: {real_gain:+.3f} | bootstrap 95% CI "
      f"[{np.percentile(gains, 2.5):+.3f}, {np.percentile(gains, 97.5):+.3f}]")

# 6) null: scrambled-label probes (trained on cities+neg_cities) in place of the real probe
null = np.array([auc(combo(margin, (H @ mm(X, np.random.RandomState(s).permutation(y))).numpy())) - auc(s_c)
                 for s in range(100)])
print(f"scrambled-label probes: median gain {np.median(null):+.3f} | 95th pct {np.percentile(null, 95):+.3f}"
      f" | frac >= real {np.mean(null >= real_gain):.3f}")

# 7) zone view (medians use no labels) and how many confident errors the alert zone catches
hc, hb = margin > np.median(margin), belief > np.median(belief)
print("\nzone                              |   n | % model wrong")
for name, m in [("high conf, LOW belief (alert)", hc & ~hb), ("high conf, high belief", hc & hb),
                ("low conf,  LOW belief", ~hc & ~hb), ("low conf,  high belief", ~hc & hb)]:
    print(f"{name:33s} | {m.sum():3d} | {100 * wrong[m].mean():.1f}")
err = hc & wrong
print(f"\nconfident picks that are wrong: {err.sum()} | in alert zone: {(err & ~hb).sum()} "
      f"({100 * (err & ~hb).sum() / err.sum():.0f}%) | alert zone is {100 * (hc & ~hb).sum() / hc.sum():.0f}% of confident picks")