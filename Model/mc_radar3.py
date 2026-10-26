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

# same items as mc_radar2.py
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

LL = []
with torch.no_grad():
    for i in tqdm(range(0, len(items), 8)):
        enc = tok([prompt(it) for it in items[i:i + 8]], return_tensors="pt", padding=True).to(model.device)
        LL.append(model(**enc).logits[:, -1, ids].float().cpu().numpy())
LL = np.concatenate(LL)
pick = LL.argmax(1); srt = np.sort(LL, 1); margin = srt[:, -1] - srt[:, -2]
wrong = np.array([k != it["correct"] for k, it in zip(pick, items)])
print(f"accuracy {1 - wrong.mean():.3f} (earlier: 0.632) | n={len(items)}")

texts = [f"Q: {it['question']}\nA: {it['opts'][k]}" for it, k in zip(items, pick)]
H = last_token_states(tok, model, texts, batch_size=8)[:, LAYER].float()
torch.save(H, "results/mc_H.pt")
np.savez("results/mc_meta.npz", margin=margin, wrong=wrong)

def load_acts(names):
    Xs, ys = [], []
    for n in names:
        d = torch.load(f"results/acts/{n}.pt")
        Xs.append(d["X"][:, LAYER].float()); ys.append(d["y"].numpy())
    return torch.cat(Xs), np.concatenate(ys)
def mm(X, y):
    m = torch.from_numpy(y == 1); return X[m].mean(0) - X[~m].mean(0)

cv = KFold(5, shuffle=True, random_state=0)
def combo(*cols):
    return cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression()),
                             np.column_stack(cols), wrong, cv=cv, method="decision_function")
base = roc_auc_score(wrong, -margin)
hc = margin > np.median(margin); err = hc & wrong
def stats(belief):                                    # median split uses no labels
    hb = belief > np.median(belief)
    return wrong[hc & ~hb].mean() - wrong[hc & hb].mean(), (err & ~hb).sum() / err.sum()
def report(label, real, null):
    null = np.array(null)
    print(f"  {label:34s} real {real:.3f} | null median {np.median(null):.3f} | "
          f"95th pct {np.percentile(null, 95):.3f} | frac >= real {np.mean(null >= real):.3f}")

# real vs scrambled-label directions, for two training sets (the second one is exploratory)
for name, sets in [("cities+neg", ["cities", "neg_cities"]),
                   ("cities+neg+larger_than", ["cities", "neg_cities", "larger_than"])]:
    X, y = load_acts(sets)
    belief = (H @ mm(X, y)).numpy()
    r_gap, r_cap = stats(belief)
    r_gain = roc_auc_score(wrong, combo(margin, belief)) - base
    nr = np.random.RandomState(1); n_gain, n_gap, n_cap = [], [], []
    for _ in range(100):
        b = (H @ mm(X, nr.permutation(y))).numpy()
        n_gain.append(roc_auc_score(wrong, combo(margin, b)) - base)
        g_, c_ = stats(b); n_gap.append(g_); n_cap.append(c_)
    print(f"\n{name}: belief alone AUROC {roc_auc_score(wrong, -belief):.3f}")
    report("gain over margin (AUROC)", r_gain, n_gain)
    report("confident-pick wrong-rate gap", r_gap, n_gap)
    report("share of confident errors in alert", r_cap, n_cap)

# supervised reference: a probe trained on this task's own right/wrong labels (rough ceiling)
sup = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.001, max_iter=2000)),
                        H.numpy(), wrong, cv=cv, method="decision_function")
print(f"\nsupervised error probe (5-fold CV, C=0.001): AUROC {roc_auc_score(wrong, sup):.3f} | margin alone {base:.3f}")