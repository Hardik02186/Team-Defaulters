import torch, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

def get(n):
    d = torch.load(f"results/acts/{n}.pt"); return d["X"], d["y"].numpy()
def mm(X, y):
    p = torch.from_numpy(y == 1); return X[p].mean(0) - X[~p].mean(0)
def wins(s, y, q):                                   # Series indexed by question: 1 if correct beats incorrect
    g = pd.DataFrame({"s": s, "y": y, "q": q}).groupby(["q", "y"]).s.mean().unstack().dropna()
    return (g[1] > g[0]).astype(float)

LAYER = 23
Xc, yc = get("cities"); Xn, yn = get("neg_cities"); Xt, yt = get("tqa")
Xtr = torch.cat([Xc, Xn])[:, LAYER].float(); ytr = np.concatenate([yc, yn])
df = pd.read_csv("results/tqa_rows.csv"); q = df.qid.values

rng = np.random.RandomState(0)                       # same split as eval_tqa2/3/4
qs = np.unique(q); rng.shuffle(qs)
A = np.isin(q, qs[: len(qs) // 2]); B = ~A
qA, qB, yA, yB = q[A], q[B], yt[A], yt[B]
X = Xt[:, LAYER].float(); XA, XB = X[torch.from_numpy(A)], X[torch.from_numpy(B)]

# text-only scores: out-of-fold on A, full-A model on B
ansA, ansB = df.answer[A].values, df.answer[B].values
txA = np.zeros(A.sum())
for tr, te in GroupKFold(5).split(ansA, yA, qA):
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    txA[te] = LogisticRegression(max_iter=1000).fit(tf.fit_transform(ansA[tr]), yA[tr]).decision_function(tf.transform(ansA[te]))
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
txB = LogisticRegression(max_iter=1000).fit(tf.fit_transform(ansA), yA).decision_function(tf.transform(ansB))

def stack(fA, fB):                                   # logistic regression fit on A, scores B
    sc = StandardScaler().fit(fA)
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(fA), yA)
    return lr.decision_function(sc.transform(fB))

auc_text = roc_auc_score(yB, txB); w_text = wins(txB, yB, qB)

# ---- real zero-shot probe ----
v0 = mm(Xtr, ytr)
sB = stack(np.c_[txA, (XA @ v0).numpy()], np.c_[txB, (XB @ v0).numpy()])
w_real = wins(sB, yB, qB)
real_auc, real_win = roc_auc_score(yB, sB) - auc_text, w_real.mean() - w_text.mean()
print(f"real gain over text-only: AUROC +{real_auc:.3f} | win rate +{real_win:.3f}")

# ---- 1) bootstrap over questions ----
rows = {u: np.where(qB == u)[0] for u in np.unique(qB)}; ul = np.array(list(rows))
common = w_text.index.intersection(w_real.index)
dw = (w_real[common] - w_text[common]).values
d_auc, d_win = [], []
for _ in range(1000):
    ri = np.concatenate([rows[u] for u in rng.choice(ul, len(ul))])
    if len(np.unique(yB[ri])) < 2: continue
    d_auc.append(roc_auc_score(yB[ri], sB[ri]) - roc_auc_score(yB[ri], txB[ri]))
    d_win.append(dw[rng.choice(len(dw), len(dw))].mean())
print(f"bootstrap 95% CI | AUROC gain [{np.percentile(d_auc, 2.5):.3f}, {np.percentile(d_auc, 97.5):.3f}]"
      f" | win-rate gain [{np.percentile(d_win, 2.5):.3f}, {np.percentile(d_win, 97.5):.3f}]")

# ---- 2) stacking null: scrambled-label probes added to the same text model ----
n_auc, n_win = [], []
for s in range(200):
    v = mm(Xtr, np.random.RandomState(s).permutation(ytr))
    sn = stack(np.c_[txA, (XA @ v).numpy()], np.c_[txB, (XB @ v).numpy()])
    n_auc.append(roc_auc_score(yB, sn) - auc_text)
    n_win.append(wins(sn, yB, qB).mean() - w_text.mean())
for name, nl, r in [("AUROC gain", n_auc, real_auc), ("win-rate gain", n_win, real_win)]:
    nl = np.array(nl)
    print(f"null {name}: median {np.median(nl):+.3f} | 95th pct {np.percentile(nl, 95):+.3f} | frac >= real {np.mean(nl >= r):.3f}")