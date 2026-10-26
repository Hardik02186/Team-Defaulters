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
def per_q(s, y, q):                                   # bool per question: correct beats incorrect?
    g = pd.DataFrame({"s": s, "y": y, "q": q}).groupby(["q", "y"]).s.mean().unstack().dropna()
    return g[1] > g[0]

LAYER = 23
Xc, yc = get("cities"); Xn, yn = get("neg_cities"); Xt, yt = get("tqa")
df = pd.read_csv("results/tqa_rows.csv"); q = df.qid.values

rng = np.random.RandomState(0)                        # same split as before
qs = np.unique(q); rng.shuffle(qs)
A = np.isin(q, qs[: len(qs) // 2]); B = ~A
qA, qB, yA, yB = q[A], q[B], yt[A], yt[B]

X = Xt[:, LAYER].float()
XA, XB = X[torch.from_numpy(A)], X[torch.from_numpy(B)]

# feature 1: zero-shot probe (cities + neg_cities), no TQA labels used
v0 = mm(torch.cat([Xc, Xn])[:, LAYER].float(), np.concatenate([yc, yn]))
zsA, zsB = (XA @ v0).numpy(), (XB @ v0).numpy()

# feature 2: text-only score, out-of-fold on A, full-A model on B
txA = np.zeros(A.sum()); ansA, ansB = df.answer[A].values, df.answer[B].values
gkf = GroupKFold(n_splits=5)
for tr, te in gkf.split(ansA, yA, qA):
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    txA[te] = LogisticRegression(max_iter=1000).fit(tf.fit_transform(ansA[tr]), yA[tr]).decision_function(tf.transform(ansA[te]))
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
txB = LogisticRegression(max_iter=1000).fit(tf.fit_transform(ansA), yA).decision_function(tf.transform(ansB))

# feature 3: probe trained on TQA-A, out-of-fold on A
tqA = np.zeros(A.sum())
for tr, te in gkf.split(ansA, yA, qA):
    tqA[te] = ((XA[te] - XA[tr].mean(0)) @ mm(XA[tr], yA[tr])).numpy()
tqB = ((XB - XA.mean(0)) @ mm(XA, yA)).numpy()

feats = {"text only": ([txA], [txB]),
         "text + zero-shot probe": ([txA, zsA], [txB, zsB]),
         "text + TQA probe": ([txA, tqA], [txB, tqB]),
         "text + both probes": ([txA, zsA, tqA], [txB, zsB, tqB])}
print("model                    | win rate | AUROC")
for name, (fa, fb) in feats.items():
    sc = StandardScaler().fit(np.c_[tuple(fa)])
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(np.c_[tuple(fa)]), yA)
    s = lr.decision_function(sc.transform(np.c_[tuple(fb)]))
    print(f"{name:24s} | {per_q(s, yB, qB).mean():.3f}    | {roc_auc_score(yB, s):.3f}")

# complementarity: probe win rate, split by whether the text model got the question right
t_ok = per_q(txB, yB, qB); z_ok = per_q(zsB, yB, qB); p_ok = per_q(tqB, yB, qB)
for lab, m in [("text model RIGHT", t_ok), ("text model WRONG", ~t_ok)]:
    print(f"\n{lab} (n={m.sum()}): zero-shot probe win rate {z_ok[m].mean():.3f} | TQA probe win rate {p_ok[m].mean():.3f}")