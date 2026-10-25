import torch, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def get(n):
    d = torch.load(f"results/acts/{n}.pt"); return d["X"], d["y"].numpy()
def mm(X, y):
    p = torch.from_numpy(y == 1); return X[p].mean(0) - X[~p].mean(0)
def win_rate(s, y, q):
    g = pd.DataFrame({"s": s, "y": y, "q": q}).groupby(["q", "y"]).s.mean().unstack().dropna()
    return (g[1] > g[0]).mean()

Xc, yc = get("cities"); Xn, yn = get("neg_cities"); Xt, yt = get("tqa")
Xtr = torch.cat([Xc, Xn]); ytr = np.concatenate([yc, yn])
df = pd.read_csv("results/tqa_rows.csv"); q = df.qid.values

rng = np.random.RandomState(0)                      # same split as eval_tqa2.py
qs = np.unique(q); rng.shuffle(qs)
A = np.isin(q, qs[: len(qs) // 2]); B = ~A
LAYER = 23

XB = Xt[torch.from_numpy(B)][:, LAYER].float()
Xtr_l = Xtr[:, LAYER].float()
yB, qB = yt[B], q[B]

# 1) the probe from cities+neg_cities
sB = (XB @ mm(Xtr_l, ytr)).numpy()
real_wr = win_rate(sB, yB, qB)
print(f"probe (cities+neg)     : win rate {real_wr:.3f} | AUROC {roc_auc_score(yB, sB):.3f}")

# 2) null: shuffled-label probes, within-question win rate
wr = np.array([win_rate((XB @ mm(Xtr_l, rng.permutation(ytr))).numpy(), yB, qB) for _ in range(200)])
print(f"shuffled-label null    : median |dev| {np.median(np.abs(wr - .5)):.3f} | 95th pct {np.percentile(np.abs(wr - .5), 95):.3f}")
print(f"   frac as extreme (two-sided) {np.mean(np.abs(wr - .5) >= abs(real_wr - .5)):.3f} | "
      f"frac >= real (one-sided) {np.mean(wr >= real_wr):.3f}")

# 3) text-only baseline, trained on half A answers, tested on half B
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
clf = LogisticRegression(max_iter=1000).fit(tf.fit_transform(df.answer[A]), yt[A])
sT = clf.decision_function(tf.transform(df.answer[B]))
print(f"text-only TF-IDF       : win rate {win_rate(sT, yB, qB):.3f} | AUROC {roc_auc_score(yB, sT):.3f}")

# 4) in-distribution probe (trained on TQA half A) = ceiling for this probe type
XA = Xt[torch.from_numpy(A)][:, LAYER].float()
sI = ((XB - XA.mean(0)) @ mm(XA, yt[A])).numpy()
print(f"probe trained on TQA-A : win rate {win_rate(sI, yB, qB):.3f} | AUROC {roc_auc_score(yB, sI):.3f}")