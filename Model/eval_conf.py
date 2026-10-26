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
def win(s, y, q):
    g = pd.DataFrame({"s": s, "y": y, "q": q}).groupby(["q", "y"]).s.mean().unstack().dropna()
    return (g[1] > g[0]).mean()

LAYER = 23
Xc, yc = get("cities"); Xn, yn = get("neg_cities"); Xt, yt = get("tqa")
df = pd.read_csv("results/tqa_rows.csv"); cf = pd.read_csv("results/tqa_conf.csv")
ok = cf.mean_lp.notna().values; q = df.qid.values

rng = np.random.RandomState(0)                         # same split as the earlier scripts
qs = np.unique(q); rng.shuffle(qs)
A = np.isin(q, qs[: len(qs) // 2]) & ok; B = ~np.isin(q, qs[: len(qs) // 2]) & ok
qA, qB, yA, yB = q[A], q[B], yt[A], yt[B]

X = Xt[:, LAYER].float(); XA, XB = X[torch.from_numpy(A)], X[torch.from_numpy(B)]
v0 = mm(torch.cat([Xc, Xn])[:, LAYER].float(), np.concatenate([yc, yn]))
prA, prB = (XA @ v0).numpy(), (XB @ v0).numpy()        # belief (zero-shot probe)
cnA, cnB = cf.mean_lp.values[A], cf.mean_lp.values[B]  # confidence (mean token logprob)

ansA, ansB = df.answer[A].values, df.answer[B].values  # text-only baseline, out-of-fold on A
txA = np.zeros(A.sum())
for tr, te in GroupKFold(5).split(ansA, yA, qA):
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    txA[te] = LogisticRegression(max_iter=1000).fit(tf.fit_transform(ansA[tr]), yA[tr]).decision_function(tf.transform(ansA[te]))
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
txB = LogisticRegression(max_iter=1000).fit(tf.fit_transform(ansA), yA).decision_function(tf.transform(ansB))

def stack(fa, fb):
    sc = StandardScaler().fit(np.c_[tuple(fa)])
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(np.c_[tuple(fa)]), yA)
    return lr.decision_function(sc.transform(np.c_[tuple(fb)]))

print("scorer                      | win rate | AUROC   (raw AUROC is NOT sign-folded)")
rows = {"confidence alone": cnB, "probe alone": prB,
        "confidence + probe": stack([cnA, prA], [cnB, prB]),
        "text only": txB,
        "text + confidence": stack([txA, cnA], [txB, cnB]),
        "text + probe": stack([txA, prA], [txB, prB]),
        "text + confidence + probe": stack([txA, cnA, prA], [txB, cnB, prB])}
for k, s in rows.items():
    print(f"{k:27s} | {win(s, yB, qB):.3f}    | {roc_auc_score(yB, s):.3f}")

# the radar's actual claim: among CONFIDENT answers, does low belief predict incorrect?
hi_c, hi_b = cnB > np.median(cnA), prB > np.median(prA)   # thresholds from half A only
print("\nzone                              |   n | % incorrect")
for name, m in [("high conf, LOW belief (alert)", hi_c & ~hi_b), ("high conf, high belief", hi_c & hi_b),
                ("low conf,  LOW belief", ~hi_c & ~hi_b), ("low conf,  high belief", ~hi_c & hi_b)]:
    print(f"{name:33s} | {m.sum():3d} | {100 * (yB[m] == 0).mean():.1f}")
z = lambda a, ref: (a - ref.mean()) / ref.std()
d = z(cnB, cnA) - z(prB, prA)                              # divergence: confident but doubted
print(f"\ndivergence score vs 'answer is incorrect' AUROC: {roc_auc_score(yB == 0, d):.3f}")