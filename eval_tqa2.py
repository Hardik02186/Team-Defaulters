import torch, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

def get(n):
    d = torch.load(f"results/acts/{n}.pt"); return d["X"], d["y"].numpy()
def mm(X, y):
    p = torch.from_numpy(y == 1); return X[p].mean(0) - X[~p].mean(0)

Xc, yc = get("cities"); Xn, yn = get("neg_cities"); Xt, yt = get("tqa")
Xtr = torch.cat([Xc, Xn]); ytr = np.concatenate([yc, yn])
df = pd.read_csv("results/tqa_rows.csv"); q = df.qid.values

# split questions into A (choose layer) and B (report)
rng = np.random.RandomState(0)
qs = np.unique(q); rng.shuffle(qs)
A = np.isin(q, qs[: len(qs) // 2]); B = ~A
At, Bt = torch.from_numpy(A), torch.from_numpy(B)

LAYERS = range(14, 36)
vs = {l: mm(Xtr[:, l].float(), ytr) for l in LAYERS}
aucA = {l: roc_auc_score(yt[A], (Xt[At][:, l].float() @ vs[l]).numpy()) for l in LAYERS}
best = max(aucA, key=aucA.get)
print(f"layer chosen on question-half A: {best} (A AUROC {aucA[best]:.3f})")

XB = Xt[Bt][:, best].float(); v = vs[best]
sB = (XB @ v).numpy(); real = roc_auc_score(yt[B], sB)
print(f"half B (unseen questions) AUROC: {real:.3f}")

# nulls: scrambled-label probes and random directions, scored on half B
Xtr_b = Xtr[:, best].float()
shuf = [roc_auc_score(yt[B], (XB @ mm(Xtr_b, rng.permutation(ytr))).numpy()) for _ in range(200)]
rand = [roc_auc_score(yt[B], (XB @ torch.randn(XB.shape[1])).numpy()) for _ in range(200)]
dev = lambda a: np.abs(np.array(a) - 0.5)
for name, nl in [("shuffled-label", shuf), ("random-direction", rand)]:
    print(f"{name:17s} median |dev| {np.median(dev(nl)):.3f} | 95th pct {np.percentile(dev(nl), 95):.3f} "
          f"| frac as extreme as real: {np.mean(dev(nl) >= abs(real - 0.5)):.3f}")

# within-question: does the mean score of correct answers beat that of incorrect ones?
g = pd.DataFrame({"s": sB, "y": yt[B], "q": q[B]}).groupby(["q", "y"]).s.mean().unstack().dropna()
print(f"within-question win rate on B: {(g[1] > g[0]).mean():.3f} over {len(g)} questions (chance 0.5)")