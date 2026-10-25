import glob, os, torch, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

LAYER = 23
FAMILY = {"cities": "geo", "neg_cities": "geo"}          # same cities -> same family; others are their own

acts = {}
for p in sorted(glob.glob("results/acts/*.pt")):
    n = os.path.basename(p)[:-3]
    if n == "tqa": continue
    d = torch.load(p); acts[n] = (d["X"][:, LAYER].float(), d["y"].numpy())
print("datasets:", {k: len(v[1]) for k, v in acts.items()})

mm = lambda X, y: X[torch.from_numpy(y == 1)].mean(0) - X[torch.from_numpy(y == 0)].mean(0)
unit = lambda v: v / v.norm()

def universal(data, svd=False):
    vs = torch.stack([unit(mm(X, y)) for X, y in data.values()])
    m = unit(vs.mean(0))
    if not svd: return m
    v1 = torch.linalg.svd(vs, full_matrices=False)[2][0]
    return v1 if float(v1 @ m) > 0 else -v1              # fix SVD sign ambiguity

def pooled(data):
    return mm(torch.cat([v[0] for v in data.values()]), np.concatenate([v[1] for v in data.values()]))

names = list(acts)
vs = torch.stack([unit(mm(*acts[n])) for n in names])
print("\ncosine between per-dataset directions:\n", pd.DataFrame((vs @ vs.T).numpy(), names, names).round(2))

# --- TruthfulQA, same question split as before; report on half B ---
d = torch.load("results/acts/tqa.pt"); Xt, yt = d["X"], d["y"].numpy()
df = pd.read_csv("results/tqa_rows.csv"); q = df.qid.values
rng = np.random.RandomState(0); qs = np.unique(q); rng.shuffle(qs)
B = ~np.isin(q, qs[: len(qs) // 2])
XB, yB, qB = Xt[torch.from_numpy(B)][:, LAYER].float(), yt[B], q[B]

def win(s):
    g = pd.DataFrame({"s": s, "y": yB, "q": qB}).groupby(["q", "y"]).s.mean().unstack().dropna()
    return (g[1] > g[0]).mean()
def tqa(v):
    s = (XB @ v).numpy(); return roc_auc_score(yB, s), win(s)

print("\nTruthfulQA half B (unseen questions)   AUROC | win rate")
for label, v in [("pooled mass-mean (all)", pooled(acts)), ("universal: mean of unit vectors", universal(acts)),
                 ("universal: SVD variant", universal(acts, svd=True))]:
    a, w = tqa(v); print(f"  {label:34s} {a:.3f} | {w:.3f}")

# --- null: scramble labels WITHIN each dataset, rebuild the universal vector ---
nr = np.random.RandomState(1)
null = np.array([tqa(universal({k: (X, nr.permutation(y)) for k, (X, y) in acts.items()}))[0] for _ in range(100)])
real = tqa(universal(acts))[0]
print(f"  null (scrambled labels): median |dev| {np.median(np.abs(null - .5)):.3f} | "
      f"frac >= real one-sided {np.mean(null >= real):.3f} | two-sided {np.mean(np.abs(null - .5) >= abs(real - .5)):.3f}")

# --- leave-one-family-out: train on other families, test on the held-out one ---
fams = sorted({FAMILY.get(n, n) for n in names})
if len(fams) < 2:
    print("\nneed at least two dataset families for leave-one-out")
else:
    print("\nleave-one-family-out AUROC (not sign-folded)   universal | pooled")
    for f in fams:
        tr = {k: v for k, v in acts.items() if FAMILY.get(k, k) != f}
        vu, vp = universal(tr), pooled(tr)
        for n in names:
            if FAMILY.get(n, n) == f:
                X, y = acts[n]
                print(f"  held out {n:14s} {roc_auc_score(y, (X @ vu).numpy()):.3f}     | {roc_auc_score(y, (X @ vp).numpy()):.3f}")