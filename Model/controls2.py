import torch, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

def get(name):
    d = torch.load(f"results/acts/{name}.pt"); return d["X"].float(), d["y"].numpy()

def mm2(X, y):                                   # X: [N, d]
    p = torch.from_numpy(y == 1); return X[p].mean(0) - X[~p].mean(0)

auc = lambda v, X, y: roc_auc_score(y, (X @ v).numpy())
cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
dev = lambda a: np.abs(np.array(a) - 0.5)

LAYER = 16
Xc, yc = get("cities"); Xl, yl = get("larger_than")
Xc, Xl = Xc[:, LAYER], Xl[:, LAYER]
perm = lambda s: np.random.RandomState(s).permutation(yc)

# A) many-seed shuffled null on larger_than
real = mm2(Xc, yc)
null = [auc(mm2(Xc, perm(s)), Xl, yl) for s in range(200)]
print(f"A) real {auc(real, Xl, yl):.3f} | shuffled |AUROC-0.5|: median {np.median(dev(null)):.3f}, "
      f"95th pct {np.percentile(dev(null), 95):.3f} | frac beyond [0.1, 0.9]: {np.mean([(n > .9) or (n < .1) for n in null]):.2f}")

# B) direction alignment: cities direction vs larger_than's own direction
own = mm2(Xl, yl)
cs = [cos(mm2(Xc, perm(s)), own) for s in range(200)]
print(f"B) cos(real, own) {cos(real, own):.3f} | shuffled |cos|: mean {np.mean(np.abs(cs)):.3f}, max {np.max(np.abs(cs)):.3f}")

# C) remove the top-k principal components of the test data, then re-test
_, _, Vh = torch.linalg.svd(Xl - Xl.mean(0), full_matrices=False)
def remove(M, k):
    if k == 0: return M
    P = Vh[:k]; return M - (M @ P.T) @ P
print("C) PC removal on larger_than test data")
for k in [0, 1, 3, 10, 30]:
    Xk = remove(Xl, k)
    r = auc(remove(real[None], k)[0], Xk, yl)
    nl = [auc(remove(mm2(Xc, perm(s))[None], k)[0], Xk, yl) for s in range(100)]
    print(f"   top {k:2d} PCs removed: real {r:.3f} | shuffled median |dev| {np.median(dev(nl)):.3f}")

# D) proper control for the headline result: shuffle labels in TRAIN cities, test on held-out cities
cities = pd.read_csv("/data3/Hardik/LSS/data/cities.csv")["city"].values
u = np.unique(cities); val = set(np.random.RandomState(0).choice(u, int(0.2 * len(u)), replace=False))
isv = np.array([c in val for c in cities]); tr, va = np.where(~isv)[0], np.where(isv)[0]
r = auc(mm2(Xc[tr], yc[tr]), Xc[va], yc[va])
nl = [auc(mm2(Xc[tr], np.random.RandomState(s).permutation(yc[tr])), Xc[va], yc[va]) for s in range(100)]
print(f"D) held-out cities: real {r:.3f} | shuffled-train median |dev| {np.median(dev(nl)):.3f}, max |dev| {np.max(dev(nl)):.3f}")