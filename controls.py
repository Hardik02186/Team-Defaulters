import torch, numpy as np
from sklearn.metrics import roc_auc_score

BEST = 16

def get(name):
    d = torch.load(f"results/acts/{name}.pt")
    return d["X"].float(), d["y"].numpy()

def mm(X, y):                       # mass-mean direction per layer: [L, d]
    pos = torch.from_numpy(y == 1)
    return X[pos].mean(0) - X[~pos].mean(0)

def auc_at(layer, v, X, y):
    return roc_auc_score(y, (X[:, layer] @ v).numpy())

Xc, yc = get("cities"); Xl, yl = get("larger_than")
L, d = Xc.shape[1], Xc.shape[2]
print("layers:", L, "| larger_than rows:", len(yl), "| true fraction:", yl.mean().round(2))

# 1) AUROC at every layer: is layer 16 special, or is it a plateau?
v = mm(Xc, yc)
print("\nlayer: cities->larger_than AUROC")
for l in range(L):
    print(l, round(auc_at(l, v[l], Xl, yl), 3))

# 2) random directions should score near chance
g = torch.Generator().manual_seed(0)
r = [auc_at(BEST, torch.randn(d, generator=g), Xl, yl) for _ in range(50)]
print(f"\nrandom directions @ layer {BEST}: mean {np.mean(r):.3f}, max {np.max(r):.3f}")

# 3) probe trained on shuffled labels should also score near chance
vs = mm(Xc, np.random.RandomState(0).permutation(yc))
print(f"shuffled-label probe @ layer {BEST}: {auc_at(BEST, vs[BEST], Xl, yl):.3f}")