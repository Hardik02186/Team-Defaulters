import torch, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ACT = "results/acts"
TRAIN = "cities"
TESTS = ["sp_en_trans", "larger_than", "neg_cities"]   # transfer sets

def get(name):
    d = torch.load(f"{ACT}/{name}.pt")
    return d["X"].float(), d["y"].numpy()

def mass_mean(X, y):                              # X: [N, L, d]
    pos = torch.from_numpy(y == 1)
    mu_t, mu_f = X[pos].mean(0), X[~pos].mean(0)  # [L, d]
    v = mu_t - mu_f                               # points from false -> true
    bias = -(((mu_t + mu_f) / 2) * v).sum(-1)     # threshold at midpoint of class means
    return v, bias

def scores(X, v, bias):                           # -> [N, L]
    return (X * v).sum(-1) + bias

def sweep(X, y, v, bias):
    s = scores(X, v, bias).numpy()
    auc = [roc_auc_score(y, s[:, l]) for l in range(s.shape[1])]
    acc = [((s[:, l] > 0) == (y == 1)).mean() for l in range(s.shape[1])]
    return np.array(auc), np.array(acc)

# 1) within-dataset: split the training set so we can pick a layer fairly
# 1) within-dataset: split BY CITY so no city appears in both train and validation
import pandas as pd
X, y = get(TRAIN)
cities = pd.read_csv(f"/data3/Hardik/LSS/data/{TRAIN}.csv")["city"].values
assert len(cities) == len(y)
uniq = np.unique(cities)
rng = np.random.RandomState(0)
val_cities = set(rng.choice(uniq, size=int(0.2 * len(uniq)), replace=False))
is_val = np.array([c in val_cities for c in cities])
tr, va = np.where(~is_val)[0], np.where(is_val)[0]
print(f"train rows: {len(tr)} | val rows: {len(va)} | val cities: {len(val_cities)}")

v, bias = mass_mean(X[tr], y[tr])
auc_val, acc_val = sweep(X[va], y[va], v, bias)
best = int(auc_val.argmax())
print(f"Best layer on held-out {TRAIN}: {best} (AUROC {auc_val[best]:.3f}, acc {acc_val[best]:.3f})")

# 2) fit on all of TRAIN, test transfer. Layer was chosen WITHOUT looking at these sets.
v, bias = mass_mean(X, y)
curves = {f"{TRAIN} (held-out)": auc_val}
for name in TESTS:
    Xt, yt = get(name)
    auc, acc = sweep(Xt, yt, v, bias)
    curves[name] = auc
    print(f"{name:14s} @ layer {best}: AUROC {auc[best]:.3f}, acc {acc[best]:.3f}")

plt.figure(figsize=(8, 4.5))
for k, c in curves.items():
    plt.plot(range(len(c)), c, label=k)
plt.axvline(best, ls="--", c="gray")
plt.axhline(0.5, ls=":", c="red", lw=0.8)
plt.xlabel("layer"); plt.ylabel("AUROC"); plt.title("Mass-mean truth probe by layer")
plt.legend(); plt.tight_layout(); plt.savefig("results/layer_sweep.png", dpi=150)
torch.save({"v": v, "bias": bias, "best_layer": best}, "results/truth_probe.pt")
print("saved results/layer_sweep.png and results/truth_probe.pt")