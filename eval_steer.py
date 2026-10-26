import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from steer import *

tok, model = load()
rng = np.random.RandomState(0); torch.manual_seed(0)

# V_truth data: Geometry of Truth only
got = []
for n in GOT:
    d = pd.read_csv(DATA / f"{n}.csv")
    got.append((cached_states(tok, model, n, d.statement.tolist()), d.label.values))

# TruthfulQA: A = tune, B = report
df = pd.read_csv(RES / "tqa_rows.csv")
X = cached_states(tok, model, "tqa", df.text.tolist())
stm, y = df.text.values, df.label.values
qids = np.unique(df.qid); np.random.RandomState(0).shuffle(qids)
isA = df.qid.isin(qids[:len(qids) // 2]).values
allA = np.where(isA)[0]
iA = rng.choice(allA, 1000, replace=False)               # tuning subset (keeps the grid fast)
iB = np.where(~isA)[0]

def metrics(s, yy):
    p = s > 0
    ta, fa = p[yy == 1].mean(), (~p[yy == 0]).mean()
    return dict(bal=(ta + fa) / 2, true_acc=ta, false_acc=fa, auc=roc_auc_score(yy, s))

def run(idx, li, v, m, tau, alpha, mode="signed"):
    log = []
    s = tf_scores(tok, model, stm[idx].tolist(), LAYERS[li], steer_hook(v, m, tau, alpha, mode, log))
    r = metrics(s, y[idx]); r["fire"] = torch.cat(log).float().mean().item()
    return r

# 1) gate check + grid on A
PCTS, ALPHAS = [30, 60, 90], [0.1, 0.25, 0.5]
base = tf_scores(tok, model, stm[iA].tolist())
print("baseline A:", {k: round(float(x), 3) for k, x in metrics(base, y[iA]).items()})
correct = (base > 0) == y[iA]
best = (-1,)
for li, L in enumerate(LAYERS):
    v, m = universal(got, li), X[allA, li].mean(0)
    c = cos_of(X[iA, li], v, m).abs().numpy()
    print(f"L{L} gate check AUROC(|cos| -> model correct): {roc_auc_score(correct, c):.3f}")
    for p in PCTS:
        for a in ALPHAS:
            r = run(iA, li, v, m, np.percentile(c, p), a)
            print(f"   L{L} pct{p} alpha{a}: bal {r['bal']:.3f} fire {r['fire']:.2f}")
            if r["bal"] > best[0]: best = (r["bal"], li, p, a)

# 2) report on B at the best config, with controls
_, li, p, a = best; print("\nbest on A -> layer", LAYERS[li], "pct", p, "alpha", a)
v, m = universal(got, li), X[allA, li].mean(0)
tau = lambda d: float(np.percentile(cos_of(X[iA, li], d, m).abs().numpy(), p))   # same fire rate for every direction
v_rand = F.normalize(torch.randn_like(v), dim=0)
v_scr = universal([(Xs, rng.permutation(ys)) for Xs, ys in got], li)
runs = {
    "real, gated, signed": (v, tau(v), "signed"),
    "real, gated, +V":     (v, tau(v), "plus"),
    "real, always on":     (v, np.inf, "signed"),
    "random direction":    (v_rand, tau(v_rand), "signed"),
    "scrambled-label V":   (v_scr, tau(v_scr), "signed"),
}
print(f"{'B baseline':22s}", {k: round(float(x), 3) for k, x in metrics(tf_scores(tok, model, stm[iB].tolist()), y[iB]).items()})
for name, (d, t, mode) in runs.items():
    print(f"{name:22s}", {k: round(float(x), 3) for k, x in run(iB, li, d, m, t, a, mode).items()})