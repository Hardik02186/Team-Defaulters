import torch, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def get(n):
    d = torch.load(f"results/acts/{n}.pt"); return d["X"], d["y"].numpy()   # stays fp16

def mm(X, y):                                       # X: [N, d] float
    p = torch.from_numpy(y == 1); return X[p].mean(0) - X[~p].mean(0)

Xc, yc = get("cities"); Xn, yn = get("neg_cities"); Xt, yt = get("tqa")
Xtr, ytr = torch.cat([Xc, Xn]), np.concatenate([yc, yn])
df = pd.read_csv("results/tqa_rows.csv"); groups = df.qid.values
gkf = GroupKFold(n_splits=5)

# --- baselines that don't use the model at all ---
a = roc_auc_score(yt, df.answer.str.len()); print(f"answer-length only AUROC: {max(a, 1 - a):.3f}")
tf, s = TfidfVectorizer(ngram_range=(1, 2), min_df=2), np.zeros(len(yt))
for tr, te in gkf.split(df, yt, groups):
    clf = LogisticRegression(max_iter=1000).fit(tf.fit_transform(df.answer.iloc[tr]), yt[tr])
    s[te] = clf.decision_function(tf.transform(df.answer.iloc[te]))
print(f"text-only (TF-IDF) group-CV AUROC: {roc_auc_score(yt, s):.3f}\n")

# --- layer sweep ---
print("layer | cities+neg -> TQA | TQA in-distribution (group CV)")
for l in range(10, Xt.shape[1]):
    v = mm(Xtr[:, l].float(), ytr)
    xt = Xt[:, l].float()
    transfer = roc_auc_score(yt, (xt @ v).numpy())          # NOT sign-folded: <0.5 means inverted
    sc = np.zeros(len(yt))
    for tr, te in gkf.split(xt, yt, groups):
        sc[te] = ((xt[te] - xt[tr].mean(0)) @ mm(xt[tr], yt[tr])).numpy()
    print(f"{l:5d} | {transfer:.3f}             | {roc_auc_score(yt, sc):.3f}")