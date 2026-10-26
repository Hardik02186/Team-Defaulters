import torch, pandas as pd, numpy as np
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []
from common import load

df = pd.read_csv("results/tqa_rows.csv")
tok, model = load()                                   # left padding, as in common.py
full = df.text.tolist()
prefix = ("Q: " + df.question + "\nA:").tolist()

mean_lp, sum_lp, n_tok, mismatches = [], [], [], 0
with torch.no_grad():
    for i in tqdm(range(0, len(df), 4)):
        f, p = full[i:i + 4], prefix[i:i + 4]
        enc = tok(f, return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits
        ids, T = enc["input_ids"], enc["input_ids"].shape[1]
        for j in range(len(f)):
            p_ids = tok(p[j])["input_ids"]
            n_f = int(enc["attention_mask"][j].sum())
            n_a = n_f - len(p_ids)                    # answer tokens sit at the END (left padding)
            if n_a <= 0 or ids[j, T - n_f:T - n_f + len(p_ids)].tolist() != p_ids:
                mismatches += 1
            if n_a <= 0:
                mean_lp.append(np.nan); sum_lp.append(np.nan); n_tok.append(0); continue
            lp = torch.log_softmax(logits[j, T - n_a - 1:T - 1].float(), -1)
            lp = lp.gather(1, ids[j, T - n_a:T].unsqueeze(1)).squeeze(1)
            mean_lp.append(lp.mean().item()); sum_lp.append(lp.sum().item()); n_tok.append(n_a)

out = pd.DataFrame({"mean_lp": mean_lp, "sum_lp": sum_lp, "n_tok": n_tok})
out.to_csv("results/tqa_conf.csv", index=False)
print("rows:", len(out), "| NaN:", int(out.mean_lp.isna().sum()), "| prefix-token mismatches:", mismatches)
print(out.describe().round(3))