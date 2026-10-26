import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from steer import (
    DATA, RES, LAYERS, GOT, load, tf_prompt, cached_states, universal, cos_of
)

tok, model = load()
rng = np.random.RandomState(0); torch.manual_seed(0)

# 1. Load cached GOT & TQA states (already extracted by eval_steer.py)
got = []
for n in GOT:
    d = pd.read_csv(DATA / f"{n}.csv")
    got.append((cached_states(tok, model, n, d.statement.tolist()), d.label.values))

df = pd.read_csv(RES / "tqa_rows.csv")
X = cached_states(tok, model, "tqa", df.text.tolist())
stm, y, qids_all = df.text.values, df.label.values, df.qid.values

qids = np.unique(qids_all); np.random.RandomState(0).shuffle(qids)
isA = df.qid.isin(qids[:len(qids) // 2]).values
allA = np.where(isA)[0]
iA = rng.choice(allA, 1000, replace=False)
iB = np.where(~isA)[0]

# 2. Build Behavioral Steering Vector (v_behav) per layer on Split A
#    Matched within-question contrast (mean(true_A) - mean(false_A) per qid in A)
#    plus Prompted Contrast (honest vs myth-repeating persona on 200 Split-A questions)
def build_q_matched_vec(X_full, li):
    vecs = []
    for q in qids[:len(qids) // 2]:
        m_q = (qids_all == q)
        pos = m_q & (y == 1)
        neg = m_q & (y == 0)
        if pos.any() and neg.any():
            vecs.append(X_full[pos, li].mean(0) - X_full[neg, li].mean(0))
    return F.normalize(torch.stack(vecs).mean(0), dim=0)

def build_persona_contrast(tok, model, df_A, n_q=200):
    f = RES / "tf_acts" / "persona_contrast_A.pt"
    if f.exists():
        return torch.load(f)
    sub_q = df_A.groupby("qid").first().iloc[:n_q]
    pos_prompts, neg_prompts = [], []
    for _, row in sub_q.iterrows():
        s = row.text
        p_msg = [{"role": "system", "content": "You are rigorously truthful and reject common myths."},
                 {"role": "user", "content": f"Is this statement true or false?\n{s}\nAnswer with one word."}]
        n_msg = [{"role": "system", "content": "You repeat plausible-sounding misconceptions and false myths."},
                 {"role": "user", "content": f"Is this statement true or false?\n{s}\nAnswer with one word."}]
        pos_prompts.append(tok.apply_chat_template(p_msg, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        neg_prompts.append(tok.apply_chat_template(n_msg, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    from common import last_token_states
    Hp = last_token_states(tok, model, pos_prompts)[:, LAYERS].float()
    Hn = last_token_states(tok, model, neg_prompts)[:, LAYERS].float()
    V = F.normalize((Hp - Hn).mean(0), dim=-1)   # [len(LAYERS), d]
    torch.save(V, f)
    return V

V_detect = [universal(got, li) for li in range(len(LAYERS))]
V_qmatch = [build_q_matched_vec(X, li) for li in range(len(LAYERS))]
V_persona = build_persona_contrast(tok, model, df.iloc[allA])
M_A = [X[allA, li].mean(0) for li in range(len(LAYERS))]

print("=== 1. Cosine Similarity to Detection Vector (V_detect) ===")
for li, L in enumerate(LAYERS):
    c_qm = F.cosine_similarity(V_detect[li], V_qmatch[li], dim=0).item()
    c_pr = F.cosine_similarity(V_detect[li], V_persona[li], dim=0).item()
    c_qp = F.cosine_similarity(V_qmatch[li], V_persona[li], dim=0).item()
    deg_qm = np.degrees(np.arccos(np.clip(c_qm, -1, 1)))
    deg_pr = np.degrees(np.arccos(np.clip(c_pr, -1, 1)))
    print(f"L{L}: cos(detect, TQA-matched) = {c_qm:+.3f} ({deg_qm:.1f} deg) | "
          f"cos(detect, persona-steer) = {c_pr:+.3f} ({deg_pr:.1f} deg) | "
          f"cos(TQA-matched, persona) = {c_qp:+.3f}")

# 3. Multi-layer & Single-layer Hook + Accuracy & Fluency Evaluation
def make_hook(v, m, tau, alpha, mode, log):
    def hook(mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == 1: return out
        h = h.clone(); vv = v.to(h.device)
        x = h[:, -1].float() - m.to(h.device)
        cos = F.cosine_similarity(x, vv.expand_as(x), dim=-1)
        fire = cos.abs() < tau
        sign = torch.sign(cos) if mode == "signed" else torch.ones_like(cos)
        step = fire.float() * sign * alpha * x.norm(dim=-1)
        h[:, -1] += (step[:, None] * vv).to(h.dtype)
        log.append(fire.cpu())
        return (h, *out[1:]) if isinstance(out, tuple) else h
    return hook

@torch.no_grad()
def eval_with_fluency(statements, labels, layer_hooks=None, bs=16):
    """
    Returns accuracy metrics AND fluency metrics:
      - tf_mass: P("True") + P("False") probability mass (checks if distribution stays valid)
      - ent: predictive entropy (nats) at the answer token
    """
    t, f = [tok.encode(w, add_special_tokens=False)[0] for w in ("True", "False")]
    handles = []
    if layer_hooks:
        for L, hk in layer_hooks:
            handles.append(model.model.layers[L].register_forward_hook(hk))
    margins, masses, ents = [], [], []
    try:
        for i in range(0, len(statements), bs):
            enc = tok([tf_prompt(tok, s) for s in statements[i:i + bs]],
                      return_tensors="pt", padding=True).to(model.device)
            lg = model(**enc).logits[:, -1].float()
            probs = F.softmax(lg, dim=-1)
            log_probs = F.log_softmax(lg, dim=-1)
            margins.append((lg[:, t] - lg[:, f]).cpu())
            masses.append((probs[:, t] + probs[:, f]).cpu())
            ents.append((-(probs * log_probs).sum(dim=-1)).cpu())
    finally:
        for h in handles: h.remove()

    s = torch.cat(margins).numpy()
    p = s > 0
    ta, fa = p[labels == 1].mean(), (~p[labels == 0]).mean()
    return dict(
        bal=round(float((ta + fa) / 2), 3),
        true_acc=round(float(ta), 3),
        false_acc=round(float(fa), 3),
        auc=round(float(roc_auc_score(labels, s)), 3),
        tf_mass=round(float(torch.cat(masses).mean()), 3),
        entropy=round(float(torch.cat(ents).mean()), 3),
    )

@torch.no_grad()
def answer_nll_fluency(idx_sub, layer_hooks=None, bs=16):
    """
    Measures free-text token fluency (mean NLL / perplexity) on true QA texts in Split B
    while steering hooks are active on every token position.
    """
    true_texts = stm[idx_sub][y[idx_sub] == 1][:200].tolist()
    handles = []
    if layer_hooks:
        for L, hk in layer_hooks:
            handles.append(model.model.layers[L].register_forward_hook(hk))
    nlls = []
    try:
        for i in range(0, len(true_texts), bs):
            enc = tok(true_texts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
            out = model(**enc, labels=enc.input_ids)
            nlls.append(float(out.loss.cpu()))
    finally:
        for h in handles: h.remove()
    mean_nll = float(np.mean(nlls))
    return round(float(np.exp(mean_nll)), 2)

# 4. Run Comparison on Held-Out Split B:
#    No steering vs Single-Layer (L23) vs Multi-Layer (L16, L20, L23)
print("\n=== 2. Split B: Accuracy & Fluency (No Steer vs Single-Layer vs Multi-Layer) ===")
li_23 = LAYERS.index(23)
PCT, ALPHA = 60, 0.25   # or plug in your winning (p, a) from eval_steer.py

def tau_for(v, li):
    return float(np.percentile(cos_of(X[iA, li], v, M_A[li]).abs().numpy(), PCT))

base_res = eval_with_fluency(stm[iB].tolist(), y[iB], layer_hooks=None)
base_res["ppl"] = answer_nll_fluency(iB, layer_hooks=None)
print(f"{'No steering (baseline)':28s}", base_res)

configs = [
    ("1L (L23) V_detect signed",  [(23, V_detect[li_23], M_A[li_23], ALPHA, "signed")]),
    ("3L (16,20,23) V_detect",    [(LAYERS[i], V_detect[i], M_A[i], ALPHA / 3, "signed") for i in range(3)]),
    ("1L (L23) V_qmatch signed",  [(23, V_qmatch[li_23], M_A[li_23], ALPHA, "signed")]),
    ("1L (L23) V_qmatch +V",      [(23, V_qmatch[li_23], M_A[li_23], ALPHA, "plus")]),
    ("3L (16,20,23) V_qmatch",    [(LAYERS[i], V_qmatch[i], M_A[i], ALPHA / 3, "signed") for i in range(3)]),
    ("1L (L23) V_persona +V",     [(23, V_persona[li_23], M_A[li_23], ALPHA, "plus")]),
    ("3L (16,20,23) V_persona +V",[(LAYERS[i], V_persona[i], M_A[i], ALPHA / 3, "plus") for i in range(3)]),
]

for name, spec in configs:
    log = []
    hooks = [(L, make_hook(v, m, tau_for(v, LAYERS.index(L)), a, mode, log)) for (L, v, m, a, mode) in spec]
    r = eval_with_fluency(stm[iB].tolist(), y[iB], layer_hooks=hooks)
    r["fire"] = round(torch.cat(log).float().mean().item(), 2)
    # All-token fluency hook for perplexity check
    def make_alltok_hook(v, a):
        def hk(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            vv = v.to(h.device, dtype=h.dtype)
            h += a * h.float().norm(dim=-1, keepdim=True).to(h.dtype) * vv
            return (h, *out[1:]) if isinstance(out, tuple) else h
        return hk
    ppl_hooks = [(L, make_alltok_hook(v, a)) for (L, v, m, a, mode) in spec]
    r["ppl"] = answer_nll_fluency(iB, layer_hooks=ppl_hooks)
    print(f"{name:28s}", r)