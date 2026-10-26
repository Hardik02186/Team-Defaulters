import numpy as np, pandas as pd, torch, torch.nn.functional as F
from pathlib import Path
from common import load, last_token_states

HERE = Path(__file__).resolve().parent
DATA, RES = HERE.parent / "Data", HERE / "results"
LAYERS = [16, 20, 23]                                   # index i = output of model.model.layers[i]
GOT = ["cities", "neg_cities", "sp_en_trans", "larger_than"]

def tf_prompt(tok, s):
    msg = [{"role": "user", "content": f"Is this statement true or false?\n{s}\nAnswer with one word."}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def cached_states(tok, model, name, statements):        # [N, len(LAYERS), d] float32; delete tf_acts/ if prompt or LAYERS change
    f = RES / "tf_acts" / f"{name}.pt"
    if f.exists(): return torch.load(f)
    f.parent.mkdir(parents=True, exist_ok=True)
    X = last_token_states(tok, model, [tf_prompt(tok, s) for s in statements])[:, LAYERS].float()
    torch.save(X, f); return X

def universal(sets, li):                                # mean of per-dataset unit (mean_true - mean_false)
    vs = []
    for X, y in sets:
        pos = torch.from_numpy(np.asarray(y) == 1); H = X[:, li]
        vs.append(F.normalize(H[pos].mean(0) - H[~pos].mean(0), dim=0))
    return F.normalize(torch.stack(vs).mean(0), dim=0)

def cos_of(H, v, m):                                    # cosine of centered states with v
    return F.cosine_similarity(H - m, v.expand_as(H), dim=-1)

def steer_hook(v, m, tau, alpha, mode, log):
    def hook(mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == 1: return out                  # decode step: not used here
        h = h.clone(); vv = v.to(h.device)
        x = h[:, -1].float() - m.to(h.device)           # centered last-token state
        cos = F.cosine_similarity(x, vv.expand_as(x), dim=-1)
        fire = cos.abs() < tau                          # uncertain -> intervene
        sign = torch.sign(cos) if mode == "signed" else torch.ones_like(cos)
        step = fire.float() * sign * alpha * x.norm(dim=-1)
        h[:, -1] += (step[:, None] * vv).to(h.dtype)
        log.append(fire.cpu())
        return (h, *out[1:]) if isinstance(out, tuple) else h
    return hook

@torch.no_grad()
def tf_scores(tok, model, statements, layer=None, hook=None, bs=16):   # logit(True) - logit(False)
    t, f = [tok.encode(w, add_special_tokens=False)[0] for w in ("True", "False")]
    handle = model.model.layers[layer].register_forward_hook(hook) if hook else None
    out = []
    try:
        for i in range(0, len(statements), bs):
            enc = tok([tf_prompt(tok, s) for s in statements[i:i + bs]],
                      return_tensors="pt", padding=True).to(model.device)
            lg = model(**enc).logits[:, -1].float()
            out.append((lg[:, t] - lg[:, f]).cpu())
    finally:
        if handle: handle.remove()
    return torch.cat(out).numpy()