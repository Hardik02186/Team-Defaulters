import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/data3/Hardik/LSS/models/qwen3-4b/"   # <- change to your actual folder name

def load(model_path=MODEL_PATH):
    # left padding so index -1 is always the real last token
    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    return tok, model

@torch.no_grad()
def last_token_states(tok, model, texts, batch_size=16):
    """Returns [N, n_layers, hidden] fp16 on CPU: last-token residual state after every layer."""
    out = []
    for i in tqdm(range(0, len(texts), batch_size)):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states   # (embeddings, layer1, ..., layerN)
        out.append(torch.stack([h[:, -1, :] for h in hs[1:]], dim=1).cpu())
    return torch.cat(out)