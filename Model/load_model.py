from common import load
import torch

tok, model = load()
cfg = model.config
print("layers:", cfg.num_hidden_layers, "| hidden size:", cfg.hidden_size)
print("GPU memory (GB):", round(torch.cuda.memory_allocated() / 1e9, 1))

# does it generate? (thinking off so we get a direct answer)
msgs = [{"role": "user", "content": "What is the capital of France?"}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
enc = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**enc, max_new_tokens=30, do_sample=False)
print(tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True))

# do hidden states have the shape we expect?
hs = model(**enc, output_hidden_states=True).hidden_states
print("hidden_states entries:", len(hs), "(should be layers + 1) | shape:", tuple(hs[-1].shape))