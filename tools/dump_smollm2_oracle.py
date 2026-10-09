"""
Dumps reference tensors from HuggingFace SmolLM2-135M for a fixed prompt.

Run from project root:
    python tools/dump_smollm2_oracle.py

Outputs to oracle/smollm2/. web/dev/verify.js diffs each stage of the forward
pass against these files, and the tokenizer self-test round-trips
tokenizer_cases.json.

Everything runs in fp32 (the checkpoint is bf16), so the diffs measure the
WebGPU implementation and not a dtype difference.
"""

import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M"
OUT = "oracle/smollm2"
PROMPT = "The capital of France is"
DECODE_STEPS = 20

os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)

print(f"Loading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
cfg = model.config

n_layer = cfg.num_hidden_layers
n_head = cfg.num_attention_heads
n_kv = cfg.num_key_value_heads
D = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_head
rope_params = getattr(cfg, "rope_parameters", None) or {}
THETA = rope_params.get("rope_theta", getattr(cfg, "rope_theta", 10000.0))

# --- Tokenize ---
# add_special_tokens=False: SmolLM2 is a base model trained with no BOS, and
# the JS tokenizer emits plain BPE ids. The two must match.
input_ids = tok(PROMPT, return_tensors="pt", add_special_tokens=False).input_ids  # [1, T]
T = input_ids.shape[1]
print(f"Prompt: {PROMPT!r}  tokens: {input_ids[0].tolist()}  T={T}")

with open(f"{OUT}/tokens.json", "w", encoding="utf-8") as f:
    json.dump({
        "prompt": PROMPT,
        "input_ids": input_ids[0].tolist(),
        "decoded": tok.decode(input_ids[0]),
        "T": T,
    }, f, indent=2)

# --- Hooks: block-0 intermediates, per-block outputs, final norm ---
captures = {}


def cap_out(name):
    def hook(_mod, _inp, out):
        x = out[0] if isinstance(out, tuple) else out
        captures[name] = x.detach().cpu().float().numpy()
    return hook


def cap_pre(name):
    def hook(_mod, inp):
        captures[name] = inp[0].detach().cpu().float().numpy()
    return hook


layers = model.model.layers
l0 = layers[0]
l0.input_layernorm.register_forward_hook(cap_out("block00_ln1"))
l0.self_attn.q_proj.register_forward_hook(cap_out("block00_q"))
l0.self_attn.k_proj.register_forward_hook(cap_out("block00_k"))
l0.self_attn.v_proj.register_forward_hook(cap_out("block00_v"))
l0.self_attn.register_forward_hook(cap_out("block00_attn_out"))
# ln_2's input is the residual stream after the attention add.
l0.post_attention_layernorm.register_forward_pre_hook(cap_pre("block00_after_attn_resid"))
l0.post_attention_layernorm.register_forward_hook(cap_out("block00_ln2"))
l0.mlp.gate_proj.register_forward_hook(cap_out("block00_gate"))
l0.mlp.up_proj.register_forward_hook(cap_out("block00_up"))
l0.mlp.register_forward_hook(cap_out("block00_mlp_out"))
model.model.norm.register_forward_hook(cap_out("final_norm"))

# Per-block outputs come from hooks, not output_hidden_states: the last entry
# of that tuple is taken after the final norm, not the raw last block output.
block_outs = [None] * n_layer


def make_block_hook(idx):
    def hook(_mod, _inp, out):
        x = out[0] if isinstance(out, tuple) else out
        block_outs[idx] = x.detach().cpu().float().numpy()
    return hook


for i, block in enumerate(layers):
    block.register_forward_hook(make_block_hook(i))

# --- Forward pass ---
print("Running forward...")
out = model(input_ids, output_hidden_states=True, use_cache=False)

# hidden_states[0] is the embedding. For Llama that is wte[ids], with no wpe.
np.save(f"{OUT}/embeddings.npy", out.hidden_states[0][0].cpu().float().numpy())
for i in range(n_layer):
    np.save(f"{OUT}/block_{i:02d}_out.npy", block_outs[i][0])


def squeeze(name):
    arr = captures[name]
    return arr[0] if arr.ndim == 3 else arr


for name in ["block00_ln1", "block00_q", "block00_k", "block00_v",
             "block00_attn_out", "block00_after_attn_resid", "block00_ln2",
             "block00_gate", "block00_up", "block00_mlp_out"]:
    np.save(f"{OUT}/{name}.npy", squeeze(name))

np.save(f"{OUT}/final_norm.npy", captures["final_norm"][0])
np.save(f"{OUT}/logits.npy", out.logits[0].cpu().float().numpy())

# --- RoPE reference, computed here, not hooked ---
# HF applies rotary inside LlamaAttention.forward, between the projections and
# the score matmul, where no module can be hooked. This implements the
# documented formula (docs/model.md) in numpy, so the JS side can diff
# Qh/K_cache right after its rope() call. If this reference were wrong,
# block00_attn_out, which is hooked from HF, would not match.
#
# Saved head-major [n_heads, T, D], the layout of the JS scratch buffers.
def to_head_major(x, heads):          # [T, heads*D] → [heads, T, D]
    return x.reshape(x.shape[0], heads, D).transpose(1, 0, 2).copy()


def apply_rope(xh):                   # [heads, T, D], split-half pairing
    half = D // 2
    pos = np.arange(xh.shape[1], dtype=np.float64)[:, None]
    inv = THETA ** (-(2.0 * np.arange(half, dtype=np.float64)) / D)[None, :]
    ang = pos * inv                                   # [T, half]
    c, s = np.cos(ang)[None], np.sin(ang)[None]
    a, b = xh[:, :, :half].astype(np.float64), xh[:, :, half:].astype(np.float64)
    return np.concatenate([a * c - b * s, b * c + a * s], axis=-1).astype(np.float32)


q_head = to_head_major(squeeze("block00_q"), n_head)
k_head = to_head_major(squeeze("block00_k"), n_kv)
np.save(f"{OUT}/block00_q_rope.npy", apply_rope(q_head))
np.save(f"{OUT}/block00_k_rope.npy", apply_rope(k_head))

# --- Next token + greedy decode ---
last_logits = out.logits[0, -1].cpu().float().numpy()
next_id = int(last_logits.argmax())
with open(f"{OUT}/next_token.json", "w", encoding="utf-8") as f:
    json.dump({
        "next_id": next_id,
        "decoded": tok.decode([next_id]),
        "top5": [
            {"id": int(i), "logit": float(last_logits[i]), "decoded": tok.decode([int(i)])}
            for i in last_logits.argsort()[-5:][::-1]
        ],
    }, f, indent=2)

print("Greedy decoding...")
gen = model.generate(
    input_ids,
    attention_mask=torch.ones_like(input_ids),
    max_new_tokens=DECODE_STEPS,
    do_sample=False,
    pad_token_id=cfg.eos_token_id,
)
gen_ids = gen[0].tolist()
new_ids = gen_ids[T:]
with open(f"{OUT}/decode_greedy.json", "w", encoding="utf-8") as f:
    json.dump({
        "prompt": PROMPT,
        "prompt_ids": input_ids[0].tolist(),
        "generated_ids": new_ids,
        "full_ids": gen_ids,
        "full_text": tok.decode(gen_ids),
    }, f, indent=2)

# --- Tokenizer fixtures for the JS self-test ---
cases = [PROMPT, "Hello world", " leading space", "SmolLM2 is a 135M-parameter model.",
         "tabs\tand\nnewlines", "unicode: café — naïve ✨"]
with open(f"{OUT}/tokenizer_cases.json", "w", encoding="utf-8") as f:
    json.dump([{"text": c, "ids": tok(c, add_special_tokens=False).input_ids} for c in cases],
              f, indent=2, ensure_ascii=False)

print(f"\nOracle written to {OUT}/")
print(f"  next token after prompt: {tok.decode([next_id])!r}")
print(f"  greedy {DECODE_STEPS} steps: {tok.decode(new_ids)!r}")
