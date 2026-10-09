"""
Dumps reference tensors for Qwen3.8-27B's hybrid Gated DeltaNet / full-
attention architecture (HF model_type "qwen3_5"), for the kernel-level checks
in web/dev/verify_qwen_spike.js and verify_qwen_dispatch.js.

Run from project root:
    python tools/dump_qwen_oracle.py

Outputs to oracle/qwen3.8-27b-spike/.

Unlike tools/dump_llama_oracle.py, this does not load the ~55 GB checkpoint.
It checks kernel math, not trained weights. It builds a 4-layer model from the
`Qwen3_5*` classes in transformers, randomly initialized, with the model's
repeating layer pattern (3x linear_attention, 1x full_attention), so layers 0
and 3 cover both kinds. Every number comes from HF's forward-pass code: hooked
module outputs, or direct calls to HF's own functions where the math is in a
plain function and not an nn.Module.forward (the causal conv1d and the gated
delta rule). Nothing is reimplemented here.

Everything runs in fp32 on CPU, with no network access.
"""

import json
import os

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_mod
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
    l2norm,
)

OUT = "oracle/qwen3.8-27b-spike"
T = 12
SEED = 0

os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)
torch.manual_seed(SEED)

# --- Tiny config: the Qwen3_5 architecture with small shapes ---
# layer_types is the real model's repeating unit (3x linear, 1x full), so
# layer 0 (linear_attention) and layer 3 (full_attention) match the kinds of
# the 64-layer model's layers 0 and 3.
LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
HEAD_DIM = 64
PARTIAL_ROTARY_FACTOR = 0.25  # from the checkpoint's config.json
ROT_DIM = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)

cfg = Qwen3_5TextConfig(
    vocab_size=256,
    hidden_size=384,
    intermediate_size=512,
    num_hidden_layers=len(LAYER_TYPES),
    num_attention_heads=6,
    num_key_value_heads=1,
    head_dim=HEAD_DIM,
    layer_types=LAYER_TYPES,
    linear_conv_kernel_dim=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_num_key_heads=4,
    linear_num_value_heads=12,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    max_position_embeddings=4096,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000000.0,  # from the checkpoint's config.json
        "partial_rotary_factor": PARTIAL_ROTARY_FACTOR,
        "mrope_section": [3, 3, 2],  # sums to ROT_DIM/2=8; the real model's is [11,11,10]
        "mrope_interleaved": True,
    },
)
print(f"layer_types: {cfg.layer_types}")

model = Qwen3_5ForCausalLM(cfg).eval()

with open(f"{OUT}/spike_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "rot_dim": ROT_DIM,
        "layer_types": cfg.layer_types,
        "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
        "linear_key_head_dim": cfg.linear_key_head_dim,
        "linear_value_head_dim": cfg.linear_value_head_dim,
        "linear_num_key_heads": cfg.linear_num_key_heads,
        "linear_num_value_heads": cfg.linear_num_value_heads,
        "rope_theta": cfg.rope_parameters["rope_theta"],
        "partial_rotary_factor": cfg.rope_parameters["partial_rotary_factor"],
        "vocab_size": cfg.vocab_size,
        "T": T,
        "seed": SEED,
    }, f, indent=2)

# --- Fixed random input (the model is synthetic, so no tokenizer) ---
torch.manual_seed(SEED)
input_ids = torch.randint(0, cfg.vocab_size, (1, T))
with open(f"{OUT}/tokens.json", "w", encoding="utf-8") as f:
    json.dump({"input_ids": input_ids[0].tolist(), "T": T}, f, indent=2)


def squeeze(x):
    return x[0] if x.ndim >= 2 and x.shape[0] == 1 else x


def save(name, x):
    # .contiguous(): several captures (conv1d pre/post, RoPE outputs) are
    # transposed views. A transposed 2-D array is F-contiguous, and numpy.save
    # would write fortran_order=True, which the loader in web/dev/npy.js
    # rejects. C order is forced here for every capture.
    arr = squeeze(x).detach().contiguous().cpu().float().numpy()
    np.save(f"{OUT}/{name}.npy", arr)


# --- Hooks: global (embeddings, per-layer output, final norm, rope cos/sin) ---
captures = {}


def cap_out(name):
    def hook(_mod, _inp, out):
        x = out[0] if isinstance(out, tuple) else out
        captures[name] = x.detach()
    return hook


def cap_pre(name):
    def hook(_mod, inp):
        captures[name] = inp[0].detach()
    return hook


model.model.embed_tokens.register_forward_hook(cap_out("embeddings"))
model.model.norm.register_forward_hook(cap_out("final_norm"))

block_outs = [None] * cfg.num_hidden_layers


def make_block_hook(idx):
    def hook(_mod, _inp, out):
        x = out[0] if isinstance(out, tuple) else out
        block_outs[idx] = x.detach()
    return hook


for i, layer in enumerate(model.model.layers):
    layer.register_forward_hook(make_block_hook(i))

rope_out = {}
model.model.rotary_emb.register_forward_hook(
    lambda _m, _i, out: rope_out.update(cos=out[0].detach(), sin=out[1].detach()))

# Checks that M-RoPE reduces to partial RoPE: captures the pre-interleave freqs
# for all 3 axes, which must be identical for text-only input.
mrope_capture = {}
_orig_interleave = Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope


def _patched_interleave(self, freqs, mrope_section):
    mrope_capture["freqs"] = freqs.detach().clone()
    return _orig_interleave(self, freqs, mrope_section)


Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope = _patched_interleave

# --- Layer 0 (linear_attention): hooks + monkeypatches ---
l0 = model.model.layers[0]
assert cfg.layer_types[0] == "linear_attention"
l0.input_layernorm.register_forward_hook(cap_out("layer00_ln1"))
l0.linear_attn.in_proj_qkv.register_forward_hook(cap_out("layer00_qkv_raw"))
l0.linear_attn.in_proj_z.register_forward_hook(cap_out("layer00_z_raw"))
l0.linear_attn.in_proj_b.register_forward_hook(cap_out("layer00_b_raw"))
l0.linear_attn.in_proj_a.register_forward_hook(cap_out("layer00_a_raw"))
l0.linear_attn.norm.register_forward_hook(cap_out("layer00_gated_norm_out"))
l0.linear_attn.register_forward_hook(cap_out("layer00_linear_attn_out"))
l0.post_attention_layernorm.register_forward_pre_hook(cap_pre("layer00_after_attn_resid"))
l0.post_attention_layernorm.register_forward_hook(cap_out("layer00_ln2"))
l0.mlp.gate_proj.register_forward_hook(cap_out("layer00_gate"))
l0.mlp.up_proj.register_forward_hook(cap_out("layer00_up"))
l0.mlp.register_forward_hook(cap_out("layer00_mlp_out"))

# The linear_attention layer whose forward is running. The monkeypatches below
# are global functions shared by layers 0, 1 and 2, so each call is tagged and
# only layer 0's is captured.
current_linear_layer = {"idx": None}


def _mark_layer(idx):
    def hook(_mod, _inp):
        current_linear_layer["idx"] = idx
    return hook


for i, lt in enumerate(cfg.layer_types):
    if lt == "linear_attention":
        model.model.layers[i].linear_attn.register_forward_pre_hook(_mark_layer(i))

# causal_conv1d_fn and torch_chunk_gated_delta_rule are module-level functions.
# Qwen3_5GatedDeltaNet.forward calls them by global name
# (causal_conv1d_fn(mixed_qkv, self.conv1d.weight...), not self.conv1d(...)),
# so no module hook sees them. Reassigning the attribute on the imported module
# changes what that global lookup resolves to.
_orig_causal_conv1d_fn = qwen_mod.causal_conv1d_fn
_orig_chunk_rule = qwen_mod.torch_chunk_gated_delta_rule
conv_capture = {}
chunk_capture = {}


def _patched_causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    out = _orig_causal_conv1d_fn(hidden_states, weight, bias, activation, **kwargs)
    if current_linear_layer["idx"] == 0:
        conv_capture["pre"] = hidden_states.detach().clone()
        conv_capture["post"] = out.detach().clone()
    return out


def _patched_chunk_rule(query, key, value, g, beta, chunk_size=64, initial_state=None,
                         output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
    out, state = _orig_chunk_rule(query, key, value, g, beta, chunk_size, initial_state,
                                   output_final_state, use_qk_l2norm_in_kernel, **kwargs)
    if current_linear_layer["idx"] == 0:
        chunk_capture["query"] = query.detach().clone()
        chunk_capture["key"] = key.detach().clone()
        chunk_capture["value"] = value.detach().clone()
        chunk_capture["g"] = g.detach().clone()
        chunk_capture["beta"] = beta.detach().clone()
        chunk_capture["use_qk_l2norm_in_kernel"] = use_qk_l2norm_in_kernel
        chunk_capture["out"] = out.detach().clone()
    return out, state


qwen_mod.causal_conv1d_fn = _patched_causal_conv1d_fn
qwen_mod.torch_chunk_gated_delta_rule = _patched_chunk_rule

# --- Layer 3 (full_attention): hooks ---
l3 = model.model.layers[3]
assert cfg.layer_types[3] == "full_attention"
l3.input_layernorm.register_forward_hook(cap_out("layer03_ln1"))
l3.self_attn.q_proj.register_forward_hook(cap_out("layer03_q_proj_raw"))
l3.self_attn.k_proj.register_forward_hook(cap_out("layer03_k_proj_raw"))
l3.self_attn.v_proj.register_forward_hook(cap_out("layer03_v"))
l3.self_attn.q_norm.register_forward_pre_hook(cap_pre("layer03_q_norm_in"))
l3.self_attn.q_norm.register_forward_hook(cap_out("layer03_q_norm_out"))
l3.self_attn.k_norm.register_forward_hook(cap_out("layer03_k_norm_out"))
l3.self_attn.register_forward_hook(cap_out("layer03_attn_out"))
l3.post_attention_layernorm.register_forward_pre_hook(cap_pre("layer03_after_attn_resid"))
l3.post_attention_layernorm.register_forward_hook(cap_out("layer03_ln2"))
l3.mlp.gate_proj.register_forward_hook(cap_out("layer03_gate"))
l3.mlp.up_proj.register_forward_hook(cap_out("layer03_up"))
l3.mlp.register_forward_hook(cap_out("layer03_mlp_out"))

# --- Forward pass ---
print("Running forward...")
out = model(input_ids, use_cache=False)

Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope = _orig_interleave
qwen_mod.causal_conv1d_fn = _orig_causal_conv1d_fn
qwen_mod.torch_chunk_gated_delta_rule = _orig_chunk_rule

# --- Save global tensors ---
save("embeddings", captures["embeddings"])
save("final_norm", captures["final_norm"])
save("logits", out.logits)
for i in range(cfg.num_hidden_layers):
    save(f"block_{i:02d}_out", block_outs[i])
save("rope_cos", rope_out["cos"])
save("rope_sin", rope_out["sin"])

# M-RoPE check: for text-only input all 3 rotary axes have the same position
# ids, so the per-axis interleave must change nothing
# (freqs[0] == freqs[1] == freqs[2] before it runs).
freqs = mrope_capture["freqs"]
assert torch.allclose(freqs[0], freqs[1]) and torch.allclose(freqs[0], freqs[2]), \
    "M-RoPE collapse claim FAILED: rotary axes differ for text-only input"
print("OK: M-RoPE collapses to plain partial RoPE for text-only input (all 3 axes identical)")

# --- Save layer 0 (linear_attention) tensors ---
for name in ["layer00_ln1", "layer00_qkv_raw", "layer00_z_raw", "layer00_b_raw",
             "layer00_a_raw", "layer00_gated_norm_out", "layer00_linear_attn_out",
             "layer00_after_attn_resid", "layer00_ln2", "layer00_gate", "layer00_up",
             "layer00_mlp_out"]:
    save(name, captures[name])

save("layer00_conv_pre", conv_capture["pre"].transpose(1, 2))
save("layer00_conv_post", conv_capture["post"].transpose(1, 2))
save("layer00_query_raw", chunk_capture["query"])
save("layer00_key_raw", chunk_capture["key"])
save("layer00_value", chunk_capture["value"])
save("layer00_g", chunk_capture["g"])
save("layer00_beta", chunk_capture["beta"])
save("layer00_A_log", l0.linear_attn.A_log)
save("layer00_dt_bias", l0.linear_attn.dt_bias)

query_l2 = l2norm(chunk_capture["query"], dim=-1)
key_l2 = l2norm(chunk_capture["key"], dim=-1)
save("layer00_query_l2norm", query_l2)
save("layer00_key_l2norm", key_l2)

# The recurrence one token at a time, calling HF's fused_recurrent function
# with a carried state. The gated_delta_net_step kernel must match this.
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule  # noqa: E402

query, key, value = chunk_capture["query"], chunk_capture["key"], chunk_capture["value"]
g, beta = chunk_capture["g"], chunk_capture["beta"]
n_v_heads, k_head_dim, v_head_dim = value.shape[2], key.shape[-1], value.shape[-1]

state = None
states = [torch.zeros(1, n_v_heads, k_head_dim, v_head_dim)]
outs = []
for t in range(T):
    core_out_t, state = torch_recurrent_gated_delta_rule(
        query=query[:, t:t + 1], key=key[:, t:t + 1], value=value[:, t:t + 1],
        g=g[:, t:t + 1], beta=beta[:, t:t + 1],
        initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
    outs.append(core_out_t)
    states.append(state.clone())

seq_out = torch.cat(outs, dim=1)  # [1, T, n_v_heads, v_head_dim]
save("layer00_recurrent_core_out_seq", seq_out)
save("layer00_recurrent_states", torch.cat(states, dim=0))  # [T+1, n_v_heads, k_dim, v_dim]

# Consistency check: the model's chunk-path output (the real forward over the
# whole sequence) must equal this token-by-token recurrence, since both are the
# same math from HF's code. A mismatch means the capture is wrong.
chunk_out = chunk_capture["out"]
d = (seq_out - chunk_out).abs().max().item()
assert d < 1e-4, f"recurrent-vs-chunk self-consistency check FAILED: max abs diff {d}"
print(f"OK: sequential recurrence matches chunked real forward (max abs diff {d:.2e})")

# --- Save layer 3 (full_attention) tensors ---
for name in ["layer03_ln1", "layer03_q_proj_raw", "layer03_k_proj_raw", "layer03_v",
             "layer03_q_norm_out", "layer03_k_norm_out", "layer03_attn_out",
             "layer03_after_attn_resid", "layer03_ln2", "layer03_gate", "layer03_up",
             "layer03_mlp_out"]:
    save(name, captures[name])

# Checks the chunk order: the query is the first head_dim half of q_proj's
# double-width output and the gate is the second (torch.chunk(..., 2, dim=-1)
# after .view(*shape, -1, head_dim*2)). The query is derived from the raw
# q_proj output and compared with the input q_norm received.
q_raw = captures["layer03_q_proj_raw"]  # [1, T, n_heads*head_dim*2]
n_heads = cfg.num_attention_heads
q_view = q_raw.view(1, T, n_heads, HEAD_DIM * 2)
query_only, gate_only = q_view.chunk(2, dim=-1)
d = (query_only - captures["layer03_q_norm_in"]).abs().max().item()
assert d < 1e-6, f"q_proj chunk-order assumption FAILED: max abs diff {d}"
print(f"OK: q_proj query/gate chunk order confirmed (max abs diff {d:.2e})")
save("layer03_gate_raw", gate_only)

# RoPE reference: HF's apply_rotary_pos_emb applied to the hooked
# q_norm/k_norm outputs.
cos, sin = rope_out["cos"], rope_out["sin"]
q_for_rope = captures["layer03_q_norm_out"].transpose(1, 2)  # [1, heads, T, head_dim]
k_for_rope = captures["layer03_k_norm_out"].transpose(1, 2)
q_rope, k_rope = apply_rotary_pos_emb(q_for_rope, k_for_rope, cos, sin)
save("layer03_q_rope", q_rope.transpose(1, 2))
save("layer03_k_rope", k_rope.transpose(1, 2))

# --- Next token + short greedy decode ---
# The weights are random, so the text is meaningless. The ids are used as an
# exact-match check.
last_logits = out.logits[0, -1]
next_id = int(last_logits.argmax())
with open(f"{OUT}/next_token.json", "w", encoding="utf-8") as f:
    json.dump({"next_id": next_id}, f, indent=2)

print("Greedy decoding...")
gen = model.generate(input_ids, max_new_tokens=5, do_sample=False,
                      pad_token_id=cfg.eos_token_id or 0)
with open(f"{OUT}/decode_greedy.json", "w", encoding="utf-8") as f:
    json.dump({"prompt_ids": input_ids[0].tolist(), "generated_ids": gen[0, T:].tolist()},
               f, indent=2)

# --- Weight export ---
# The activation dumps above are compared against. The verify pages also need
# the weights that produced them: plain f32 .npy in HF layout, which the verify
# pages split as tools/convert_hf.py does.


def save_param(name, p):
    np.save(f"{OUT}/{name}.npy", p.detach().contiguous().cpu().float().numpy())


save_param("weight_embed_tokens", model.model.embed_tokens.weight)
save_param("weight_final_norm", model.model.norm.weight)
save_param("weight_lm_head", model.lm_head.weight)

save_param("weight_layer00_input_layernorm", l0.input_layernorm.weight)
save_param("weight_layer00_post_attention_layernorm", l0.post_attention_layernorm.weight)
save_param("weight_layer00_in_proj_qkv", l0.linear_attn.in_proj_qkv.weight)
save_param("weight_layer00_in_proj_z", l0.linear_attn.in_proj_z.weight)
save_param("weight_layer00_in_proj_b", l0.linear_attn.in_proj_b.weight)
save_param("weight_layer00_in_proj_a", l0.linear_attn.in_proj_a.weight)
save_param("weight_layer00_conv1d", l0.linear_attn.conv1d.weight.squeeze(1))  # [conv_dim, K]
save_param("weight_layer00_A_log", l0.linear_attn.A_log)
save_param("weight_layer00_dt_bias", l0.linear_attn.dt_bias)
save_param("weight_layer00_gated_norm", l0.linear_attn.norm.weight)
save_param("weight_layer00_out_proj", l0.linear_attn.out_proj.weight)
save_param("weight_layer00_gate_proj", l0.mlp.gate_proj.weight)
save_param("weight_layer00_up_proj", l0.mlp.up_proj.weight)
save_param("weight_layer00_down_proj", l0.mlp.down_proj.weight)

save_param("weight_layer03_input_layernorm", l3.input_layernorm.weight)
save_param("weight_layer03_post_attention_layernorm", l3.post_attention_layernorm.weight)
save_param("weight_layer03_q_proj", l3.self_attn.q_proj.weight)
save_param("weight_layer03_k_proj", l3.self_attn.k_proj.weight)
save_param("weight_layer03_v_proj", l3.self_attn.v_proj.weight)
save_param("weight_layer03_q_norm", l3.self_attn.q_norm.weight)
save_param("weight_layer03_k_norm", l3.self_attn.k_norm.weight)
save_param("weight_layer03_o_proj", l3.self_attn.o_proj.weight)
save_param("weight_layer03_gate_proj", l3.mlp.gate_proj.weight)
save_param("weight_layer03_up_proj", l3.mlp.up_proj.weight)
save_param("weight_layer03_down_proj", l3.mlp.down_proj.weight)

print(f"\nOracle written to {OUT}/")
