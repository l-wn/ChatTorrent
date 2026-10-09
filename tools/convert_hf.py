"""
Converts a HuggingFace decoder-only transformer into flat binary files and a
manifest. The WebGPU client (web/model/loader.js) fetches them and uploads them
to GPU buffers.

    python tools/convert_hf.py                                  # SmolLM2-135M
    python tools/convert_hf.py --model HuggingFaceTB/SmolLM2-360M
    python tools/convert_hf.py --model unsloth/Llama-3.2-1B --out weights/llama32
    python tools/convert_hf.py --model Qwen/Qwen3.8-27B --dtype q8

Every dimension is read from config.json and written to the manifest, which
web/model/loader.js uses to size its buffers. The kernels implement a fixed set
of architecture features, and this script rejects anything outside that set
with a reason, so it never exports weights the browser would run incorrectly.

Supported (check_supported below verifies each):

    * Llama-shaped blocks: RMSNorm → GQA/MHA attention → RMSNorm → SwiGLU MLP,
      pre-norm residual, as in Llama and SmolLM2.
    * Grouped-query attention (n_kv_head <= n_head) and plain MHA.
    * Tied or untied LM heads.
    * Any RoPE theta, any even head_dim, any rms_norm_eps.
    * Llama 3.1+ "llama3" RoPE scaling (the NTK-aware frequency rescale). The
      four rope_scaling numbers are written to the manifest and applied on the
      CPU when the frequency table is built.

Rejected, because no kernel implements it:

    * Attention biases (Qwen2 has q/k/v biases)     matmul_nobias only
    * QK-norm (Qwen3), outside the hybrid path      no kernel (see below)
    * MoE FFNs (Mixtral, Qwen3-MoE)                 no routing/expert kernels
    * Any RoPE scaling other than "llama3"          the rope kernel has no other form
    * Interleaved RoPE                              the kernel is Llama split-half
    * Sliding-window attention (Mistral)            causal_softmax is dense
    * Gemma-family                                  (1+w) norm + embedding scale
    * Any dtype but f32/f16/q8                      the storage dtypes the kernels read

Also supported: Qwen3.8-27B's hybrid architecture (model_type "qwen3_5(_text)",
`layer_types` a mix of "linear_attention"/"full_attention"; see
docs/architecture.md). Most layers use Gated DeltaNet, a linear/recurrent
attention with no KV cache. The rest use standard GQA with partial RoPE,
QK-norm, and an attention-output gate stored in a double-width q_proj. QK-norm
is therefore accepted on this path and rejected elsewhere. Each layer is
checked against the tensor table for its kind (FULL_ATTN_BLOCK_TENSORS /
LINEAR_ATTN_BLOCK_TENSORS), not against BLOCK_TENSORS.
tools/make_qwen_fixture.py builds a tiny checkpoint of this shape, to test
this path without the ~55GB model.

Layout: every weight is an nn.Linear weight stored [out, in], and nothing is
transposed. The matmul kernel computes Y[M,N] = X[M,K] @ W[N,K]^T and reads
[out, in] directly, `wte` included.

Tensor names in the manifest are the keys of web/model/model.js's weights
object (q_proj, input_layernorm, ...), so the loader needs no rename table.
"""

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"

# config.json's rope_scaling block for rope_type "llama3". These four numbers
# are what llama3Scale() in web/model/kernels.js needs, and they are the whole
# rope_scaling entry in the manifest.
LLAMA3_ROPE_KEYS = ("factor", "low_freq_factor", "high_freq_factor",
                    "original_max_position_embeddings")

# name in model.js's weights object  →  suffix under model.layers.{i}.
BLOCK_TENSORS = [
    ("input_layernorm",          "input_layernorm.weight"),
    ("q_proj",                   "self_attn.q_proj.weight"),
    ("k_proj",                   "self_attn.k_proj.weight"),
    ("v_proj",                   "self_attn.v_proj.weight"),
    ("o_proj",                   "self_attn.o_proj.weight"),
    ("post_attention_layernorm", "post_attention_layernorm.weight"),
    ("gate_proj",                "mlp.gate_proj.weight"),
    ("up_proj",                  "mlp.up_proj.weight"),
    ("down_proj",                "mlp.down_proj.weight"),
]

# --- Qwen3.8-27B hybrid-attention support (model_type "qwen3_5") ---
# Most layers (`layer_types`) use Gated DeltaNet, a linear/recurrent attention.
# The rest use standard GQA with partial RoPE and an attention-output gate
# stored in a double-width q_proj. The names below match the Qwen3_5* HF
# modeling code (tools/dump_qwen_oracle.py hooks the same module attributes).

FULL_ATTN_BLOCK_TENSORS = BLOCK_TENSORS + [
    ("q_norm", "self_attn.q_norm.weight"),
    ("k_norm", "self_attn.k_norm.weight"),
]

LINEAR_ATTN_BLOCK_TENSORS = [
    ("input_layernorm",          "input_layernorm.weight"),
    ("post_attention_layernorm", "post_attention_layernorm.weight"),
    ("gate_proj",                "mlp.gate_proj.weight"),
    ("up_proj",                  "mlp.up_proj.weight"),
    ("down_proj",                "mlp.down_proj.weight"),
    ("in_proj_qkv",               "linear_attn.in_proj_qkv.weight"),
    ("in_proj_z",                "linear_attn.in_proj_z.weight"),
    ("in_proj_b",                "linear_attn.in_proj_b.weight"),
    ("in_proj_a",                "linear_attn.in_proj_a.weight"),
    ("conv1d",                   "linear_attn.conv1d.weight"),
    ("A_log",                    "linear_attn.A_log"),
    ("dt_bias",                  "linear_attn.dt_bias"),
    ("gated_norm",               "linear_attn.norm.weight"),
    ("out_proj",                 "linear_attn.out_proj.weight"),
]

# conv1d is small (width 4) and precision-sensitive, so a q8 build keeps it
# f32. It is 2-D ([conv_dim, 4]), so the ndim<2 rule does not cover it.
FORCE_F32_NAMES = {"conv1d_q", "conv1d_k", "conv1d_v"}

# Top-level state-dict prefixes that are dropped, not exported: the vision
# tower and the multi-token-prediction head, which text-only chat does not use.
# AutoModelForCausalLM instantiates only the text backbone, so these never
# appear in model.state_dict(). They appear in from_pretrained's
# unexpected_keys report, which main() checks against this list.
KNOWN_DROP_PREFIXES = ("visual.", "vision_tower.", "mtp.")


def linear_attn_dims(cfg):
    """Gated DeltaNet's head and width config, separate from full-attention's
    num_attention_heads/head_dim."""
    nK = cfg.linear_num_key_heads
    nV = cfg.linear_num_value_heads
    Dk = cfg.linear_key_head_dim
    Dv = cfg.linear_value_head_dim
    key_dim = nK * Dk
    value_dim = nV * Dv
    return dict(nK=nK, nV=nV, Dk=Dk, Dv=Dv, key_dim=key_dim, value_dim=value_dim,
                conv_dim=2 * key_dim + value_dim, K4=cfg.linear_conv_kernel_dim)


def split_rows(arr, ranges):
    """[rows, ...] -> {name: arr[r0:r1]}. An nn.Linear's output rows are the
    concatenation of the rows of separate smaller Linears, so slicing rows is
    exact (as in splitRows in web/dev/verify_qwen_spike.js)."""
    return {name: np.ascontiguousarray(arr[r0:r1]) for name, (r0, r1) in ranges.items()}


def split_q_gate(arr, n_heads, head_dim):
    """q_proj's rows interleave query and gate per head ([q0..q63,g0..g63,
    q64..], not [q0..q383,g0..g383]). torch.chunk(view(*,heads,2*D),2,dim=-1)
    splits the last axis, which after the reshape is two halves within each
    head's block. tools/dump_qwen_oracle.py checks this with torch.chunk (max
    abs diff < 1e-6 against the real q_norm input)."""
    k = arr.shape[1]
    view = arr.reshape(n_heads, 2, head_dim, k)
    q = np.ascontiguousarray(view[:, 0].reshape(n_heads * head_dim, k))
    gate = np.ascontiguousarray(view[:, 1].reshape(n_heads * head_dim, k))
    return q, gate


def full_attn_layer_tensors(sd, i, cfg):
    """One full_attention layer's (name, array) pairs. The double-width q_proj
    is split into the query and the attention-output gate. The RMSNorm weights
    get +1 added (see linear_attn_layer_tensors)."""
    p = f"model.layers.{i}"
    n_head = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_head
    q, gate = split_q_gate(f32(sd[f"{p}.self_attn.q_proj.weight"]), n_head, head_dim)
    return [
        ("input_layernorm", f32(sd[f"{p}.input_layernorm.weight"]) + 1.0),
        ("q_proj", q),
        ("attn_gate", gate),
        ("k_proj", f32(sd[f"{p}.self_attn.k_proj.weight"])),
        ("v_proj", f32(sd[f"{p}.self_attn.v_proj.weight"])),
        ("q_norm", f32(sd[f"{p}.self_attn.q_norm.weight"]) + 1.0),
        ("k_norm", f32(sd[f"{p}.self_attn.k_norm.weight"]) + 1.0),
        ("o_proj", f32(sd[f"{p}.self_attn.o_proj.weight"])),
        ("post_attention_layernorm", f32(sd[f"{p}.post_attention_layernorm.weight"]) + 1.0),
        ("gate_proj", f32(sd[f"{p}.mlp.gate_proj.weight"])),
        ("up_proj", f32(sd[f"{p}.mlp.up_proj.weight"])),
        ("down_proj", f32(sd[f"{p}.mlp.down_proj.weight"])),
    ]


def linear_attn_layer_tensors(sd, i, dims):
    """One linear_attention (Gated DeltaNet) layer's (name, array) pairs, split
    and packed for the Gated DeltaNet kernels in web/model/kernels.js:
    in_proj_qkv becomes three projections, and conv1d is split into the same
    q/k/v row ranges (in a depthwise conv, output channel c depends only on
    input channel c, so splitting rows before or after the conv is
    equivalent). A_log and dt_bias are packed into one [2*nV] buffer, so
    gatedDeltaNetStep fits WebGPU's limit of 8 storage buffers per stage."""
    p = f"model.layers.{i}"
    key_dim, value_dim, conv_dim = dims["key_dim"], dims["value_dim"], dims["conv_dim"]
    ranges = {"q": (0, key_dim), "k": (key_dim, 2 * key_dim), "v": (2 * key_dim, conv_dim)}

    qkv = split_rows(f32(sd[f"{p}.linear_attn.in_proj_qkv.weight"]), ranges)
    conv = split_rows(f32(sd[f"{p}.linear_attn.conv1d.weight"]).squeeze(1), ranges)
    gate_params = np.concatenate([f32(sd[f"{p}.linear_attn.A_log"]),
                                   f32(sd[f"{p}.linear_attn.dt_bias"])])

    return [
        ("input_layernorm", f32(sd[f"{p}.input_layernorm.weight"]) + 1.0),
        ("post_attention_layernorm", f32(sd[f"{p}.post_attention_layernorm.weight"]) + 1.0),
        ("in_proj_q", qkv["q"]), ("in_proj_k", qkv["k"]), ("in_proj_v", qkv["v"]),
        ("in_proj_z", f32(sd[f"{p}.linear_attn.in_proj_z.weight"])),
        ("in_proj_b", f32(sd[f"{p}.linear_attn.in_proj_b.weight"])),
        ("in_proj_a", f32(sd[f"{p}.linear_attn.in_proj_a.weight"])),
        ("conv1d_q", conv["q"]), ("conv1d_k", conv["k"]), ("conv1d_v", conv["v"]),
        ("gate_params", gate_params),
        ("gated_norm", f32(sd[f"{p}.linear_attn.norm.weight"])),  # RMSNormGated: no +1
        ("out_proj", f32(sd[f"{p}.linear_attn.out_proj.weight"])),
        ("gate_proj", f32(sd[f"{p}.mlp.gate_proj.weight"])),
        ("up_proj", f32(sd[f"{p}.mlp.up_proj.weight"])),
        ("down_proj", f32(sd[f"{p}.mlp.down_proj.weight"])),
    ]


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert a HF decoder-only model for the WebGPU client.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"HF model id or local path (default: {DEFAULT_MODEL})")
    p.add_argument("--out", default=None,
                   help="output dir (default: weights/<model-name-lowercased>)")
    p.add_argument("--force", action="store_true",
                   help="overwrite a non-empty output dir")
    p.add_argument("--skip-tokenizer", action="store_true",
                   help="don't export tokenizer.json")
    p.add_argument("--dtype", default="f32", choices=("f32", "f16", "q8"),
                   help="weight storage. f16 halves the download and the VRAM a "
                        "block costs; the kernels still accumulate in f32, so "
                        "only the stored weights lose precision.")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the round-trip check (faster, less safe)")
    return p.parse_args()


def default_out(model_id):
    """weights/<slug>, e.g. 'HuggingFaceTB/SmolLM2-135M' → 'weights/smollm2-135m'."""
    slug = re.sub(r"[^a-z0-9.-]+", "-", model_id.split("/")[-1].lower()).strip("-")
    return os.path.join("weights", slug)


def rope_settings(cfg):
    """transformers 5 nests RoPE config; older versions had flat attributes."""
    rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    if not isinstance(rope, dict):
        rope = {}
    theta = rope.get("rope_theta", getattr(cfg, "rope_theta", 10000.0))
    rope_type = rope.get("rope_type") or rope.get("type") or "default"
    return float(theta), rope_type, rope


def check_supported(cfg, sd, n_layer, head_dim, layer_types=None):
    """Returns every reason this checkpoint cannot run, so one run reports all.

    The main check is a per-layer key diff. It does not allow-list model types.
    It lists what each layer contains and compares that with the tensors the
    forward pass uses. Any extra tensor (a bias, a q_norm, a router, an expert)
    means the architecture does something the kernels do not, and it is
    reported by name.

    `layer_types` (Qwen3.8-27B's mix of Gated DeltaNet and full attention):
    when present and not uniform, each layer is checked against the tensor
    table for its kind (FULL_ATTN_BLOCK_TENSORS / LINEAR_ATTN_BLOCK_TENSORS),
    not against BLOCK_TENSORS. A full-attention layer with q_norm/k_norm is
    therefore accepted here, where a kernel exists for it. A non-hybrid model
    with the same extra tensors is rejected by the BLOCK_TENSORS check below.
    """
    errs = []
    model_type = getattr(cfg, "model_type", "?")
    hybrid = bool(layer_types) and len(set(layer_types)) > 1

    # --- architecture family ---
    if model_type.startswith("gemma"):
        errs.append(f"model_type {model_type}: Gemma normalises with (1 + weight) and "
                    f"scales embeddings by sqrt(hidden); the rmsnorm kernel does neither")

    for attr in ("num_experts", "num_local_experts", "n_routed_experts"):
        if getattr(cfg, attr, None):
            errs.append(f"{attr}={getattr(cfg, attr)}: mixture-of-experts FFN, "
                        f"but the MLP is a dense gate/up/down SwiGLU")
            break

    # --- attention shape ---
    n_head = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_head
    if n_head % n_kv:
        errs.append(f"num_attention_heads {n_head} not divisible by "
                    f"num_key_value_heads {n_kv}")
    if head_dim % 2:
        errs.append(f"head_dim {head_dim} is odd; RoPE pairs channel i with i+head_dim/2")
    # Llama-shaped blocks assume n_head*head_dim equals hidden_size, and the
    # kernels and scratch buffers for them are sized that way. The hybrid
    # full-attention layer's o_proj (FULL_ATTN_BLOCK_TENSORS) maps
    # n_head*head_dim to hidden explicitly (Qwen3.8-27B: 24*256=6144 -> 5120),
    # so the check applies only to the non-hybrid path below.
    if not hybrid and n_head * head_dim != cfg.hidden_size:
        errs.append(f"n_head*head_dim ({n_head * head_dim}) != hidden_size "
                    f"({cfg.hidden_size}); the kernels assume attention output is "
                    f"exactly the residual width")

    sw = getattr(cfg, "sliding_window", None)
    uses_sw = sw and getattr(cfg, "use_sliding_window", True)
    if uses_sw and sw < getattr(cfg, "max_position_embeddings", 1 << 30):
        errs.append(f"sliding_window={sw}: causal_softmax masks only the future, "
                    f"not a window, so long contexts would attend too far back")

    # --- RoPE ---
    theta, rope_type, rope = rope_settings(cfg)
    if rope_type == "llama3":
        missing = [k for k in LLAMA3_ROPE_KEYS if k not in rope]
        if missing:
            errs.append(f"rope_type 'llama3' but rope_scaling is missing {missing}")
    elif rope_type != "default":
        detail = {k: v for k, v in rope.items() if k != "rope_type"}
        errs.append(f"rope_type {rope_type!r} {detail}: the rope kernel implements "
                    f"only the plain form and Llama 3's NTK-aware ('llama3') scaling")
    if getattr(cfg, "rope_interleaved", False):
        errs.append("rope_interleaved: the kernel pairs (i, i+D/2), not (2i, 2i+1)")

    # --- activation / norm ---
    act = getattr(cfg, "hidden_act", None) or getattr(cfg, "hidden_activation", None)
    if act not in (None, "silu", "swish"):
        errs.append(f"hidden_act {act!r}: the MLP kernel is silu(gate)*up")

    # --- the per-layer tensor diff ---
    if hybrid:
        nK = getattr(cfg, "linear_num_key_heads", None)
        nV = getattr(cfg, "linear_num_value_heads", None)
        if not nK or not nV:
            errs.append("layer_types includes linear_attention but "
                        "linear_num_key_heads/linear_num_value_heads are missing")
        elif nV % nK:
            errs.append(f"linear_num_value_heads {nV} not divisible by "
                        f"linear_num_key_heads {nK}: l2NormalizeHeads' "
                        f"repeat-interleave folding needs an integer group factor")

        for i, kind in enumerate(layer_types):
            prefix = f"model.layers.{i}."
            present = {k[len(prefix):] for k in sd if k.startswith(prefix)}
            table = FULL_ATTN_BLOCK_TENSORS if kind == "full_attention" else LINEAR_ATTN_BLOCK_TENSORS
            expected = {suffix for _, suffix in table}
            missing = expected - present
            extra = present - expected
            if missing:
                errs.append(f"layer {i} ({kind}) is missing {sorted(missing)}")
            if extra:
                errs.append(f"layer {i} ({kind}) carries tensors the forward pass "
                            f"ignores: {sorted(extra)}. Exporting would silently drop them")
    else:
        prefix = "model.layers.0."
        present = {k[len(prefix):] for k in sd if k.startswith(prefix)}
        expected = {suffix for _, suffix in BLOCK_TENSORS}
        missing = expected - present
        extra = present - expected
        if missing:
            errs.append(f"layer 0 is missing {sorted(missing)}: not a Llama-shaped block")
        if extra:
            errs.append(f"layer 0 carries tensors the forward pass ignores: {sorted(extra)}. "
                        f"Exporting would silently drop them")

    # --- what the loader needs ---
    for key in ("model.embed_tokens.weight", "model.norm.weight"):
        if key not in sd:
            errs.append(f"state dict has no {key}")

    if not getattr(cfg, "tie_word_embeddings", False) and "lm_head.weight" not in sd:
        errs.append("untied embeddings but no lm_head.weight in the state dict")

    return errs


# Every tensor is read from the checkpoint as f32 and cast once, when packed.
# NUMPY_DTYPE is the storage dtype. The kernels convert to f32 on read and
# accumulate in f32, so the dtype affects storage only.
NUMPY_DTYPE = {"f32": np.float32, "f16": np.float16}

# q8: symmetric int8 weights with one f32 scale per group of 64 along the K
# (input) axis. Scales are per group, not per tensor or per row, because
# Llama-family activations have a few very large channels (verify.js notes one
# reaching ~19000 by block 11). A scale wide enough for an outlier would
# quantize everything else to a couple of levels. With groups of 64, an outlier
# affects only its own group.
#
# The cost is one f32 per 64 weights, 1.0625 bytes per weight in total: a
# quarter of f32 and half of f16. q8 also needs no `shader-f16` feature, so it
# runs on GPUs that cannot read f16.
Q8_GROUP = 64


def quantize_q8(arr, group=Q8_GROUP):
    """f32 [N, K] -> (int8 [N, K], f32 scales [N, ceil(K/group)]).

    Symmetric and clipped to [-127, 127], not [-128, 127]. The extra negative
    level would make the range asymmetric about zero, which adds a small bias
    to every dequantized group.
    """
    if arr.ndim != 2:
        raise ValueError(f"q8 expects a 2-D weight, got shape {arr.shape}")
    n, k = arr.shape
    if k % 4 != 0:
        raise ValueError(f"q8 needs K divisible by 4 (4 weights per u32), got K={k}")

    ngroup = (k + group - 1) // group
    pad = ngroup * group - k
    a = np.pad(arr, ((0, 0), (0, pad))) if pad else arr
    a = a.reshape(n, ngroup, group)

    scale = np.abs(a).max(axis=2) / 127.0
    # An all-zero group would divide by zero. Any non-zero scale reproduces it.
    scale[scale == 0] = 1.0
    q = np.rint(a / scale[:, :, None]).clip(-127, 127).astype(np.int8)

    return np.ascontiguousarray(q.reshape(n, ngroup * group)[:, :k]), \
           scale.astype(np.float32)


def q8_error(arr, q, scale, group=Q8_GROUP):
    """Max relative error of the round trip, for the converter's report."""
    n, k = arr.shape
    ngroup = scale.shape[1]
    pad = ngroup * group - k
    qq = np.pad(q, ((0, 0), (0, pad))) if pad else q
    back = (qq.astype(np.float32).reshape(n, ngroup, group)
            * scale[:, :, None]).reshape(n, ngroup * group)[:, :k]
    denom = np.abs(arr).max()
    return float(np.abs(back - arr).max() / denom) if denom else 0.0


def chat_format(model_id):
    """Returns the prompt format the client should wrap a turn in.

    An instruction-tuned checkpoint has a chat template and an end-of-turn
    token. A base checkpoint has neither and can only be prompted with a plain
    User:/Assistant: transcript that it continues. The browser cannot tell the
    two apart from the weights, and with the wrong format the model echoes the
    question. So the format is written to the manifest here, where the
    tokenizer is available. web/prompt.js reads it.

    ChatML, Qwen3's variant of it with a thinking mode, and Llama 3's
    header-token format are recognized, because web/prompt.js implements those
    wrappers. Anything else is reported and falls back to the transcript, as
    for a base checkpoint.
    """
    try:
        template = AutoTokenizer.from_pretrained(model_id).chat_template
    except Exception as e:                                   # noqa: BLE001
        print(f"  WARNING: could not read the tokenizer ({e}); "
              f"assuming a base checkpoint")
        return None
    if not template:
        return None
    if "<|im_start|>" in template:
        # Qwen3's ChatML-shaped template also has a thinking mode
        # (enable_thinking), which plain ChatML (e.g. SmolLM2) lacks.
        # web/prompt.js needs to tell them apart to handle the reasoning
        # instruction and <think>.
        return "qwen" if "enable_thinking" in template else "chatml"
    if "<|start_header_id|>" in template and "<|eot_id|>" in template:
        return "llama3"
    print("  WARNING: this checkpoint's chat template is not ChatML or Llama 3's "
          "header format. web/prompt.js has no wrapper for it, so the client "
          "will fall back to the User:/Assistant: transcript and replies will "
          "be poor.")
    return None


def f32(t):
    return t.detach().cpu().to(torch.float32).numpy().astype(np.float32, copy=False)


def cast(arr, dtype):
    """f32 array -> the storage dtype, checking that nothing overflowed to inf.

    f16's maximum is 65504. Real weights are far below it, so an inf means the
    checkpoint is wrong, not that a value was rounded. An inf weight would
    produce NaN in the first matmul that reads it, and downstream peers could
    not tell that from a valid activation.
    """
    out = arr.astype(NUMPY_DTYPE[dtype], copy=False)
    if dtype != "f32":
        bad = np.isinf(out) & ~np.isinf(arr)
        if bad.any():
            raise ValueError(f"{bad.sum()} weights overflowed {dtype} "
                             f"(max |w| = {float(np.abs(arr).max())})")
    return out


def pack_file(out_dir, filename, tensors, dtype="f32", force_f32=frozenset()):
    """Write tensors consecutively into one .bin and return manifest entries.

    Each entry has its own `dtype`, because a q8 file is mixed: the 2-D
    projections are quantized and the 1-D RMSNorm weights stay f32. The norm
    weights are a tiny part of the file (576 floats next to a 1.3 MB
    projection) and they multiply every channel of the residual stream, so
    quantizing them would cost accuracy for no gain. `force_f32` keeps
    further tensors f32 by name: ones that are 2-D but small and
    precision-sensitive (Qwen3.8-27B's causal-conv1d weight).

    A q8 tensor is written as its int8 block followed by its f32 scales. The
    int8 block is N*K bytes with K divisible by 4, so the scales are 4-byte
    aligned.
    """
    entries = []
    offset = 0
    worst = 0.0
    with open(f"{out_dir}/{filename}", "wb") as f:
        for name, arr in tensors:
            # 1-D tensors are norm weights and stay f32 in every build.
            per_tensor = "f32" if (dtype == "q8" and (arr.ndim < 2 or name in force_f32)) else dtype

            if per_tensor == "q8":
                q, scale = quantize_q8(arr)
                worst = max(worst, q8_error(arr, q, scale))
                f.write(np.ascontiguousarray(q).tobytes())
                entry = {
                    "name": name,
                    "shape": list(arr.shape),
                    "offset": offset,
                    "size": q.nbytes,
                    "dtype": "q8",
                    "group": Q8_GROUP,
                    "scales_offset": offset + q.nbytes,
                    "scales_size": scale.nbytes,
                }
                f.write(np.ascontiguousarray(scale).tobytes())
                offset += q.nbytes + scale.nbytes
            else:
                out = cast(arr, per_tensor)
                out = np.ascontiguousarray(out)
                f.write(out.tobytes())
                entry = {
                    "name": name,
                    "shape": list(out.shape),
                    "offset": offset,
                    "size": out.nbytes,
                    "dtype": per_tensor,
                }
                offset += out.nbytes
            entries.append(entry)

    extra = f", worst q8 error {worst * 100:.2f}% of max|w|" if dtype == "q8" else ""
    print(f"  {filename}: {len(tensors)} tensors, {offset / 1e6:.2f} MB{extra}")
    return entries


def main():
    args = parse_args()
    out_dir = args.out or default_out(args.model)
    torch.set_grad_enabled(False)

    # A Windows console defaults to cp1252, which cannot encode characters
    # outside Latin-1, and HF model ids can contain them. Without this a
    # print() can raise UnicodeEncodeError and end the run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if os.path.isdir(out_dir) and os.listdir(out_dir) and not args.force:
        sys.exit(f"{out_dir} exists and is not empty. Pass --force to overwrite.")
    os.makedirs(out_dir, exist_ok=True)

    # Load the config first. AutoConfig fetches a few KB, so a rejected model
    # does not cost a multi-GB weights download.
    print(f"Reading config for {args.model}...")
    cfg = AutoConfig.from_pretrained(args.model)
    # Qwen3.8-27B's checkpoint config is a composite VLM wrapper (model_type
    # "qwen3_5"). Its top-level fields are architectures, text_config,
    # vision_config and so on, and hidden_size, layer_types etc. are on
    # cfg.text_config, a Qwen3_5TextConfig (as in the Qwen/Qwen3.8-27B repo's
    # config.json). get_text_config() is the HF API for this: it returns
    # cfg.text_config for a composite config and cfg itself for any other
    # model.
    cfg = cfg.get_text_config()
    if getattr(cfg, "is_encoder_decoder", False):
        sys.exit("encoder-decoder models are not supported; this is a decoder-only stack")

    n_layer = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    n_head = cfg.num_attention_heads
    n_kv_head = getattr(cfg, "num_key_value_heads", None) or n_head
    head_dim = getattr(cfg, "head_dim", None) or hidden // n_head
    ffn = cfg.intermediate_size
    vocab = cfg.vocab_size
    theta, rope_type, rope = rope_settings(cfg)
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    # Qwen3.8-27B's mix of Gated DeltaNet and full attention (model_type
    # "qwen3_5"). Absent or uniform on every other supported model, which uses
    # the non-hybrid path below.
    layer_types = list(getattr(cfg, "layer_types", None) or [])
    hybrid = len(set(layer_types)) > 1
    n_full = layer_types.count("full_attention") if hybrid else 0
    n_lin = n_layer - n_full if hybrid else 0

    # f32 everywhere: the browser download is 4 bytes per parameter.
    if hybrid:
        dims = linear_attn_dims(cfg)
        full_params = (2 * hidden + 2 * n_head * head_dim * hidden  # q_proj (doubled) + o_proj-ish
                       + 2 * n_kv_head * head_dim * hidden + hidden * hidden + 3 * ffn * hidden)
        lin_params = (2 * hidden + dims["conv_dim"] * hidden + dims["value_dim"] * hidden
                      + 2 * dims["nV"] * hidden + hidden * dims["value_dim"] + 3 * ffn * hidden)
        approx = vocab * hidden * (1 if tied else 2) + hidden + n_full * full_params + n_lin * lin_params
    else:
        approx = vocab * hidden * (1 if tied else 2) + hidden + n_layer * (
            2 * hidden + n_head * head_dim * hidden + 2 * n_kv_head * head_dim * hidden
            + hidden * hidden + 3 * ffn * hidden)
    rope_desc = f"rope_theta {theta:g}" + (f" ({rope_type} scaling)" if rope_type != "default" else "")
    hybrid_desc = f", {n_full} full-attn / {n_lin} linear-attn (Gated DeltaNet) layers" if hybrid else ""
    print(f"  {getattr(cfg, 'model_type', '?')}: {n_layer} layers, hidden {hidden}, "
          f"{n_head} Q / {n_kv_head} KV heads, head_dim {head_dim}, ffn {ffn}, "
          f"vocab {vocab}, {rope_desc}, {'tied' if tied else 'untied'} head{hybrid_desc}")
    # q8 is one byte per weight plus one f32 scale per group: 1.0625 bytes at
    # group 64. The reported size uses that figure, not 4 bytes.
    width = {"f16": 2, "q8": 1 + 4 / Q8_GROUP}.get(args.dtype, 4)
    print(f"  ~{approx / 1e6:.0f} M params -> {approx * width / 1e6:.0f} MB as {args.dtype}")

    print("Loading weights...")
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", output_loading_info=True)
    model = model.eval()
    sd = model.state_dict()

    # A multimodal checkpoint's vision-tower and MTP tensors are not in sd,
    # since AutoModelForCausalLM instantiates only the text backbone. They are
    # listed in from_pretrained's unexpected-keys report. Each is checked
    # against KNOWN_DROP_PREFIXES, so a naming mismatch that dropped real
    # text-model tensors is a hard error.
    if hybrid:
        unexpected = loading_info.get("unexpected_keys") or []
        missing = loading_info.get("missing_keys") or []
        known_dropped = [k for k in unexpected if k.startswith(KNOWN_DROP_PREFIXES)]
        unknown = [k for k in unexpected if k not in known_dropped]
        if known_dropped:
            extra = f" (+{len(known_dropped) - 1} more)" if len(known_dropped) > 1 else ""
            print(f"  stripping {len(known_dropped)} vision/MTP checkpoint tensors "
                  f"(text-only chat): {known_dropped[0]}{extra}")

    errs = check_supported(cfg, sd, n_layer, head_dim, layer_types)
    if hybrid:
        if unknown:
            errs.append(f"checkpoint carries tensors this converter doesn't recognize and "
                        f"can't classify as known-dropped vision/MTP weights: {unknown[:10]}"
                        + (" ..." if len(unknown) > 10 else ""))
        if missing:
            errs.append(f"checkpoint is missing tensors the model expects (random-init "
                        f"parts would be exported): {missing[:10]}"
                        + (" ..." if len(missing) > 10 else ""))
    if errs:
        print(f"\n{args.model} cannot run on these kernels:\n")
        for e in errs:
            print(f"  x {e}")
        print("\nNothing was written. See the module docstring for what is supported.")
        sys.exit(1)

    print("Reading the tokenizer's chat template...")
    fmt = chat_format(args.model)
    print(f"  chat_format: {fmt or 'transcript (no chat template: base checkpoint)'}")

    manifest = {
        "model": args.model,
        "arch": getattr(cfg, "model_type", "llama"),
        "hidden": hidden,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_kv_head": n_kv_head,
        "head_dim": head_dim,
        "group": n_head // n_kv_head,
        "ffn": ffn,
        "vocab": vocab,
        "n_pos": getattr(cfg, "max_position_embeddings", 2048),
        "rms_eps": getattr(cfg, "rms_norm_eps", 1e-5),
        "rope_theta": theta,
        # Always the llama3 form (check_supported rejects any other): the four
        # numbers llama3Scale() in web/model/kernels.js applies to the plain
        # frequency table. Absent on a plain-RoPE checkpoint.
        "rope_scaling": (
            {k: rope[k] for k in LLAMA3_ROPE_KEYS} if rope_type == "llama3" else None
        ),
        "norm": "rmsnorm",
        "act": "silu",
        "dtype": args.dtype,
        "layout": "row_major",
        "weight_layout": "out_in",   # [out, in], no transpose
        "tied_lm_head": tied,
        # The wrapper web/prompt.js uses for a turn. None means the
        # User:/Assistant: transcript, for a base checkpoint.
        "chat_format": fmt,
        "bos_token_id": getattr(cfg, "bos_token_id", None),
        "eos_token_id": getattr(cfg, "eos_token_id", None),
        "files": {},
    }
    # Some configs list several EOS ids. The client compares against one.
    if isinstance(manifest["eos_token_id"], (list, tuple)):
        manifest["eos_token_id"] = manifest["eos_token_id"][0]
    if fmt in ("chatml", "qwen", "llama3"):
        # cfg.eos_token_id can list a different token first from the one the
        # chat template ends a turn with (Llama 3: <|end_of_text|> vs.
        # <|eot_id|>; Qwen3.8-27B: <|endoftext|> vs. <|im_end|>). The
        # tokenizer's `eos_token` is the one the template uses. web/route.js's
        # `eos` is a single id compared with ===, so with the wrong id
        # generation would not stop at a turn boundary.
        manifest["eos_token_id"] = AutoTokenizer.from_pretrained(args.model).eos_token_id

    if hybrid:
        # rot_dim: the client's partial rotary takes a dimension, not a
        # fraction. The fraction is in rope_parameters/rope_scaling
        # (rope_settings()'s `rope` dict), not a top-level config attribute.
        rot_factor = rope.get("partial_rotary_factor", 1.0)
        manifest["layer_types"] = layer_types
        manifest["linear_key_head_dim"] = dims["Dk"]
        manifest["linear_value_head_dim"] = dims["Dv"]
        manifest["linear_num_key_heads"] = dims["nK"]
        manifest["linear_num_value_heads"] = dims["nV"]
        manifest["linear_conv_kernel_dim"] = dims["K4"]
        manifest["rot_dim"] = int(head_dim * rot_factor)

    # --- Client weights: embeddings + final norm (+ lm_head if untied) ---
    # Qwen3.5's RMSNorm computes norm(x)*(1+weight) (modeling_qwen3_5.py), and
    # kernels.js's rmsnorm computes norm(x)*weight. The +1 is added to the
    # weight here, so the kernel needs no per-architecture branch.
    # full_attn_layer_tensors and linear_attn_layer_tensors do the same for the
    # per-layer norms.
    # The verify step below reloads and compares only a few blocks: the first,
    # the last, and one of each layer kind. They are chosen here, from n_layer,
    # hybrid and layer_types, so the packing loop keeps only those in memory.
    # Keeping every block's f32 arrays (4 bytes/param, about 4x a q8 block's
    # packed size) for all 64 layers would add tens of GB on top of the loaded
    # checkpoint.
    verify_block_idxs = set()
    if not args.no_verify:
        verify_block_idxs = {0, n_layer - 1}
        if hybrid:
            for kind in set(layer_types):
                verify_block_idxs.add(layer_types.index(kind))

    print("Packing client.bin...")
    packed = {}  # filename -> {tensor name: array}, kept only for blocks the verify step below checks
    norm_arr = f32(sd["model.norm.weight"])
    if hybrid:
        norm_arr = norm_arr + 1.0
    client = [
        ("wte",  f32(sd["model.embed_tokens.weight"])),
        ("norm", norm_arr),
    ]
    if not args.no_verify:
        packed["client.bin"] = dict(client)
    manifest["files"]["client.bin"] = pack_file(out_dir, "client.bin", client, args.dtype)

    # An untied lm_head gets its own file and is not appended to client.bin.
    # On a large-vocab checkpoint wte and lm_head together can exceed the
    # ~2 GiB several browsers allow for one fetched body (Qwen3.8-27B: 2.5 GiB
    # at q8). See loadClient in web/model/loader.js.
    if not tied:
        print("Packing lm_head.bin...")
        lm_head = [("lm_head", f32(sd["lm_head.weight"]))]
        if not args.no_verify:
            packed["lm_head.bin"] = dict(lm_head)
        manifest["files"]["lm_head.bin"] = pack_file(out_dir, "lm_head.bin", lm_head, args.dtype)

    # --- Per-block weights ---
    for i in range(n_layer):
        filename = f"block_{i:02d}.bin"
        if hybrid:
            kind = layer_types[i]
            tensors = (full_attn_layer_tensors(sd, i, cfg) if kind == "full_attention"
                       else linear_attn_layer_tensors(sd, i, dims))
            print(f"Packing {filename} ({kind})...")
            manifest["files"][filename] = pack_file(out_dir, filename, tensors, args.dtype,
                                                      force_f32=FORCE_F32_NAMES)
        else:
            p = f"model.layers.{i}"
            tensors = [(name, f32(sd[f"{p}.{key}"])) for name, key in BLOCK_TENSORS]
            print(f"Packing {filename}...")
            manifest["files"][filename] = pack_file(out_dir, filename, tensors, args.dtype)
        if i in verify_block_idxs:
            packed[filename] = dict(tensors)

    with open(f"{out_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {out_dir}/manifest.json")

    # --- Tokenizer ---
    # web/model/tokenizer.js reads tokenizer.json from next to the weights. It
    # implements byte-level BPE only, so a SentencePiece checkpoint exports
    # here and fails in the browser. That is a warning, not an error, since
    # the weights are still usable.
    if not args.skip_tokenizer:
        print("\nSaving tokenizer...")
        tok_dir = f"{out_dir}/tokenizer"
        if os.path.isdir(tok_dir):
            shutil.rmtree(tok_dir)
        tok = AutoTokenizer.from_pretrained(args.model)
        tok.save_pretrained(tok_dir)
        if not os.path.exists(f"{tok_dir}/tokenizer.json"):
            print("  WARNING: no tokenizer.json (slow/SentencePiece tokenizer). "
                  "web/model/tokenizer.js needs one.")
        else:
            with open(f"{tok_dir}/tokenizer.json", encoding="utf-8") as f:
                tj = json.load(f)
            pre = (tj.get("pre_tokenizer") or {}).get("type")
            if pre != "ByteLevel" and pre != "Sequence":
                print(f"  WARNING: pre_tokenizer is {pre!r}, not ByteLevel. "
                      f"web/model/tokenizer.js assumes byte-level BPE and will refuse it.")
            print(f"  {tok_dir}/tokenizer.json")

    # --- Verify: reload tensors from disk and compare ---
    if not args.no_verify:
        print("\nVerifying round-trip...")

        def reload(filename, name):
            """Read one tensor back from disk, dequantized if it was quantized.

            Uses the entry's dtype, not the file's. A q8 build keeps its norm
            weights in f32, and reading them by the file's dtype would decode
            576 floats as 2304 int8s.
            """
            entry = next(e for e in manifest["files"][filename] if e["name"] == name)
            with open(f"{out_dir}/{filename}", "rb") as f:
                f.seek(entry["offset"])
                buf = f.read(entry["size"])
                if entry.get("dtype") == "q8":
                    f.seek(entry["scales_offset"])
                    sbuf = f.read(entry["scales_size"])

            if entry.get("dtype") != "q8":
                dt = NUMPY_DTYPE[entry.get("dtype", args.dtype)]
                return np.frombuffer(buf, dtype=dt).reshape(entry["shape"])

            n, k = entry["shape"]
            group = entry["group"]
            ngroup = (k + group - 1) // group
            q = np.frombuffer(buf, dtype=np.int8).reshape(n, k)
            scale = np.frombuffer(sbuf, dtype=np.float32).reshape(n, ngroup)
            pad = ngroup * group - k
            qq = np.pad(q, ((0, 0), (0, pad))) if pad else q
            return (qq.astype(np.float32).reshape(n, ngroup, group)
                    * scale[:, :, None]).reshape(n, ngroup * group)[:, :k]

        checks = [("client.bin", "wte"), ("client.bin", "norm")]
        if not tied:
            checks.append(("lm_head.bin", "lm_head"))

        seen = set()

        def add_block_checks(i):
            if hybrid:
                names = (["input_layernorm", "q_proj", "attn_gate", "k_proj", "down_proj"]
                          if layer_types[i] == "full_attention" else
                          ["input_layernorm", "in_proj_q", "in_proj_k", "conv1d_v",
                           "gate_params", "down_proj"])
            else:
                names = ["input_layernorm", "q_proj", "k_proj", "down_proj"]
            for name in names:
                key = (f"block_{i:02d}.bin", name)
                if key not in seen:
                    seen.add(key)
                    checks.append(key)

        # The first and last block, plus one of each layer kind, chosen before
        # packing as verify_block_idxs.
        for i in verify_block_idxs:
            add_block_checks(i)

        # Compared bit for bit against `cast(orig)`, not the f32 original. The
        # check is on the memory -> disk -> memory round trip, and on an f16
        # build the rounding is expected. Its size is reported separately.
        #
        # q8 has no bit-exact comparison, since quantization changes the
        # values. The check is that the dequantized tensor is within the error
        # the quantizer guarantees. A transposed scale index or an off-by-one
        # group would put whole groups on the wrong scale and fail this, where
        # a loose absolute threshold would not catch it.
        worst = 0.0
        for fname, tname in checks:
            loaded = reload(fname, tname)
            orig = packed[fname][tname]
            entry_dtype = next(e for e in manifest["files"][fname]
                               if e["name"] == tname).get("dtype", args.dtype)

            if entry_dtype == "q8":
                span = float(np.abs(orig).max()) or 1.0
                rel = float(np.abs(loaded - orig).max()) / span
                # 127 levels over a group's range, so half a step is 1/254. A
                # small margin is allowed for rounding.
                if rel > 1.5 / 254:
                    sys.exit(f"FAIL {fname}:{tname}  q8 round trip off by "
                             f"{rel * 100:.2f}% of max|w|. Scales misaligned?")
                worst = max(worst, rel)
                print(f"  ok  {fname}:{tname}  shape {list(loaded.shape)}"
                      f"  max Δ {rel * 100:.3f}% of max|w|")
                continue

            if not np.array_equal(loaded, cast(orig, entry_dtype)):
                sys.exit(f"FAIL {fname}:{tname}  round trip is not bit-exact")
            err = float(np.abs(loaded.astype(np.float32) - orig).max())
            worst = max(worst, err)
            print(f"  ok  {fname}:{tname}  shape {list(loaded.shape)}"
                  + (f"  max |Δ| {err:.2e}" if entry_dtype != "f32" else ""))
        if args.dtype == "q8":
            print(f"  worst dequantisation error: {worst * 100:.3f}% of max|w|")
        elif args.dtype != "f32":
            print(f"  worst rounding error across the sampled tensors: {worst:.2e}")

    total = sum(os.path.getsize(f"{out_dir}/{f}") for f in manifest["files"])
    print(f"\nAll checks passed.  {total / 1e6:.1f} MB across "
          f"{len(manifest['files'])} files in {out_dir}/")
    print(f"\nTo serve it, set checkpoint.json to "
          f"{{\"url\": \"/{out_dir.replace(os.sep, '/')}\"}} and restart the signal server.")


if __name__ == "__main__":
    main()
