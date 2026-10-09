"""
Dumps reference tensors from the real Qwen/Qwen3.8-27B checkpoint over the full
64-layer forward pass, with a greedy decode. Used by dev.html's
?stream=1&oracle=/oracle/qwen3.8-27b.

Run from project root:
    python tools/dump_qwen_oracle_full.py [--decode-steps N] [--out DIR]

Outputs to oracle/qwen3.8-27b/ by default. Two differences from
tools/dump_llama_oracle.py:

  * It runs in the checkpoint's bf16 and does not upcast to fp32, which would
    need ~104 GB of RAM. The oracle therefore has bf16 rounding. That is
    negligible next to q8's ~0.4% error, which TOL_Q8 in web/dev/verify.js
    allows for.

  * It captures only embeddings, per-block outputs, the final norm and logits.
    Kernel-level sub-stages are covered by oracle/qwen3.8-27b-spike/
    (tools/dump_qwen_oracle.py).

Needs no GPU. It is slow: on CPU, the prefill and every decode step are each a
full forward pass of a 27B-parameter model. Keep --decode-steps small and run
it as a background job.
"""

import argparse
import json
import os

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_OUT = "oracle/qwen3.8-27b"
PROMPT = "The capital of France is"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decode-steps", type=int, default=5,
                   help="greedy-decode steps for the regression fixture (default: 5). "
                        "Each is a full CPU forward pass over a 27B-param model, so keep "
                        "this small; raise it only once the short run is green)")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"output dir (default: {DEFAULT_OUT})")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    print(f"Reading config for {MODEL}...")
    cfg = AutoConfig.from_pretrained(MODEL).get_text_config()
    n_layer = cfg.num_hidden_layers
    layer_types = list(getattr(cfg, "layer_types", None) or [])
    print(f"  {n_layer} layers"
          + (f", {layer_types.count('full_attention')} full-attn / "
             f"{layer_types.count('linear_attention')} linear-attn (Gated DeltaNet)"
             if layer_types else ""))

    print(f"Loading {MODEL} (native bf16, CPU; this is the slow part)...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    # dtype="auto" keeps the checkpoint's stored dtype (bf16) and does not
    # upcast (see the module docstring). tools/convert_hf.py loads the model
    # the same way.
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto").eval()

    # AutoModelForCausalLM instantiates only the text backbone for a composite
    # (VLM-wrapper) checkpoint, so model.model.layers below is the 64-layer
    # text decoder. The assert fails if the module structure differs.
    assert hasattr(model, "model") and hasattr(model.model, "layers"), \
        f"unexpected module structure on {type(model).__name__}. Inspect and update this script."
    assert len(model.model.layers) == n_layer, \
        f"model has {len(model.model.layers)} layers, config says {n_layer}"

    # --- Tokenize ---
    # add_special_tokens=False: web/model/tokenizer.js is a plain BPE encoder
    # that adds no BOS. The ids must match, or every later position is off by
    # one.
    input_ids = tok(PROMPT, return_tensors="pt", add_special_tokens=False).input_ids  # [1, T]
    T = input_ids.shape[1]
    print(f"Prompt: {PROMPT!r}  tokens: {input_ids[0].tolist()}  T={T}")

    with open(f"{args.out}/tokens.json", "w", encoding="utf-8") as f:
        json.dump({
            "prompt": PROMPT,
            "input_ids": input_ids[0].tolist(),
            "decoded": tok.decode(input_ids[0]),
            "T": T,
        }, f, indent=2)

    # --- Hooks: per-block output only, plus final norm ---
    # The same hook works for both layer kinds: linear_attention and
    # full_attention decoder layers return their hidden-state output the same
    # way.
    final_norm_out = {}
    model.model.norm.register_forward_hook(
        lambda _m, _i, out: final_norm_out.update(x=(out[0] if isinstance(out, tuple) else out)))

    block_outs = [None] * n_layer

    def make_block_hook(idx):
        def hook(_mod, _inp, out):
            x = out[0] if isinstance(out, tuple) else out
            block_outs[idx] = x.detach().cpu().float().numpy()
        return hook

    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(make_block_hook(i))

    # --- Forward pass ---
    print("Running forward (prefill)...")
    out = model(input_ids, output_hidden_states=True, use_cache=False)

    np.save(f"{args.out}/embeddings.npy", out.hidden_states[0][0].cpu().float().numpy())
    for i in range(n_layer):
        np.save(f"{args.out}/block_{i:02d}_out.npy", block_outs[i][0])
    np.save(f"{args.out}/final_norm.npy", final_norm_out["x"][0].cpu().float().numpy())
    np.save(f"{args.out}/logits.npy", out.logits[0].cpu().float().numpy())

    # --- Next token ---
    last_logits = out.logits[0, -1].cpu().float().numpy()
    next_id = int(last_logits.argmax())
    with open(f"{args.out}/next_token.json", "w", encoding="utf-8") as f:
        json.dump({
            "next_id": next_id,
            "decoded": tok.decode([next_id]),
            "top5": [
                {"id": int(i), "logit": float(last_logits[i]), "decoded": tok.decode([int(i)])}
                for i in last_logits.argsort()[-5:][::-1]
            ],
        }, f, indent=2)

    # --- Greedy decode ---
    print(f"Greedy decoding {args.decode_steps} step(s) "
          f"(each is a full forward pass; this is the slow part)...")
    eos = cfg.eos_token_id
    gen = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=args.decode_steps,
        do_sample=False,
        pad_token_id=eos if isinstance(eos, int) else eos[0],
    )
    gen_ids = gen[0].tolist()
    new_ids = gen_ids[T:]
    with open(f"{args.out}/decode_greedy.json", "w", encoding="utf-8") as f:
        json.dump({
            "prompt": PROMPT,
            "prompt_ids": input_ids[0].tolist(),
            "generated_ids": new_ids,
            "full_ids": gen_ids,
            "full_text": tok.decode(gen_ids),
        }, f, indent=2)

    # --- Tokenizer fixtures for selfTest() in web/model/tokenizer.js ---
    # dev/main.js runs a tokenizer self-test against
    # `${oracle}/tokenizer_cases.json` whenever ?oracle= is passed, which
    # ?stream=1 requires. Without this file the self-test gets a 404 and the
    # page stops before the streaming verification.
    cases = [PROMPT, "Hello world", " leading space", "Qwen3.8-27B is a hybrid-attention model.",
             "tabs\tand\nnewlines", "unicode: café — naïve ✨"]
    with open(f"{args.out}/tokenizer_cases.json", "w", encoding="utf-8") as f:
        json.dump([{"text": c, "ids": tok(c, add_special_tokens=False).input_ids} for c in cases],
                  f, indent=2, ensure_ascii=False)

    print(f"\nOracle written to {args.out}/")
    print(f"  next token after prompt: {tok.decode([next_id])!r}")
    print(f"  greedy {args.decode_steps} step(s): {tok.decode(new_ids)!r}")


if __name__ == "__main__":
    main()
