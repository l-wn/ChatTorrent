"""
Builds a tiny Qwen3_5 HuggingFace checkpoint on disk (Qwen3.8-27B's
architecture, model_type "qwen3_5"): config.json and safetensors, with no
tokenizer. It lets tools/convert_hf.py's hybrid-attention path run unmodified
on a small model. The model is built as in tools/dump_qwen_oracle.py. That code
is copied, not imported, because that script runs its whole dump at import
time.

Run from project root:
    python tools/make_qwen_fixture.py
    python tools/convert_hf.py --model oracle/qwen3.8-27b-spike/hf-fixture \
        --out /tmp/qwen-fixture-out --skip-tokenizer --dtype q8

--extra-tensors also writes two extra tensors into the saved safetensors file:
a dummy vision-tower tensor, which convert_hf.py must drop, and an unrecognized
tensor, which must make it fail. This tests convert_hf.py's unexpected-keys
check without a real multimodal checkpoint.
"""

import argparse
import os

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

OUT_DEFAULT = "oracle/qwen3.8-27b-spike/hf-fixture"
SEED = 0

# The same small shapes as the config in tools/dump_qwen_oracle.py.
# layer_types is the real model's repeating unit (3x linear, 1x full).
LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
HEAD_DIM = 64
PARTIAL_ROTARY_FACTOR = 0.25


def build_config():
    return Qwen3_5TextConfig(
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
            "rope_theta": 10000000.0,
            "partial_rotary_factor": PARTIAL_ROTARY_FACTOR,
            "mrope_section": [3, 3, 2],
            "mrope_interleaved": True,
        },
    )


def inject_extra_tensors(out_dir):
    from safetensors.torch import load_file, save_file

    path = f"{out_dir}/model.safetensors"
    tensors = load_file(path)
    tensors["visual.blocks.0.weight"] = torch.randn(4, 4)
    tensors["some_unrecognized.weight"] = torch.randn(4, 4)
    save_file(tensors, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--extra-tensors", action="store_true",
                     help="inject a dummy vision-tower tensor + one unrecognized "
                          "tensor, to test the unexpected-keys audit's known-drop "
                          "vs. hard-error classification")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(SEED)
    cfg = build_config()
    model = Qwen3_5ForCausalLM(cfg).eval()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)

    # The Qwen/Qwen3.8-27B repo's config.json is a composite VLM wrapper
    # (model_type "qwen3_5"), with hidden_size, layer_types etc. nested under
    # text_config. model.save_pretrained above wrote a flat text-only config.
    # config.json is overwritten here with the composite shape (the safetensors
    # file is not changed), so the fixture goes through the same
    # cfg.get_text_config() unwrapping that tools/convert_hf.py does for the
    # real checkpoint.
    composite_cfg = Qwen3_5Config(text_config=cfg, tie_word_embeddings=cfg.tie_word_embeddings)
    composite_cfg.architectures = ["Qwen3_5ForConditionalGeneration"]
    composite_cfg.save_pretrained(args.out)

    if args.extra_tensors:
        inject_extra_tensors(args.out)

    print(f"Fixture written to {args.out}/ ({cfg.model_type}, {len(LAYER_TYPES)} layers, "
          f"hidden {cfg.hidden_size}, layer_types {cfg.layer_types})")


if __name__ == "__main__":
    main()
