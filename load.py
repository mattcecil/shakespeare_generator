"""
load.py — unified model loader.

load_model(source, source_type) returns (model, tokenizer).

source_type options:
  "custom"  — .pt checkpoint path (+ _cfg.json sidecar); uses model.py + data.py
  "hf"      — HuggingFace model hub name or local path
  "fft"     — fine-tuned HF model saved with save_pretrained()
  "lora"    — LoRA adapter directory (requires peft)
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(source, source_type, device="cuda"):
    """Load a model and tokenizer.

    Args:
        source:      checkpoint path, HF model name, or directory
        source_type: "custom", "hf", "fft", or "lora"
        device:      device to move the model to

    Returns:
        (model, tokenizer)
    """
    if source_type == "custom":
        return _load_custom(source, device)
    elif source_type == "hf":
        return _load_hf(source, device)
    elif source_type == "fft":
        return _load_fft(source, device)
    elif source_type == "lora":
        return _load_lora(source, device)
    else:
        raise ValueError(f"Unknown source_type: {source_type!r}. Use 'custom', 'hf', 'fft', or 'lora'.")


def _load_custom(pt_path, device):
    """Load a custom Transformer from a .pt checkpoint + _cfg.json sidecar."""
    from model import Config, Transformer
    from data import load_tokenizer

    cfg_path = pt_path.replace(".pt", "_cfg.json")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    cfg_dict.pop("d_k", None)
    cfg_dict.pop("d_v", None)
    cfg = Config(**cfg_dict)

    model = Transformer(cfg)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.to(device).eval()

    tokenizer = load_tokenizer()
    return model, tokenizer


def _load_hf(name, device):
    """Load a HuggingFace model by hub name."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    return model, tokenizer


def _load_fft(directory, device):
    """Load a fine-tuned HF model saved with save_pretrained()."""
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForCausalLM.from_pretrained(directory).to(device).eval()
    return model, tokenizer


def _load_lora(directory, device):
    """Load a LoRA adapter on top of its base model (requires peft)."""
    import json, os
    from peft import PeftModel

    with open(os.path.join(directory, "adapter_config.json")) as f:
        adapter_cfg = json.load(f)
    base_name = adapter_cfg["base_model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(base_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_name).to(device)
    model = PeftModel.from_pretrained(base_model, directory).eval()
    return model, tokenizer
