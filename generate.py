"""
generate.py — unified text generation utilities.

Handles two model types:
  - Custom Transformer (nn.Module): encode with HF tokenizers → model.generate() → decode
  - HuggingFace model: tokenizer() → model.generate() with HF sampling args → decode

Note: transformer.ipynb has its own standalone generate() that operates directly
on raw tensor model internals — it cannot be unified here without rewriting it.
"""

import torch
from tokenizers import Tokenizer as HFTokenizersTokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0,
                  top_k=None):
    """Encode prompt, generate tokens, decode and return the generated text.

    Detects model type from the tokenizer: HF tokenizers.Tokenizer → custom
    Transformer path; transformers tokenizer → HuggingFace path.

    Args:
        model:          custom Transformer (nn.Module) or HuggingFace model
        tokenizer:      HF tokenizers.Tokenizer (custom) or transformers tokenizer (HF)
        prompt:         input string
        max_new_tokens: number of new tokens to generate
        temperature:    sampling temperature (1.0 = unchanged, <1 = sharper, >1 = more random)
        top_k:          if set, restrict sampling to top-k logits

    Returns:
        str: generated text (not including the prompt)
    """
    if isinstance(tokenizer, HFTokenizersTokenizer):
        # Custom Transformer path
        prompt_ids = tokenizer.encode(prompt).ids
        output_ids = model.generate(prompt_ids, max_new_tokens=max_new_tokens,
                                    temperature=temperature, top_k=top_k)
        return tokenizer.decode(output_ids)
    else:
        # HuggingFace model path
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=temperature != 1.0 or top_k is not None,
        )
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
