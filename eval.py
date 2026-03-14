"""
eval.py — unified evaluation utilities.

Two functions:
  eval_loss(model, data_loader, loss_fn, vocab_size)   — custom nn.Module Transformer
  evaluate(model, val_path, tokenizer, source_type)    — any model type via source_type flag

source_type mirrors load.py: "custom" | "hf" | "fft" | "lora"
"""

import torch


def eval_loss(model, data_loader, loss_fn, vocab_size, device="cuda"):
    """Run an eval loop over a DataLoader and return average cross-entropy loss.

    Args:
        model:       nn.Module Transformer in eval mode.
        data_loader: DataLoader of (B, n) token ID tensors.
        loss_fn:     instantiated loss function, e.g. nn.CrossEntropyLoss().
        vocab_size:  number of output classes; used to reshape logits for loss_fn.
        device:      device to move batches to.

    Returns:
        float: average loss over all batches.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            logits, targets = model(batch.to(device))
            loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
            losses.append(loss.item())
    return sum(losses) / len(losses)


def evaluate(model, val_path, tokenizer, source_type, cfg=None, batch_size=8, device="cuda"):
    """Run evaluation on a text file and return average cross-entropy loss.

    Args:
        model:       trained model (nn.Module Transformer or HuggingFace causal LM)
        val_path:    path to validation text file
        tokenizer:   tokenizer matching the model
        source_type: "custom" for nn.Module Transformer; "hf" / "fft" / "lora" for HF models
        cfg:         Config — required when source_type is "custom"
        batch_size:  batch size for HF evaluation
        device:      device string, e.g. "cuda"

    Returns:
        float: average cross-entropy loss
    """
    if source_type == "custom":
        from data import make_dataloaders
        import torch.nn as nn
        val_loader = make_dataloaders(val_path, tokenizer, cfg, shuffle=False)
        return eval_loss(model, val_loader, nn.CrossEntropyLoss(), cfg.vocab_size, device)
    else:
        from utils import eval_loss_hf
        with open(val_path) as f:
            text = f.read()
        return eval_loss_hf(model, tokenizer, text, batch_size=batch_size, device=device)
