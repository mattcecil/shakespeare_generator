"""
Shared utilities for Shakespeare Generator: tokenization, data loading, and evaluation.

Used by transformer.ipynb (scratch training), sft.ipynb, and inference.ipynb.
"""

import os
import math
import csv
import json
from datetime import datetime

import torch
from collections import Counter

ASCII_VOCAB = [chr(i) for i in range(128)]  # stable base vocab for all corpora

# ── BPE Tokenization ──────────────────────────────────────────────────────────

def train_bpe(text, num_merges):
    """Learn BPE merge rules from text.

    Starts from the full ASCII vocabulary and greedily merges the most frequent
    adjacent pair at each step. Returns the full vocab (ASCII + merged tokens)
    and the ordered list of merge rules needed to reproduce the encoding.
    """
    vocab = ASCII_VOCAB[:]
    tokens = list(text)
    merges = []

    for _ in range(num_merges):
        pairs = Counter(zip(tokens, tokens[1:]))
        if not pairs:
            break
        top_pair = pairs.most_common(1)[0][0]
        new_token = ''.join(top_pair)
        tokens = merge_pair(tokens, top_pair, new_token)
        vocab.append(new_token)
        merges.append(top_pair)

    return vocab, merges


def encode(text, merges, vocab):
    """Encode a string to a list of token IDs.

    Works for both BPE (pass merge rules) and char-level (pass merges=[]).
    Applies merge rules in order, then maps each token to its vocab index.
    OOV tokens (not in vocab) are silently dropped.
    """
    tokens = list(text)
    for pair in merges:
        tokens = merge_pair(tokens, pair, ''.join(pair))
    token_to_id = {t: i for i, t in enumerate(vocab)}
    return [token_to_id[t] for t in tokens if t in token_to_id]


def decode(ids, vocab):
    """Decode a list of token IDs back to a string."""
    return ''.join(vocab[i] for i in ids)


def merge_pair(tokens, pair, new_token):
    """Replace all non-overlapping occurrences of pair in tokens with new_token."""
    merged, i = [], 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
            merged.append(new_token)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(train_path, val_path, n, bpe_merges=0):
    """Load pre-split data files and tokenize, returning train/val chunk tensors.

    Args:
        train_path: path to training text file
        val_path:   path to validation text file
        n:          chunk (sequence) length
        bpe_merges: number of BPE merges; 0 = char-level tokenization

    Returns:
        train_chunks, val_chunks: LongTensors of shape (n_chunks, n)
        vocab:                list of token strings (index = token ID)
        merges:               BPE merge rules (empty list for char-level)
        avg_chars_per_token:  used to convert cross-entropy loss to BPC
    """
    with open(train_path) as f: train_text = f.read()
    with open(val_path)   as f: val_text   = f.read()

    # Vocab and BPE merges are learned from training data only;
    # base vocab is always full ASCII so IDs are stable across corpora.
    if bpe_merges > 0:
        print(f"Training BPE with {bpe_merges} merges...")
        vocab, merges = train_bpe(train_text, bpe_merges)
    else:
        vocab = ASCII_VOCAB[:]
        merges = []

    print(f"Vocab size: {len(vocab)}")
    avg_chars_per_token = len(train_text) / len(encode(train_text, merges, vocab))
    print(f"Avg chars/token: {avg_chars_per_token:.3f}")

    def to_chunks(text):
        ids = encode(text, merges, vocab)
        tokens = torch.tensor(ids)
        n_chunks = len(tokens) // n
        return tokens[:n_chunks * n].view(n_chunks, n)

    return to_chunks(train_text), to_chunks(val_text), vocab, merges, avg_chars_per_token


def encode_split(path, merges, vocab, n):
    """Encode a single pre-split text file using an existing vocab/merges.

    Used to load a secondary validation set (e.g. tiny_val when training on full).
    """
    with open(path) as f:
        text = f.read()
    ids = encode(text, merges, vocab)
    tokens = torch.tensor(ids)
    n_chunks = len(tokens) // n
    return tokens[:n_chunks * n].view(n_chunks, n)


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_loss_custom(model, chunks, cfg):
    """Evaluate average cross-entropy loss over all complete batches in chunks.

    For use with the custom Transformer class.
    """
    model.training = False
    n_batches = chunks.shape[0] // cfg.B
    batches = chunks[:n_batches * cfg.B].view(n_batches, cfg.B, cfg.n)
    total_loss = 0
    with torch.no_grad():
        for batch in batches:
            logits, targets = model.forward_pass(batch.to(cfg.device))
            total_loss += model.CrossEntropyLoss(logits, targets).item()
    model.training = True
    return total_loss / n_batches


def eval_loss_hf(model, tokenizer, text, batch_size=8, device="cpu"):
    """Evaluate average cross-entropy loss for a HuggingFace causal LM on raw text.

    Splits text into non-overlapping windows of the model's max context length,
    then evaluates loss (with labels = input_ids, so loss is next-token prediction).
    """
    import torch
    model.eval()
    n = tokenizer.model_max_length
    ids = tokenizer.encode(text)
    # Trim to complete windows
    n_chunks = len(ids) // n
    ids = ids[:n_chunks * n]
    chunks = torch.tensor(ids).view(n_chunks, n)

    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, n_chunks, batch_size):
            batch = chunks[i:i + batch_size].to(device)
            out = model(input_ids=batch, labels=batch)
            total_loss += out.loss.item()
            n_batches += 1

    model.train()
    return total_loss / n_batches


# ── Logging ───────────────────────────────────────────────────────────────────

CSV_FILE   = "training_log.csv"

CSV_HEADER = [
    "name",
    "tiny_val_loss", "tiny_val_bpc", "full_val_loss", "full_val_bpc",
    "train_loss", "timestamp",
    "vocab_size", "avg_chars_per_token",
    "layers", "d_model", "d_ff", "H",
    "B", "n", "lr", "epochs",
    "time_per_epoch_s", "total_time_s",
    "notes"
]

def log_results(cfg, train_loss, val_loss, tiny_val_loss=None, avg_chars_per_token=1.0,
                time_per_epoch=0, total_time=0, note="", name="",
                csv_path=CSV_FILE):
    """Append a training run summary to the CSV log.

    BPC (bits per character) = cross-entropy loss / log(2) / avg_chars_per_token.
    val_loss / val_bpc come from full_val (per-epoch monitor).
    tiny_val_loss / tiny_val_bpc come from tiny_val (cross-dataset benchmark).
    """
    bpc_val      = val_loss   / math.log(2) / avg_chars_per_token
    bpc_tiny_val = tiny_val_loss / math.log(2) / avg_chars_per_token if tiny_val_loss is not None else None

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        tiny_val_loss_r = round(tiny_val_loss, 6) if tiny_val_loss is not None else ""
        tiny_val_bpc_r  = round(bpc_tiny_val, 3)  if bpc_tiny_val is not None else ""
        writer.writerow([
            name,
            tiny_val_loss_r, tiny_val_bpc_r, round(val_loss, 6), round(bpc_val, 3),
            round(train_loss, 6), datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            cfg.vocab_size, round(avg_chars_per_token, 3),
            cfg.layers, cfg.d_model, cfg.d_ff, cfg.H,
            cfg.B, cfg.n, cfg.lr, cfg.epochs,
            round(time_per_epoch, 1), round(total_time, 1),
            note
        ])
    print(f"CSV row appended to {csv_path}")


# ── Checkpoint save (nn.Module) ───────────────────────────────────────────────

def save_model(model, cfg, name, directory="checkpoints"):
    """Save an nn.Module model's state_dict and config. Returns (path, name).

    Writes:
      checkpoints/<name>.pt       — state_dict
      checkpoints/<name>_cfg.json — hyperparameters
    """
    os.makedirs(directory, exist_ok=True)
    base = os.path.join(directory, name)
    torch.save(model.state_dict(), base + ".pt")
    cfg_dict = {k: v for k, v in cfg.__dict__.items()
                if not k.startswith("_") and isinstance(v, (int, float, str, bool))}
    with open(base + "_cfg.json", "w") as f:
        json.dump(cfg_dict, f)
    print(f"Saved model to {base}.pt")
    return base + ".pt", name
