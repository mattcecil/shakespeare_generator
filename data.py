import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import BPEDecoder
from torch.utils.data import DataLoader


def train_tokenizer(train_path, vocab_size=1000, save_path="tokenizer.json"):
    """Train a BPE tokenizer on the training corpus and save it to disk.

    vocab_size = 128 base ASCII chars + n_merges, so vocab_size=628 ~ 500 merges.
    Only needs to be run once; reload with load_tokenizer(save_path) after.
    Returns tokenizer.
    """
    tokenizer = Tokenizer(BPE())
    tokenizer.decoder = BPEDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size)
    tokenizer.train(files=[train_path], trainer=trainer)
    tokenizer.save(save_path)
    return tokenizer


def load_tokenizer(path="tokenizer.json"):
    """Load a previously trained BPE tokenizer from disk."""
    return Tokenizer.from_file(path)


def make_dataloaders(path, tokenizer, cfg, shuffle=False):
    """Tokenize a text file, chunk into sequences of length cfg.n, return a DataLoader.

    Steps:
        - Read text, encode with tokenizer
        - Split into non-overlapping chunks of length n
        - Convert to tensors
        - Return DataLoader with drop_last=True

    Args:
        path:      path to text file
        tokenizer: HuggingFace tokenizers.Tokenizer instance
        cfg:       Config with fields B (batch size), n (seq length)
        shuffle:   whether to shuffle (True for train, False for val)
    """
    with open(path) as f:
        text = f.read()

    ids = tokenizer.encode(text).ids
    tokens = torch.tensor(ids)
    n_chunks = len(tokens) // cfg.n
    truncated_tokens = tokens[:n_chunks * cfg.n]
    tokens_matrix = truncated_tokens.view(n_chunks, cfg.n)
    return DataLoader(tokens_matrix, batch_size=cfg.B, shuffle=shuffle,
                      drop_last=True, num_workers=4, pin_memory=True)
