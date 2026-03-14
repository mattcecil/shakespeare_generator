# Shakespeare Generator

A decoder-only transformer trained on Shakespeare, built as a learning project. Covers pretraining from scratch, evaluation, inference, and fine-tuning across custom and HuggingFace models.

## Code Structure

| File | Contents |
|---|---|
| `model.py` | `Config`, `CausalSelfAttention`, `TransformerBlock`, `Transformer` |
| `data.py` | `train_tokenizer`, `load_tokenizer`, `make_dataloaders` |
| `utils.py` | `log_results`, `save_model`, `eval_loss_hf`, `eval_loss_custom` |
| `eval.py` | `eval_loss`, `evaluate()` |
| `generate.py` | `generate_text()` |
| `load.py` | `load_model(source, source_type)` |
| `pretrain.ipynb` | scratch training with PyTorch Lightning |
| `eval.ipynb` | evaluate any model, log BPC to `training_log.csv` |
| `inference.ipynb` | load any model, generate text |
| `finetune.ipynb` | fine-tuning |
| `transformer.ipynb` | earlier hand-rolled implementation (self-contained) |

## Architecture

Decoder-only transformer with RoPE positional encoding, pre-norm, and weight-tied embedding/lm_head.

```
Transformer
├── nn.Embedding               (token → d_model)
├── TransformerBlock × N
│   ├── nn.LayerNorm           (pre-attention norm)
│   ├── CausalSelfAttention    (RoPE, Flash Attention via F.scaled_dot_product_attention)
│   │   └── nn.Linear × 4     (Wq, Wk, Wv, Wo)
│   ├── nn.LayerNorm           (pre-FFN norm)
│   └── nn.Linear × 2         (FFN in/out, GELU)
├── nn.LayerNorm               (final norm)
└── nn.Linear                  (lm_head, weight-tied to Embedding)
```

Data flow: `(B, n)` token IDs → embedding → N transformer blocks → final norm → lm_head → `(B, n, vocab_size)` logits. Targets are inputs shifted by one; loss is cross-entropy on `logits[:, :-1]` vs `targets[:, 1:]`.

## Tokenization

BPE tokenizer trained on `full_train.txt` via HuggingFace `tokenizers`. Saved to `tokenizer.json` on first run and reloaded for all subsequent runs.

## BPC

BPC (bits per character) normalises cross-entropy loss back to characters for apples-to-apples comparison regardless of tokenizer:

```
bpc = cross_entropy_loss / log(2) / avg_chars_per_token
```

Evaluated on `tiny_val.txt` after every run. Results logged to `training_log.csv` and wandb project `shakespeare`.

## Workflows

**Pretrain from scratch** — set `cfg` in `pretrain.ipynb`, run all cells. After training, the eval cell loads the best checkpoint, computes BPC on `tiny_val`, and saves a `.pt` checkpoint to `checkpoints/`.

**Evaluate any model** — set `SOURCE` and `SOURCE_TYPE` in `eval.ipynb`, run all cells.

**Generate text** — set `SOURCE` and `SOURCE_TYPE` in `inference.ipynb`, run all cells.

**Fine-tune** — `finetune.ipynb`.

### Model source types

| source_type | source |
|---|---|
| `"custom"` | `"checkpoints/my-run.pt"` |
| `"hf"` | `"gpt2"`, `"EleutherAI/pythia-70m"`, etc. |
| `"fft"` | path to directory saved with `save_pretrained()` |
| `"lora"` | path to LoRA adapter directory (requires `peft`) |

## Stopping training early

Touch the sentinel file from a terminal to stop cleanly at the end of the current epoch:

```bash
touch /teamspace/studios/this_studio/shakespeare_generator/stop_training
```

This triggers normal checkpoint saving and wandb finalisation.

## Data

Data lives in `../data/` (outside this repo):

- `full_train.txt` — full training corpus
- `tiny_val.txt` — small held-out set used for BPC benchmarking across all runs

## Training callbacks

- **`EpochPrinter`** — prints per-batch loss on a single overwriting line, then a per-epoch summary. Accumulates batch losses directly rather than relying on Lightning's `callback_metrics` (since `on_validation_epoch_end` fires before `on_train_epoch_end`).
- **`StopOnFile`** — checks for the sentinel file at the end of each epoch and sets `trainer.should_stop`.
