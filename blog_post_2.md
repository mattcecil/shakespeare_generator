# What the Data Was Telling Me All Along: Training on All of Shakespeare

The first post in this series ended with a section called "Things I Would Do Differently." The first item on the list:

> **More data.** The clearest lever available. The entire TinyShakespeare corpus is only 1.1M characters — a single short novel. Training on the full Works of Shakespeare (~5M characters) would likely push val BPC below 2.0 and make BPE tokenization competitive.

That prediction turned out to be exactly right, and the story of what happened next is mostly the story of testing it — along with a few detours that were worth taking.

---

## What the Training Log Says

Before getting into the narrative, here's the full picture laid out chronologically. The CSV tells a story if you read it carefully.

| Run | Corpus | Tiny val BPC | Train BPC | Epochs | Time/epoch |
|---|---|---|---|---|---|
| v1-0 | TinyShakespeare (1.1M) | 4.789 | 4.712 | 5 | — |
| v1-1 | TinyShakespeare (1.1M) | 4.742 | 4.671 | 6 | 18.8s |
| v1-2 | TinyShakespeare (1.1M) | 3.000 | 2.937 | 8 | 47.0s |
| v1-3 | TinyShakespeare (1.1M) | 2.492 | 1.901 | 16 | 15.1s |
| v1-4 | TinyShakespeare (1.1M) | 2.381 | 1.741 | 15 | 20.8s |
| v1-5 | TinyShakespeare (1.1M) | 2.369 | 1.800 | 18 | 25.4s |
| v1-6 | TinyShakespeare (1.1M) | 2.219 | 1.617 | 20 | 32.6s |
| v1-7 | TinyShakespeare (1.1M) | **2.201** | 1.670 | 18 | 32.2s |
| v1-8 | TinyShakespeare, BPE-566 | 2.254 | 2.374 | 14 | 10.7s |
| **v1-9** | **Full Shakespeare (3.5M)** | **1.819** | **1.568** | **15** | **184s** |
| v1-10 | Full Shakespeare, 862K subset (different scenes) | 2.480 | 1.721 | 18 | 47.0s |
| v1-11 | Full Shakespeare, 862K subset (same scenes as tiny_train) | 2.273 | 1.719 | 17 | 47.2s |
| gpt2 | — (baseline) | 1.880 | — | — | — |

A few things jump out.

**The jump from v1-7 to v1-9 is the single largest improvement in the log.** Val BPC went from 2.201 to 1.819 — a drop of 0.382 — with no architecture changes at all. Everything else in the first phase of the project (adding RoPE, switching to AdamW, tuning learning rates, adding dropout, scaling layers) added up to a total improvement of about 2.5 BPC points across many experiments. The data scaling did 0.382 in a single run. The model hadn't changed. The architecture hadn't changed. Three times more training data, and the model just... got much better.

This is the empirical version of a point that's easy to state abstractly: at small data scales, the binding constraint is almost always data, not architecture. Every architecture improvement I made from v1-0 to v1-7 was fighting uphill against a fundamentally data-starved model.

**v1-9 beats untuned GPT-2 on Shakespeare text.** This is the number I didn't expect. GPT-2, a model with ~117M parameters trained on ~40GB of web text, achieves BPC 1.880 on TinyShakespeare with zero fine-tuning. v1-9, a 6-layer model with ~10M parameters trained from scratch on 3.5M characters of Shakespeare, achieves 1.819 on the same benchmark. Domain specificity is a real effect: a small model that knows Shakespeare deeply can outperform a large model that knows Shakespeare only incidentally.

**v1-10 and v1-11 reveal a measurement trap.** After training on the full corpus, I ran two ablations: same architecture and hyperparameters, but trained on 862K-character subsets of the full text — approximately matching the volume of TinyShakespeare. v1-10 used scenes not in TinyShakespeare; v1-11 used scenes that overlap with TinyShakespeare.

The results are almost backwards from what you'd expect at first glance. v1-11 gets *better* tiny_val BPC (2.273) than v1-10 (2.480), despite training on the same volume of data. The reason: the tiny_val split comes from the same original corpus as tiny_train. A model trained on similar scenes has seen similar text patterns, so it performs better on that specific benchmark. This isn't generalization — it's measurement contamination. The right benchmark is full_val (held out from the full corpus), where v1-10 (2.362) actually slightly outperforms v1-11 (2.410). Subtle, but it matters: if I had only been watching tiny_val, I would have drawn the wrong conclusion about which training corpus was better.

---

## Phase 1: Full Shakespeare

Getting the full Works of Shakespeare into a usable training set required more care than I expected.

TinyShakespeare is a convenient excerpt — 1.1M characters pulled from a mix of plays. The full Works is approximately 5.3M characters. But simply dumping the full text into the training loop would contaminate the benchmark: the TinyShakespeare validation split I'd been using throughout the project comes from the same text as TinyShakespeare's training split, meaning the full Works already contains it. Training on that would be training on the validation set.

The solution was to construct a corpus that excludes the plays covered in TinyShakespeare. The resulting training corpus was 3.5M characters — about 3x the original — with the TinyShakespeare validation set kept entirely held out.

But simply excluding the right plays wasn't enough. The raw t8.shakespeare.txt source is formatted completely differently from TinyShakespeare. It contains Dramatis Personae sections (cast lists at the top of each play), Act and Scene headings, inline and standalone stage directions, and speaker names in a mix of formats depending on the play — full ALL CAPS in most, abbreviated Title Case in Romeo and Juliet (`Rom.`, `Jul.`), and so on. TinyShakespeare, by contrast, is pure dialogue: speaker name followed by a colon, then speech text, with nothing else. Training on the raw source would have introduced characters and patterns the model had never seen in the validation data, making any performance comparison meaningless.

The fix was a purpose-built formatting pipeline (`make_tinyshakespeare.py`) that converts the complete works into TinyShakespeare format:

- **Dramatis Personae stripped** — regex removal of all cast-list sections
- **Stage directions stripped** — standalone lines starting with Enter, Exit, Exeunt, Flourish, Alarum, etc., plus inline `[Aside]` / `[To BRUTUS]` annotations removed from dialogue lines
- **Act/Scene headings stripped** — left no structural markers in the output text
- **Speaker names normalized** — proper personal names kept ALL CAPS (`HAMLET`, `BRUTUS`); role/descriptor names converted to Title Case (`First Citizen`, `Friar Laurence`, `Messenger`). Romeo and Juliet's abbreviated speakers (`Rom.`, `Jul.`) mapped to their full normalized forms
- **Splits snapped to speech boundaries** — train/val/test splits aligned to blank lines so no speech is cut mid-utterance

The result: the full corpus and TinyShakespeare share identical formatting conventions. The character-level vocabulary is the same 69-character set in both. Any BPC difference between a model trained on TinyShakespeare and one trained on the full corpus reflects what was actually learned from the data, not formatting artifacts.

The practical cost was runtime. Each training epoch went from ~32 seconds to ~184 seconds, a 5.75x increase. Fifteen epochs took 48 minutes. Not prohibitive, but enough to make each experiment a real time commitment rather than something you fire off while thinking about the next change.

The result: **val BPC 1.819 on the original TinyShakespeare benchmark, and 1.953 on the held-out full-corpus validation set.** Both substantially better than the previous best of 2.201.

---

## Phase 2: The Data Volume Question

The jump from v1-7 to v1-9 raised an obvious question: was the improvement from the *volume* of data, or from the *content* (more plays, more diverse Shakespeare)?

To isolate this, I ran two controlled experiments at 862K characters — roughly matching the original TinyShakespeare volume:

- **v1-10**: 862K characters drawn from plays *not* in TinyShakespeare
- **v1-11**: 862K characters from plays that *overlap* with TinyShakespeare

Both converged around BPC 2.35–2.48, roughly matching the original TinyShakespeare results. This is a clean confirmation that volume was the driver, not content diversity. The model isn't learning "better Shakespeare" from different plays — it's learning better Shakespeare from more of it. More tokens per parameter during training is what moves the needle.

The overfitting gap also tells the same story. v1-9 has a train-val BPC gap of 0.385 (1.568 train, 1.953 val). v1-10 and v1-11, with one-quarter the data, have gaps of 0.641 and 0.691. The model capacity (same architecture throughout) is unchanged; the ratio of data to parameters is what determines how much the model overfits.

---

## Phase 3: Rebuilding in PyTorch Lightning (and a larger BPE experiment)

With the data scaling question answered, the next step was rebuilding in PyTorch Lightning. The motivation was practical: the hand-rolled training loop had grown into something functional but not particularly clean. Lightning handles device placement, gradient accumulation, logging, and checkpointing in a structured way — and for anything beyond toy experiments, it's the right foundation.

The Lightning rebuild also came with an opportunity to revisit BPE tokenization with a larger vocabulary. The original BPE experiment used 500 merges (vocab size 566). The rebuild used 1000 merges (vocab size 1000, avg 2.45 chars/token).

The results were mixed. BPC came down to 2.062 on tiny_val and 2.27 on full_val — worse than v1-9, but this isn't a fair comparison, since the Lightning runs hadn't yet been trained on the full 3.5M-character corpus. The rebuild was primarily infrastructure work, with the BPE experiment as a secondary thing to try while the new training loop was being validated.

One thing that has become clear: BPE at this dataset size is consistently worse in practice than the theory would suggest. A vocab of 1000 tokens means the model needs to learn embedding relationships for 1000 items rather than 128 ASCII characters. With only ~3.5M characters of training data, many of those token types don't appear often enough to learn robust representations. The parameter cost is real; the benefit (compressed sequences, better long-range structure) doesn't fully materialize at this scale.

---

## Phase 4: Implementing LoRA

The most recent work is implementing LoRA (Low-Rank Adaptation) from scratch, extending the same "no `nn.Module`" approach that characterized the whole project.

The idea is efficient fine-tuning: instead of updating all parameters of a pretrained model, freeze them and inject small trainable adaptations into specific weight matrices. For a matrix $\mathbf{W} \in \mathbb{R}^{d \times d}$, the adapted weight is:

$$
\mathbf{W}' = \mathbf{W} + \frac{\alpha}{r} \mathbf{A} \mathbf{B}
$$

where $\mathbf{A} \in \mathbb{R}^{d \times r}$ and $\mathbf{B} \in \mathbb{R}^{r \times d}$, with $r \ll d$. At rank 4 with `d_model=384`, each LoRA pair adds $2 \times 384 \times 4 = 3072$ trainable parameters. Applied to all 6 weight matrices in each of 6 layers: $6 \times 6 \times 3072 = 110592$ trainable parameters, compared to ~10M in the full model. A 99% reduction.

A critical implementation detail: **B matrices must be initialized to zero**. This ensures that the LoRA correction $\frac{\alpha}{r}\mathbf{A}\mathbf{B}$ starts at exactly zero, so the model's outputs at the start of fine-tuning are identical to the pretrained base. The original LoRA paper specifies this, and it matters: if B is initialized randomly, fine-tuning begins from a perturbed starting point rather than the pretrained checkpoint, which degrades both stability and final quality. (A matrices can be initialized with small random values; B's zero init is the load-bearing constraint.)

The `LoRATransformer` is a separate class from `Transformer`. All of the attention and FFN computations are modified to add the low-rank correction inline:

```python
Q = X @ self.Wq[i] + (cfg.lora_alpha / cfg.lora_rank) * (X @ self.Aq[i]) @ self.Bq[i]
```

The frozen parameters (`Wq`, `Wk`, etc.) have `requires_grad_(False)`; the A/B pairs have `requires_grad_(True)`. The `params()` method returns only the A/B pairs, so the optimizer never sees the frozen weights.

The natural next step — loading pretrained weights from v1-9 into the frozen parameters before fine-tuning — is set up but not yet run. This is the interesting experiment: take a model that has already learned Shakespeare's patterns at the character level, freeze that knowledge, and ask whether LoRA fine-tuning can adapt it further with minimal parameter updates.

---

## Where Things Stand

The trajectory of the project, in BPC terms:

```
v1-0  (baseline, manual SGD):              4.789
v1-7  (best TinyShakespeare architecture): 2.201
v1-9  (full Shakespeare, 3.5M chars):      1.819
gpt2  (untuned baseline):                  1.880
```

The model trained from scratch on Shakespeare now outperforms an untuned GPT-2 on Shakespeare-specific text. The remaining gap below human-level compression (~1.0–1.2 BPC for fluent English) is mostly explained by model size: GPT-2 has ~117M parameters; this model has ~10M. To close that gap meaningfully would require either more parameters or a fundamentally different approach — fine-tuning a large pretrained model rather than training from scratch.

Which is exactly what LoRA enables. The experiment waiting to be run: initialize the frozen weights of `LoRATransformer` from v1-9, train only the 110K LoRA parameters on a specific subset of Shakespeare — a single play, a particular style, a constrained domain — and see what fine-tuning at minimal computational cost can actually achieve.

That's the next post.

---

## The Division of Labor, Revisited

The first post described a four-part role for AI assistance: syntax and typos, infrastructure, deep conversations, and writing. That division still holds. The full-corpus data pipeline, the training loop modifications, the Lightning rebuild, and the LoRA implementation all followed the same pattern: architecture and math decisions made manually, AI used to accelerate the parts that are orthogonal to understanding.

One addition worth noting: this post was drafted with AI assistance — but the training log analysis, the conclusions about what the data says, and the framing of what comes next were worked out before a word was written. The data is the data. The patterns in it are there whether or not you have help articulating them. Getting the articulation right is a different skill from getting the analysis right, and it's the one where AI assistance has the highest leverage.

---

## Code

Full implementation: [github.com/mattcecil/ml_models](https://github.com/mattcecil/ml_models)

Key additions since the first post:
- Full Shakespeare corpus construction and training (v1-9)
- Data volume ablations (v1-10, v1-11)
- PyTorch Lightning rebuild
- `LoRATransformer` class with rank-4 low-rank adaptation on all attention and FFN weight matrices

---

*Written March 2026. Model trained on an NVIDIA GPU via Lightning AI studio.*
