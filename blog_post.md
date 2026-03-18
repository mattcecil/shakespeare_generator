# Closer to the Metal, Faster With AI: Building a Transformer From Scratch

I've been working in ML, AI, and NLU for over a decade — including several years building production systems at Amazon Alexa. As an IC I was fluent in the technical details: statistical NLU models, logistic regression classifiers, grammar-based language understanding systems, the kind of thing you could sit down and implement from scratch. Then I moved into engineering management, and something else happened at the same time: the field transformed around me. The models we were building went from interpretable statistical systems to deep neural networks to transformers to full-scale generative AI. Each shift raised the floor of what "understanding the model" even means. As a manager I could stay current at the level of research papers and architectural decisions, but the gap between that and being able to implement it from scratch kept widening. This project was about closing that gap: getting back to a level of technical detail where I actually understand what's happening inside the model, not just how to deploy it.

There are two failure modes for learning a technical skill with AI coding assistants. The first: let the AI write everything while you review and ship. You move fast, but understanding doesn't accumulate — the AI did the hard parts. The second: refuse to use AI on principle, treating it as a crutch. You do the work yourself, but you're leaving a genuinely powerful learning tool on the table.

This post is about a third path. The goal was to implement a decoder-only Transformer language model from scratch — every matrix multiply, every attention head, every gradient update, without delegating the core architecture to `nn.Module` or `nn.MultiheadAttention`. AI coding assistants (primarily Claude Code) were used throughout, but deliberately: for boilerplate and infrastructure, for catching typos, and most valuably as a patient tutor for *why* questions at any hour. The hypothesis was that the right kind of AI use doesn't prevent deep understanding — it accelerates it.

The task was decoder-only Transformer language modeling on the TinyShakespeare dataset — about 1.1 million characters of Shakespeare's plays, the same dataset Karpathy uses in his *makemore* and *nanoGPT* tutorials. It's small enough to train on a single GPU in minutes per epoch, large enough to produce outputs that feel evaluable, and the domain is distinctive enough that you can tell at a glance whether the model has learned something real.

The model itself is a plain Python class with raw `torch.Tensor` parameters. This project was the last in a progression: I first implemented gradient descent with no autograd at all (manually deriving gradients for a Multi Layer Perceptron classifier), then built a small Transformer that used `loss.backward()` but applied weight updates by hand per parameter, and finally arrived here — using `torch.optim.AdamW` only after understanding exactly what it was replacing. Each step was a deliberate choice to understand an abstraction before using it.

This post documents both threads: what I built and how AI made the building deeper and faster.

## How AI Was (and Wasn't) Used

The assistant played four distinct roles:

1. **Syntax and typos.** When a reshape was wrong or a variable name was inconsistent, AI caught it faster than rereading. This is table stakes.
2. **Infrastructure.** Checkpoint saving, training log formatting, the inference pipeline, BPC metric calculation — these are important but not intellectually core. I described what I wanted and let AI implement it, reviewing and modifying the result.
3. **Deep conversations.** The most valuable use: probing questions. *Why does weight tying work mathematically? What's the information-theoretic meaning of BPC? Why do transformers underperform LSTMs on small datasets?* Having a precise, patient conversation partner available while the model is training accelerates the parts of learning that are about understanding, not coding.
4. **Writing.** I asked the assistant to generate a first draft of this blog post based on our full conversation history, the git history of code changes, and the training result logs. From there I pushed back on inaccuracies, suggested improvements, and took the driver's seat to make it more closely represent my voice, with the post evolving through that back-and-forth.

What AI was **not** used for: the architecture itself, the math derivations, the hyperparameter intuitions, or any of the "what should I try next" decisions. The goal was to use AI to go *deeper* into the metal, not to stay further from it.

---

## Architecture

The model is a standard decoder-only Transformer with pre-norm residual connections:

$$
\mathbf{X}^{(l)} = \mathbf{X}^{(l-1)} + \text{Dropout}\left(\text{Attention}\left(\text{LayerNorm}\left(\mathbf{X}^{(l-1)}\right)\right)\right)
$$

$$
\mathbf{X}^{(l)} \leftarrow \mathbf{X}^{(l)} + \text{Dropout}\left(\text{FFN}\left(\text{LayerNorm}\left(\mathbf{X}^{(l)}\right)\right)\right)
$$

Key components:

| Component | Choice | Rationale |
|---|---|---|
| Positional encoding | RoPE | Relative position generalization |
| Activation | GELU | Smoother than ReLU, standard in GPT-family |
| Normalization | Pre-norm LayerNorm | More stable training than post-norm |
| Output projection | Weight tying ($\mathbf{W}_{\text{emb}}^\top$) | Reduces parameters, improves generalization |
| Final layer | LayerNorm before output projection | Matches GPT-2 architecture |
| Optimizer | AdamW | Decoupled weight decay |

### RoPE: Rotary Position Embedding

Instead of adding fixed sinusoidal embeddings to token representations, as in the original Attention Is All You Need, RoPE rotates query and key vectors in the complex plane based on their absolute position. For my use case both approaches work equally well, but for longer context lengths and larger models, RoPE scales better because it encodes *relative* position between tokens, not just *absolute* position in the sequence. For a query vector $\mathbf{q}$ at position $m$:

$$
\mathbf{q}_m = \mathbf{q} \cdot e^{im\theta}
$$

The dot product $\mathbf{q}_m \cdot \mathbf{k}_n^\top$ then depends only on the relative position $m - n$, giving the model translational equivariance for free without learned positional embeddings. Implemented manually as paired cosine/sine rotations over adjacent dimensions.

### Weight Tying

Instead of a separate output projection matrix $\mathbf{W}_{\text{out}} \in \mathbb{R}^{d_{\text{model}} \times V}$, the transposed embedding matrix $\mathbf{W}_{\text{emb}}^\top$ is reused:

$$
\text{logits} = \mathbf{X}_{\text{final}} \cdot \mathbf{W}_{\text{emb}}^\top
$$

This is principled: the embedding matrix encodes "what does token $v$ mean as an input?" and its transpose encodes "how similar is this hidden state to token $v$?" — which is exactly what a softmax classifier needs. It also reduces the parameter count by $V \times d_{\text{model}}$ (about 25K parameters at our scale), which matters for small datasets.

---

## Training Setup

```
vocab_size:  66  (character-level)
d_model:     384
d_ff:        1536  (4× d_model)
H:           8 heads
d_k = d_v:  48
layers:      6
n (seq len): 256
B (batch):   32
dropout:     0.1
lr:          3e-4
optimizer:   AdamW (weight_decay=0.01)
scheduler:   ReduceLROnPlateau (factor=0.5, patience=2)
grad clip:   1.0
```

Dataset: TinyShakespeare (~1.1M characters, 66 unique chars). Split 90/5/5 train/val/test.

---

## Phase 1: Getting the Basics Working

Before the main Transformer, two earlier experiments built up to it:

**No-autograd linear regression.** The first experiment (`linear_regression.ipynb`) implemented gradient descent with no PyTorch autograd at all — manually deriving the MSE gradient and applying updates directly:

```python
W = W - lr * X_batch.T @ (y_hat - y_batch) / len(X_batch)
```

No `loss.backward()`. No `.grad`. Just math. The point was to understand what autograd is actually doing before letting it do it.

**Manual SGD Transformer.** The second experiment (`SGDtransformer.ipynb`) used `loss.backward()` for gradient computation but applied weight updates by hand for every parameter:

```python
with torch.no_grad():
    model.W_emb -= cfg.lr * model.W_emb.grad
    model.W_emb.grad = None
    for i in range(cfg.layers):
        model.Wq[i] -= cfg.lr * model.Wq[i].grad
        model.Wq[i].grad = None
        # ... repeated for every parameter
```

Having written this loop once, `optimizer.step()` is no longer a black box — it's just that loop, plus momentum terms and weight decay, done for you.

**The main Transformer.** The first working version of the final model was tiny: `d_model=64`, 2 layers, sequence length 50, batch size 16, now using `torch.optim.AdamW`. Val loss ~3.3. Not impressive, but it ran.

<!-- INSERT FIGURE: early loss curves, small model -->

Early experiments revealed something important: **the optimizer matters enormously.** With a fixed learning rate of `lr=0.001`, val loss stagnated around 3.2–3.3 across many runs. Bumping to `lr=0.01` got loss down to ~2.08, but training was unstable. Switching to AdamW dropped val loss to **1.79 in just 4 epochs** — a dramatic jump that no amount of plain SGD tuning had achieved.

This is the adaptive gradient methods lesson in practice. AdamW's per-parameter learning rates let the model escape flat regions that simple SGD gets stuck in, especially with the diverse gradient magnitudes that arise in deep networks.

---

## Phase 2: Scaling Up

With AdamW working, the obvious move was to scale: 6 layers, `d_model=384`, sequence length 256. Each architectural addition was added deliberately:

**RoPE** was added after training the initial model with no positional encoding at all. Without positional information, the model is theoretically permutation-invariant — it can't distinguish "the cat sat" from "sat cat the" except through what it learns from context statistics. In practice the model still learned reasonable structure (language has strong local patterns that survive without explicit position), but adding RoPE gave a measurable improvement. The improvement at small scale was modest (~0.05 val loss), but the theoretical appeal of relative position generalization justified the implementation effort. It was also the most mathematically interesting piece to implement — rotating paired dimensions by position-dependent angles, applied to Q and K but not V.

**Activation function.** The feed-forward layers started with ReLU and were switched to GELU — smoother, and standard in the GPT family. In practice the improvement was negligible at this scale, which is consistent with the literature: the differences between ReLU and GELU tend to show up more clearly at larger model sizes and longer training runs.

**Gradient clipping** (`max_norm=1.0`) was added following standard practice for Transformer training. With fixed batch sizes and a well-tuned learning rate, the practical impact was modest — loss curves didn't change dramatically before and after. It's the kind of defensive measure that matters more in longer runs or with noisier data, and costs nothing to include.

**ReduceLROnPlateau** provided a form of adaptive scheduling without needing to specify a decay schedule in advance. When val loss plateaued for 2 consecutive epochs, the learning rate halved. This naturally extended useful training beyond what a fixed schedule would allow.

---

## Phase 3: Comparing to nanoGPT

After reaching val loss ~1.65, I compared my implementation against Karpathy's nanoGPT (which achieves ~1.47 on the same dataset). The gap traced back to several concrete differences:

| Feature | My initial impl | nanoGPT |
|---|---|---|
| Context length | 100 | 256 |
| Layers | 3 | 6 |
| LR scheduler | None | Cosine decay |
| Gradient clipping | None | 1.0 |
| Final LayerNorm | No | Yes |
| Weight tying | No | Yes |

Adding each of these iteratively drove val loss from ~1.65 down to **1.53**. The remaining gap is likely explained by nanoGPT using `nn.CrossEntropyLoss` (which uses a more numerically stable log-sum-exp implementation) and potentially different random seeds / initialization.

The lesson: these "small" implementation details collectively matter more than architecture choices at this scale.

---

## Phase 4: Inference and Generation

Autoregressive generation requires no new training — just run the forward pass one token at a time, sample from the output distribution, append the token, repeat.

Two knobs control output quality:

**Temperature** $T$ scales logits before softmax:
$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

As $T \to 0$, the distribution collapses to argmax (greedy). As $T \to \infty$, it approaches uniform. In practice, $T \in [0.7, 1.0]$ produces the best Shakespeare-sounding text.

**Top-$k$ sampling** restricts sampling to the $k$ most probable tokens, setting all others to $-\infty$ before softmax. This prevents the model from occasionally emitting very low-probability tokens that break the illusion of coherent text.

An interesting empirical observation: **output quality improves as generation continues.** The cause is a padding warm-up effect: the model was trained on 256-token sequences of dense real text, but inference starts with a short prompt left-padded with dummy token 0 up to the full context length of 256. The first tokens are generated while the context is mostly padding — out-of-distribution for the model. As real generated text fills the context window, the model operates in its trained distribution and output quality improves.

This is easiest to see by comparing a single-newline prompt against a long Shakespeare passage.

**Short prompt** (`"\n"`, temperature=0.8, top-k=30):

```
the ba-pnte, b'A'Be hCe, The cut'Than Ceappnocinges burbites baral,
he g not brone, and pluckbbbeats nose, bribburnes, But geal, mains, an
Upon aral, and bringly, galces 'O'' but-sing; and, why are you moth
upon it, two lives 'Tis 'll followards wrongers slikenes!
Have for doing them, by the born sweet-wenchered with
must be not black-a-tore of gribles could never
thunder these wines with unjudged block our highth,
And buy to die my refer wearing in the mighty.

BENVOLIO:
Come, the way I must not say 'My lord, where I may becomeful
speech aption, but yet we fear these counsel
are nor day to save the boin of men of them two much
master, by their the fields as lies to thy fruit
wrong the foe of heard! Let's he should be good.

LUCIO:
Come, mistress; I pardere the gentle shepherd...
```

The first ~100 characters are essentially noise — the model has no real context, so it produces garbled fragments. By ~200 characters it has found some rhythm, and by the time it emits `BENVOLIO:` it's operating much closer to its trained distribution.

**Long prompt** (21 lines of actual Shakespeare, temperature=0.8, top-k=30):

```
SEBASTIAN:                                          ← prompt starts
You were kneel'd to and importuned otherwise
By all of us, and the fair soul herself
Weigh'd between loathness and obedience, at
Which end o' the beam should bow. We have lost your
son,
I fear, for ever: Milan and Naples have
More widows in them of this business' making
Than we bring men to comfort them:
The fault's your own.

ALONSO:
So is the dear'st o' the loss.

GONZALO:
My lord Sebastian,
The truth you speak doth lack some gentleness
And time to speak it in: you rub the sore,
When you should bring the plaster.

SEBASTIAN:
Very well.--                                        ← prompt ends, generation begins

ESCALUS:
Tell him to you, no: approachance?
Appose you all descend all conclusters,
And well I be such a secret some frowns.
His too first be a great a taste in honest
Flenge some fludess, but not by the fire-place,
They have minded in the honour Menenius are well fellow
Measure of beauty, and it wisden, then, when
nothing shall you have recorted to hidst our enemies of my
made the trimment wretched the saw of the bosom,
he were to two with a fears. What is now no more than
some captain good courone! Besent challenge you see
how he bears mine of two might after she is nurse.
3 KING HENRY VI

KING EDWARD IV:
But who made the queen in Richmond's lights,
Remememanded upon soldiers, which had a gentlewomen
Hath that the wife washed the father of fly,
Hath cominrey to the Towards of Aufidius,
For she was franch'd in this great friends,
And speaked him law as we have drops tails,
Look on the heavy are and a king maughter
Of my present suspicion and providence with war
Make chamber there, whether I cannot see:
To having cowards the heavens of men.
```

The difference is immediate. With a full-length real-text prompt, the very first generated token is a character name — the model is already in-distribution. The output still has errors ("approachance", "conclusters", "Flenge") but these are model capacity limitations, not warm-up artifacts.

The practical takeaway: **prompt length is a form of context conditioning.** A longer, more coherent prompt doesn't just provide more information — it positions the model's hidden state closer to where it was during training on real text, producing qualitatively better output from the start.

---

## Phase 5: Byte-Pair Encoding

The natural next step after character-level tokenization was BPE, implemented from scratch. The algorithm is elegant:

1. Start with the character vocabulary
2. Count all adjacent token pairs
3. Merge the most frequent pair into a new token
4. Repeat for $N$ merge operations

With 500 merges, the vocabulary grows from 66 to 566 tokens, and the average token covers ~2.3 characters. This compresses the sequence length (more information per token) and lets the model attend over more context in characters while using fewer tokens.

### The Apples-to-Apples Comparison Problem

Cross-entropy loss measures **bits per token** (after dividing by $\log 2$). But char-level and BPE models have different token granularities, making their raw losses incomparable. The correct metric is **bits per character (BPC)**:

$$
\text{BPC} = \frac{\mathcal{L}_{\text{CE}}}{\log 2 \cdot \bar{c}}
$$

where $\bar{c}$ is the average number of characters per token. For char-level, $\bar{c} = 1$ and BPC = loss/log(2). For BPE with 500 merges, $\bar{c} \approx 2.3$, so the BPC is meaningfully lower than the raw loss suggests.

| Model | val loss | avg chars/token | val BPC |
|---|---|---|---|
| Char-level, 18 epochs | 1.526 | 1.000 | 2.201 |
| BPE-500, 8 epochs (best) | 3.551 | ~2.27 | ~2.25 |

### Why BPE Struggled Here

With the corrected BPC, the gap is smaller than the raw loss makes it appear — BPE val BPC (~2.25) is close to char-level (~2.20). But the overfitting is severe: train BPC was 1.64 while val BPC was 2.25, a gap of 0.61 vs 0.53 for the char-level model. BPE adds complexity without a clear generalization benefit on this dataset.

The core issue: **BPE pays off at scale**. With a larger vocabulary (566 vs 66), the model needs to learn more embedding relationships from the same amount of data. The effective training set size in tokens also shrinks proportionally to $\bar{c}$ — each token type appears less often. On 1.1M characters this means substantially fewer training examples per token type. BPE amplifies model capacity (larger vocab, richer token space) without increasing the data. For a dataset this small, character-level is the better fit.

---

## Results Summary

<!-- INSERT FIGURE: val loss over epochs, char-level best run -->
<!-- INSERT FIGURE: train vs val loss gap showing overfitting for BPE -->
<!-- INSERT TABLE: full hyperparameter sweep from training_log.txt -->

**Starting config:** 2 layers, d_model=64, d_ff=256, H=4, B=16, n=50, lr=0.001, manual SGD, LayerNorm, no RoPE, no dropout, ReLU activation.

| Change | val loss | val BPC | epochs |
|---|---|---|---|
| Baseline | 3.319 | 4.789 | 5 |
| + RoPE | 3.287 | 4.742 | 6 |
| d_model 64→256; lr 0.001→0.01 | 2.080 | 3.000 | 8 |
| lr 0.01→0.001 | 1.727 | 2.492 | 16 |
| AdamW; 3L; d_model 256→384, d_ff 256→1536; B 16→32, n 50→100 | 1.651 | 2.381 | 15 |
| + Dropout; ReLU→GELU | 1.642 | 2.369 | 18 |
| 6L; n 100→256; + weight tying, grad clip, dropout; lr 0.001→0.0003 | **1.526** | **2.201** | 18 |
| BPE tokenization; vocab 66→566; n 256→128 | 3.551 | 2.254 | 14 |



---

## Things I Would Do Differently

**More data.** The clearest lever available. The entire TinyShakespeare corpus is only 1.1M characters — a single short novel. Training on the full Works of Shakespeare (~5M characters) would likely push val BPC below 2.0 and make BPE tokenization competitive.

**Bias terms.** I chose not to add bias parameters to QKV projections, feed-forward layers, or LayerNorm. This is defensible (some modern architectures omit them) but adds a few percent of parameters that can help, especially in the feed-forward layers.

**Cosine LR decay** rather than ReduceLROnPlateau. The plateau scheduler is reactive and conservative; cosine decay with a warmup period often reaches lower loss because it anneals the learning rate more smoothly through the loss landscape. The tradeoff is that cosine decay requires committing to a total number of steps upfront — you can't do early stopping, because the schedule is calibrated to the full run. ReduceLROnPlateau sidesteps this by reacting to what the model is actually doing, which made it a better fit for exploratory training where the right number of epochs wasn't known in advance.

**Flash Attention.** The attention implementation here is $O(n^2)$ in memory and compute. Flash Attention computes the same result in $O(n)$ memory by fusing the softmax into a single kernel pass. At `n=256` this doesn't matter; at `n=2048` it's the difference between running and not running.

---

## Conclusion

Building a Transformer from scratch — every matrix multiply, every gradient, every hyperparameter choice made consciously — produces a different kind of understanding than using high-level libraries. The model isn't better than nanoGPT; it's worse. But I understand exactly why it's worse, and I know what would close the gap. The AI didn't shortcut that understanding — it helped build it.

The division of labor that worked: AI handled infrastructure and boilerplate, caught mistakes faster than rereading, answered theory questions on demand, and helped me write this post. Everything else — the architecture, the math, the decisions about what to try and why — stayed manual. That boundary is the key. AI is most useful for learning when it clears away the friction that's orthogonal to what you're trying to understand, so you can spend more time on the parts that actually matter.

If you're learning something technical and avoiding AI out of principle, I'd encourage you to reconsider. The question isn't whether to use it — it's *how*. Use it to go deeper, not to go around.

---

## Code

The full implementation is available at: [github.com/mattcecil/ml_models](https://github.com/mattcecil/ml_models)

Key files:
- `transformer.ipynb` — model, training loop, BPE, inference
- `training_log.txt` — run-by-run results with epoch-level detail
- `SGDtransformer.ipynb` — earlier Transformer with `loss.backward()` + manual per-parameter SGD
- `linear_regression.ipynb` — gradient descent from scratch, no autograd

---

*Written March 2026. Model trained on an NVIDIA GPU via Lightning AI studio.*

---

**Part 2** covers what happened next: training on the full Works of Shakespeare, a data volume ablation that validated the predictions above, and implementing LoRA from scratch. [Read it here.](blog_post_2.md)
