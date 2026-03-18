"""
Build two experiment corpora:

  full_train_1mil  — same size as tiny_train (~1M chars), sourced from
                     full_train_minus_tiny_val with all tiny_train content
                     removed. Tests: does a different corpus of the same size
                     improve over tiny_train?

  tiny_train_t8    — the t8-edition passages that correspond to tiny_train's
                     scenes. Tests: does the edition (TinyShakespeare vs t8)
                     matter at all?

Run from data/:
    python3 process_data/make_experiments.py
"""

import re
from collections import defaultdict

DATA          = "."
SHINGLE       = 8
STEP          = 4
MAX_GAP       = 30
CHAR_PAD      = 300
MIN_RUN_WORDS = SHINGLE


# ── shared helpers (same as deduplicate.py) ───────────────────────────────────

def tokenize(text):
    return [(m.group().lower(), m.start()) for m in re.finditer(r'[a-zA-Z]+', text)]


def build_shingle_index(tokens, W):
    idx = defaultdict(list)
    words = [w for w, _ in tokens]
    for i in range(len(words) - W):
        idx[tuple(words[i:i+W])].append(i)
    return idx


def find_runs(tv_tokens, ft_index, W, step, max_gap):
    tv_words = [w for w, _ in tv_tokens]
    runs = []
    i = 0
    while i <= len(tv_words) - W:
        shingle = tuple(tv_words[i:i+W])
        hits = ft_index.get(shingle)
        if not hits:
            i += step
            continue
        ft_wstart = hits[0]
        tv_run_start = i
        tv_run_end = i + W
        ft_run_end = ft_wstart + W
        tv_pos = i + step
        ft_pos = ft_wstart + step
        while tv_pos <= len(tv_words) - W:
            next_shingle = tuple(tv_words[tv_pos:tv_pos+W])
            next_hits = ft_index.get(next_shingle)
            if not next_hits:
                tv_pos += step
                ft_pos += step
                if tv_pos - tv_run_end > max_gap:
                    break
                continue
            best = min(next_hits, key=lambda h: abs(h - ft_pos))
            if abs(best - ft_pos) > max_gap * 3:
                break
            ft_pos = best + step
            tv_run_end = tv_pos + W
            ft_run_end = best + W
            tv_pos += step
        if tv_run_end - tv_run_start >= MIN_RUN_WORDS:
            runs.append((tv_run_start, tv_run_end, ft_wstart, ft_run_end))
        i = tv_run_end
    return runs


def runs_to_ft_char_spans(runs, ft_tokens, char_pad, ft_len):
    ft_char_starts = [c for _, c in ft_tokens]
    spans = []
    for _, _, fw_start, fw_end in runs:
        cs = max(0, ft_char_starts[fw_start] - char_pad)
        ce_idx = min(fw_end, len(ft_char_starts) - 1)
        ce = min(ft_len, ft_char_starts[ce_idx] + char_pad + 200)
        spans.append((cs, ce))
    return spans


def merge_spans(spans):
    if not spans:
        return []
    spans = sorted(spans)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(s) for s in merged]


def remove_spans(text, spans):
    result, prev = [], 0
    for s, e in spans:
        result.append(text[prev:s])
        prev = e
    result.append(text[prev:])
    return ''.join(result)


def extract_spans(text, spans):
    """Extract and concatenate the given char spans from text."""
    return '\n\n'.join(text[s:e].strip() for s, e in spans)


def fuzzy_match_rate(a_text, b_text, window=15):
    def words(t): return [m.group().lower() for m in re.finditer(r'[a-zA-Z]+', t)]
    aw, bw = words(a_text), words(b_text)
    shingles = set(tuple(bw[i:i+window]) for i in range(len(bw) - window))
    hits = sum(1 for i in range(0, len(aw) - window, window)
               if tuple(aw[i:i+window]) in shingles)
    total = len(range(0, len(aw) - window, window))
    return hits, total


# ── load files ────────────────────────────────────────────────────────────────

print("Reading files...")
with open(f"{DATA}/tiny_train.txt")               as f: tiny_train            = f.read()
with open(f"{DATA}/full_train_minus_tiny_val.txt") as f: full_train_minus_tiny_val = f.read()
with open(f"{DATA}/full_train.txt")               as f: full_train            = f.read()

TARGET_SIZE = len(tiny_train)
print(f"Target size: {TARGET_SIZE:,} chars (= tiny_train)")


# ── full_train_1mil: full_train_minus_tiny_val minus tiny_train content ───────

print("\n── Building full_train_1mil ──")
print("Tokenizing...")
tt_tokens   = tokenize(tiny_train)
fmtv_tokens = tokenize(full_train_minus_tiny_val)

print("Building shingle index over full_train_minus_tiny_val...")
fmtv_index = build_shingle_index(fmtv_tokens, SHINGLE)

print("Finding tiny_train runs in full_train_minus_tiny_val...")
runs = find_runs(tt_tokens, fmtv_index, SHINGLE, STEP, MAX_GAP)
print(f"  {len(runs)} runs")

spans = merge_spans(runs_to_ft_char_spans(runs, fmtv_tokens, CHAR_PAD, len(full_train_minus_tiny_val)))
removed = sum(e - s for s, e in spans)
print(f"  removing {removed:,} chars ({100*removed/len(full_train_minus_tiny_val):.1f}%)")

deduped = remove_spans(full_train_minus_tiny_val, spans)
print(f"  deduped corpus: {len(deduped):,} chars available")

# Snap to speech boundary (blank line) near TARGET_SIZE
snap = deduped.rfind('\n\n', 0, TARGET_SIZE)
full_train_1mil = deduped[:snap if snap > 0 else TARGET_SIZE]

with open(f"{DATA}/full_train_1mil.txt", 'w') as f:
    f.write(full_train_1mil)
print(f"  full_train_1mil: {len(full_train_1mil):,} chars")

hits, total = fuzzy_match_rate(tiny_train, full_train_1mil)
print(f"  overlap check (tiny_train → full_train_1mil): {hits}/{total} = {100*hits/total:.1f}%")


# ── tiny_train_t8: extract t8 passages matching tiny_train ───────────────────

print("\n── Building tiny_train_t8 ──")
print("Tokenizing full_train (t8)...")
ft_tokens = tokenize(full_train)

print("Building shingle index over full_train...")
ft_index = build_shingle_index(ft_tokens, SHINGLE)

print("Finding tiny_train runs in full_train...")
runs_t8 = find_runs(tt_tokens, ft_index, SHINGLE, STEP, MAX_GAP)
print(f"  {len(runs_t8)} runs")
for r in runs_t8:
    print(f"    tv words {r[0]:6}–{r[1]:6} ({r[1]-r[0]:5} words) → ft words {r[2]:6}–{r[3]:6}")

spans_t8 = merge_spans(runs_to_ft_char_spans(runs_t8, ft_tokens, CHAR_PAD, len(full_train)))
extracted = sum(e - s for s, e in spans_t8)
print(f"  extracting {extracted:,} chars from full_train")

tiny_train_t8 = extract_spans(full_train, spans_t8)
with open(f"{DATA}/tiny_train_t8.txt", 'w') as f:
    f.write(tiny_train_t8)
print(f"  tiny_train_t8: {len(tiny_train_t8):,} chars")

hits, total = fuzzy_match_rate(tiny_train, tiny_train_t8)
print(f"  overlap check (tiny_train → tiny_train_t8): {hits}/{total} = {100*hits/total:.1f}%")

print("\nDone.")
