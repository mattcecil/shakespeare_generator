"""
Build full_train_minus_tiny_val.txt by removing all tiny_val content from full_train.

Algorithm:
1. Use alpha-word tokenization (re.finditer) so char offsets are exact.
2. Build 8-word shingle index over full_train word tokens.
3. Slide an 8-word window over tiny_val; every hit gives a (tv_word, ft_word) anchor.
4. Group anchors into contiguous runs where both tv and ft word indices advance together.
5. For each run, expand the ft span by padding to catch partial-match edges.
6. Merge overlapping ft spans and delete them from full_train.

Run from data/:
    python3 process_data/deduplicate.py
"""

import re
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────────────────
SHINGLE        = 8       # words per shingle for initial matching
STEP           = 4       # stride when sliding shingle window over tiny_val
MAX_GAP        = 30      # max word gap before a run is considered broken
CHAR_PAD       = 300     # extra chars to remove either side of each matched span
MIN_RUN_WORDS  = SHINGLE # ignore isolated single-shingle hits shorter than this

DATA = "."   # relative to data/; run from there with: python3 process_data/deduplicate.py

# ── Helpers ───────────────────────────────────────────────────────────────────

def tokenize(text):
    """Return list of (word_lower, char_start) from contiguous alpha sequences."""
    return [(m.group().lower(), m.start()) for m in re.finditer(r'[a-zA-Z]+', text)]


def build_shingle_index(tokens, W):
    """Map W-word tuple → list of start word indices."""
    idx = defaultdict(list)
    words = [w for w, _ in tokens]
    for i in range(len(words) - W):
        idx[tuple(words[i:i+W])].append(i)
    return idx


def find_runs(tv_tokens, ft_index, W, step, max_gap):
    """
    Slide over tiny_val and collect (tv_wstart, tv_wend, ft_wstart, ft_wend) runs.
    A run is a sequence of consecutive shingle hits where both tv and ft word
    indices advance roughly in sync (gap < max_gap).
    """
    tv_words = [w for w, _ in tv_tokens]
    runs = []

    i = 0
    while i <= len(tv_words) - W:
        shingle = tuple(tv_words[i:i+W])
        hits = ft_index.get(shingle)

        if not hits:
            i += step
            continue

        # Start a new run from the first (closest to expected) hit
        ft_wstart = hits[0]
        tv_run_start = i
        ft_run_start = ft_wstart

        # Extend the run greedily
        tv_pos = i + step
        ft_pos = ft_wstart + step
        tv_run_end = i + W
        ft_run_end = ft_wstart + W

        while tv_pos <= len(tv_words) - W:
            next_shingle = tuple(tv_words[tv_pos:tv_pos+W])
            next_hits = ft_index.get(next_shingle)
            if not next_hits:
                tv_pos += step
                ft_pos += step
                if tv_pos - tv_run_end > max_gap:
                    break
                continue
            # Accept hit if ft index is close to where we expect it
            best = min(next_hits, key=lambda h: abs(h - ft_pos))
            if abs(best - ft_pos) > max_gap * 3:
                break  # jumped to a different part of the text
            ft_pos = best + step
            tv_run_end = tv_pos + W
            ft_run_end = best + W
            tv_pos += step

        if tv_run_end - tv_run_start >= MIN_RUN_WORDS:
            runs.append((tv_run_start, tv_run_end, ft_run_start, ft_run_end))

        i = tv_run_end  # skip past matched region in tiny_val

    return runs


def runs_to_char_spans(runs, ft_tokens, char_pad, ft_len):
    """Convert (ft_wstart, ft_wend) word-index spans to (char_start, char_end) spans."""
    ft_char_starts = [c for _, c in ft_tokens]
    spans = []
    for _, _, fw_start, fw_end in runs:
        cs = max(0, ft_char_starts[fw_start] - char_pad)
        ce_idx = min(fw_end, len(ft_char_starts) - 1)
        ce = min(ft_len, ft_char_starts[ce_idx] + char_pad + 200)
        spans.append((cs, ce))
    return spans


def merge_spans(spans):
    """Merge overlapping or adjacent char spans."""
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def remove_spans(text, spans):
    """Delete all (start, end) spans from text."""
    result = []
    prev = 0
    for s, e in spans:
        result.append(text[prev:s])
        prev = e
    result.append(text[prev:])
    return ''.join(result)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tiny_val_path  = f"{DATA}/tiny_val.txt"
    full_train_path = f"{DATA}/full_train.txt"
    out_path        = f"{DATA}/full_train_minus_tiny_val.txt"

    print("Reading files...")
    with open(tiny_val_path)   as f: tiny_val   = f.read()
    with open(full_train_path) as f: full_train = f.read()

    print("Tokenizing...")
    tv_tokens = tokenize(tiny_val)
    ft_tokens = tokenize(full_train)
    print(f"  tiny_val:  {len(tv_tokens):,} word tokens")
    print(f"  full_train: {len(ft_tokens):,} word tokens")

    print(f"Building {SHINGLE}-word shingle index over full_train...")
    ft_index = build_shingle_index(ft_tokens, SHINGLE)
    print(f"  {len(ft_index):,} unique shingles")

    print("Finding matching runs...")
    runs = find_runs(tv_tokens, ft_index, SHINGLE, STEP, MAX_GAP)
    print(f"  {len(runs)} runs found")
    for r in runs:
        tv_w = r[1] - r[0]
        ft_w = r[3] - r[2]
        tv_c_start = tv_tokens[r[0]][1]
        tv_c_end   = tv_tokens[min(r[1], len(tv_tokens)-1)][1]
        print(f"    tv words {r[0]:6}–{r[1]:6} ({tv_w:4} words) | "
              f"ft words {r[2]:6}–{r[3]:6} ({ft_w:4} words) | "
              f"tiny_val chars {tv_c_start}–{tv_c_end}")

    print("\nConverting to char spans with padding...")
    spans = runs_to_char_spans(runs, ft_tokens, CHAR_PAD, len(full_train))
    spans = merge_spans(spans)
    removed_chars = sum(e - s for s, e in spans)
    print(f"  {len(spans)} merged spans, {removed_chars:,} chars to remove "
          f"({100*removed_chars/len(full_train):.1f}% of full_train)")

    print(f"\nWriting {out_path}...")
    result = remove_spans(full_train, spans)
    with open(out_path, 'w') as f:
        f.write(result)
    print(f"  full_train_minus_tiny_val: {len(result):,} chars (was {len(full_train):,})")
    print("Done.")
