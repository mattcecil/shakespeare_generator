"""
Creates train/val/test text files from the two Shakespeare datasets.

  tiny_train.txt / tiny_val.txt / tiny_test.txt
      Source : shakespeare/shakespeare.csv  (TinyShakespeare, 90/5/5)
      Matches the proportions used by load_data()

  full_train.txt / full_val.txt / full_test.txt
      Source : shakespeare_tinyshakespeare.txt  (complete works, 90/5/5)

Splits are snapped to the nearest speech boundary (blank line) so no speech
is cut in half.

Run from data/:
    python3 process_data/make_splits.py
"""

import re

DATA = '.'


def snap_to_boundary(text, pos):
    """Move pos to the nearest blank line (speech boundary)."""
    # Search up to 2000 chars in either direction for '\n\n'
    search_range = 2000
    lo = max(0, pos - search_range)
    hi = min(len(text), pos + search_range)
    region = text[lo:hi]
    best = None
    best_dist = float('inf')
    for m in re.finditer(r'\n\n', region):
        candidate = lo + m.end()
        dist = abs(candidate - pos)
        if dist < best_dist:
            best_dist = dist
            best = candidate
    return best if best is not None else pos


def split_and_write(source_path, out_prefix, ratios):
    with open(source_path) as f:
        text = f.read()

    n = len(text)
    r1, r2 = ratios  # e.g. (0.9, 0.05) → train=90%, val=5%, test=5%

    p1 = snap_to_boundary(text, int(n * r1))
    p2 = snap_to_boundary(text, int(n * (r1 + r2)))

    splits = {
        'train': text[:p1],
        'val':   text[p1:p2],
        'test':  text[p2:],
    }

    for split, content in splits.items():
        path = f'{out_prefix}_{split}.txt'
        with open(path, 'w') as f:
            f.write(content)
        pct = 100 * len(content) / n
        print(f'  {path}: {len(content):>10,} chars ({pct:.1f}%)')


print('=== TinyShakespeare  (90 / 5 / 5) ===')
split_and_write(
    f'{DATA}/shakespeare/shakespeare.csv',
    f'{DATA}/tiny',
    ratios=(0.90, 0.05),
)

# Post-split cleanup: remove the CSV header/wrapper that load_data never stripped.
# The raw CSV is one big quoted field:  text\n"<content>"
# After splitting: tiny_train.txt starts with 'text\n"', tiny_test.txt ends with '"'
print('\nCleaning CSV artifacts from tiny_*.txt files...')
for fname in [f'{DATA}/tiny_train.txt', f'{DATA}/tiny_val.txt', f'{DATA}/tiny_test.txt']:
    with open(fname) as f:
        content = f.read()
    before = len(content)
    # Strip CSV header from the very start of the first file
    content = re.sub(r'^text\n', '', content)
    # Remove all " characters — they are CSV escape artifacts (escaped "" → "),
    # not real dialogue content (Shakespeare's text uses no double-quotes)
    content = content.replace('"', '')
    with open(fname, 'w') as f:
        f.write(content)
    removed = before - len(content)
    if removed:
        print(f'  {fname}: removed {removed} chars of CSV markup')

print('\n=== Complete Works  (90 / 5 / 5) ===')
split_and_write(
    f'{DATA}/shakespeare_tinyshakespeare.txt',
    f'{DATA}/full',
    ratios=(0.90, 0.05),
)
