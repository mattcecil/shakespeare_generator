"""
Converts t8.shakespeare.txt to TinyShakespeare format:
  - All dialogue preserved as:  Speaker Name:\nDialogue text\n\nNext Speaker:\n...
  - Stage directions, act/scene headings, cast lists, play titles stripped
  - Speaker names: ALL CAPS for proper names, Title Case for roles/descriptions
  - Single blank line between speeches

Run from data/:
    python3 process_data/make_tinyshakespeare.py
"""

import re

# Words that indicate a role/description rather than a proper name.
# If any word in a speaker name is in this set, the name is Title-cased.
# Otherwise it stays ALL CAPS (proper personal name: HAMLET, BRUTUS, etc.)
_ROLE_WORDS = {
    # Ordinals
    'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh',
    # Quantifiers / articles
    'all', 'both', 'another', 'other', 'several', 'a', 'an',
    # Aristocratic / military titles
    'lord', 'lady', 'duke', 'duchess', 'count', 'countess', 'earl', 'baron',
    'sir', 'dame', 'captain', 'general', 'lieutenant', 'sergeant', 'ensign',
    'admiral', 'mayor', 'sheriff', 'constable', 'provost', 'marshal',
    # Religious
    'friar', 'abbess', 'abbot', 'bishop', 'cardinal', 'priest', 'archbishop',
    'monk', 'nun', 'deacon',
    # Civic / occupational roles
    'citizen', 'soldier', 'officer', 'servant', 'clown', 'nurse', 'page',
    'herald', 'messenger', 'senator', 'doctor', 'keeper', 'attendant',
    'widow', 'gaoler', 'jailer', 'porter', 'gardener', 'apothecary',
    'player', 'musician', 'gentleman', 'gentlewoman', 'ambassador',
    'aedile', 'tribune', 'patrician', 'plebeian', 'watchman', 'tailor',
    'tinker', 'weaver', 'joiner', 'carpenter', 'cobbler', 'shepherd',
    'huntsman', 'hostess', 'host', 'vintner', 'drawer', 'carrier',
    'chamberlain', 'steward', 'bailiff', 'master', 'boatswain', 'sailor',
    'pirate', 'guard', 'prisoner', 'beggar', 'courtezan', 'courtesan',
    'pedant', 'pedlar', 'scrivener', 'notary', 'clerk', 'jailor',
    # Gender / age
    'man', 'woman', 'boy', 'girl', 'child', 'maiden', 'maid', 'youth',
    # Supernatural
    'ghost', 'spirit', 'fairy', 'nymph', 'goddess', 'witch', 'shadow',
    # Collective
    'chorus', 'people', 'crowd',
    # Nationality / origin adjectives
    'roman', 'french', 'english', 'scotch', 'welsh', 'venetian', 'greek',
    'trojan', 'spanish', 'italian', 'danish', 'scotch', 'irish',
}


def format_speaker(name_upper: str) -> str:
    """
    Convert an ALL-CAPS speaker name to TinyShakespeare casing:
      - Title Case if the name contains any role/descriptor word (First Citizen,
        Messenger, Duke Of Florence, All, ...)
      - ALL CAPS if it's a proper personal name (HAMLET, BRUTUS, MENENIUS, ...)
    """
    word_set = {w.lower() for w in name_upper.split()}
    if word_set & _ROLE_WORDS:
        return name_upper.title()
    return name_upper  # keep ALL CAPS

INPUT  = 't8.shakespeare.txt'
OUTPUT = 'shakespeare_tinyshakespeare.txt'

STAGE_DIRECTION_RE = re.compile(
    r'^(?:Enter|Exit|Exeunt|Re-enter|Flourish|Alarum|Alarums|Retreat|March|'
    r'Noise|Thunder|Music|Sennet|Tucket|Shout|Cry|Sound|'
    r'They |He |She |Both |All )',
    re.IGNORECASE
)

# Patterns that look like scene headings even when indented
HEADING_RE = re.compile(
    r'^(?:SCENE\s+\d+|SCENE\s+[IVX]+|ACT\s+[IVX]+|PROLOGUE|EPILOGUE|'
    r'INDUCTION|SCENE\s*:)',
    re.IGNORECASE
)

with open(INPUT) as f:
    text = f.read()

# ── Phase 1: bulk regex removals ──────────────────────────────────────────────

# Remove <<...>> copyright blocks
text = re.sub(r'<<.*?>>', '', text, flags=re.DOTALL)

# Remove Dramatis Personae / DRAMATIS PERSONAE sections.
# Each section runs until the next SCENE: or ACT heading.
text = re.sub(r'(?is)dramatis personae\.?\s*.*?(?=SCENE\s*:|ACT\s+[IVX])', '', text)

# ── Phase 2: line-by-line conversion ──────────────────────────────────────────

# Speaker pattern: 1-4 leading spaces, ALL-CAPS name, period, optional dialogue
SPEAKER_RE = re.compile(r'^ {1,4}([A-Z][A-Z0-9 \'\-]+)\. *(.*)')

# Abbreviated speaker pattern for Romeo and Juliet (Title-case short names, e.g. Rom. Jul.)
ABBREV_SPEAKER_RE = re.compile(r'^ {1,4}([A-Z][a-z]+)\. *(.*)')
CAP_WIFE_RE       = re.compile(r'^ {1,4}(Cap\. Wife)\. *(.*)')  # Lady Capulet

# Romeo and Juliet abbreviated speaker → final formatted name
ROMEO_JULIET_ABBREVS = {
    'Rom':      'ROMEO',          'Jul':      'JULIET',
    'Mer':      'MERCUTIO',       'Ben':      'BENVOLIO',
    'Samp':     'SAMPSON',        'Greg':     'GREGORY',
    'Tyb':      'TYBALT',         'Cap':      'CAPULET',
    'Cap. Wife':'Lady Capulet',   'Mon':      'MONTAGUE',
    'Par':      'PARIS',          'Laur':     'Friar Laurence',
    'John':     'Friar John',     'Abr':      'ABRAM',
    'Bal':      'BALTHASAR',      'Pet':      'PETER',
    'Apoth':    'Apothecary',     'Chor':     'Chorus',
    'Serv':     'Servant',        'Nurse':    'Nurse',
    'Peter':    'PETER',          'Prince':   'PRINCE',
    'Boy':      'Boy',            'Man':      'Man',
    'Officer':  'Officer',        'Page':     'Page',
    'Citizen':  'Citizen',        'Citizens': 'Citizens',
    'Friar':    'Friar Laurence', 'Chorus':   'Chorus',
    'Wife':     'Lady Capulet',   'Lady':     'Lady Capulet',
    'Mother':   'Lady Capulet',   'Father':   'CAPULET',
    'Fellow':   'Fellow',
}

output      = []
in_speech   = False
play_stats  = {}   # play_title → speech count
current_play = 'UNKNOWN'

for line in text.splitlines():
    stripped = line.strip()

    # Skip blank lines here; we'll control spacing ourselves
    if not stripped:
        continue

    # Skip non-indented lines: play titles, ACT/SCENE headings, year lines,
    # "by William Shakespeare", THE END, boilerplate, etc.
    if not line.startswith(' '):
        # Detect play title lines (all-caps, multi-word, no punctuation)
        if stripped not in ('THE END', 'DRAMATIS PERSONAE', 'FINIS') and (
                re.match(r'^THE [A-Z ]+$', stripped) or re.match(r'^[A-Z ]{15,}$', stripped)):
            current_play = stripped
            play_stats.setdefault(current_play, 0)
        in_speech = False
        continue

    # Skip standalone stage direction lines (any indent level)
    if STAGE_DIRECTION_RE.match(stripped):
        continue

    # Skip lines that are entirely a [stage direction]
    if re.match(r'^\[.*\]$', stripped):
        continue

    # Skip indented scene/act headings (can appear with heavy indentation)
    if HEADING_RE.match(stripped):
        in_speech = False
        continue

    # Skip ALL-CAPS-only lines (song titles, split stage direction fragments,
    # character labels like "JOAN LA PUCELLE" on its own line)
    if stripped == stripped.upper() and re.search(r'[A-Z]{2}', stripped):
        # But don't skip if it matched speaker pattern above (already handled)
        in_speech = False
        continue

    # ── Speaker line ──────────────────────────────────────────────────────────
    m = SPEAKER_RE.match(line)
    if m:
        speaker  = format_speaker(m.group(1).strip())
        dialogue = m.group(2).strip()

        # Strip inline [stage directions] (e.g. [Aside], [To BERTRAM])
        dialogue = re.sub(r'\s*\[[^\]]*\]\s*', ' ', dialogue).strip()
        # Strip trailing Exit / Exeunt annotation
        dialogue = re.sub(r'\s+(?:Exit|Exeunt)\b.*$', '', dialogue, flags=re.IGNORECASE).strip()

        if output:
            output.append('')          # blank line between speeches
        output.append(f'{speaker}:')
        if dialogue:
            output.append(dialogue)
        play_stats[current_play] = play_stats.get(current_play, 0) + 1
        in_speech = True
        continue

    # ── Abbreviated speaker (Romeo and Juliet title-case names) ───────────────
    am = CAP_WIFE_RE.match(line) or ABBREV_SPEAKER_RE.match(line)
    if am:
        abbrev = am.group(1)
        if abbrev in ROMEO_JULIET_ABBREVS:
            speaker  = ROMEO_JULIET_ABBREVS[abbrev]
            dialogue = am.group(2).strip()
            dialogue = re.sub(r'\s*\[[^\]]*\]\s*', ' ', dialogue).strip()
            dialogue = re.sub(r'\s+(?:Exit|Exeunt)\b.*$', '', dialogue, flags=re.IGNORECASE).strip()
            if output:
                output.append('')
            output.append(f'{speaker}:')
            if dialogue:
                output.append(dialogue)
            play_stats[current_play] = play_stats.get(current_play, 0) + 1
            in_speech = True
            continue

    # ── Continuation line (any indented non-speaker line while in a speech) ───
    if line.startswith(' ') and in_speech:
        cont = stripped
        cont = re.sub(r'\s*\[[^\]]*\]\s*', ' ', cont).strip()
        cont = re.sub(r'\s+(?:Exit|Exeunt)\b.*$', '', cont, flags=re.IGNORECASE).strip()
        if cont:
            output.append(cont)
        continue

    # Anything else (Dramatis Personae remnant, SCENE: location, etc.) – drop
    in_speech = False

# ── Phase 3: collapse consecutive blank lines to one ─────────────────────────
result = []
prev_blank = False
for line in output:
    if line == '':
        if not prev_blank:
            result.append(line)
        prev_blank = True
    else:
        prev_blank = False
        result.append(line)

# ── Phase 4: strip remaining formatting artifacts ────────────────────────────
cleaned = []
for line in result:
    if line == '':
        cleaned.append(line)
        continue
    # Remove orphaned [ to end-of-line (split stage direction, opening bracket only)
    line = re.sub(r'\s*\[[^\]]*$', '', line).strip()
    # Remove start-of-line to ] (split stage direction, closing bracket only)
    line = re.sub(r'^[^\[]*\]\s*', '', line).strip()
    # Remove backtick and } formatting artifacts
    line = line.replace('`', '').replace('}', '').strip()
    if line:
        cleaned.append(line)

# Re-collapse blank lines after cleanup
result = []
prev_blank = False
for line in cleaned:
    if line == '':
        if not prev_blank:
            result.append(line)
        prev_blank = True
    else:
        prev_blank = False
        result.append(line)

final = '\n'.join(result).strip() + '\n'

with open(OUTPUT, 'w') as f:
    f.write(final)

in_chars  = len(open(INPUT).read())
out_lines = final.count('\n')
out_chars = len(final)
print(f"Input : {in_chars:,} chars")
print(f"Output: {out_chars:,} chars, {out_lines:,} lines")
print(f"Removed {in_chars - out_chars:,} chars ({(in_chars-out_chars)/in_chars*100:.1f}%)")

# ── Sanity checks ─────────────────────────────────────────────────────────────
print(f"\n── Sanity checks ──")
print(f"Plays detected: {len(play_stats)}")
total_speeches = sum(play_stats.values())
print(f"Total speeches: {total_speeches:,}")
plays_sorted = sorted(play_stats.items(), key=lambda x: -x[1])
for play, count in plays_sorted:
    print(f"  {count:5d} speeches  {play}")

# Flag plays with suspiciously few speeches
thin = [p for p, c in play_stats.items() if c < 50]
if thin:
    print(f"\nWARNING: plays with <50 speeches (may be incomplete):")
    for p in thin:
        print(f"  {p}: {play_stats[p]} speeches")
else:
    print("\nAll plays have >=50 speeches. OK.")

# Verify Romeo and Juliet is present
rj_key = next((k for k in play_stats if 'ROMEO' in k), None)
if rj_key:
    print(f"\nRomeo and Juliet: {play_stats[rj_key]} speeches. OK.")
else:
    print("\nERROR: Romeo and Juliet not found in output!")
