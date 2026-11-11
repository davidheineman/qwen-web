"""Build src/rl/english-words.txt, the vocabulary behind the RL page's "is English" reward.

Words = GPT-2 tokens that are a whole lowercase word (so: reasonably common) and that are either in
Webster's Second International (/usr/share/dict/web2 on macOS/BSD; 1934, copyright lapsed) or an
inflection of a word that is (cats, running, happily, ...).

    python3 scripts/make_english_words.py path/to/gpt2/tokenizer.json
"""

import json
import re
import sys
from pathlib import Path

tok_path = sys.argv[1]
dict_path = sys.argv[2] if len(sys.argv) > 2 else "/usr/share/dict/web2"
out = Path(__file__).resolve().parent.parent / "src" / "rl" / "english-words.txt"

webster = {w.strip().lower() for w in open(dict_path) if w.strip()}
vocab = json.load(open(tok_path))["model"]["vocab"]

SUFFIXES = ["s", "es", "ed", "d", "ing", "ly", "er", "est", "ies", "ied", "ier", "iest", "ily", "'s"]


def stems(w):
    yield w
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 2:
            base = w[: -len(suf)]
            yield base
            if suf.startswith("i"):
                yield base + "y"  # happiest -> happy
            if suf in ("ed", "ing", "er", "est") and len(base) > 2 and base[-1] == base[-2]:
                yield base[:-1]  # running -> run
            if suf in ("ed", "ing", "er", "est"):
                yield base + "e"  # making -> make


words = set()
for t in vocab:
    if not t.startswith("Ġ"):  # GPT-2's leading-space marker: the token starts a word
        continue
    w = t[1:].lower()
    if not re.fullmatch(r"[a-z]+", w):
        continue
    if len(w) == 1 and w not in ("a", "i"):
        continue
    if any(s in webster for s in stems(w)):
        words.add(w)

out.write_text("\n".join(sorted(words)) + "\n")
print(f"{len(words)} words -> {out} ({out.stat().st_size // 1024} KB)")
