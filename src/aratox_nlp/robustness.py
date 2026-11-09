"""Deterministic text perturbations for robustness stress tests.

Each perturbation keeps the meaning a human reader would take from the
comment. The labels stay the original AraTox labels.
"""

from __future__ import annotations

import random
import re

ARABIC_LETTER = re.compile(r"[ء-ي]")
DIACRITICS = ["َ", "ُ", "ِ", "ْ", "ّ"]
TATWEEL = "ـ"
EMOJI = ["😂", "😡", "🙄", "🔥", "👍"]
ENGLISH_FILLERS = ["bro", "lol", "seriously", "omg", "honestly"]

# Common chat transliteration (Arabizi). Approximate by design.
ARABIZI = {
    "ا": "a", "أ": "2", "إ": "2", "آ": "2a", "ء": "2", "ؤ": "2", "ئ": "2", "ب": "b", "ت": "t", "ث": "th",
    "ج": "j", "ح": "7", "خ": "5", "د": "d", "ذ": "th", "ر": "r", "ز": "z", "س": "s", "ش": "sh",
    "ص": "9", "ض": "9'", "ط": "6", "ظ": "6'", "ع": "3", "غ": "3'", "ف": "f", "ق": "8", "ك": "k",
    "ل": "l", "م": "m", "ن": "n", "ه": "h", "ة": "a", "و": "w", "ي": "y", "ى": "a",
}


def _rng(text, seed):
    return random.Random(f"{seed}:{text}")


def add_diacritics(text, seed=0):
    rng = _rng(text, seed)
    return "".join(ch + rng.choice(DIACRITICS) if ARABIC_LETTER.match(ch) and rng.random() < 0.5 else ch for ch in text)


def add_tatweel(text, seed=0):
    rng = _rng(text, seed)
    return "".join(ch + TATWEEL * 2 if ARABIC_LETTER.match(ch) and rng.random() < 0.25 else ch for ch in text)


def elongate_letters(text, seed=0):
    """Repeat one letter in about half of the words: حمار -> حماااار."""
    rng = _rng(text, seed)
    words = []
    for word in text.split(" "):
        positions = [i for i, ch in enumerate(word) if ARABIC_LETTER.match(ch)]
        if positions and rng.random() < 0.5:
            i = rng.choice(positions)
            word = word[: i + 1] + word[i] * 3 + word[i + 1 :]
        words.append(word)
    return " ".join(words)


def append_emoji(text, seed=0):
    rng = _rng(text, seed)
    return f"{text} {rng.choice(EMOJI)}{rng.choice(EMOJI)}"


def code_switch(text, seed=0):
    rng = _rng(text, seed)
    return f"{rng.choice(ENGLISH_FILLERS)} {text} {rng.choice(ENGLISH_FILLERS)}"


def arabizi(text, seed=0):
    return "".join(ARABIZI.get(ch, ch) for ch in text)


def typo_noise(text, seed=0):
    """Swap or drop one Arabic letter in about a third of the words."""
    rng = _rng(text, seed)
    words = []
    for word in text.split(" "):
        if len(word) > 3 and rng.random() < 0.33:
            i = rng.randrange(len(word) - 1)
            word = word[:i] + word[i + 1] + word[i] + word[i + 2 :] if rng.random() < 0.5 else word[:i] + word[i + 1 :]
        words.append(word)
    return " ".join(words)


PERTURBATIONS = {
    "Diacritics added": add_diacritics,
    "Tatweel (ـ) added": add_tatweel,
    "Elongated letters": elongate_letters,
    "Emoji appended": append_emoji,
    "English words mixed in": code_switch,
    "Typos (swap/drop)": typo_noise,
    "Arabizi (Latin script)": arabizi,
}
