"""Approximate normalization for Fake.br synthetic texts (not the original corpus pipeline)."""
import re
import unicodedata

# ponytail: original Fake.br preprocessing code is unavailable; this is only a
# sensitivity analysis. Prefer raw real + raw synthetic when source becomes available.
STOPWORDS = frozenset("a o as os um uma uns umas de da do das dos em no na nos nas e ou que por para com sem ao aos a sua seu suas seus se nao este esta estes estas foi sao ser ter ha mais muito como ja tambem mas ate entre sobre".split())


def normalize_fakebr(text):
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    words = re.findall(r"[a-z0-9]+", text)
    return " ".join(w for w in words if w not in STOPWORDS)
