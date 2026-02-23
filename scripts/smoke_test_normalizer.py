from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "normalizer"))

from normalizer.hebrew_tts_normalizer import (
    TTSNormalizeOptions,
    load_word_replacements,
    normalize_tts_text,
)

texts = [
    "אני אוהבת קפה ורוצה ללכת לספר",
    "אני אוהב קפה ורוצה ללכת לספר",
    "אני אוהבת קפה ורוצה ללכת לקרוא ספר",
]

opts = TTSNormalizeOptions(
    apply_word_replacements=True,
    expand_numbers=True,
    keep_punctuations=True,
    attach_punctuations_to_token=True,
    stt_compat_mode=False,
    remove_parentheses=False,
)
repl = load_word_replacements()

for i, text in enumerate(texts, 1):
    out = normalize_tts_text(text, options=opts, word_replacements=repl)
    print(f"{i}. IN : {text}")
    print(f"   OUT: {out}")
