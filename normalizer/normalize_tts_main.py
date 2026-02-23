# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import sys
from pathlib import Path

from f5_tts.train.datasets.heb_norm.hebrew_tts_normalizer import (
    TTSNormalizeOptions,
    load_word_replacements,
    normalize_tts_text,
)


if __name__ == "__main__":
    # Ensure Hebrew prints correctly on Windows consoles.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    raw_text = "משנת 1997 כולם עושים בדיקות DNA. בדיקה אחת עולה 1542 שקל. זה משתלם אם אתם 3 אחים."

    word_replacements = load_word_replacements()

    options = TTSNormalizeOptions(
        # conservative defaults for TTS:
        apply_word_replacements=False,  # turn on only if your TSV matches your recordings
        expand_numbers=False,  # turn on only if speakers actually say expanded numbers
        keep_punctuations=True,
        attach_punctuations_to_token=True,
    )

    normalized = normalize_tts_text(
        raw_text, options=options, word_replacements=word_replacements
    )

    print(raw_text)
    print("-->")
    print(normalized)
