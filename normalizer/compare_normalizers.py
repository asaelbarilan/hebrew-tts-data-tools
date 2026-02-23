# -*- coding: utf-8 -*-

import sys
from pathlib import Path

from f5_tts.train.datasets.heb_norm.hebrew_text_normalizer import normalize_text
from f5_tts.train.datasets.heb_norm.hebrew_tts_normalizer import TTSNormalizeOptions, load_word_replacements_tsv, normalize_tts_text


def _utf8_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    _utf8_stdout()

    # Load optional replacement dictionary (used by both demos below)
    reps_path = Path(__file__).resolve().parent / "ex_reps_all.tsv"
    word_replacements = load_word_replacements_tsv(reps_path)

    examples = [
        "משנת 1997 כולם עושים בדיקות DNA. בדיקה אחת עולה 1542 שקל. זה משתלם אם אתם 3 אחים.",
        "ב-27/01/2026 בשעה 14:30 שילמתי 15% ו-1542₪.",
    ]

    for raw_text in examples:
        print("=" * 80)
        print("RAW:")
        print(raw_text)
        print()

        # STT normalizer (aggressive: converts numbers, splits on / and -, punctuation becomes separate token)
        stt = normalize_text(
            raw_text,
            split_markers=["/", "-"],
            remove_parenthesis=True,
            remove_brackets=True,
            keep_punctuations=True,
            word_replacements=word_replacements,
        )
        print("STT normalizer (hebrew_text_normalizer.normalize_text):")
        print(stt)
        print()

        # TTS normalizer (conservative defaults)
        tts_default = normalize_tts_text(raw_text, options=TTSNormalizeOptions(), word_replacements=word_replacements)
        print("TTS normalizer (conservative default):")
        print(tts_default)
        print()

        # TTS normalizer with optional expansions enabled
        tts_expanding = normalize_tts_text(
            raw_text,
            options=TTSNormalizeOptions(
                apply_word_replacements=True,
                expand_numbers=True,
                keep_punctuations=True,
                attach_punctuations_to_token=True,
            ),
            word_replacements=word_replacements,
        )
        print("TTS normalizer (with replacements + number expansion ON):")
        print(tts_expanding)
        print()

