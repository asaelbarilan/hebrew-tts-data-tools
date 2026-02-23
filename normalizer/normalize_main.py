import csv
import json
import sys
from pathlib import Path
from f5_tts.train.datasets.heb_norm.hebrew_text_normalizer import normalize_text
from f5_tts.train.datasets.heb_norm.hebrew_tts_normalizer import TTSNormalizeOptions, normalize_tts_text

if __name__ == "__main__":
    # Ensure Hebrew prints correctly on Windows consoles.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    raw_texts = [
        'הרו"ח שלי עו"ד בצה"ל',
        "שלום—מה נשמע??",
        "10km ממני",
        "❤️🙂",
        "הטמפרטורה היא 20°C",
        'אני גר בת"א',
        "המחיר הוא 12.5$ (לא כולל מע\"מ).",
        "משנת 1997 כולם עושים בדיקות DNA. בדיקה אחת עולה 1542 שקל. זה משתלם אם אתם 3 אחים.",
        "ב-27/01/2026 בשעה 14:30 שילמתי 15% ו-1542₪.",
        "המחיר הוא 12.5$ (לא כולל מע\"מ).",
        "היא אמרה: \"זה עובד!!!\" ואז הלכה...",
        "זה (מיותר) וזה [גם] צריך להיעלם.",
        "יש לי 2 ילדים ו-3 ילדות.",
    ]
    language_code = "he"

    normalization_params = {
        "split_markers": ["/", "-"],
        "remove_parenthesis": True,
        "remove_brackets": True,
        "keep_punctuations": True,
    }

    # Read the word replacements dictionary.
    word_replacements = {}
    reps_path = Path(__file__).resolve().parent / "ex_reps_all.tsv"
    with reps_path.open("r", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file, delimiter="\t")
        for entry in reader:
            source = entry["SOURCE"]
            target = entry["TARGET"]
            word_replacements[source] = target
    normalization_params["word_replacements"] = word_replacements

    tts_options_conservative = TTSNormalizeOptions(
        apply_word_replacements=False,
        expand_numbers=False,
        keep_punctuations=True,
        attach_punctuations_to_token=True,
    )

    tts_options_expanding = TTSNormalizeOptions(
        apply_word_replacements=True,
        expand_numbers=True,
        keep_punctuations=True,
        attach_punctuations_to_token=True,
    )

    for i, raw_text in enumerate(raw_texts, start=1):
        stt_normalized = normalize_text(raw_text, **normalization_params)
        tts_normalized = normalize_tts_text(
            raw_text,
            options=tts_options_conservative,
            word_replacements=word_replacements,
        )
        tts_normalized_expanding = normalize_tts_text(
            raw_text,
            options=tts_options_expanding,
            word_replacements=word_replacements,
        )

        print("=" * 80)
        print(f"EXAMPLE {i}")
        print("RAW:")
        print(raw_text)
        print()

        print("STT normalize_text:")
        print(stt_normalized)
        print()

        print("TTS normalize_tts_text (conservative):")
        print(tts_normalized)
        print()

        print("TTS normalize_tts_text (replacements + number expansion ON):")
        print(tts_normalized_expanding)
    print()