# -*- coding: utf-8 -*-
"""
Hebrew TTS-oriented text normalizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Set
import csv
import re

from hebrew_spoken_form import get_spoken_form

# --- Constants & Sets ---

__HEBREW_MULTIPLY_WORD = "פי"
__HEBREW_CENTURY_WORD = "מאה"
__HEBREW_MASCULINE_PLURAL_SUFFIX = "ים"
__HEBREW_FEMININE_PLURAL_SUFFIX = "ות"
__HEBREW_THE_PREFIX = "ה"

# Taken from: https://safa-ivrit.org/irregulars/pluralml.php
__HEBREW_EX_FEMININE_PLURALS = {
    "אבנים", "ביצים", "גפנים", "דבורים", "דבלים", "דרכים", "יונים", "כינים",
    "כבשים", "מחטים", "מילים", "נשים", "עדשים", "עיזים", "ערים", "פילגשים",
    "פנינים", "פעמים", "ציפורים", "צפרדעים", "שיבולים", "שנים", "שקמים",
    "תאנים", "תולעים"
}

# Taken from: https://www.safa-ivrit.org/irregulars/pluralfm.php
__HEBREW_EX_MASCULINE_PLURALS = {
    "אבות", "אולמות", "אוצרות", "אורות", "אותות", "אילנות", "אסונות",
    "ארונות", "אריות", "ארמונות", "בורות", "ביזיונות", "גבולות", "גגות",
    "גייסות", "גיליונות", 'דו"חות', "דורות", "דיברות", "וילונות", "זיכרונות",
    "זנבות", "חלומות", "חלונות", "חלזונות", "חסרונות", "חשבונות", "חשדות",
    "חששות", "יינות", "יתרונות", "כוחות", "כינורות", "כיסאות", "כסאות", "לוחות",
    "לילות", "מוחות", "מוסדות", "מזלגות", "מזלות", "מחוזות", "מחנות",
    "מטבעות", "מכרות", "מלונות", "מסעות", "מעונות", "מעיינות", "מעמדות",
    "מקומות", "מקורות", "משקאות", "נהרות", "ניירות", "נרות", "סודות",
    "סולמות", "ספקות", "עופות", "עורות", "עפרונות", "עקרונות", "פיקדונות",
    "פירות", "פתרונות", "צבאות", "צינורות", "צרורות", "קולות", "קירות",
    "קצוות", "קרבות", "קרונות", "רגשות", "רחובות", "ריאיונות", "רעיונות",
    "רצונות", "שבועות", "שדות", "שולחנות", "שופרות", "שטרות", "שיטפונות",
    "שמות"
}

__HEBREW_KNOWN_MASCULINE = {
    "קילוגרם", "קילומטר", "מטר", "סנטימטר", "מילימטר", "שקל", "דולר", "אירו", "אחוז",
    "יום", "חודש", "שבוע"
}

_CHAR_CONVERSION_DICT = {
    # Convert white spaces into blanks:
    "\u0001": "", "\u0007": "", "\t": " ", "\n": " ",
    # Remove non-standard blanks:
    "\u200a": "", "\u200b": "", "\u200c": "", "\u200d": "", "\u200e": "", "\u200f": "",
    # Remove left-to-right and right-to-left markers:
    "\u202a": "", "\u202b": "", "\u202c": "", "\u202d": "", "\u202e": "",
    # Convert Unicode hyphens to a simple dash:
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u2015": "-", "\u05be": "-",
    # Convert Unicode quotation marks to simple quote:
    "\u00ab": '"', "\u00bb": '"', "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u201f": '"', "\u05f4": '"',
    # Convert Unicode apostrophes marks to simple apostrophe:
    "\u2018": "'", "\u2019": "'", "\u201b": "'", "\u05f3": "'",
    # Remove Backslashes
    "\\": "",
}

_SUFFIX_PUNCT_MAP = {
    ".": ".", "..": ".", "...": ".", ",": ",", ";": ",", ":": ",",
    "?": "?", "?!": "?", "!?": "?", "??": "?", "!": "!", "!!": "!",
    "!!!": "!", "…": ".",
}

_DROP_PREFIX_MARKS = ('"', "(", "[", "{")
_DROP_SUFFIX_MARKS = ('"', ")", "]", "}")

_BUILTIN_ABBREVIATIONS: Dict[str, str] = {
    'רו"ח': "רואה חשבון",
    'עו"ד': "עורך דין",
    'ד"ר': "דוקטור",
    'פרופ\'': "פרופסור",
    'מס\'': "מספר",
    'רח\'': "רחוב",
    'בע"מ': "בעמ",
    'ת"א': "תל אביב",
    'מע"מ': "מעמ",
}

# Map unit symbol to (Hebrew Name, Gender)
_UNIT_MAP: Dict[str, Tuple[str, str]] = {
    "km": ("קילומטר", "m"),
    "m": ("מטר", "m"),
    "cm": ("סנטימטר", "m"),
    "mm": ("מילימטר", "m"),
    "kg": ("קילוגרם", "m"),
    "g": ("גרם", "m"),
    "ml": ("מיליליטר", "m"),
    "l": ("ליטר", "m"),
    "L": ("ליטר", "m"),
    "GB": ("גיגה בייט", "m"),
    "Mb": ("מגה ביט", "m"),
    "MB": ("מגה בייט", "m"),
    "%": ("אחוז", "m"),
}

_EMOJI_RE = re.compile(
    "["  # noqa: W605
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F780-\U0001F7FF"  # geometric extended
    "\U0001F800-\U0001F8FF"  # arrows C
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA00-\U0001FAFF"  # symbols & pictographs extended-A
    "\U00002700-\U000027BF"  # dingbats
    "\U00002600-\U000026FF"  # misc symbols
    "]+"
)

# Date regexes
# Short: 27/01 or 27-01 (require / or -)
_DATE_SHORT_RE = re.compile(r"^(\d{1,2})([/-])(\d{1,2})$")
# Long: 27/01/2026 or 27.01.2026 (allow .)
_DATE_LONG_RE = re.compile(r"^(\d{1,2})([./-])(\d{1,2})\2(\d{2,4})$")

_MONTH_ORDINAL_MAP = {
    1: "לראשון", 2: "לשני", 3: "לשלישי", 4: "לרביעי", 5: "לחמישי",
    6: "לשישי", 7: "לשביעי", 8: "לשמיני", 9: "לתשיעי", 10: "לעשירי",
    11: "לאחד עשר", 12: "לשנים עשר"
}

_NUM_UNIT_RE = re.compile(r"^([+-]?\d[\d,]*(?:\.\d+)?)(°[CF]|[A-Za-z%]{1,4})$")
_PHONE_RE = re.compile(r"^0\d{1,2}-?\d{7}$")

@dataclass(frozen=True)
class TTSNormalizeOptions:
    # cleanup
    normalize_unicode: bool = True
    remove_parentheses: bool = True
    remove_brackets: bool = True
    collapse_whitespace: bool = True

    # punctuation
    keep_punctuations: bool = True
    attach_punctuations_to_token: bool = True
    
    # special handling
    strip_parentheses: bool = True # If remove_parentheses=False, do we still strip '(' and ')' chars?

    # text rewriting (off by default for safety)
    apply_word_replacements: bool = False
    expand_numbers: bool = False

    # tts-specific expansions
    expand_abbreviations: bool = True
    translate_units: bool = True
    remove_emojis: bool = True
    split_dashes_between_words: bool = True

    # splitting (rarely needed; leave conservative)
    split_markers: Tuple[str, ...] = ("/", "-")

    # compatibility mode
    stt_compat_mode: bool = False


def load_word_replacements_tsv(path: str | Path) -> Dict[str, str]:
    path = Path(path)
    repl: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            src = (row.get("SOURCE") or "").strip()
            tgt = (row.get("TARGET") or "").strip()
            if not src or not tgt:
                continue
            repl[src] = tgt
    return repl


def _apply_unicode_cleanup(text: str) -> str:
    out_chars: List[str] = []
    for ch in text:
        out_chars.append(_CHAR_CONVERSION_DICT.get(ch, ch))
    return "".join(out_chars)


def _strip_outer_marks(token: str) -> str:
    # Remove wrapper punctuation like quotes/brackets; keep the core content.
    while token.startswith(_DROP_PREFIX_MARKS):
        token = token[1:]
    while token.endswith(_DROP_SUFFIX_MARKS):
        token = token[:-1]
    return token


def _remove_emojis(text: str) -> str:
    return _EMOJI_RE.sub("", text)


def _split_suffix_punct(token: str) -> Tuple[str, str]:
    token = token.strip()
    if not token:
        return "", ""
    for punct in sorted(_SUFFIX_PUNCT_MAP.keys(), key=len, reverse=True):
        if token.endswith(punct) and len(token) > len(punct):
            core = token[: -len(punct)]
            return core, _SUFFIX_PUNCT_MAP[punct]
    return token, ""


def _normalize_abbrev_key(token: str) -> str:
    return token.replace("״", '"').replace("׳", "'")


def _expand_abbreviation(token: str, repl: Dict[str, str]) -> Optional[str]:
    tok = _normalize_abbrev_key(token)
    if tok in repl: return repl[tok]
    if tok in _BUILTIN_ABBREVIATIONS: return _BUILTIN_ABBREVIATIONS[tok]
    if len(tok) >= 2 and tok[0] in ("ה", "ו", "ב", "ל", "כ", "מ"):
        prefix = tok[0]
        rest = tok[1:]
        rest = _normalize_abbrev_key(rest)
        if rest in repl: return prefix + repl[rest]
        if rest in _BUILTIN_ABBREVIATIONS: return prefix + _BUILTIN_ABBREVIATIONS[rest]
    return None


def _translate_unit_suffix(token: str) -> Optional[str]:
    m = _NUM_UNIT_RE.match(token)
    if not m: return None
    num_str = m.group(1)
    unit = m.group(2)
    
    # Identify unit name and gender
    target_unit = ""
    gender = "m" # Default

    if unit in ("°C", "°c"):
        target_unit = "מעלות"
        gender = "f"
    elif unit in ("°F", "°f"):
        target_unit = "מעלות פרנהייט"
        gender = "f"
    else:
        # Check map
        found = False
        if unit in _UNIT_MAP:
            target_unit, gender = _UNIT_MAP[unit]
            found = True
        else:
            unit_lower = unit.lower()
            if unit_lower in _UNIT_MAP:
                target_unit, gender = _UNIT_MAP[unit_lower]
                found = True
        
        if not found:
            return None

    # Expand number part
    # For units like KG (Masculine), "2.5" should be "SNEIEM va-hetzi".
    # For degrees (Feminine), "2.5" should be "SHTEIEM va-hetzi".
    # We pass 'mn' or 'fn' if we want construct state? No, "2.5 Kg" is absolute number + unit noun.
    # But "2 Kg" -> "Shnei Kilogram". This is construct.
    # "2.5 Kg" -> "Shnayim va-hetzi kilogram" (Absolute).
    # Heuristic: If integer, use construct? If float, use absolute?
    # Usually "Shnei Kilogram" (2 kg). "Shnayim va-hetzi kilogram" (2.5 kg).
    # So if it has a fraction, it behaves like absolute.
    # If integer, behaves like construct.
    
    is_integer = '.' not in num_str
    
    eff_gender = gender
    if is_integer:
         if gender == 'm': eff_gender = 'mn'
         if gender == 'f': eff_gender = 'fn'
    
    expanded_num = get_spoken_form(num_str, eff_gender)
    if not expanded_num:
        expanded_num = num_str
        
    return f"{expanded_num} {target_unit}"


def _expand_phone(token: str) -> Optional[str]:
    if _PHONE_RE.match(token):
        digits = []
        for ch in token:
            if ch.isdigit():
                d_spoken = get_spoken_form(ch, "f")
                if d_spoken:
                    digits.append(d_spoken)
        return " ".join(digits)
    return None


def _maybe_expand_number(token: str, gender: str = "f") -> Optional[str]:
    return get_spoken_form(token, gender)


def _expand_date(token: str) -> Optional[str]:
    # Check long regex first
    m = _DATE_LONG_RE.match(token)
    year_str = None
    if m:
        day = int(m.group(1))
        month = int(m.group(3))
        year_str = m.group(4)
    else:
        # Check short regex
        m = _DATE_SHORT_RE.match(token)
        if m:
            day = int(m.group(1))
            month = int(m.group(3))
        else:
            return None
    
    if day < 1 or day > 31 or month < 1 or month > 12:
        return None
        
    # Day
    if day == 1:
        day_str = "הראשון"
    else:
        # Date numbers are usually masculine "Ehad", "Shnayim" (absolute)? 
        # "Shisha be-mai". "Esrim ve-shiva be-mai".
        # User requested "Esrim ve-sheva" (F).
        # We will use 'f' as per feedback.
        day_str = get_spoken_form(str(day), "f")
    
    # Month
    month_str = _MONTH_ORDINAL_MAP.get(month)
    if not month_str:
        return None
        
    # Year
    year_spoken = ""
    if year_str:
        year_spoken = get_spoken_form(year_str, "f")
        
    parts = [day_str, month_str]
    if year_spoken:
        parts.append(year_spoken)
        
    return " ".join(parts)


def load_word_replacements(path: Path | None = None) -> Dict[str, str]:
    path = path or Path(__file__).resolve().parent / "ex_reps_all.tsv"
    return load_word_replacements_tsv(path)

def normalize_tts_text(
    text: str,
    *,
    options: TTSNormalizeOptions = TTSNormalizeOptions(),
    word_replacements: Optional[Dict[str, str]] = None,
) -> str:
    proc = text
    if options.normalize_unicode:
        proc = _apply_unicode_cleanup(proc)
    if options.remove_emojis:
        proc = _remove_emojis(proc)
    
    # Strip parentheses/brackets content if requested
    if options.remove_parentheses:
        proc = re.sub(r"\(.*?\)", "", proc)
    if options.remove_brackets:
        proc = re.sub(r"\[.*?\]", "", proc)
        
    if options.split_dashes_between_words:
        # Only split if BOTH sides are NOT digits (avoid splitting 050-123 or 2026-01)
        # Regex: Lookbehind for non-digit, Lookahead for non-digit
        # Note: \D matches any non-digit character (including space, punctuation, letters)
        # We want to split "Word-Word" but not "0-1" or "0-Word"?
        # Actually usually "0-Word" should split.
        # Only "Digit-Digit" should stay.
        def repl(m):
            # Check if both sides are digits
            if m.group(1).isdigit() and m.group(2).isdigit():
                return m.group(0) # Keep hyphen
            return f"{m.group(1)} {m.group(2)}" # Replace hyphen with space
        
        proc = re.sub(r"(\S)-(\S)", repl, proc)

    if options.collapse_whitespace:
        proc = re.sub(r"\s+", " ", proc).strip()

    if not proc:
        return ""

    repl = word_replacements or {}

    # STT Compat Mode (omitted for brevity, assume not used in this context)
    if options.stt_compat_mode:
        # ... logic ...
        pass

    # TTS Logic (Main)
    raw_tokens = proc.split(" ")
    out_tokens: List[str] = []
    
    punctuation_marks = set([".", ",", ":", ";", "!", "?", '"', "…", ")", "]", "}"])

    for ind, raw_tok in enumerate(raw_tokens):
        # 1. Strip outer marks (quotes, parens if enabled)
        # Note: If remove_parentheses=False, we still want to strip the CHARACTERS '(' and ')'
        # _strip_outer_marks uses _DROP_PREFIX_MARKS which includes '('.
        tok = _strip_outer_marks(raw_tok)
        if not tok: continue
        
        tok, suffix_punct = _split_suffix_punct(tok)
        tok = tok.strip()
        tok = _strip_outer_marks(tok)
        if not tok: continue

        tok_out = tok

        # Gender Detection
        gender = "f"
        prev_tok = out_tokens[-1] if out_tokens else ""
        if prev_tok == __HEBREW_MULTIPLY_WORD:
             gender = "m"
        elif prev_tok.endswith(__HEBREW_CENTURY_WORD):
             gender = "f0"
        
        if ind + 1 < len(raw_tokens):
            next_word = raw_tokens[ind + 1]
            while len(next_word) > 1 and next_word[-1] in punctuation_marks:
                next_word = next_word[:-1]
            next_word_no_the = next_word[len(__HEBREW_THE_PREFIX):] if next_word.startswith(__HEBREW_THE_PREFIX) else None
            
            if next_word in __HEBREW_KNOWN_MASCULINE:
                 gender = "mn"
            elif next_word.endswith(__HEBREW_MASCULINE_PLURAL_SUFFIX):
                 if (next_word in __HEBREW_EX_FEMININE_PLURALS) or (next_word_no_the and next_word_no_the in __HEBREW_EX_FEMININE_PLURALS):
                     gender = "fn"
                 else:
                     gender = "mn"
            elif next_word.endswith(__HEBREW_FEMININE_PLURAL_SUFFIX):
                 if (next_word in __HEBREW_EX_MASCULINE_PLURALS) or (next_word_no_the and next_word_no_the in __HEBREW_EX_MASCULINE_PLURALS):
                     gender = "mn"
                 else:
                     gender = "fn"
        
        # 1. Abbreviations
        if options.expand_abbreviations:
            expanded = _expand_abbreviation(tok_out, repl if options.apply_word_replacements else {})
            if expanded: tok_out = expanded
        elif options.apply_word_replacements and tok_out in repl:
            tok_out = repl[tok_out]

        # 2. Units (handles "2.5kg", "-5C")
        # NOTE: This consumes the number!
        if options.translate_units:
            translated = _translate_unit_suffix(tok_out)
            if translated:
                tok_out = translated
                # Skip number expansion for this token as it's already done
                if options.keep_punctuations and suffix_punct:
                    if options.attach_punctuations_to_token:
                        tok_out = f"{tok_out}{suffix_punct}"
                    else:
                        out_tokens.append(tok_out)
                        out_tokens.append(suffix_punct)
                        continue
                out_tokens.append(tok_out)
                continue

        if options.expand_numbers:
            # 3. Dates
            date_expanded = _expand_date(tok_out)
            if date_expanded:
                tok_out = date_expanded
            else:
                # 4. Phone numbers
                phone_expanded = _expand_phone(tok_out)
                if phone_expanded:
                    tok_out = phone_expanded
                else:
                    # 5. Numbers
                    spoken = _maybe_expand_number(tok_out, gender)
                    if spoken:
                        tok_out = spoken

        # 6. Split markers
        if options.expand_numbers and options.split_markers:
            for marker in options.split_markers:
                if marker in tok_out:
                    parts = tok_out.split(marker)
                    spoken_parts: List[str] = []
                    ok = True
                    for p in parts:
                        p = p.strip()
                        if not p:
                             ok = False
                             break
                        s = _maybe_expand_number(p, "f")
                        if not s:
                             ok = False
                             break
                        spoken_parts.append(s)
                    if ok:
                        tok_out = " ".join(spoken_parts)
                    break

        if options.keep_punctuations and suffix_punct:
            if options.attach_punctuations_to_token:
                tok_out = f"{tok_out}{suffix_punct}"
            else:
                out_tokens.append(tok_out)
                out_tokens.append(suffix_punct)
                continue

        out_tokens.append(tok_out)

    out = " ".join(out_tokens)
    if options.keep_punctuations:
        out = re.sub(r"\s+([.,?!])", r"\1", out)
    return out.strip()
