# -*- coding: utf-8 -*-
"""
Created on Tue May 27 2025

@author: Ron Wein
"""
from typing import List, Dict
import re

from f5_tts.train.datasets.heb_norm.hebrew_spoken_form import get_spoken_form, __HEBREW_PREPOSITION_PREFIXES

# Taken from: https://en.wikipedia.org/wiki/Template:Punctuation_marks_in_Unicode
__CHAR_CONVERSION_DICT = {
    # Convert white spaces into blanks:
    "\u0001": "",
    "\u0007": "",
    "\t": " ",
    "\n": " ",
    # Remove non-standard blanks:
    "\u200a": "",
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\u200e": "",
    "\u200f": "",
    # Remove left-to-right and right-to-left markers:
    "\u202a": "",
    "\u202b": "",
    "\u202c": "",
    "\u202d": "",
    "\u202e": "",
    # Convert Unicode hyphens to a simple dash:
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u05be": "-",
    # Convert Unicode quotation marks to simple quote:
    "\u00ab": '"',
    "\u00bb": '"',
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u05f4": '"',
    # Convert Unicode apostrophes marks to simple apostrophe:
    "\u2018": "'",
    "\u2019": "'",
    "\u201b": "'",
    "\u05f3": "'",
}

__TOKEN_PREFIX_MARKS = [
    "...",
    "..",
    '"',
    "'",
    "(",
    "[",
    ",",
    ".",
    ":",
    ";",
    "?",
    "!",
    "…",
]
__TOKEN_SUFFIX_MARKS = [
    "...",
    "!!!",
    "..",
    "??",
    "?!",
    "!?",
    "!!",
    '"',
    ".",
    ",",
    ":",
    ";",
    "?",
    "!",
    ")",
    "]",
    "…",
]

__PUNCTUATION_MAP_BRACKETED = {
    ".": "[.]",
    "..": "[.]",
    ",": "[,]",
    ";": "[,]",
    ":": "[,]",
    "?": "[?]",
    "?!": "[?]",
    "!?": "[?]",
    "??": "[?]",
    "!": "[!]",
    "!!": "[!]",
    "!!!": "[!]",
    "…": "[.]",
}
__PUNCTUATION_MAP = {
    ".": ".",
    "..": ".",
    ",": ",",
    ";": ",",
    ":": ",",
    "?": "?",
    "?!": "?",
    "!?": "?",
    "??": "?",
    "!": "!",
    "!!": "!",
    "!!!": "!",
    "…": ".",
}

__HEBREW_MULTIPLY_WORD = "פי"
__HEBREW_CENTURY_WORD = "מאה"

__HEBREW_MASCULINE_PLURAL_SUFFIX = "ים"
__HEBREW_FEMININE_PLURAL_SUFFIX = "ות"
__HEBREW_THE_PREFIX = "ה"

# Taken from: https://safa-ivrit.org/irregulars/pluralml.php
__HEBREW_EX_FEMININE_PLURALS = set(
    [
        "אבנים",
        "ביצים",
        "גפנים",
        "דבורים",
        "דבלים",
        "דרכים",
        "יונים",
        "כינים",
        "כבשים",
        "מחטים",
        "מילים",
        "נשים",
        "עדשים",
        "עיזים",
        "ערים",
        "פילגשים",
        "פנינים",
        "פעמים",
        "ציפורים",
        "צפרדעים",
        "שיבולים",
        "שנים",
        "שקמים",
        "תאנים",
        "תולעים",
    ]
)

# Taken from: https://www.safa-ivrit.org/irregulars/pluralfm.php
__HEBREW_EX_MASCULINE_PLURALS = set(
    [
        "אבות",
        "אולמות",
        "אוצרות",
        "אורות",
        "אותות",
        "אילנות",
        "אסונות",
        "ארונות",
        "אריות",
        "ארמונות",
        "בורות",
        "ביזיונות",
        "גבולות",
        "גגות",
        "גייסות",
        "גיליונות",
        'דו"חות',
        "דורות",
        "דיברות",
        "וילונות",
        "זיכרונות",
        "זנבות",
        "חלומות",
        "חלונות",
        "חלזונות",
        "חסרונות",
        "חשבונות",
        "חשדות",
        "חששות",
        "יינות",
        "יתרונות",
        "כוחות",
        "כינורות",
        "כיסאות",
        "לוחות",
        "לילות",
        "מוחות",
        "מוסדות",
        "מזלגות",
        "מזלות",
        "מחוזות",
        "מחנות",
        "מטבעות",
        "מכרות",
        "מלונות",
        "מסעות",
        "מעונות",
        "מעיינות",
        "מעמדות",
        "מקומות",
        "מקורות",
        "משקאות",
        "נהרות",
        "ניירות",
        "נרות",
        "סודות",
        "סולמות",
        "ספקות",
        "עופות",
        "עורות",
        "עפרונות",
        "עקרונות",
        "פיקדונות",
        "פירות",
        "פתרונות",
        "צבאות",
        "צינורות",
        "צרורות",
        "קולות",
        "קירות",
        "קצוות",
        "קרבות",
        "קרונות",
        "רגשות",
        "רחובות",
        "ריאיונות",
        "רעיונות",
        "רצונות",
        "שבועות",
        "שדות",
        "שולחנות",
        "שופרות",
        "שטרות",
        "שיטפונות",
        "שמות",
    ]
)


def normalize_text(
    text: str,
    remove_parenthesis: bool,
    remove_brackets: bool,
    keep_punctuations: bool,
    split_markers: List[str],
    word_replacements: Dict[str, str],
) -> str:
    # First perform character conversion to get rid of special Unicode characters.
    proc_text = ""
    for ch in text:
        if ch in __CHAR_CONVERSION_DICT:
            # Convert the character.
            target_ch = __CHAR_CONVERSION_DICT[ch]
            if len(target_ch) > 0:
                proc_text += target_ch

        else:
            # Convert the currect character as is.
            proc_text += ch

    # Remove portions of the text between parenthesis and brackets, if required.
    if remove_parenthesis:
        proc_text = re.sub(r"\(.*?\)", "", proc_text)

    if remove_brackets:
        proc_text = re.sub(r"\[.*?\]", "", proc_text)

    # Remove redundant consecutive blanks.
    while proc_text.find("  ") > 0:
        proc_text = proc_text.replace("  ", " ")

    # Split the text into tokens and normalize each one separately.
    punctuation_marks = set([".", ",", ":", ";", "!", "?", '"', "…"])

    in_tokens = proc_text.split(" ")
    num_in_tokens = len(in_tokens)

    out_tokens = []
    for ind, token in enumerate(in_tokens):
        # Ignore empty tokens, caused by multiple consecutive blanks.
        if len(token) == 0:
            continue

        # Try to determine the gender of the current token.
        # This is important only if the current token is converted as a number.
        gender = "f"
        if ind > 0 and in_tokens[ind - 1] == __HEBREW_MULTIPLY_WORD:
            # The correct for is having a masculine number after "pi".
            gender = "m"

        elif ind > 0 and in_tokens[ind - 1].endswith(__HEBREW_CENTURY_WORD):
            # Use the orinal feminine form for the century.
            gender = "f0"

        elif ind + 1 != num_in_tokens:
            # Check the next word, if one is available, and determine its gender
            # based on the plural suffixes.
            next_word = in_tokens[ind + 1]
            while len(next_word) > 1 and next_word[-1] in punctuation_marks:
                next_word = next_word[0:-1]

            next_word_sin_heb_the_prefix = (
                next_word[len(__HEBREW_THE_PREFIX) :]
                if next_word.startswith(__HEBREW_THE_PREFIX)
                else None
            )
            if next_word.endswith(__HEBREW_MASCULINE_PLURAL_SUFFIX):
                # The next word looks like a masculine plural form, but check ofr exceptions.
                if (next_word in __HEBREW_EX_FEMININE_PLURALS) or (
                    next_word_sin_heb_the_prefix
                    and next_word_sin_heb_the_prefix in __HEBREW_EX_FEMININE_PLURALS
                ):
                    gender = "f"
                else:
                    gender = "m"

            elif next_word.endswith(__HEBREW_FEMININE_PLURAL_SUFFIX):
                # The next word looks like a feminine plural form, but check ofr exceptions.
                if (next_word in __HEBREW_EX_MASCULINE_PLURALS) or (
                    next_word_sin_heb_the_prefix
                    and next_word_sin_heb_the_prefix in __HEBREW_EX_MASCULINE_PLURALS
                ):
                    gender = "m"
                else:
                    gender = "f"

        # Convert the token.
        out_token = _convert_token(
            token, keep_punctuations, gender, split_markers, word_replacements
        )

        if out_token:
            out_tokens.append(out_token)

    # Create the output text.
    return " ".join(out_tokens)


def _convert_token(
    token: str,
    keep_punct: bool,
    gender: str,
    split_markers: List[str],
    word_replacements: Dict[str, str],
) -> str:
    # First check if the token can be replaced as is.
    if token in word_replacements:
        return word_replacements[token]

    clean_token = token

    # Remove redudant characters that may appear as token prefixes.
    check_prefix = True
    while check_prefix:
        check_prefix = False
        for prefix in __TOKEN_PREFIX_MARKS:
            if clean_token.startswith(prefix) and clean_token != prefix:
                clean_token = clean_token[len(prefix) :]
                check_prefix = True
                break

    # Remove redudant characters that may appear as token suffixes.
    norm_punct_mark = None

    check_suffix = True
    while check_suffix:
        check_suffix = False
        for suffix in __TOKEN_SUFFIX_MARKS:
            if clean_token.endswith(suffix):
                if not norm_punct_mark and suffix in __PUNCTUATION_MAP:
                    norm_punct_mark = __PUNCTUATION_MAP[suffix]

                clean_token = clean_token[0 : -len(suffix)]
                check_suffix = True
                break

    # Check if the token can be replaced after it cleaning.
    target_token = word_replacements.get(clean_token, None)
    if target_token:
        if keep_punct and norm_punct_mark:
            return target_token + " " + norm_punct_mark
        else:
            return target_token

    # Check if the token can be converted into a spoken form of a numeral.
    spoken_form = get_spoken_form(clean_token, gender)
    if spoken_form:
        if keep_punct and norm_punct_mark:
            spoken_form += " " + norm_punct_mark

        return spoken_form

    # Check if the token contains slashes.
    for marker in split_markers:
        if clean_token.find(marker) >= 0:
            sub_forms = []
            success = True
            for sub_token in clean_token.split(marker):
                sub_form = _convert_token(sub_token, False, "f", [], {})
                if sub_form:
                    sub_forms.append(sub_form)
                else:
                    success = False
                    break

            if success:
                spoken_form = " ".join(sub_forms)

                if keep_punct and norm_punct_mark:
                    spoken_form += " " + norm_punct_mark

                return spoken_form

    """
    if clean_token.find('/') >= 0:
        sub_forms = []
        for sub_token in clean_token.split('/'):
            sub_form = _convert_token(sub_token, False, 'f')
            if sub_form:
                sub_forms.append(sub_form)
        spoken_form = ' '.join(sub_forms)
        
        if keep_punct and norm_punct_mark:
            spoken_form += ' ' + norm_punct_mark

        return spoken_form

    # Check if the token contains dashes.
    if clean_token.find('-') >= 0:
        sub_forms = []
        for sub_token in clean_token.split('-'):
            sub_form = _convert_token(sub_token, False, 'f')
            if sub_form:
                sub_forms.append(sub_form)
        spoken_form = ' '.join(sub_forms)
        
        if keep_punct and norm_punct_mark:
            spoken_form += ' ' + norm_punct_mark

        return spoken_form
    """

    # Check if the token contains a quotation mark after a preposinal prefix.
    # If so, remove this quotation mark.
    quote_pos = clean_token.find('"')
    if quote_pos > 0:
        is_plural = clean_token.endswith(
            __HEBREW_MASCULINE_PLURAL_SUFFIX
        ) or clean_token.endswith(__HEBREW_FEMININE_PLURAL_SUFFIX)

        if (not is_plural and quote_pos != len(clean_token) - 2) or (
            is_plural and quote_pos != len(clean_token) - 4
        ):
            token_prefix = clean_token[0:quote_pos]
            if (token_prefix + "-") in __HEBREW_PREPOSITION_PREFIXES:
                clean_token = token_prefix + clean_token[quote_pos + 1 :]

    # Otherwise, return the clean token as is.
    if keep_punct and norm_punct_mark:
        clean_token += " " + norm_punct_mark

    return clean_token
