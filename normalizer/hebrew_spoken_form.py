# -*- coding: utf-8 -*-
"""
Created on Sun May 25 2025

@author: Ron Wein
"""
from typing import Tuple, Optional
import re

__TIME_REGEX = r'^([\d]\d)\:(\d\d)$'

__SEPARATED_NUMBER_REGEX = r'^(\-)?[1-9]\d{0,2}(\,\d{3})*(\.\d+)?$'
__PLAIN_NUMBER_REGEX = r'^(\-)?\d+(\.\d+)?$'

__HEBREW_TO_20_FEM = [
'אפס', 'אחת', 'שתיים', 'שלוש', 'ארבע', 'חמש', 'שש', 'שבע', 'שמונה', 'תשע',
'עשר', 'אחת עשרה', 'שתים עשרה', 'שלוש עשרה', 'ארבע עשרה', 'חמש עשרה',
'שש עשרה', 'שבע עשרה', 'שמונה עשרה', 'תשע עשרה'
]

__HEBREW_TO_20_MASC = [
'אפס', 'אחד', 'שניים', 'שלושה', 'ארבעה', 'חמישה', 'שישה', 'שבעה', 'שמונה', 'תשעה',
'עשרה', 'אחד עשר', 'שנים עשר', 'שלושה עשר', 'ארבעה עשר', 'חמישה עשר',
'שישה עשר', 'שבעה עשר', 'שמונה עשר', 'תשעה עשר'
]

__HEBREW_TENS = [
'', '','עשרים', 'שלושים', 'ארבעים', 'חמישים', 'שישים', 'שבעים', 'שמונים', 'תשעים'
]

__HEBREW_HUNDREDS = [
'', 'מאה', 'מאתיים', 'שלוש מאות', 'ארבע מאות', 'חמש מאות',
'שש מאות', 'שבע מאות', 'שמונה מאות', 'תשע מאות'
]

__HEBREW_THOUSANDS = [
'', 'אלף','אלפיים', 'שלושת אלפים', 'ארבעת אלפים', 'חמשת אלפים',
'ששת אלפים', 'שבעת אלפים', 'שמונת אלפים', 'תשעת אלפים', 'עשרת אלפים'
]

__HEBREW_MILLION = 'מיליון'
__HEBREW_BILLION = 'מיליארד'

__HEBREW_PREPOSITION_PREFIXES = [
'מ-', 'ש-', 'ה-', 'ו-', 'כ-', 'ל-', 'ב-',
 'מה-', 'מכ-', 'שמ-', 'שה-', 'שכ-', 'של-', 'שב-',
'ומ-', 'וש-', 'וה-', 'וכ-', 'ול-', 'וב-', 'כש-', 'וכש-', 'כשה-', 'כשמ-', 'כשב-', 'כשבכ-'
 'לכ-', 'בכ-', 'כשבכ-', 'כשלכ-', 'שכ-', 'של-', 'שב-',
]

__HEBREW_UNITS_DICT = {
        '%': ('אחוז', 'm'),
                       '°': ('מעלות', 'f'),
                       '₪': ('שקלים', 'm'),
                       '$': ('דולר', 'm'),
                       '€': ('אירו', 'm')
}

__HEBREW_AND = 'ו'
__HEBREW_MINUS = 'מינוס'
__HEBREW_DECIAL_POINT = 'נקודה'

__HEBREW_TIME_AM = 'בבוקר'
__HEBREW_TIME_PM = 'אחר הצוהריים'
__HEBREW_TIME_HALF = 'וחצי'
__HEBREW_TIME_QUARTER = 'ורבע'

__HEBREW_ORDINALS_FEM = [
'', 'ראשונה', 'שנייה', 'שלישית', 'רביעית', 'חמישית',
'שישית', 'שביעית', 'שמינית', 'תשיעית', 'עשירית'
]

__HEBREW_ORDINALS_MASC = [
'', 'ראשון', 'שני', 'שלישי', 'רביעי', 'חמישי',
'שישי', 'שביעי', 'שמיני', 'תשיעי', 'עשירי'
]

def get_spoken_form(token: str, gender: str) -> Optional[str]:
    """
    Try converting the input token and obtain a string representation of spoken form.
    
    @param token The input token.
    @param gender The gender of the token ('m' or 'f' for cardinal numbers,
                  or 'm0' and 'f0' for ordinal numbers).
    @return The spoken form of the token (a string containing several words)
            in case of success, or None if the token is not a valid numeral.  
    """
    if not token or len(token) == 0:
        return None

    # Check if the token starts with a prepositional prefix.
    for prefix in __HEBREW_PREPOSITION_PREFIXES:
        if token.startswith(prefix):
            suffix = token[len(prefix): ]
            if len(suffix) > 0:
                # Check if the suffix after the dash can be coverted.
                # If so, concatenate the prefix (without the dash) to the spoken form.
                suffix_spoken_form = get_spoken_form(suffix, gender)
                
                if suffix_spoken_form:
                    return prefix[0: -1] + suffix_spoken_form
                
    # Check if the token looks like a time expression.
    if re.match(__TIME_REGEX, token):
        # Check that we have a valid time expression.
        hours = int(token[0: 2])
        minutes = int(token[3: ])
        
        if hours >= 24 or minutes >= 60:
            return None
        
        # Make sure the hours are normalized to the range of 1--12.
        pm_time = False
        if hours == 0:
            hours = 12
        elif hours > 12:
            hours -= 12
            pm_time = True
            
        time_str = __HEBREW_TO_20_FEM[hours]
        
        # Add the minutes, if necessary.
        add_am_pm = True
        if minutes == 15:
            time_str += ' ' + __HEBREW_TIME_QUARTER
        elif minutes == 30:
            time_str += ' ' + __HEBREW_TIME_HALF
        elif minutes > 0:
            if minutes % 10 == 0:
                time_str += ' ' + __HEBREW_AND + _convert_integer(minutes, 'm')
            else:
                time_str += ' ' + _convert_integer(minutes, 'f')
            add_am_pm = False
            
        if add_am_pm:
            if pm_time:
                time_str += ' ' + __HEBREW_TIME_PM
            else:
                time_str += ' ' + __HEBREW_TIME_AM
                
        return time_str

    # Check for symbols that may appear as unit suffixes.
    last_char = token[-1]
    if last_char in __HEBREW_UNITS_DICT:
        unit_str, unit_gender = __HEBREW_UNITS_DICT[last_char]
        
        # Try converting the prefix of the token.
        # In case of success, append the spoken form of the unit.
        spoken_num = _convert_number(token[0: -1], unit_gender)
        if spoken_num:
            return spoken_num + ' ' + unit_str
        
    # Check if the entire token is numeral.
    spoken_num = _convert_number(token, gender)
    if spoken_num:
        return spoken_num

    # If we reached here, the token is not a valid numeral form.
    return None

# Convert a non-negative integer value into its spoken form.
def _convert_integer(value: int, gender: str) -> str:
    spoken_form = ''

    # Check if the integer value is above 1 billion.
    if value >= 1_000_000_000:
        billions = value // 1_000_000_000
        if billions == 1:
            # Just use the plain form of "one billion".
            spoken_form = __HEBREW_BILLION

        else:
            # Obtain the spoken form of the number of billions.
            spoken_form = _convert_integer(billions, 'm') + \
                          ' ' + __HEBREW_BILLION

        # Continue with the remainder.
        value %= 1_000_000_000

        if value == 0:
            return spoken_form

    # Check if the (remaining) integer value is above 1 million.
    if value >= 1_000_000:
        if len(spoken_form) > 0 and spoken_form[-1] != ' ':
            spoken_form += ' '

        millions = value // 1_000_000
        if millions == 1:
            # Just use the plain form of "one million".
            spoken_form += __HEBREW_MILLION

        else:
            # Obtain the spoken form of the number of millions.
            spoken_form += _convert_integer(millions, 'm') + \
                           ' ' + __HEBREW_MILLION

        # Continue with the remainder.
        value %= 1_000_000

        if value == 0:
            return spoken_form

    # Check if the (remaining) integer value is above 1 thousand.
    if value >= 1000:
        if len(spoken_form) > 0 and spoken_form[-1] != ' ':
            spoken_form += ' '

        thousands = value // 1000
        if thousands <= 10:
            # Just use a form the thousands' table.
            spoken_form += __HEBREW_THOUSANDS[thousands]

        else:
            # Obtain the spoken form of the number of thousands.
            spoken_form += _convert_integer(thousands, 'm') + \
                           ' ' + __HEBREW_THOUSANDS[1]

        # Continue with the remainder.
        value %= 1000

        if value == 0:
            return spoken_form

    # If we reached here, the (remaining) numerical value is less than 1000.
    if len(spoken_form) > 0 and spoken_form[-1] != ' ':
           spoken_form += ' '

    add_and = False
    if value >= 100:
        hundreds = value // 100
        spoken_form += __HEBREW_HUNDREDS[hundreds]
        add_and = (hundreds == 1)
        
        # Continue with the remainder.
        value %= 100

        if value == 0:
            return spoken_form

    # If we reached here, the (remaining) numerical value is less than 100.
    if len(spoken_form) > 0:
        if spoken_form[-1] != ' ':
            spoken_form += ' '
        
        if add_and or (value < 20) or (value % 10 == 0):
            spoken_form += __HEBREW_AND
        
    if value < 20:
        # Special construct forms for 2
        if value == 2:
            if gender == 'mn': return 'שני'
            if gender == 'fn': return 'שתי'

        # Just take the spoken form the correct table, according to the gender.
        if value > 0 and value <= 10 and gender == 'm0':
            return spoken_form + __HEBREW_ORDINALS_MASC[value]            
        elif gender == 'm' or gender == 'mn':
            return spoken_form + __HEBREW_TO_20_MASC[value]
        if value > 0 and value <= 10 and gender == 'f0':
            return spoken_form + __HEBREW_ORDINALS_FEM[value]            
        else:
            return spoken_form + __HEBREW_TO_20_FEM[value]
            
    # If we reached here, the number is between 20 and 99.
    # Split it into tens and into units.
    tens = value // 10
    units = value % 10
    
    spoken_form += __HEBREW_TENS[tens]
    
    if units > 0:
        # Use the correct gender for the units.
        spoken_form += ' ' + __HEBREW_AND
        if gender == 'm' or gender == 'm0' or gender == 'mn':
            if units == 2 and gender == 'mn':
                spoken_form += 'שני'
            else:
                spoken_form += __HEBREW_TO_20_MASC[units]
        else:
            if units == 2 and gender == 'fn':
                spoken_form += 'שתי'
            else:
                spoken_form += __HEBREW_TO_20_FEM[units]

    return spoken_form

# Auxiliary function:
# Convert the digits after the decimal point into their spoken form.
def _convert_decimal_fraction(digits: str) -> str:
    # Special case for .5 -> "and a half"
    if digits == '5':
        return __HEBREW_TIME_HALF

    # Start with the decimal point.
    spoken_form = __HEBREW_DECIAL_POINT

    # If there are exactly two digits after the decimal point, read them as a two-digit number.
    if len(digits) == 2 and digits[0] != '0':
        value = int(digits)
        spoken_form += ' ' + _convert_integer(value, 'f')

        return spoken_form

    # Otherwise, just read the digits one by one.
    for dig in digits:
        value = int(dig)
        spoken_form += ' ' + __HEBREW_TO_20_FEM[value]

    return spoken_form

# Auxiliary function:
# Check if the given token is a valid numeral: either an integer number of a decimal number.
# In case of success, the function returns the integer part and a string
# containing the digits after the decimal point (this string is empty in case
# of an integer number).
# Otherwise, the function returns None.
def _is_numeral_token(token: str) -> Optional[Tuple[int, str]]:
    # Check if the token contains commas, acting thousand separators.
    proc_token = None
    if re.search(__SEPARATED_NUMBER_REGEX, token):
        # Remove all commas, acting as thousand separators.
        proc_token = token.replace(',', '')

    elif re.search(__PLAIN_NUMBER_REGEX, token):
        # There are no thousand separators: Just leave the number as is.
        proc_token = token

    # If we have not processed form of the token, it does not represent a number.
    if not proc_token:
        # Special case: check for phone numbers or hyphenated numbers that are NOT dates.
        # Simple phone regex: starts with 0, 9-10 digits, optional hyphen.
        # But here we only handle single token. "050-1234567" is one token.
        # We should NOT return it as a numeral tuple (int, frac) because that treats it as a single integer.
        return None

    # Check if we have a decimal point.
    point_pos = proc_token.find('.')

    if (point_pos < 0) or (point_pos + 1 == len(proc_token)):
        # We have an integer number.
        value = int(proc_token)
        return (value, '')

    # Split the token to its integer part and is fractional part.
    if point_pos == 0:
        int_value = 0

    else:
        int_value = int(proc_token[0: point_pos])

    frac_digits = proc_token[point_pos + 1: ]
    return (int_value, frac_digits)

# Auxiliary function:
# Try converting the given string into its spoken form, or return None if the
# string does not represent a valid number. 
def _convert_number(token: str, gender: str) -> Optional[str]:
    # Check if the token represents a valid number.
    res_pair = _is_numeral_token(token)
        
    if not res_pair:
        return None

    # Convert the integer part.
    int_value = res_pair[0]
    frac_part = res_pair[1]
    has_frac_part = (not frac_part is None) and (len(frac_part) > 0)

    if int_value < 0:
        # Negate the integer part, so that _convert_integer() always accepts a non-positive value.
        # Note that we always treat a negative number as feminine.
        spoken_form = __HEBREW_MINUS + ' ' + _convert_integer(-int_value, 'f')
        
    else:
        # The integer part is positive:
        # We generally use the provided gender (which defaults to 'f' for bare numbers,
        # but matches the unit gender if one was provided).
        spoken_form = _convert_integer(int_value, gender)

    # Append the fractional part, if needed.
    if has_frac_part:
        if int_value == 0 and frac_part == '5':
             return 'חצי'
        spoken_form += ' ' + _convert_decimal_fraction(frac_part)

    return spoken_form
