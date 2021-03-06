# coding: utf-8

"""
@File   : strings.py
@Author : garnet
@Time   : 2020/10/17 10:58
"""

import re
import string
import unicodedata

CJK_PUNCTUATION = '\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\u00b7\uff01\uff1f\uff61\u3002'


def keyword_match(s, keywords):
    r"""At least one keyword in keywords list is contained in input string.

    Args:
        :param s: string to match
        :param keywords: keywords list

    Returns:
        bool.
    """
    for key in keywords:
        if re.search(key, s):
            return True
    return False


def is_space_character(ch):
    r"""Whether input is a space character.
    """
    return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or unicodedata.category(ch) == 'Zs'


def is_punctuation_character(ch):
    r"""Whether input is a punctuation character.

    Including fullwidth(全角) and halfwidth(半角) punctuation.
    """
    code = ord(ch)
    return 33 <= code <= 47 or \
           58 <= code <= 64 or \
           91 <= code <= 96 or \
           123 <= code <= 126 or \
           unicodedata.category(ch).startswith('P')


def is_cjk_character(ch):
    r"""Whether input is a cjk character.

    See: https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    """
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or \
           0x3400 <= code <= 0x4DBF or \
           0x20000 <= code <= 0x2A6DF or \
           0x2A700 <= code <= 0x2B73F or \
           0x2B740 <= code <= 0x2B81F or \
           0x2B820 <= code <= 0x2CEAF or \
           0xF900 <= code <= 0xFAFF or \
           0x2F800 <= code <= 0x2FA1F


def is_control_character(ch):
    r"""Whether input is a control character.
    """
    return unicodedata.category(ch) in ('Cc', 'Cf')


def text_segment(text: str, max_len: int = -1, seps='\n', strips=None):
    text = text.strip().strip(strips)
    if not text:
        return []

    if seps and len(text) > max_len:
        sep = seps[0]
        pieces = text.split(sep)
        num_pieces = len(pieces)
        current, segments = '', []
        for i, p in enumerate(pieces):
            if current and p and len(current) + len(p) > max_len - 1:
                # cannot combine former text and current piece together
                segments.extend(text_segment(current, max_len, seps[1:], strips))
                current = ''

            current += (p + ('' if i + 1 == num_pieces else sep))

        if current:
            segments.extend(text_segment(current, max_len, seps[1:], strips))
        return segments
    else:
        return [text]
