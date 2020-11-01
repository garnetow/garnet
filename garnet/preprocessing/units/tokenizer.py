# coding: utf-8

"""
@File   : tokenizer.py
@Author : garnet
@Time   : 2020/10/14 23:50
"""

import re
import typing
import unicodedata

from .base import StatefulUnit
from .vocabulary import Vocabulary, BertVocabulary, SOS, EOS, UNK, PAD, MASK, SEP, CLS
from ...utils.strings import CJK_PUNCTUATION
from ...utils.strings import is_cjk_character, is_punctuation_character, is_space_character, is_control_character


class Tokenizer(StatefulUnit):
    def __init__(self,
                 with_sos=True,
                 with_eos=True,
                 token_start=SOS,
                 token_end=EOS,
                 token_unknown=UNK,
                 token_pad=PAD,
                 token_mask=MASK,
                 max_length=None,
                 **kwargs):
        super(Tokenizer, self).__init__(**kwargs)
        self.with_sos = with_sos
        self.with_eos = with_eos
        self.max_length = max_length

        self.token_start = token_start
        self.token_end = token_end
        self.token_pad = token_pad
        self.token_unknown = token_unknown
        self.token_mask = token_mask

        self.vocab = None

    def token2id(self, token):
        return self.vocab[token]

    def id2token(self, index):
        return self.vocab.i2w(index)

    def tokens2ids(self, tokens):
        return [self.token2id(token) for token in tokens]

    def ids2tokens(self, indices):
        return [self.id2token(index) for index in indices]

    def token_pad(self, tokens, seq_length=None, padding='post'):
        if padding is None:
            return tokens

        if seq_length and len(tokens) < seq_length:
            if padding == 'post':
                tokens.extend([self.token2id(self.token_pad)] * (seq_length - len(tokens)))
            else:
                t = [self.token2id(self.token_pad)] * (seq_length - len(tokens))
                t.extend(tokens)
                tokens = t
        return tokens

    def truncate_pad(self, tokens, seq_length=None, truncate='post', padding='post'):
        r"""Truncate and pad token sequence to specified length.

        Argument:
            :param tokens: token sequence.
            :param seq_length: final token sequence length. `None` means do nothing with raw sequence.
            :param truncate: truncate position. Must be one of `'post'` and `'pre'`.
            :param padding: pad position. Must be one of `'post'`, `'pre'` or `None`, and `None` means do not pad.

        Return:
            Token list.
        """
        if seq_length:
            tokens = tokens[:seq_length] if truncate == 'post' else tokens[-seq_length:]
            if padding:
                tokens = self.token_pad(tokens, seq_length=seq_length, padding=padding)
        return tokens

    def _tokenize(self, text):
        return [t for t in text.split(' ') if t]

    def tokenize(self, text, max_length, truncate='post', padding='post'):
        r"""Tokenize text into token sequence, and option can be chosen whether sequence should be truncated.s

        Argument:
            :param text: text string.
            :param max_length: (int, optional, default: `None`) max length of final token sequence. `None` means
                no limits.
            :param truncate: truncate token sequence when length of sequence is greater than max threshold.
                `pre`, `post` and `None` is available.
            :param padding: pad position. Must be one of `'post'`, `'pre'` or `None`, and `None` means do not pad.
        """
        tokens = self._tokenize(text)

        max_length = max_length or self.max_length
        if max_length:
            residual_length = max_length - int(self.with_sos) - int(self.with_eos)
            tokens = self.truncate_pad(tokens, seq_length=residual_length, truncate=truncate, padding=padding)

        if self.with_sos:
            tokens.insert(0, self.token_start)
        if self.with_eos:
            tokens.append(self.token_end)

        return tokens

    def fit(self,
            vocab: Vocabulary = None,
            vocab_path: str = None,
            token_start: typing.Union[bool, str] = False,
            token_end: typing.Union[bool, str] = False,
            token_pad: typing.Union[bool, str] = False,
            token_unknown: typing.Union[bool, str] = False,
            token_mask: typing.Union[bool, str] = False,
            ignore_case=False,
            encoding='utf-8',
            special_tokens=None,
            **kwargs):
        if vocab is not None:
            self.vocab = vocab
        elif vocab_path is not None:
            self.vocab = Vocabulary(ignore_case=ignore_case,
                                    token_start=token_start,
                                    token_end=token_end,
                                    token_unknown=UNK,
                                    token_pad=PAD,
                                    token_mask=MASK,
                                    **kwargs)
            self.vocab.fit(vocab_path=vocab_path, encoding=encoding, special_tokens=special_tokens)
        else:
            raise ValueError("`vocab` and `vocab_path` can't be `None` as the same time.")

        self.fitted = True


class BertTokenizer(Tokenizer):
    def __init__(self,
                 vocab_path,
                 ignore_case=True,
                 token_start=CLS,
                 token_end=SEP,
                 encoding='utf-8',
                 max_length=None,
                 simplified=False,
                 start_tokens=None,
                 extended_tokens=None,
                 **kwargs):
        super(BertTokenizer, self).__init__(with_sos=True,
                                            with_eos=True,
                                            token_start=token_start,
                                            token_end=token_end,
                                            token_pad=PAD,
                                            token_unknown=UNK,
                                            token_mask=MASK,
                                            max_length=max_length,
                                            **kwargs)
        self.vocab = None
        self.ignore_case = ignore_case

        self.simplified_mapping = None

        self.fit(vocab_path,
                 ignore_case=ignore_case,
                 token_start=token_start,
                 token_end=token_end,
                 encoding=encoding,
                 simplified=simplified,
                 start_tokens=start_tokens,
                 extended_tokens=extended_tokens)

        self.token_start = self.vocab.start_token
        self.token_end = self.vocab.end_token
        self.token_pad = self.vocab.pad_token
        self.token_unknown = self.vocab.unknown_token
        self.token_mask = self.vocab.mask_token
        self.token_sep = SEP
        self.token_cls = CLS

        self.punctuation = r'+-/={(<[' + CJK_PUNCTUATION

    def fit(self,
            vocab_path,
            ignore_case=False,
            token_start=SOS,
            token_end=EOS,
            encoding='utf-8',
            simplified=False,
            start_tokens=None,
            extended_tokens=None):
        self.vocab = BertVocabulary(
            vocab_path,
            ignore_case=ignore_case,
            token_start=token_start,
            token_end=token_end,
            encoding=encoding,
            simplified=simplified,
            start_tokens=start_tokens,
            extended_tokens=extended_tokens,
        )
        self.simplified_mapping = self.vocab.simplified_mapping
        self.fitted = True

    def transform(self, first_text, second_text=None, max_length=None, first_length=None, second_length=None):
        r"""
        Argument:
            :param first_text: string of first sentence.
            :param second_text: string of second sentence. `None` is available.
            :param max_length: length of total sequence.
            :param first_length: length of tokens in first sentence.
            :param second_length: length of tokens in second sentence.

        Return:
            Tuple. First element is token ids sequence, second element is segmentation ids sequence.
        """
        first_tokens = self.tokenize(first_text)
        second_tokens = self.tokenize(second_text)[1:] if second_text is not None else None

        max_length = max_length or self.max_length
        if max_length:
            first_tokens, second_tokens = self.truncate_sequence(first_tokens,
                                                                 second_seq=second_tokens,
                                                                 max_length=max_length,
                                                                 pop_index=-2)

        first_token_ids = self.tokens2ids(first_tokens)
        if first_length is not None:
            first_token_ids = self.truncate_pad(first_token_ids,
                                                seq_length=first_length,
                                                truncate='post',
                                                padding='post')
        first_segment_ids = [0] * len(first_token_ids)

        if second_tokens:
            second_token_ids = self.tokens2ids(second_tokens)
            if second_length is not None:
                second_token_ids = self.truncate_pad(second_token_ids,
                                                     seq_length=second_length,
                                                     truncate='post',
                                                     padding='post')
            second_segment_ids = [1] * len(second_token_ids)

            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)
        return first_token_ids, first_segment_ids

    def reverse_transform(self, indices, tokens=None):
        tokens = tokens or self.ids2tokens(indices)
        tokens = [token for token in tokens if not self.vocab.is_special_token(token)]

        text = ""
        for i, token in enumerate(tokens):
            if token[:2] == '##':
                text += token[2:]
            elif len(token) == 1 and is_cjk_character(token):
                text += token
            elif len(token) == 1 and is_punctuation_character(token):
                text += token + ' '
            elif i > 0 and is_cjk_character(text[-1]):
                text += token
            else:
                text += ' ' + token

        text = re.sub(r' +', ' ', text)  # eliminate continuous space characters
        text = re.sub(r'\' (re|m|s|t|ve|d|ll) ', '\'\\1 ', text)
        punctuation_pattern = '|'.join([re.escape(p) for p in self.punctuation])
        punctuation_pattern = '(%s) ' % punctuation_pattern
        text = re.sub(punctuation_pattern, '\\1', text)
        text = re.sub(r'(\d\.) (\d)', '\\1\\2', text)
        return text.strip()

    def tokenize(self, text, max_length=None, **kwargs):
        tokens = self._tokenize(text)
        tokens.insert(0, self.token_start)  # insert start token
        tokens.append(self.token_end)  # insert end token

        max_length = max_length or self.max_length
        tokens, _ = self.truncate_sequence(tokens, max_length=max_length, pop_index=-2)
        return tokens

    def rematch(self, text, tokens):
        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self.ignore_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or is_control_character(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))
        text = normalized_text.lower()

        token_mapping, offset = [], 0
        for token in tokens:
            if self.vocab.is_special_token(token):
                token_mapping.append([])
            else:
                token = self.vocab.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping

    def match_tokenize(self, text, max_length=None):
        token = self.tokenize(text, max_length=max_length)
        mapping = self.rematch(text, token)
        return token, mapping

    @staticmethod
    def truncate_sequence(first_seq, second_seq=None, max_length=None, pop_index=-1):
        second_seq = second_seq or []
        while True:
            total_length = len(first_seq) + len(second_seq)
            if not max_length or total_length <= max_length:
                break
            elif len(first_seq) > len(second_seq):
                first_seq.pop(pop_index)
            else:
                second_seq.pop(pop_index)
        return first_seq, second_seq or None

    def _tokenize(self, text):
        if self.ignore_case:  # text clean process
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
            text = text.lower()

        spaced = ''
        for ch in text:
            if is_punctuation_character(ch) or is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif is_space_character(ch):
                spaced += ' '
            elif is_control_character(ch) or ord(ch) == 0 or ord(ch) == 0xfffd:
                continue
            else:
                spaced += ch  # number and alphabet letter will stick together

        tokens = []
        for word in spaced.strip().split():
            tokens.extend(self._word_piece_tokenize(word))
        return tokens

    def _word_piece_tokenize(self, word):
        r"""Divide normal word into sub-word.
        """
        if word in self.vocab:
            return [word]

        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            sub = word[start:stop]
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self.vocab:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop

        return tokens
