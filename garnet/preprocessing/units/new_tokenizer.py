# coding: utf-8

"""
@File   : new_tokenizer.py
@Author : garnet
@Time   : 2020/11/24 23:21
"""

from .base import StatefulUnit
from .vocabulary import Vocabulary, BertVocabulary
from .vocabulary import SOS, EOS, UNK, PAD, MASK, SEP, CLS


class BaseTokenizer(StatefulUnit):
    def __init__(self,
                 token_start=SOS,
                 token_end=EOS,
                 token_unknown=UNK,
                 token_pad=PAD,
                 token_mask=MASK,
                 max_length=None,
                 with_start=True,
                 with_end=True,
                 ignore_case=True,
                 **kwargs):
        super(BaseTokenizer, self).__init__(**kwargs)
        self.token_start = token_start
        self.token_end = token_end
        self.token_pad = token_pad
        self.token_unknown = token_unknown
        self.token_mask = token_mask

        self.max_length = max_length
        self.with_start = with_start
        self.with_end = with_end

        self.ignore_case = ignore_case

    def token2id(self, token):
        raise NotImplementedError

    def id2token(self, index):
        raise NotImplementedError

    def tokens2ids(self, tokens):
        return [self.token2id(token) for token in tokens]

    def ids2tokens(self, indices):
        return [self.id2token(index) for index in indices]

    @property
    def start(self):
        return self.token_start

    @property
    def start_id(self):
        return self.token2id(self.start)

    @property
    def end(self):
        return self.token_end

    @property
    def end_id(self):
        return self.token2id(self.end)

    @property
    def pad(self):
        return self.token_pad

    @property
    def pad_id(self):
        return self.token2id(self.pad)

    @property
    def unknown(self):
        return self.token_unknown

    @property
    def unknown_id(self):
        return self.token2id(self.unknown)

    @property
    def mask(self):
        return self.token_mask

    @property
    def mask_id(self):
        return self.token2id(self.mask)

    @property
    def special_tokens(self):
        return [self.start, self.end, self.pad, self.unknown, self.mask]

    def real_length(self, max_length=None):
        max_length = max_length or self.max_length
        if max_length:
            return max_length - int(self.with_start) - int(self.with_end)
        return max_length

    def sequence_pad(self, tokens, max_length=None, padding='post'):
        if not padding:
            return tokens

        max_length = max_length or self.max_length
        raw_length = len(tokens)
        if max_length and raw_length < max_length:
            if padding == 'post':
                tokens.extend([self.pad_id] * (max_length - raw_length))
            else:
                prefix = [self.pad_id] * (max_length - raw_length)
                prefix.extend(tokens)
                tokens = prefix
        return tokens

    def sequence_truncate(self, tokens, max_length=None, truncate='post'):
        if not truncate:
            return tokens

        max_length = max_length or self.max_length
        raw_length = len(tokens)
        if max_length and raw_length > max_length:
            tokens = tokens[:max_length] if truncate == 'post' else tokens[-max_length:]
        return tokens

    def sequence_fix_length(self, tokens, max_length=None, truncate='post', padding='post'):
        r"""Truncate or pad token sequence to specified length.

        Arguments:
            :param tokens: token sequence.
            :param max_length: final token sequence length. `None` means do nothing with raw sequence.
            :param truncate: truncate position. Must be one of `'post'` and `'pre'`.
            :param padding: pad position. Must be one of `'post'`, `'pre'` or `None`, and `None` means do not pad.

        Return:
            Token list.
        """
        if truncate:
            tokens = self.sequence_truncate(tokens, max_length=max_length, truncate=truncate)
        if padding:
            tokens = self.sequence_pad(tokens, max_length=max_length, padding=padding)
        return tokens

    def tokenize_performer(self, text, **kwargs):
        raise NotImplementedError

    def tokenize(self, text, max_length=None, truncate='post', padding='post', **kwargs):
        r"""Transfer text into sequence of text tokens.

        Arguments:
            :param text: text string.
            :param max_length: (int, optional, default: `None`) max length of final token sequence. `None` means
                no limits.
            :param truncate: truncate token sequence when length of sequence is greater than max threshold.
                `pre`, `post` and `None` is available.
            :param padding: pad position. Must be one of `'post'`, `'pre'` or `None`, and `None` means do not pad.
        """
        tokens = self.tokenize_performer(text, **kwargs)

        residual_length = self.real_length(max_length or self.max_length)
        tokens = self.sequence_fix_length(tokens, max_length=residual_length, truncate=truncate, padding=padding)

        if self.with_start:
            tokens.insert(0, self.start)
        if self.with_end:
            tokens.append(self.end)

        return tokens

    def encode(self, *args, **kwargs):
        r"""Transfer tokens into token ids.
        """
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        r"""Transfer ids into text tokens.
        """
        raise NotImplementedError


class BertLikeTokenizer(BaseTokenizer):
    r"""Tokenizer used for bert-like models.

    Arguments:
        vocab_path: (:obj:`str`, optional, default: `None`) vocabulary file path. Usually contained in pre-trained
            bert-like model package.
        encoding: (:obj:`str`, optional, default: `'utf-8'`) encoding of vocabulary file.
        simplified: (:obj:`bool`, optional, default: `False`) whether simplify original vocabulary. Can't be used in
            the same time with :arg:`keep_tokens`.
        keep_tokens: (:obj:`list`, optional, default: `None`) simplified tokens list obtained from original vocabulary.
            Can't be used in the same time with :arg:`simplified`.
        start_tokens: (:obj:`list`, optional, default: `None`) vocabulary starts with these tokens, that is to say
            they will have smallest token indices.
        extended_tokens: (:obj:`list`, optional, default: `None`) vocabulary ends with these tokens, that is to say
            they will have largest token indices.
    """

    def __init__(self,
                 vocab_path=None,
                 token_start=CLS,
                 token_end=SEP,
                 encoding='utf-8',
                 simplified=False,
                 keep_tokens=None,
                 start_tokens=None,
                 extended_tokens=None,
                 **kwargs):
        kwargs['token_pad'] = PAD
        kwargs['token_unknown'] = UNK
        kwargs['token_mask'] = MASK
        super(BertLikeTokenizer, self).__init__(token_start=token_start,
                                                token_end=token_end,
                                                **kwargs)
