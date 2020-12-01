# coding: utf-8

"""
@File   : new_tokenizer.py
@Author : garnet
@Time   : 2020/11/24 23:21
"""

import re
import typing
import unicodedata
import sentencepiece as spm

from .base import StatefulUnit
from .vocabulary import Vocabulary, BertVocabulary
from .vocabulary import SOS, EOS, UNK, PAD, MASK, SEP, CLS
from ...utils.strings import CJK_PUNCTUATION
from ...utils.strings import is_cjk_character, is_punctuation_character, is_space_character, is_control_character
from ...utils.decorator import safe_return


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


class VocabTokenizer(BaseTokenizer):
    r"""Tokenizer which tokenize text into token with the help of vocabulary.

    Arguments:
        vocab: (:obj:`Vocabulary`, optional, default: `None`) vocabulary instance.
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
                 vocab=None,
                 vocab_path=None,
                 encoding='utf-8',
                 simplified=False,
                 keep_tokens=None,
                 start_tokens=None,
                 extended_tokens=None,
                 **kwargs):
        super(VocabTokenizer, self).__init__(**kwargs)
        self.vocab = None

        if vocab is not None or vocab_path is not None:
            self.fit(
                vocab=vocab,
                vocab_path=vocab_path,
                encoding=encoding,
                simplified=simplified,
                keep_tokens=keep_tokens,
                start_tokens=start_tokens,
                extended_tokens=extended_tokens,
                **kwargs
            )

    @property
    def vocab_size(self):
        return len(self.vocab)

    def token2id(self, token):
        return self.vocab[token]

    def id2token(self, index):
        return self.vocab.i2w(index)

    def create_vocab(self,
                     vocab_path,
                     encoding='utf-8',
                     simplified=False,
                     keep_tokens=None,
                     start_tokens=None,
                     extended_tokens=None,
                     **kwargs):
        raise NotImplementedError

    def fit(self,
            vocab=None,
            vocab_path=None,
            encoding='utf-8',
            simplified=False,
            keep_tokens=None,
            start_tokens=None,
            extended_tokens=None,
            **kwargs):
        assert vocab is not None or vocab_path is not None, "`vocab` and `vocab_path` can't both be `None` at the " \
                                                            "same time"
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = self.create_vocab(
                vocab_path,
                encoding=encoding,
                simplified=simplified,
                keep_tokens=keep_tokens,
                start_tokens=start_tokens,
                extended_tokens=extended_tokens,
                **kwargs
            )

        super(VocabTokenizer, self).fit(**kwargs)


class BertLikeTokenizer(VocabTokenizer):
    r"""Tokenizer used for bert-like models.
    """

    punctuation = r'+-/={(<[' + CJK_PUNCTUATION

    def __init__(self,
                 vocab_path,
                 token_start=CLS,
                 token_end=SEP,
                 **kwargs):
        kwargs['token_pad'] = PAD
        kwargs['token_unknown'] = UNK
        kwargs['token_mask'] = MASK
        super(BertLikeTokenizer, self).__init__(
            vocab_path=vocab_path,
            token_start=token_start,
            token_end=token_end,
            **kwargs
        )

        self.token_sep = SEP
        self.token_cls = CLS

    @property
    def sep(self):
        return self.token_sep

    @property
    def sep_id(self):
        return self.token2id(self.sep)

    @property
    def cls(self):
        return self.token_cls

    @property
    def cls_id(self):
        return self.token2id(self.cls)

    def create_vocab(self,
                     vocab_path,
                     encoding='utf-8',
                     simplified=False,
                     keep_tokens=None,
                     start_tokens=None,
                     extended_tokens=None,
                     **kwargs):
        if 'token_start' not in kwargs:
            kwargs['token_start'] = self.start
        if 'token_end' not in kwargs:
            kwargs['token_end'] = self.end
        if 'ignore_case' not in kwargs:
            kwargs['ignore_case'] = self.ignore_case

        return BertVocabulary(
            vocab_path,
            encoding=encoding,
            simplified=simplified,
            keep_tokens=keep_tokens,
            start_tokens=start_tokens,
            extended_tokens=extended_tokens,
            **kwargs
        )

    @property
    def keep_tokens(self):
        return self.vocab.keep_tokens

    def tokenize_performer(self, text, **kwargs):
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

    @staticmethod
    def truncate_sequence_pairs(first_seq, second_seq=None, max_length=None, pop_index=-1):
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

    def encode(self, first_text, second_text=None, max_length=None, first_length=None, second_length=None):
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
        second_tokens = None
        if second_text:
            fid = int(bool(self.start))
            second_tokens = self.tokenize(second_text)[fid:] if second_text is not None else None

        max_length = max_length or self.max_length
        if max_length:
            first_tokens, second_tokens = self.truncate_sequence_pairs(first_tokens,
                                                                       second_seq=second_tokens,
                                                                       max_length=max_length,
                                                                       pop_index=-2)

        first_token_ids = self.tokens2ids(first_tokens)
        if first_length is not None:
            first_token_ids = self.sequence_fix_length(first_token_ids,
                                                       max_length=first_length,
                                                       truncate='post',
                                                       padding='post')
        first_segment_ids = [0] * len(first_token_ids)

        if second_tokens:
            second_token_ids = self.tokens2ids(second_tokens)
            if second_length is not None:
                second_token_ids = self.sequence_fix_length(second_token_ids,
                                                            max_length=second_length,
                                                            truncate='post',
                                                            padding='post')
            second_segment_ids = [1] * len(second_token_ids)

            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)
        return first_token_ids, first_segment_ids

    def transform(self, first_text, second_text=None, max_length=None, first_length=None, second_length=None):
        return self.encode(
            first_text,
            second_text=second_text,
            max_length=max_length,
            first_length=first_length,
            second_length=second_length,
        )

    def decode(self, indices, tokens=None):
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

    def reverse_transform(self, indices, tokens=None):
        return self.decode(indices, tokens=tokens)

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


class SentencePieceTokenizer(BertLikeTokenizer):
    def __init__(self, model_path, **kwargs):
        kwargs['with_start'] = False
        super(SentencePieceTokenizer, self).__init__(vocab_path=None, **kwargs)
        self.model = spm.SentencePieceProcessor()
        self.model.Load(model_path)

        self.token_pad = self.model.IdToPiece(self.model.pad_id())
        self.token_unknown = self.model.IdToPiece(self.model.unk_id())

    @property
    def vocab_size(self):
        return self.model.GetPieceSize()

    @safe_return
    @property
    def start_id(self):
        return self.model.PieceToId(self.start)

    @safe_return
    @property
    def end_id(self):
        return self.model.PieceToId(self.end)

    @safe_return
    @property
    def pad_id(self):
        return self.model.PieceToId(self.pad)

    @safe_return
    @property
    def mask_id(self):
        return self.model.PieceToId(self.mask)

    @safe_return
    @property
    def unknown_id(self):
        return self.model.PieceToId(self.unknown)

    def token2id(self, token):
        return self.model.PieceToId(token)

    def id2token(self, index):
        return self.model.IdToPiece(index) if index < self.vocab_size else ''

    def tokenize_performer(self, text, **kwargs):
        tokens = self.model.EncodeAsPieces(text)
        return tokens

    def decode(self, indices, tokens=None):
        tokens = [token for token in self.ids2tokens(indices)]
        text = self.model.DecodePieces(tokens)
        return text
