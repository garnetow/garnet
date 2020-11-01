# coding: utf-8

"""
@File   : test_tokenizer_simplify.py
@Author : garnet
@Time   : 2020/10/31 12:34
"""

import unittest

from garnet.preprocessing import BertTokenizer
from garnet.utils.strings import is_cjk_character, is_punctuation_character


class TokenizerSimplifyTestCase(unittest.TestCase):
    def test_simplify(self):
        dict_path = '../models/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'
        token = '[unused1]'

        tokenizer = BertTokenizer(dict_path, simplified=True, start_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]'])

        for char in tokenizer.vocab.stem(token):
            r1, r2 = is_cjk_character(char), is_punctuation_character(char)
            print(char, r1, r2, not r1 and not r2)

        r3 = all([True if not is_cjk_character(char) and not is_punctuation_character(char) else False for char in
                  tokenizer.vocab.stem(token)])
        self.assertFalse(r3)


if __name__ == '__main__':
    unittest.main()
