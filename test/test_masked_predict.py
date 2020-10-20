# coding: utf-8

"""
@File   : test_masked_predict.py
@Author : garnet
@Time   : 2020/10/20 16:45
"""

import unittest
import numpy as np

from bert4keras.snippets import to_array

from garnet.models.build import build_transformer_model
from garnet.preprocessing.units.tokenizer import BertTokenizer

config_path = '../demo/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../demo/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../demo/chinese_L-12_H-768_A-12/vocab.txt'


class MaskedPredictTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = BertTokenizer(dict_path, ignore_case=True)
        cls.model = build_transformer_model(config_path, checkpoint_path, with_mlm=True)

    def test_masked_predict(self):
        text = "科学技术是第一生产力"
        tokens, segs = self.tokenizer.transform(text)
        print(tokens)
        tokens[3] = tokens[4] = self.tokenizer.token2id(self.tokenizer.token_mask)
        print(tokens)
        tokens, segs = to_array([tokens], [segs])
        probs = self.model.predict([tokens, segs])[1][0]
        pred_ids = probs.argmax(axis=1)
        print(pred_ids)
        text = self.tokenizer.reverse_transform(list(pred_ids))
        print(text)
        self.assertEqual(text[3:5], '技术')


if __name__ == '__main__':
    unittest.main()
