# coding: utf-8

"""
@File   : test_output_vectors.py
@Author : garnet
@Time   : 2020/10/20 14:38
"""

import gc
import unittest
import numpy as np
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array

from garnet.preprocessing.units.tokenizer import BertTokenizer

config_path = '../demo/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../demo/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../demo/chinese_L-12_H-768_A-12/vocab.txt'


class BertTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.my_tokenizer = BertTokenizer(dict_path, ignore_case=True)
        cls.sjl_tokenizer = Tokenizer(dict_path, do_lower_case=True)

    def test_vector(self):
        text = '语言模型'
        tokens1, segs1 = self.sjl_tokenizer.encode(text)
        tokens2, segs2 = self.my_tokenizer.transform(text)
        self.assertEqual(tokens1, tokens2)
        self.assertEqual(segs1, segs2)
        tokens1, segs1 = to_array([tokens1], [segs1])
        tokens2, segs2 = to_array([tokens2], [segs2])

        from bert4keras.models import build_transformer_model
        print(build_transformer_model.__module__)
        model = build_transformer_model(config_path, checkpoint_path)
        res1 = model.predict([tokens1, segs1])
        del model
        gc.collect()

        from garnet.models.build import build_transformer_model
        print(build_transformer_model.__module__)
        model = build_transformer_model(config_path, checkpoint_path)
        res2 = model.predict([tokens2, segs2])
        del model
        gc.collect()

        shape = res1.shape

        self.assertEqual(np.sum(res1), np.sum(res2))

        for k in range(shape[0]):
            for i in range(shape[1]):
                for j in range(shape[2]):
                    self.assertAlmostEqual(res1[k, i, j], res2[k, i, j])


if __name__ == '__main__':
    unittest.main()
