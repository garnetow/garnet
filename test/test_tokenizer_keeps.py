# coding: utf-8

"""
@File   : test_tokenizer_keeps.py
@Author : garnet
@Time   : 2020/11/30 23:13
"""

import json
import unittest
from bert4keras.tokenizers import Tokenizer, load_vocab
from garnet.preprocessing.units.tokenizer import BertLikeTokenizer


class TokenizerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        vocab_path = '../models/mixed_corpus_bert_base_model/vocab.txt'
        keep_path = '../models/mixed_corpus_bert_base_model/token_dict_keep_tokens.json'

        token_dict, keep_tokens = json.load(open(keep_path))
        cls.my_tokenizer = BertLikeTokenizer(vocab_path, keep_tokens=keep_tokens, ignore_case=True)
        cls.raw_tokenizer = Tokenizer(token_dict, do_lower_case=True)

    def test_length(self):
        size1 = self.my_tokenizer.vocab_size
        size2 = self.raw_tokenizer._vocab_size
        self.assertEqual(size1, size2)

    def test_vocab(self):
        vocab1 = self.my_tokenizer.vocab.vocab
        vocab2 = self.raw_tokenizer._token_dict
        self.assertEqual(len(vocab1), len(vocab2))
        print(len(vocab1))

        for k1, k2 in zip(sorted(vocab1.items(), key=lambda x: x[1]), sorted(vocab2.items(), key=lambda x: x[1])):
            self.assertEqual(k1, k2)

    def test_single_chinese(self):
        text = '科学技术是第一生产力'
        tokens1 = self.my_tokenizer.transform(text)
        tokens2 = self.raw_tokenizer.encode(text)
        self.assertEqual(tokens1, tokens2)
        text1 = self.my_tokenizer.reverse_transform(tokens1[0])
        text2 = self.raw_tokenizer.decode(tokens2[0])
        self.assertEqual(text1, text2)

    def test_single_long_chinese(self):
        text = r"""机器学习是人工智能的一个分支。人工智能的研究历史有着一条从以“推理”为重点，到以“知识”为重点，再到以“学习”为重点的自然、清晰的脉络。显然，机器学习是实现人工智能的一个途径，即以机器学习为手段解决人工智能中的问题。机器学习在近30多年已发展为一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、计算复杂性理论等多门学科。机器学习理论主要是设计和分析一些让计算机可以自动“学习”的算法。机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。因为学习算法中涉及了大量的统计学理论，机器学习与推断统计学联系尤为密切，也被称为统计学习理论。算法设计方面，机器学习理论关注可以实现的，行之有效的学习算法。很多推论问题属于无程序可循难度，所以部分的机器学习研究是开发容易处理的近似算法。"""
        tokens1 = self.my_tokenizer.transform(text)
        tokens2 = self.raw_tokenizer.encode(text)
        self.assertEqual(tokens1, tokens2)
        text1 = self.my_tokenizer.reverse_transform(tokens1[0])
        text2 = self.raw_tokenizer.decode(tokens2[0])
        self.assertEqual(text1, text2)

    def test_single_mixture(self):
        text = r"""一种经常引用的英文定义是：A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."""
        tokens1 = self.my_tokenizer.transform(text)
        tokens2 = self.raw_tokenizer.encode(text)
        self.assertEqual(tokens1, tokens2)
        text1 = self.my_tokenizer.reverse_transform(tokens1[0])
        text2 = self.raw_tokenizer.decode(tokens2[0])
        self.assertEqual(text1, text2)

    def test_single_long_english(self):
        text = r"""Machine learning (ML) is the study of computer algorithms that improve automatically through experience.[1][2] It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so.[3] Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or infeasible to develop conventional algorithms to perform the needed tasks.

Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a related field of study, focusing on exploratory data analysis through unsupervised learning.[5][6] In its application across business problems, machine learning is also referred to as predictive analytics.

Simple Definition: Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves."""
        tokens1 = self.my_tokenizer.transform(text)
        tokens2 = self.raw_tokenizer.encode(text)
        self.assertEqual(tokens1, tokens2)
        text1 = self.my_tokenizer.reverse_transform(tokens1[0])
        text2 = self.raw_tokenizer.decode(tokens2[0])
        self.assertEqual(text1, text2)

    def test_second(self):
        text1 = r"""机器学习是人工智能的一个分支。人工智能的研究历史有着一条从以“推理”为重点，到以“知识”为重点，再到以“学习”为重点的自然、清晰的脉络。显然，机器学习是实现人工智能的一个途径，即以机器学习为手段解决人工智能中的问题。机器学习在近30多年已发展为一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、计算复杂性理论等多门学科。机器学习理论主要是设计和分析一些让计算机可以自动“学习”的算法。机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。因为学习算法中涉及了大量的统计学理论，机器学习与推断统计学联系尤为密切，也被称为统计学习理论。算法设计方面，机器学习理论关注可以实现的，行之有效的学习算法。很多推论问题属于无程序可循难度，所以部分的机器学习研究是开发容易处理的近似算法。"""
        text2 = r"""一种经常引用的英文定义是：A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."""
        tokens1 = self.my_tokenizer.transform(text1, text2)
        tokens2 = self.raw_tokenizer.encode(text1, text2)
        self.assertEqual(tokens1, tokens2)
        text1 = self.my_tokenizer.reverse_transform(tokens1[0])
        text2 = self.raw_tokenizer.decode(tokens2[0])
        self.assertEqual(text1, text2)

    def test_padding(self):
        text = '科学技术是第一生产力'
        tokens1 = self.my_tokenizer.transform(text, max_length=50)
        tokens2 = self.raw_tokenizer.encode(text, maxlen=50)
        self.assertEqual(tokens1, tokens2)
        text1 = self.my_tokenizer.reverse_transform(tokens1[0])
        text2 = self.raw_tokenizer.decode(tokens2[0])
        self.assertEqual(text1, text2)


if __name__ == '__main__':
    unittest.main()
