# coding: utf-8

"""
@File   : task_similar_sentence_generating.py
@Author : garnet
@Time   : 2020/10/21 11:45
"""

import random
import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Nadam

from garnet.preprocessing import BertTokenizer
from garnet.models.build import build_transformer_model
from garnet.layers import SimBertLoss

from garnet.preprocessing.data import JsonFileDataset, Collator, DataLoader
from garnet.utils import text_segment, sequence_padding
from garnet.utils.decoder import AutoRegressiveDecoder


class SimBertCollator(Collator):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def truncate(text):
        return text_segment(text, 30, seps='\n。！？!?；;，, ', strips='；;，, ')[0]

    def collate_fn(self, batch_data):
        batch_final = []
        for sample in batch_data:
            text, synonyms = sample['text'], sample['synonyms']
            synonyms.append(text)
            s1, s2 = random.sample(synonyms, k=2)
            s1, s2 = self.truncate(s1), self.truncate(s2)

            batch_final.append(tokenizer.transform(s1, s2, max_length=64))
            batch_final.append(tokenizer.transform(s2, s1, max_length=64))
        token_ids, seg_ids = list(list(t) for t in zip(*batch_final))
        token_ids, seg_ids = sequence_padding(token_ids), sequence_padding(seg_ids)
        return [token_ids, seg_ids], None


class SynonymsDecoder(AutoRegressiveDecoder):
    @AutoRegressiveDecoder.predict_wraps(default_return_type='probas')
    def predict(self, inputs, output_indices, states=None, return_type='probas'):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_indices], axis=1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_indices)], axis=1)
        probas = seq2seq_model.predict([token_ids, segment_ids])
        return probas[:, -1]

    def generate(self, text, n=5, top_k=5, top_p=None, max_length=None, mode='random'):
        token_ids, segment_ids = tokenizer.transform(text, max_length=max_length)
        if mode == 'random':
            outputs = self.random_sample([token_ids, segment_ids], n=n, top_k=top_k, top_p=top_p)
        else:
            outputs = self.beam_search([token_ids, segment_ids], top_k=top_k)
        return [tokenizer.reverse_transform(sample) for sample in outputs]


if __name__ == '__main__':
    bert_path = 'E:/Models/chinese_simbert_L-12_H-768_A-12/'
    # bert_path = '../demo/chinese_simbert_L-12_H-768_A-12/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'bert_model.ckpt'
    dict_path = bert_path + 'vocab.txt'

    tokenizer = BertTokenizer(dict_path, ignore_case=True)
    collator = SimBertCollator(tokenizer)

    model = build_transformer_model(config_path,
                                    checkpoint_path,
                                    application='unilm',
                                    with_pool=True,
                                    return_keras_model=False)
    print(model.outputs)

    encoder_model = Model(model.inputs, model.outputs[1])  # [CLS] vector
    seq2seq_model = Model(model.inputs, model.outputs[2])  # vocabulary probabilities

    outputs = SimBertLoss()(model.inputs + model.outputs[1:])

    model = Model(model.inputs, outputs[-2:])
    model.compile(optimizer=Nadam(0.001))
    model.summary()

    # train_dataset = JsonFileDataset('./similar_sentence_sample.json')
    # train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, collator=collator)

    # graph = tf.get_default_graph()
    # op = graph.get_operation_by_name('sim_bert_loss_1/mul_3')

    # model.fit_generator(train_dataloader, steps_per_epoch=100, epochs=2)

    decoder = SynonymsDecoder(end_index=tokenizer.token2id(tokenizer.token_end), max_length=32)

    text = "车险理赔报案"
    synonyms = decoder.generate(text, n=20, top_k=5, mode='random')
    print("Raw text: {}".format(text))
    print("Synonyms:")
    for t in synonyms:
        print(t, end='\n')
