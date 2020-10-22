# coding: utf-8

"""
@File   : task_similar_sentence_generating.py
@Author : garnet
@Time   : 2020/10/21 11:45
"""

import random
import keras
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Nadam

from garnet.preprocessing import BertTokenizer
from garnet.models.build import build_transformer_model
from garnet.layers import SimBertLoss

from garnet.preprocessing.data import JsonFileDataset, Collator, DataLoader
from garnet.utils import text_segment, sequence_padding


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


if __name__ == '__main__':
    bert_path = 'E:/Models/chinese_simbert_L-12_H-768_A-12/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'bert_model.ckpt'
    dict_path = bert_path + 'vocab.txt'

    tokenizer = BertTokenizer(dict_path, ignore_case=True)
    collator = SimBertCollator(tokenizer)

    train_dataset = JsonFileDataset('./similar_sentence_sample.json')
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, collator=collator)

    for batch in train_dataloader:
        print(batch[0][0].shape)
        print('-' * 32)
    print('*' * 32)
    for batch in train_dataloader:
        print(batch[0][0].shape)
        print('-' * 32)

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

    graph = tf.get_default_graph()
    op = graph.get_operation_by_name('sim_bert_loss_1/mul_3')


    model.fit_generator(train_dataloader, steps_per_epoch=100, epochs=2)