# coding: utf-8

"""
@File   : task_similar_sentence_generating.py
@Author : garnet
@Time   : 2020/10/21 11:45
"""

from keras.models import Model
from keras.optimizers import Nadam

from garnet.preprocessing import BertTokenizer
from garnet.models.build import build_transformer_model
from garnet.layers import SimBertLoss

if __name__ == '__main__':
    bert_path = 'E:/Models/chinese_simbert_L-12_H-768_A-12/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'bert_model.ckpt'
    dict_path = bert_path + 'vocab.txt'

    tokenizer = BertTokenizer(dict_path, ignore_case=True)

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
