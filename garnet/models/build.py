# coding: utf-8

"""
@File   : build.py
@Author : garnet
@Time   : 2020/10/13 22:34
"""

import json

from .transformer import Bert


def build_transformer_model(
        config_path=None,
        checkpoint_path=None,
        model='bert',
        application='encoder',
        return_keras_model=True,
        **kwargs):
    configs = dict()
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)  # you can assign parameters to override parameters in configuration file

    if 'max_position_embeddings' not in configs:
        configs['max_position_embeddings'] = 512
    if 'hidden_dropout_prob' not in configs:
        configs['hidden_dropout_prob'] = 0.
    if 'attention_dropout_prob' not in configs:
        configs['attention_dropout_prob'] = configs.get('hidden_dropout_prob', 0.)
    if 'segment_vocab_size' not in configs:
        configs['segment_vocab_size'] = configs.get('type_vocab_size', 2)

    model_mapping = {
        'bert': Bert,
    }

    Model = model_mapping[model.lower()]

    transformer = Model(**configs)
    transformer.build(**configs)

    if checkpoint_path is not None:
        transformer.load_weights_from_checkpoint(checkpoint_path)

    return transformer.model if return_keras_model else transformer
