# coding: utf-8

"""
@File   : transformer.py
@Author : garnet
@Time   : 2020/9/29 8:48
"""

import copy
import keras
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Add, Lambda, Activation

from .model import WrappedModel
from .transformer_mixin import LanguageModelMixin
from ..layers.core import BiasAdd
from ..layers.bert import PositionEmbedding
from ..layers.bert import LayerNormalization
from ..layers.bert import MultiHeadAttention
from ..layers.bert import FeedForward
from ..layers.bert import DenseEmbedding
from ..layers.t5 import RelativePositionEmbeddingT5
from ..utils.functions.normalization import truncated_normal


class Transformer(WrappedModel):
    def __init__(self,
                 vocab_size,  # 词表大小
                 hidden_size,  # 编码维度
                 num_hidden_layers,  # 隐层数
                 num_attention_heads,  # Multi-head attention中head的数量
                 intermediate_size,  # FeedForward的隐层维度
                 attention_key_size=None,  # Attention中Q, K的head_size
                 hidden_act='gelu',  # FeedForward隐层的激活函数
                 embedding_size=None,  # 指定的embedding输出大小
                 attention_dropout_prob=None,  # Attention层dropout比例
                 hidden_dropout_prob=None,  # 其他层dropout比例
                 max_position_embeddings=512,  # 位置编码最大范围
                 fixed_sequence_length=None,  # 固定的序列长度
                 reserved_tokens=None,  # 要保留的词ID列表
                 extended_tokens=None,  # 扩展的词ID列表
                 prefix=None,  # 层名称前缀
                 name=None,  # 模型名称
                 **kwargs):
        r"""Transformer model.

        Arguments:
            :param vocab_size: vocabulary size.
            :param hidden_size: encoding size of token.
            :param num_hidden_layers: number of hidden layers in the Transformer encoder.
            :param num_attention_heads: number of attention heads for each attention layer in the Transformer encoder.
            :param intermediate_size: dimensionality of the feed-forward layer in the Transformer encoder.
            :param attention_key_size(:obj:`int`, optional, default: None):
                specified hidden size of `query` and `head` in attention.
            :param hidden_act (:obj:`str` or :obj:`Callable`, optional, default: `'gelu'`):
                non-linear activation function in the encoder and pooler.
            :param attention_dropout_prob (:obj:`float`, optional, default: `None`):
                the dropout ratio for the attention probabilities.
            :param hidden_dropout_prob (:obj:`float`, optional, default: `None`):
                the dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            :param embedding_size (:obj:`int`, optional, default: `None`):
                fixed embedding output size. If is `None`, embedding output will be `hidden_size`.
            :param max_position_embeddings (:obj:`int`, optional, defaults: 512):
                the maximum sequence length that this model might ever be used with.
                Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
            :param fixed_sequence_length (:obj:`int`, optional, defaults: None):
                fixed length of token sequence. Default is `None`, which means dynamic sequence length in each batch.
            :param reserved_tokens: (:obj:`list`, optional, defaults: None):
                list of tokens representing a simplified subset of all tokens in transformer-like model vocabulary.
            :param extended_tokens: (:obj:`list`, optional, defaults: None):
                list of extra extended tokens, which are not included in original model vocabulary.
            :param prefix (:obj:`str`, optional, default: `None`):
                prefix of the every layers' name in the model.
            :param name (:obj:`str`, optional, default: `None`):
                name of the model.
        """
        if reserved_tokens is not None:
            vocab_size = len(reserved_tokens)
        if extended_tokens is not None:
            vocab_size += len(extended_tokens)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.attention_dropout_prob = attention_dropout_prob or 0.
        self.hidden_dropout_prob = hidden_dropout_prob or 0.
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.fixed_sequence_length = fixed_sequence_length
        self.reserved_tokens = reserved_tokens
        self.extended_tokens = extended_tokens
        self.prefix = prefix or ''
        self.name = name

        self.input = None
        self.inputs = None
        self.output = None
        self.outputs = None
        self.layers = dict()
        self.attention_mask = None
        self.relative_position = None
        self.model = None
        self.built = False

    def build(self,
              attention_mask=None,
              head_mask=None,
              input_embeds=None,
              custom_position_ids=None,
              encoder_hidden_context=None,
              encode_attention_mask=None,
              layer_norm_cond_inputs=None,
              layer_norm_cond_hidden_size=None,
              layer_norm_cond_hidden_act=None,
              additional_inputs=None,
              **kwargs):
        r"""Build transformer-based model. Overwrite this method if the model is complex, such as `Seq2Seq` model.

        Argument:
            :param attention_mask (Tensor with shape (batch_size, query_length, key_length), optional):
                mask of attention matrix. `1` for not masked, 0 for masked.
            :param head_mask (Tensor with shape (num_heads,) or (num_layers, num_heads), optional):
                mask of heads in multi-head attention.
            :param input_embeds (Tensor with shape (batch_size, seq_length, hidden_size), optional):
                instead of passing token ids, directly pass an embedded representation of tokens. This is useful if
                you want to control token embeddings than the model's internal embedding lookup matrix.
            :param custom_position_ids (Tensor with shape (batch_size, seq_length), optional):
                use custom positions ids instead of internal generated position ids.
            :param encoder_hidden_context (Tensor with shape (batch_size, seq_length, hidden_size), optional):
                sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                if the model is configured as a decoder.
            :param encode_attention_mask (Tensor with shape (batch_size, seq_length), optional):
                mask to avoid performing attention on the padding token indices of the encoder input. This mask
                is used in the cross-attention if the model is configured as a decoder.
            :param layer_norm_cond_inputs (Tensor with shape (batch_size, hidden_size), optional):
                condition inputs for `Conditional Layer Normalization`.
            :param layer_norm_cond_hidden_size (:obj:`int`, optional):
                project hidden size of `layer_norm_cond_inputs` into `layer_norm_cond_hidden_size` if this is not None.
            :param layer_norm_cond_hidden_act (:obj:`str` or :obj:`Callable`, optional):
                activation of the dense layer of hidden size of `layer_norm_cond_inputs` projection.
            :param additional_inputs (Tensor or list of Tensors, optional):
                addition inputs.
        """
        if self.built:
            return

        # get inputs
        inputs = self.get_inputs(
            input_embeds=input_embeds,
            **kwargs
        )
        self.set_inputs(inputs, additional_inputs=additional_inputs)

        # apply embedding layer
        outputs = self.apply_embeddings(
            inputs,
            input_embeds=input_embeds,
            custom_position_ids=custom_position_ids,
            layer_norm_cond_inputs=layer_norm_cond_inputs,
            layer_norm_cond_hidden_size=layer_norm_cond_hidden_size,
            layer_norm_cond_hidden_act=layer_norm_cond_hidden_act,
            **kwargs
        )

        # apply transformer layers
        for i in range(self.num_hidden_layers):
            outputs = self.apply_main_layers(
                outputs,
                i,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_context=encoder_hidden_context,
                encode_attention_mask=encode_attention_mask,
                layer_norm_cond_inputs=layer_norm_cond_inputs,
                layer_norm_cond_hidden_size=layer_norm_cond_hidden_size,
                layer_norm_cond_hidden_act=layer_norm_cond_hidden_act,
                **kwargs
            )

        # apply output layer
        outputs = self.apply_output_layers(
            outputs,
            layer_norm_cond_inputs=layer_norm_cond_inputs,
            layer_norm_cond_hidden_size=layer_norm_cond_hidden_size,
            layer_norm_cond_hidden_act=layer_norm_cond_hidden_act,
            **kwargs
        )
        self.set_outputs(outputs)

        self.model = Model(self.inputs, self.outputs, name=self.name)
        self.built = True

    def apply(self, inputs=None, layer=None, arguments=None, name=None, **kwargs):
        r"""Apply layer to inputs, include layer building and calling.

        :param inputs: output of last layer.
        :param layer: class of the layer to apply.
        :param arguments: parameters delivered to :meth:`layer.call` method.
        :param name: name of the layer.
        :param kwargs: parameters delivered to layer initialization.
        """

        arguments = arguments or dict()
        name = self.add_prefix(name)
        kwargs['name'] = name
        if name not in self.layers:
            layer = layer(**kwargs)
            name = layer.name
            self.layers[name] = layer

        if inputs is None:
            return self.layers[name]
        else:
            return self.layers[name](inputs, **arguments)

    def get_inputs(self, **kwargs):
        raise NotImplementedError

    def apply_embeddings(self, inputs, **kwargs):
        raise NotImplementedError

    def apply_main_layers(self, inputs, index, **kwargs):
        raise NotImplementedError

    def apply_output_layers(self, inputs, **kwargs):
        raise NotImplementedError

    def set_inputs(self, inputs, additional_inputs=None):
        if inputs is None:
            inputs = []
        elif not isinstance(inputs, list):
            inputs = [inputs]

        inputs = inputs[:]  # shallow copy

        if additional_inputs is not None:
            if isinstance(additional_inputs, list):
                inputs.extend(additional_inputs)
            else:
                inputs.append(additional_inputs)

        self.inputs = inputs
        if len(inputs) > 1:
            self.input = self.inputs
        elif len(inputs) == 1:
            self.input = self.inputs[0]

    def set_outputs(self, outputs):
        if not isinstance(outputs, list):
            outputs = [outputs]

        outputs = outputs[:]  # shallow copy
        self.outputs = outputs

        if len(outputs) > 1:
            self.output = outputs
        elif len(outputs) == 1:
            self.output = outputs[0]

    def add_prefix(self, name):
        if name:
            return self.prefix + name

    def variable_mapping(self):
        return dict()

    def load_weights_from_checkpoint(self, checkpoint, mapping=None):
        r"""Load parameter weights form checkpoint according to mapping relationship.
        """

        mapping = mapping or self.variable_mapping()
        mapping = {self.add_prefix(k): v for k, v in mapping.items()}
        mapping = {k: v for k, v in mapping.items() if k in self.layers}

        weight_value_pairs = []
        for layer_name, variables in mapping.items():
            layer = self.layers[layer_name]
            weights = layer.trainable_weights
            values = [self.load_variable(checkpoint, v) for v in variables]
            weight_value_pairs.extend(zip(weights, values))
        K.batch_set_value(weight_value_pairs)

    def save_weights_as_checkpoint(self, filename, mapping=None):
        mapping = mapping or self.variable_mapping()
        mapping = {self.add_prefix(k): v for k, v in mapping.items()}
        mapping = {k: v for k, v in mapping.items() if k in self.layers}

        with tf.Graph().as_default():
            all_variables, all_values = [], []
            for layer_name, variables in mapping.items():
                layer = self.layers[layer_name]
                values = K.batch_get_value(layer.trainable_weights)
                for name, value in zip(variables, values):
                    all_variables.append(self.create_variable(name, value))
                    all_values.append(value)
            with tf.Session() as sess:
                K.batch_set_value(zip(all_variables, all_values))
                saver = tf.train.Saver()
                saver.save(sess, filename)

    def load_variable(self, checkpoint, name):
        r"""Load the weight of a single variable.
        """
        return tf.train.load_variable(checkpoint, name)

    def load_embedding(self, embeddings):
        if self.reserved_tokens is not None:
            embeddings = embeddings[self.reserved_tokens]
        if self.extended_tokens is not None:
            extended_embeddings = truncated_normal(shape=(len(self.extended_tokens), embeddings.shape[1]),
                                                   mean=0.,
                                                   stddev=0.02,
                                                   lower=2,
                                                   upper=2)
            embeddings = np.concatenate([embeddings, extended_embeddings], 0)
        return embeddings

    def create_variable(self, name, value):
        r"""Initial variable with truncated normal distribution.
        """
        return K.variable(self.initializer(value.shape), name=name)

    @property
    def initializer(self):
        return keras.initializers.TruncatedNormal(stddev=0.02)

    def compute_attention_mask(self, inputs=None, **kwargs):
        r"""Calculate attention mask
        """
        return None


class Bert(Transformer):
    def __init__(self,
                 segment_vocab_size=2,  # Segment总数目
                 with_pool=False,  # 输出是否增加[CLS]部分
                 with_nsp=False,  # 输出是否增加 Next Sentence Prediction 部分
                 with_mlm=False,  # 输出是否增加 Masked Language Model 部分
                 pooler_activation='tanh',
                 shared_segment_embeddings=False,  # segment跟token是否共享embedding参数
                 **kwargs):
        r"""Bert model. See more in [Attention is all you need](https://arxiv.org/abs/1706.03762).

        Argument:
            :param segment_vocab_size (:obj:`int`, optional, defaults: 2):
                number of segments, also the vocabulary size of the :obj:`token_type_ids` passed.
            :param with_pool (:obj:`bool`, optional, default: False):
                if `True`, outputs including `[CLS]` vector.
            :param with_nsp (:obj:`bool`, optional, default: False):
                if `True`, outputs including the probabilities of Next Sentence Prediction.
            :param with_mlm (:obj:`bool`, optional, default: False):
                if `True`, outputs including the whole vocabulary probabilities of masked token(`[MASK]`). This also is
                the output of Masked Language Model.
            :param pooler_activation (:obj:`str`, optional, default: `tanh`):
                activation of pooler in output layer.
            :param shared_segment_embeddings (:obj:`bool`, optional, default: False):
                if `True`, segment tokens share the embedding matrix with normal tokens.
        """
        super().__init__(**kwargs)
        self.segment_vocab_size = segment_vocab_size
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.pooler_activation = keras.activations.get(pooler_activation)
        self.shared_segment_embeddings = shared_segment_embeddings
        if self.with_nsp:
            self.with_pool = True

    def get_inputs(self,
                   input_embeds=None,
                   **kwargs):
        inputs = []
        if input_embeds is None:
            token_in = self.apply(
                layer=Input,
                shape=(self.fixed_sequence_length,),
                name='Input-Token',
            )
            inputs.append(token_in)

        if self.segment_vocab_size > 0:
            seg_in = self.apply(
                layer=Input,
                shape=(self.fixed_sequence_length,),
                name='Input-Segment',
            )
            inputs.append(seg_in)

        return inputs

    def apply_embeddings(self,
                         inputs,
                         input_embeds=None,
                         custom_position_ids=None,
                         layer_norm_cond_inputs=None,
                         layer_norm_cond_hidden_size=None,
                         layer_norm_cond_hidden_act=None,
                         **kwargs):
        inputs = inputs[:]  # shallow copy

        if input_embeds is not None:
            x = input_embeds
        else:
            x = inputs.pop(0)
            x = self.apply(
                inputs=x,
                layer=DenseEmbedding,
                input_dim=self.vocab_size,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                mask_zero=True,
                name='Embedding-Token',
            )

        if self.segment_vocab_size > 0:
            s = inputs.pop(0)

        p = custom_position_ids or None

        if self.segment_vocab_size > 0:
            name = 'Embedding-Token' if self.shared_segment_embeddings else 'Embedding-Segment'
            s = self.apply(
                inputs=s,
                layer=DenseEmbedding,
                input_dim=self.segment_vocab_size,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                name=name,
            )
            x = self.apply(
                inputs=[x, s],
                layer=Add,
                name='Embedding-Token-Segment',
            )

        # apply position embedding
        x = self.apply(
            inputs=x if p is None else [x, p],
            layer=PositionEmbedding,
            input_dim=self.max_position_embeddings,
            output_dim=self.embedding_size,
            merge_mode='add',
            embeddings_initializer=self.initializer,
            name='Embedding-Position',
        )

        # apply layer normalization
        x = self.apply(
            inputs=x if layer_norm_cond_inputs is None else [x, layer_norm_cond_inputs],
            layer=LayerNormalization,
            conditional=layer_norm_cond_inputs is not None,
            cond_hidden_units=layer_norm_cond_hidden_size,
            cond_hidden_activation=layer_norm_cond_hidden_act,
            cond_hidden_initializer=self.initializer,
            name='Embedding-Norm'
        )

        # apply dropout
        if self.hidden_dropout_prob:
            x = self.apply(
                inputs=x,
                layer=Dropout,
                rate=self.hidden_dropout_prob,
                name='Embedding-Dropout'
            )

        # apply hidden size projection
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x

    def apply_main_layers(self,
                          inputs,
                          index,
                          attention_mask=None,
                          head_mask=None,
                          encoder_hidden_context=None,
                          encode_attention_mask=None,
                          layer_norm_cond_inputs=None,
                          layer_norm_cond_hidden_size=None,
                          layer_norm_cond_hidden_act=None,
                          **kwargs):
        r"""Each single layer in transformer is based on `Self-attention` layer.

        In each layer, tensors flow in the following order:
        Attention -> Add -> Layer Normalization -> Feed Forward -> Add -> Layer Normalization

        :param inputs: embedding inputs
        :param index: index of this layer
        """

        x = inputs

        attention_name = 'Transformer-{}-MultiHeadSelfAttention'.format(index)
        feed_forward_name = 'Transformer-{}-FeedForward'.format(index)

        if attention_mask is None:
            attention_mask = self.compute_attention_mask(inputs,
                                                         seg_ids=self.inputs[1],
                                                         index=index,
                                                         **kwargs)

        attention_args = {
            'head_mask': head_mask is not None,
            'attention_mask': attention_mask is not None,
            'encoder_hidden_context': encoder_hidden_context,
            'encode_attention_mask': encode_attention_mask,
        }

        # apply self-attention
        x_inputs = [x, x, x]
        if attention_mask is not None:
            x_inputs.append(attention_mask)
        if head_mask is not None:
            x_inputs.append(head_mask)

        xt = self.apply(
            inputs=x_inputs,
            layer=MultiHeadAttention,
            head_num=self.num_attention_heads,
            head_size=self.attention_head_size,
            kernel_initializer=self.initializer,
            arguments=attention_args,
            name=attention_name,
        )

        if self.attention_dropout_prob:
            xt = self.apply(
                inputs=xt,
                layer=Dropout,
                rate=self.attention_dropout_prob,
                name='{}-Dropout'.format(attention_name),
            )

        x = self.apply(
            inputs=[x, xt],
            layer=Add,
            name='{}-Add'.format(attention_name),
        )

        # apply layer normalization
        x = self.apply(
            inputs=x if layer_norm_cond_inputs is None else [x, layer_norm_cond_inputs],
            layer=LayerNormalization,
            conditional=layer_norm_cond_inputs is not None,
            cond_hidden_units=layer_norm_cond_hidden_size,
            cond_hidden_activation=layer_norm_cond_hidden_act,
            cond_hidden_initializer=self.initializer,
            name='{}-Norm'.format(attention_name),
        )

        # apply feed-forward
        xt = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name,
        )
        if self.hidden_dropout_prob:
            xt = self.apply(
                inputs=xt,
                layer=Dropout,
                rate=self.hidden_dropout_prob,
                name='{}-Dropout'.format(feed_forward_name),
            )

        x = self.apply(
            inputs=[x, xt],
            layer=Add,
            name='{}-Add'.format(feed_forward_name),
        )

        # apply layer normalization
        x = self.apply(
            inputs=x if layer_norm_cond_inputs is None else [x, layer_norm_cond_inputs],
            layer=LayerNormalization,
            conditional=layer_norm_cond_inputs is not None,
            cond_hidden_units=layer_norm_cond_hidden_size,
            cond_hidden_activation=layer_norm_cond_hidden_act,
            cond_hidden_initializer=self.initializer,
            name='{}-Norm'.format(feed_forward_name),
        )

        return x

    def apply_output_layers(self,
                            inputs,
                            layer_norm_cond_inputs=None,
                            layer_norm_cond_hidden_size=None,
                            layer_norm_cond_hidden_act=None,
                            **kwargs):
        outputs = [inputs]

        if self.with_pool:  # add [CLS] vector into output list
            x = self.apply(
                inputs=inputs,
                layer=Lambda,
                function=lambda xt: xt[:, 0],
                name='Pooler',
            )
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                activation=self.pooler_activation,
                kernel_initializer=self.initializer,
                name='Pooler-Dense',
            )

            if self.with_nsp:  # add Next Sentence Prediction prediction probabilities into output list
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=2,
                    activation='softmax',
                    kernel_initializer=self.initializer,
                    name='NSP-Proba',
                )
            outputs.append(x)

        if self.with_mlm:  # add Masked Language Model prediction probabilities into output list
            x = self.apply(
                inputs=inputs,
                layer=Dense,
                units=self.embedding_size,
                activation=self.hidden_act,
                kernel_initializer=self.initializer,
                name='MLM-Dense',
            )

            x = self.apply(
                inputs=x if layer_norm_cond_inputs is None else [x, layer_norm_cond_inputs],
                layer=LayerNormalization,
                conditional=layer_norm_cond_inputs is not None,
                cond_hidden_units=layer_norm_cond_hidden_size,
                cond_hidden_activation=layer_norm_cond_hidden_act,
                cond_hidden_initializer=self.initializer,
                name='MLM-Norm',
            )

            # get token probabilities for each step
            x = self.apply(
                inputs=x,
                layer=DenseEmbedding,
                arguments={'mode': 'dense'},
                name='Embedding-Token'
            )
            x = self.apply(
                inputs=x,
                layer=BiasAdd,
                name='MLM-Bias',
            )
            x = self.apply(
                inputs=x,
                layer=Activation,
                activation='softmax',
                name='MLM-Activation'
            )
            outputs.append(x)

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def load_variable(self, checkpoint, name):
        variable = super(Bert, self).load_variable(checkpoint, name)
        if name in ['bert/embeddings/word_embeddings', 'cls/predictions/output_bias']:
            return self.load_embedding(variable)
        if name == 'cls/seq_relationship/output_weights':
            return variable.T
        else:
            return variable

    def create_variable(self, name, value):
        if name == 'cls/seq_relationship/output_weights':
            value = value.T
        return super(Bert, self).create_variable(name, value)

    def variable_mapping(self):
        r"""Map standard weight with weight in current class.
        """
        mapping = {
            'Embedding-Token': ['bert/embeddings/word_embeddings'],
            'Embedding-Segment': ['bert/embeddings/token_type_embeddings'],
            'Embedding-Position': ['bert/embeddings/position_embeddings'],
            'Embedding-Norm': [
                'bert/embeddings/LayerNorm/beta',
                'bert/embeddings/LayerNorm/gamma',
            ],
            'Embedding-Mapping': [
                'bert/encoder/embedding_hidden_mapping_in/kernel',
                'bert/encoder/embedding_hidden_mapping_in/bias',
            ],
            'Pooler-Dense': [
                'bert/pooler/dense/kernel',
                'bert/pooler/dense/bias',
            ],
            'NSP-Proba': [
                'cls/seq_relationship/output_weights',
                'cls/seq_relationship/output_bias',
            ],
            'MLM-Dense': [
                'cls/predictions/transform/dense/kernel',
                'cls/predictions/transform/dense/bias',
            ],
            'MLM-Norm': [
                'cls/predictions/transform/LayerNorm/beta',
                'cls/predictions/transform/LayerNorm/gamma',
            ],
            'MLM-Bias': ['cls/predictions/output_bias'],
        }

        for i in range(self.num_hidden_layers):
            prefix = 'bert/encoder/layer_%d/' % i
            mapping.update({
                'Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'attention/self/query/kernel',
                    prefix + 'attention/self/query/bias',
                    prefix + 'attention/self/key/kernel',
                    prefix + 'attention/self/key/bias',
                    prefix + 'attention/self/value/kernel',
                    prefix + 'attention/self/value/bias',
                    prefix + 'attention/output/dense/kernel',
                    prefix + 'attention/output/dense/bias',
                ],
                'Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'attention/output/LayerNorm/beta',
                    prefix + 'attention/output/LayerNorm/gamma',
                ],
                'Transformer-%d-FeedForward' % i: [
                    prefix + 'intermediate/dense/kernel',
                    prefix + 'intermediate/dense/bias',
                    prefix + 'output/dense/kernel',
                    prefix + 'output/dense/bias',
                ],
                'Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'output/LayerNorm/beta',
                    prefix + 'output/LayerNorm/gamma',
                ],
            })

        return mapping


class T5Base(Transformer):
    r"""T5 model with version t5.1.1.

    See more at:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/master/released_checkpoints.md#t511
    """

    def __init__(self, version='t5.1.1', **kwargs):
        super(T5Base, self).__init__(**kwargs)
        self.version = version

    def load_variable(self, checkpoint, name):
        variable = super(T5Base, self).load_variable(checkpoint, name)
        if name == 'shared/embedding':
            return self.load_embedding(variable)
        elif name == 'decoder/logits/kernel':
            return self.load_embedding(variable.T).T
        elif 'relative_attention_bias' in name:
            return variable.T
        else:
            return variable

    def create_variable(self, name, value):
        if 'relative_attention_bias' in name:
            value = value.T
        return super(T5Base, self).create_variable(name, value)

    def variable_mapping(self):
        mapping = {
            'Embedding-Token': ['shared/embedding'],
            'Encoder-Embedding-Relative-Position': [
                'encoder/block_000/layer_000/SelfAttention/relative_attention_bias'
            ],
            'Encoder-Output-Norm': ['encoder/final_layer_norm/scale'],
            'Decoder-Embedding-Relative-Position': [
                'decoder/block_000/layer_000/SelfAttention/relative_attention_bias',
            ],
            'Decoder-Output-Norm': ['decoder/final_layer_norm/scale'],
        }

        for i in range(self.num_hidden_layers):
            # encoder parameters
            prefix = 'encoder/block_%03d/' % i
            mapping.update({
                'Encoder-Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'layer_000/SelfAttention/q',
                    prefix + 'layer_000/SelfAttention/k',
                    prefix + 'layer_000/SelfAttention/v',
                    prefix + 'layer_000/SelfAttention/o',
                ],
                'Encoder-Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'layer_000/layer_norm/scale',
                ],
                'Encoder-Transformer-%d-FeedForward' % i: [
                    prefix + 'layer_001/DenseReluDense/wi/kernel',
                    prefix + 'layer_001/DenseReluDense/wo/kernel',
                ],
                'Encoder-Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'layer_001/layer_norm/scale',
                ],
            })

            # decoder parameters
            prefix = 'decoder/block_%03d/' % i
            mapping.update({
                'Decoder-Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'layer_000/SelfAttention/q',
                    prefix + 'layer_000/SelfAttention/k',
                    prefix + 'layer_000/SelfAttention/v',
                    prefix + 'layer_000/SelfAttention/o',
                ],
                'Decoder-Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'layer_000/layer_norm/scale',
                ],
                'Decoder-Transformer-%d-MultiHeadCrossAttention' % i: [
                    prefix + 'layer_001/EncDecAttention/q',
                    prefix + 'layer_001/EncDecAttention/k',
                    prefix + 'layer_001/EncDecAttention/v',
                    prefix + 'layer_001/EncDecAttention/o',
                ],
                'Decoder-Transformer-%d-MultiHeadCrossAttention-Norm' % i: [
                    prefix + 'layer_001/layer_norm/scale',
                ],
                'Decoder-Transformer-%d-FeedForward' % i: [
                    prefix + 'layer_002/DenseReluDense/wi/kernel',
                    prefix + 'layer_002/DenseReluDense/wo/kernel',
                ],
                'Decoder-Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'layer_002/layer_norm/scale',
                ],
            })

        if self.version == 't5.1.1':
            mapping['Encoder-Output-Norm'] = ['encoder/rms_norm/scale']
            mapping['Decoder-Output-Norm'] = ['decoder/rms_norm/scale']
            mapping['Decoder-Output-LM'] = ['decoder/logits/kernel']
            mapping = {
                k: [i.replace('layer_norm', 'rms_norm') for i in v]
                for k, v in mapping.items()
            }
            for i in range(self.num_hidden_layers):
                for layer in [
                    'Encoder-Transformer-%d-FeedForward' % i,
                    'Decoder-Transformer-%d-FeedForward' % i
                ]:
                    mapping[layer] = [
                        mapping[layer][0][:-7] + '_0' + mapping[layer][0][-7:],
                        mapping[layer][0][:-7] + '_1' + mapping[layer][0][-7:],
                        mapping[layer][1]
                    ]

        return mapping


class T5Encoder(T5Base):
    def compute_relative_position(self, inputs=None):
        if self.relative_position is None:
            self.relative_position = self.apply(
                inputs=[inputs, inputs],
                layer=RelativePositionEmbeddingT5,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=True,
                embeddings_initializer=self.initializer,
                name='Encoder-Embedding-Relative-Position',
            )
        return self.relative_position

    def get_inputs(self, **kwargs):
        x_token = self.apply(
            layer=Input,
            shape=(self.fixed_sequence_length,),
            name='Encoder-Input-Token'
        )

        return x_token

    def apply_embeddings(self, inputs, **kwargs):
        x = self.apply(
            inputs=inputs,
            layer=DenseEmbedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token',
        )

        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.hidden_dropout_prob,
            name='Encoder-Embedding-Dropout'
        )

        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Encoder-Embedding-Mapping'
            )

        return x

    def apply_main_layers(self,
                          inputs,
                          index,
                          layer_norm_cond_inputs=None,
                          layer_norm_cond_hidden_size=None,
                          layer_norm_cond_hidden_act=None,
                          **kwargs):
        r"""
        In each layer, tensors flow in the following order:
        Layer Normalization --> Attention --> Add --> Layer Normalization --> Feed Forward --> Add
        """
        attention_name = 'Encoder-Transformer-{}-MultiHeadSelfAttention'.format(index)
        feed_forward_name = 'Encoder-Transformer-{}-FeedForward'.format(index)
        relative_position = self.compute_relative_position(inputs)

        x = inputs

        xt = self.apply(
            inputs=x if layer_norm_cond_inputs is None else [x, layer_norm_cond_inputs],
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=layer_norm_cond_inputs is not None,
            cond_hidden_units=layer_norm_cond_hidden_size,
            cond_hidden_activation=layer_norm_cond_hidden_act,
            cond_hidden_initializer=self.initializer,
            name='{}-Norm'.format(attention_name),
        )

        xt = self.apply(
            inputs=[xt, xt, xt, relative_position],
            layer=MultiHeadAttention,
            head_num=self.num_attention_heads,
            head_size=self.attention_head_size,
            key_size=self.attention_key_size,
            use_bias=False,
            attention_scale=False,
            kernel_initializer=self.initializer,
            arguments={'relative_position': 't5'},
            name=attention_name,
        )

        if self.attention_dropout_prob:
            xt = self.apply(
                inputs=xt,
                layer=Dropout,
                rate=self.attention_dropout_prob,
                name='{}-Dropout'.format(attention_name),
            )

        x = self.apply(
            inputs=[x, xt],
            layer=Add,
            name='{}-Add'.format(attention_name),
        )

        xt = self.apply(
            inputs=x if layer_norm_cond_inputs is None else [x, layer_norm_cond_inputs],
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=layer_norm_cond_inputs is not None,
            cond_hidden_units=layer_norm_cond_hidden_size,
            cond_hidden_activation=layer_norm_cond_hidden_act,
            cond_hidden_initializer=self.initializer,
            name='{}-Norm'.format(feed_forward_name),
        )

        xt = self.apply(
            inputs=xt,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            use_bias=False,
            kernel_initializer=self.initializer,
            name=feed_forward_name,
        )

        if self.hidden_dropout_prob:
            xt = self.apply(
                inputs=xt,
                layer=Dropout,
                rate=self.hidden_dropout_prob,
                name='{}-Dropout'.format(feed_forward_name),
            )

        x = self.apply(
            inputs=[x, xt],
            layer=Add,
            name='{}-Add'.format(feed_forward_name),
        )

        return x

    def apply_output_layers(self,
                            inputs,
                            layer_norm_cond_inputs=None,
                            layer_norm_cond_hidden_size=None,
                            layer_norm_cond_hidden_act=None,
                            **kwargs):
        x = self.apply(
            inputs=inputs if layer_norm_cond_inputs is None else [inputs, layer_norm_cond_inputs],
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=layer_norm_cond_inputs is not None,
            cond_hidden_units=layer_norm_cond_hidden_size,
            cond_hidden_activation=layer_norm_cond_hidden_act,
            cond_hidden_initializer=self.initializer,
            name='Encoder-Output-Norm',
        )

        if self.hidden_dropout_prob:
            x = self.apply(
                inputs=x,
                layer=Dropout,
                rate=self.hidden_dropout_prob,
                name='Encoder-Output-Dropout'
            )

        return x


class T5Decoder(LanguageModelMixin, T5Base):
    def __init__(self, with_lm=True, **kwargs):
        super(T5Decoder, self).__init__(**kwargs)
        self.with_lm = with_lm

    def compute_attention_mask(self, inputs=None, **kwargs):
        mask = super(T5Decoder, self).compute_attention_mask(self.inputs[1])
        return mask

    def compute_relative_position(self, inputs=None):
        if self.relative_position is None:
            context, x = inputs
            rp1 = self.apply(
                inputs=[x, x],
                layer=RelativePositionEmbeddingT5,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=False,
                embeddings_initializer=self.initializer,
                name='Decoder-Embedding-Relative-Position',
            )
            rp2 = self.apply(
                inputs=[x, context],
                layers=RelativePositionEmbeddingT5,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=False,
                embeddings_initializer=self.initializer,
                name='Decoder-Embedding-Relative-Position',
            )

            self.relative_position = (rp1, rp2)

        return self.relative_position

    def get_inputs(self, **kwargs):
        context_in = self.apply(
            layer=Input,
            shape=(self.fixed_sequence_length, self.hidden_size),
            name='Input-Context',
        )

        x_in = self.apply(
            layer=Input,
            shape=(self.fixed_sequence_length,),
            name='Decoder-Input-Token',
        )

        return [context_in, x_in]

    def apply_embeddings(self, inputs, **kwargs):
        context, x = inputs

        x = self.apply(
            inputs=x,
            layer=DenseEmbedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token',
        )

        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.hidden_dropout_prob,
            name='Decoder-Embedding-Dropout',
        )

        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Decoder-Embedding-Mapping'
            )

        return [context, x]

    def apply_main_layers(self,
                          inputs,
                          index,
                          layer_norm_cond_inputs=None,
                          layer_norm_cond_hidden_size=None,
                          layer_norm_cond_hidden_act=None,
                          **kwargs):
        r"""
        In each layer, tensors flow in the following order:
        Layer Normalization --> Attention1 --> Add --> Layer Normalization --> Attention2 --> Add -->
        Layer Normalization --> Feed Forward --> Add
        """
        context, x = inputs

        self_attention_name = 'Decoder-Transformer-{}-MultiHeadSelfAttention'.format(index)
        cross_attention_name = 'Decoder-Transformer-{}-MultiHeadCrossAttention'.format(index)
        feed_forward_name = 'Decoder-Transformer-{}-FeedForward'.format(index)
        attention_mask = self.compute_attention_mask()
        self_rp, cross_rp = self.compute_relative_position(inputs)

        # self attention
        xt = self.apply(
            inputs=x if layer_norm_cond_inputs is None else [x, layer_norm_cond_inputs],
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=layer_norm_cond_inputs is not None,
            cond_hidden_units=layer_norm_cond_hidden_size,
            cond_hidden_activation=layer_norm_cond_hidden_act,
            cond_hidden_initializer=self.initializer,
            name='{}-Norm'.format(self_attention_name),
        )

        xt = self.apply(
            inputs=[xt, xt, xt, attention_mask, self_rp],
            layer=MultiHeadAttention,
            head_num=self.num_attention_heads,
            head_size=self.attention_head_size,
            key_size=self.attention_key_size,
            use_bias=False,
            attention_scale=False,
            kernel_initializer=self.initializer,
            arguments={
                'attention_mask': True,
                'relative_position': 't5'
            },
            name=self_attention_name,
        )

        if self.attention_dropout_prob:
            xt = self.apply(
                inputs=xt,
                layer=Dropout,
                rate=self.attention_dropout_prob,
                name='{}-Dropout'.format(self_attention_name),
            )

        x = self.apply(
            inputs=[x, xt],
            layer=Add,
            name='{}-Add'.format(self_attention_name),
        )

        # cross attention
        xt = self.apply(
            inputs=x if layer_norm_cond_inputs is None else [x, layer_norm_cond_inputs],
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=layer_norm_cond_inputs is not None,
            cond_hidden_units=layer_norm_cond_hidden_size,
            cond_hidden_activation=layer_norm_cond_hidden_act,
            cond_hidden_initializer=self.initializer,
            name='{}-Norm'.format(cross_attention_name),
        )

        xt = self.apply(
            inputs=[xt, context, context, cross_rp],
            layer=MultiHeadAttention,
            head_num=self.num_attention_heads,
            head_size=self.attention_head_size,
            key_size=self.attention_key_size,
            use_bias=False,
            attention_scale=False,
            kernel_initializer=self.initializer,
            arguments={
                'relative_position': 't5'
            },
            name=cross_attention_name,
        )

        if self.attention_dropout_prob:
            xt = self.apply(
                inputs=xt,
                layer=Dropout,
                rate=self.attention_dropout_prob,
                name='{}-Dropout'.format(cross_attention_name),
            )

        x = self.apply(
            inputs=[x, xt],
            layer=Add,
            name='{}-Add'.format(cross_attention_name),
        )

        # feed forward
        xt = self.apply(
            inputs=x if layer_norm_cond_inputs is None else [x, layer_norm_cond_inputs],
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=layer_norm_cond_inputs is not None,
            cond_hidden_units=layer_norm_cond_hidden_size,
            cond_hidden_activation=layer_norm_cond_hidden_act,
            cond_hidden_initializer=self.initializer,
            name='{}-Norm'.format(feed_forward_name),
        )

        xt = self.apply(
            inputs=xt,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            use_bias=False,
            kernel_initializer=self.initializer,
            name=feed_forward_name,
        )

        if self.hidden_dropout_prob:
            xt = self.apply(
                inputs=xt,
                layer=Dropout,
                rate=self.hidden_dropout_prob,
                name='{}-Dropout'.format(feed_forward_name),
            )

        x = self.apply(
            inputs=[x, xt],
            layer=Add,
            name='{}-Add'.format(feed_forward_name),
        )

        return [context, x]

    def apply_output_layers(self,
                            inputs,
                            layer_norm_cond_inputs=None,
                            layer_norm_cond_hidden_size=None,
                            layer_norm_cond_hidden_act=None,
                            **kwargs):
        context, x = inputs

        x = self.apply(
            inputs=x if layer_norm_cond_inputs is None else [x, layer_norm_cond_inputs],
            layer=LayerNormalization,
            center=False,
            epsilon=1e-6,
            conditional=layer_norm_cond_inputs is not None,
            cond_hidden_units=layer_norm_cond_hidden_size,
            cond_hidden_activation=layer_norm_cond_hidden_act,
            cond_hidden_initializer=self.initializer,
            name='Decoder-Output-Norm',
        )

        if self.hidden_dropout_prob:
            x = self.apply(
                inputs=x,
                layer=Dropout,
                rate=self.hidden_dropout_prob,
                name='Decoder-Output-Dropout'
            )

        x = self.apply(
            inputs=x,
            layer=Lambda,
            function=lambda x1: x1 / np.sqrt(self.hidden_size),
            mask=lambda i, m: m,
            name='Decoder-Output-Scale'
        )

        if self.with_lm:
            # predict tokens' probabilities
            if self.embedding_size != self.hidden_size:
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=self.embedding_size,
                    kernel_initializer=self.initializer,
                    name='Decoder-Output-Mapping'
                )

            if self.version == 't5.1.1':
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=self.vocab_size,
                    activation='softmax',
                    use_bias=False,
                    kernel_initializer=self.initializer,
                    name='Decoder-Output-LM',
                )
            else:
                x = self.apply(
                    inputs=x,
                    layer=DenseEmbedding,
                    arguments={'mode': 'dense'},
                    name='Embedding-Token'
                )
                x = self.apply(
                    inputs=x,
                    layer=Activation,
                    activation='softmax',
                    name='Decoder-Output-LM-Activation'
                )

        return x


class T5(T5Base):
    r"""T5 model.
    """

    def __init__(self, **kwargs):
        super(T5, self).__init__(**kwargs)
        kwargs['layers'] = self.layers

        encoder_name = 'Encoder'
        decoder_name = 'Decoder'
        if 'name' in kwargs:
            encoder_name = '{}_{}'.format(kwargs['name'], encoder_name)
            decoder_name = '{}_{}'.format(kwargs['name'], decoder_name)
            del kwargs['name']

        self.encoder_model = T5Encoder(name=encoder_name, **kwargs)
        self.decoder_model = T5Decoder(name=decoder_name, **kwargs)

    def build(self, **kwargs):
        encoder_kwargs = copy.copy(kwargs)
        decoder_kwargs = copy.copy(kwargs)

        if 'additional_inputs' in kwargs:
            total_additional_inputs = []
            encoder_additional_inputs = []
            decoder_additional_inputs = []
            if 'encoder' in kwargs['additional_inputs']:
                add_inputs = kwargs['additional_inputs']['encoder']
                if isinstance(add_inputs, list):
                    total_additional_inputs.extend(add_inputs)
                    encoder_additional_inputs.extend(add_inputs)
                else:
                    total_additional_inputs.append(add_inputs)
                    encoder_additional_inputs.append(add_inputs)
            if 'decoder' in kwargs['additional_inputs']:
                add_inputs = kwargs['additional_inputs']['decoder']
                if isinstance(add_inputs, list):
                    total_additional_inputs.extend(add_inputs)
                    decoder_additional_inputs.extend(add_inputs)
                else:
                    total_additional_inputs.append(add_inputs)
                    decoder_additional_inputs.append(add_inputs)
            if 'both' in kwargs['additional_inputs']:
                add_inputs = kwargs['additional_inputs']['both']
                if isinstance(add_inputs, list):
                    total_additional_inputs.extend(add_inputs)
                    encoder_additional_inputs.extend(add_inputs)
                    decoder_additional_inputs.extend(add_inputs)
                else:
                    total_additional_inputs.append(add_inputs)
                    encoder_additional_inputs.append(add_inputs)
                    decoder_additional_inputs.append(add_inputs)
            encoder_kwargs['additional_inputs'] = encoder_additional_inputs or None
            decoder_kwargs['additional_inputs'] = decoder_additional_inputs or None

        self.encoder_model.build(**encoder_kwargs)
        self.decoder_model.build(**decoder_kwargs)

        inputs = self.encoder_model.inputs + self.decoder_model.inputs[1:]
        if 'additional_inputs' in kwargs:
            inputs = [i for i in inputs if i not in total_additional_inputs] + total_additional_inputs
        self.inputs = inputs
        self.outputs = self.decoder_model.model(self.encoder_model.outputs + self.decoder_model.inputs[1:])
        self.model = Model(self.inputs, self.outputs)

    @property
    def encoder(self):
        return self.encoder_model.model

    @property
    def decoder(self):
        return self.decoder_model.model
