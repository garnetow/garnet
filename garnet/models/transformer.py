# coding: utf-8

"""
@File   : transformer.py
@Author : garnet
@Time   : 2020/9/29 8:48
"""

import numpy as np
import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Add, Lambda, Activation

from .model import WrappedModel
from ..layers.core import BiasAdd
from ..layers.bert import PositionEmbedding
from ..layers.bert import LayerNormalization
from ..layers.bert import MultiHeadAttention
from ..layers.bert import FeedForward
from ..layers.bert import DenseEmbedding
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
        self.position_bias = None
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
            additional_inputs=additional_inputs,
            **kwargs
        )
        self.set_inputs(inputs)

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

    def set_inputs(self, inputs):
        if inputs is None:
            inputs = []
        elif not isinstance(inputs, list):
            inputs = [inputs]

        inputs = inputs[:]  # shallow copy
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
                   additional_inputs=None,
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

        if additional_inputs is not None:
            if isinstance(additional_inputs, list):
                inputs.extend(additional_inputs)
            else:
                inputs.append(additional_inputs)
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
            'head_mask': head_mask,
            'attention_mask': attention_mask,
            'encoder_hidden_context': encoder_hidden_context,
            'encode_attention_mask': encode_attention_mask,
        }

        # apply self-attention
        xt = self.apply(
            inputs=[x, x, x],
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
