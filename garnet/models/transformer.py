# coding: utf-8

"""
@File   : transformer.py
@Author : garnet
@Time   : 2020/9/29 8:48
"""

import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, Add

from .model import WrappedModel
from ..layers import PositionEmbedding


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
            :param prefix (:obj:`str`, optional, default: `None`):
                prefix of the every layers' name in the model.
            :param name (:obj:`str`, optional, default: `None`):
                name of the model.
        """
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
        self.inputs = inputs

        if len(inputs) > 1:
            self.input = self.inputs
        elif len(inputs) == 1:
            self.input = self.inputs[0]

    def set_outputs(self, outputs):
        if not isinstance(outputs, list):
            outputs = [outputs]
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

    @staticmethod
    def load_variable(self, checkpoint, name):
        r"""Load weight of single variable.
        """
        return tf.train.load_variable(checkpoint, name)

    def create_variable(self, name, value):
        r"""Initial variable with truncated normal distribution.
        """
        return K.variable(self.initializer(value.shape), name=name)

    @property
    def initializer(self):
        return keras.initializers.TruncatedNormal(stddev=0.02)


class Bert(Transformer):
    def __init__(self,
                 segment_vocab_size=2,  # Segment总数目
                 with_pool=False,  # 输出是否增加[CLS]部分
                 with_nsp=False,  # 输出是否增加 Next Sentence Prediction 部分
                 with_mlm=False,  # 输出是否增加 Masked Language Model 部分
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
            :param shared_segment_embeddings (:obj:`bool`, optional, default: False):
                if `True`, segment tokens share the embedding matrix with normal tokens.
        """
        super().__init__(**kwargs)
        self.segment_vocab_size = segment_vocab_size
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.shared_segment_embeddings = shared_segment_embeddings

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

    def apply_embeddings(self, inputs, input_embeds=None, custom_position_ids=None, **kwargs):
        if input_embeds is not None:
            x = input_embeds
        else:
            x = inputs.pop(0)
            x = self.apply(
                inputs=x,
                layer=Embedding,
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
                layer=Embedding,
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
            inputs=[x, p] if p is not None else x,
            layer=PositionEmbedding,
            input_dim=self.max_position_embeddings,
            output_dim=self.embedding_size,
            merge_mode='add',
            embeddings_initializer=self.initializer,
            name='Embedding-Position',
        )
