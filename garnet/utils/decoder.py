# coding: utf-8

"""
@File   : decoder.py
@Author : garnet
@Time   : 2020/10/26 9:05
"""

import numpy as np
from functools import wraps

from .functions.activations import softmax


class AutoRegressiveDecoder(object):
    r"""Base model of general auto-regressive decoder for generative models.

    Generating strategy including:
        - beam search
        - random sample
    """

    def __init__(self, end_index, max_length, min_length=1, start_index=None):
        self.start_index = start_index
        self.end_index = end_index
        self.max_length = max_length
        self.min_length = min_length
        self.first_output_index = np.array([[self.start_index]]) if self.start_index is not None else \
            np.empty((1, 0), dtype='int')

    @staticmethod
    def predict_wraps(default_return_type='probas', use_states=False):
        assert default_return_type in ('logits', 'probas'), "`return_type` must be one of `logits` or `probas`, " \
                                                            "got {} instead".format(default_return_type)

        def decorator(predict):
            @wraps(predict)
            def wrapper(self, inputs, output_indices, states=None, return_type=default_return_type):
                res = predict(self, inputs, output_indices, states=states, return_type=return_type)
                if not use_states:
                    res = (res, None)

                if default_return_type == 'logits':
                    res = (softmax(res[0]), res[1])  # unified into probas
                if return_type == 'probas':
                    return res
                else:
                    return np.log(res[0] + 1e-12), res[1]

            return wrapper

        return decorator

    def predict(self, inputs, output_indices, states=None, return_type='probas'):
        r"""Custom prediction method.

        Predict probas or logits of the next token in given samples.

        Args:
            :param inputs: known tokens.
            :param output_indices: predicted tokens of last step.
            :param states: some prediction needs states.
            :param return_type: output type, `logits` or `probas`.
        Returns:
            logits or probas
        """
        raise NotImplementedError

    def random_sample(self, inputs, n, top_k=None, top_p=None, states=None, min_ends_per_sample=1):
        r"""Generate next sentence using random sample method.

        Get more information in below webs:
        - [如何应对Seq2Seq中的“根本停不下来”问题？](https://kexue.fm/archives/7500)
        - [top-k random search(Hierarchical Neural Story Generation)](https://arxiv.org/abs/1805.04833)
        - [Nucleus random search(The Curious Case of Neural Text Degeneration)](https://arxiv.org/abs/1904.09751)

        Args:
            :param inputs: tokens which have been already known.
            :param n: number of samples taken from each sample process.
            :param top_k: if not `None`, reserve `k` tokens with highest probabilities.
            :param top_p: if not `None`, reserve highest probabilities tokens which cumsum probabilities exactly
                over `p`.
            :param states: some prediction needs states.
            :param min_ends_per_sample: number of end token coming up times which controls the end of prediction of
                one sample.
        Returns:
            Decode sequence list with `n` samples.
        """
        result = []
        if not top_p:
            top_k = top_k or n

        inputs = [np.repeat(np.array([ipt]), n, axis=0) for ipt in inputs]
        output_indices = np.repeat(self.first_output_index, n, axis=0)

        for step in range(self.max_length):
            probas, states = self.predict(inputs, output_indices=output_indices, states=states, return_type='probas')
            probas /= probas.sum(axis=1, keepdims=True)  # probas normalization

            if top_k:
                k_indices = probas.argpartition(-top_k, axis=1)[:, -top_k:]  # top k token indices
                probas = np.take_along_axis(probas, k_indices, axis=1)  # top k probas
                probas /= probas.sum(axis=1, keepdims=True)  # probas re-normalization
            if top_p:
                p_indices = probas.argsort(axis=1)[:, ::-1]  # sort from high probabilities to low
                probas = np.take_along_axis(probas, p_indices, axis=1)
                cum_probas = np.cumsum(probas, axis=1)  # accumulative probas
                excluded = np.roll(cum_probas >= top_p, shift=1, axis=1)
                excluded[:, 0] = False
                probas[excluded] = 0.
                probas /= probas.sum(axis=1, keepdims=True)  # probas re-normalization

            # sample process
            sample_indices = np.apply_along_axis(lambda p: np.random.choice(len(p), p=p), axis=1, arr=probas)
            sample_indices = sample_indices.reshape((-1, 1))

            if top_p:
                sample_indices = np.take_along_axis(p_indices, sample_indices, axis=1)
            if top_k:
                sample_indices = np.take_along_axis(k_indices, sample_indices, axis=1)

            # update decode output sequence
            output_indices = np.concatenate([output_indices, sample_indices], axis=1)

            end_count = np.sum(output_indices == self.end_index, axis=1)
            if output_indices.shape[1] >= self.min_length:
                current_step_completed = end_count == min_ends_per_sample  # number of prediction completed samples
                if current_step_completed.any():
                    for index_sequence in output_indices[current_step_completed]:
                        result.append(index_sequence)

                    uncompleted = ~current_step_completed
                    inputs = [i[uncompleted] for i in inputs]
                    output_indices = output_indices[uncompleted]
                    if len(output_indices) == 0:
                        break

        for index_sequence in output_indices:
            result.append(index_sequence)
        return result

    def beam_search(self, inputs, return_k, cache_k=None, states=None, min_ends_per_sample=1, mode='total'):
        r"""Generate next sentence using beam search method.

        Args:
            :param inputs: tokens which have been already known.
            :param return_k: return top k highest score samples.
            :param cache_k: keep `cache_k` samples with highest scores during decoding search. If `None`, it
                will be assigned as `return_k`.
            :param states: some prediction needs states.
            :param min_ends_per_sample: number of end token coming up times which controls the end of prediction of
                one sample.
        Returns:
            Decode sequence list with `top_k` samples.
        """
        results, result_scores = [], []
        if not cache_k:
            cache_k = return_k

        inputs = [np.array([ipt]) for ipt in inputs]
        output_indices = self.first_output_index
        output_scores = np.zeros(1)
        for step in range(self.max_length):
            scores, states = self.predict(inputs, output_indices=output_indices, states=states, return_type='logits')

            scores = output_scores.reshape((-1, 1)) + scores  # accumulated token scores
            indices = scores.argpartition(-cache_k, axis=None)[-cache_k:]  # flatten array
            indices_row = indices // scores.shape[1]  # indices of row, point out which sample
            indices_col = np.reshape(indices % scores.shape[1], (-1, 1))  # token index
            output_indices = np.concatenate([output_indices[indices_row], indices_col], axis=1)
            output_scores = np.take_along_axis(scores, indices, axis=None)  # get accumulated scores
            inputs = [ipt[indices_row] for ipt in inputs]

            end_count = np.sum(output_indices == self.end_index, axis=1)
            if output_indices.shape[1] >= min_ends_per_sample:
                uncompleted = end_count < min_ends_per_sample
                if not uncompleted.all():
                    best_index = output_scores.argmax()
                    if end_count[best_index] == min_ends_per_sample:
                        results.append(output_indices[best_index])
                        result_scores.append(output_scores[best_index])
                        if len(results) == return_k:
                            break

                    inputs = [ipt[uncompleted] for ipt in inputs]
                    output_indices = output_indices[uncompleted]
                    output_scores = output_scores[uncompleted]
                    if len(output_indices) == 0:
                        break

        if len(results) < return_k:
            for index in np.argsort(-output_scores)[:(return_k - len(results))]:
                results.append(output_indices[index])
                result_scores.append(output_scores[index])

        sorted_results = self.sort_scores(results, result_scores, mode=mode)
        return sorted_results

    @staticmethod
    def sort_scores(samples, scores, mode='total'):
        s = sorted(zip(samples, scores), key=lambda x: x[1], reverse=True)
        return list(zip(*s))[0]
