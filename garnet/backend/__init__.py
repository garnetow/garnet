# coding: utf-8

"""
@File   : __init__.py
@Author : garnet
@Time   : 2020/10/12 8:47
"""

import os
import sys

is_tf_keras = os.environ.get('TF_KERAS', '0')
if is_tf_keras == '1':  # use tf.keras
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
    sys.modules['keras'] = keras
else:  # use original keras
    import keras
    import keras.backend as K

from .mask import sequence_masking
from .mask import unilm_mask
from .mask import language_model_mask

from .recompute import recompute_grad
