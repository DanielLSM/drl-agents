#Some models can be found here: https://github.com/openai/baselines

import tensorflow as tf
import tensorflow.contrib.layers as layers

from drl.tools.math_util import *
from drl.tools.tf_util import *


def q_mlp(hiddens, input_, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = input_
        for hidden in hiddens:
            out = layers.fully_connected(
                out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)
        q_out = layers.fully_connected(
            out, num_outputs=num_actions, activation_fn=None)
        return q_out
