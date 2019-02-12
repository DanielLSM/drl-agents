#Some models can be found here: https://github.com/openai/baselines

import tensorflow as tf
import tensorflow.contrib.layers as layers

from drl.tools.math_util import *
from drl.tools.tf_util import *


#TODO: inputting a different activation than tf.nn.relu would make sense
def q_mlp(hiddens,
          input_,
          num_actions,
          scope='action_value_function',
          reuse=None,
          layer_norm=False):
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


def q_target_update(q_vars, qt_vars):
    update_target_expr = []
    for var, var_target in zip(
            sorted(q_vars, key=lambda v: v.name),
            sorted(qt_vars, key=lambda v: v.name)):
        update_target_expr.append(var_target.assign(var))
    update_target_expr = tf.group(*update_target_expr)
    return update_target_expr
