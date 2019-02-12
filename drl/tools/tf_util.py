#code and authorship on https://github.com/openai/baselines
import os
import joblib
import multiprocessing

import numpy as np
import tensorflow as tf

# ================================================================
# Session related functions
# ================================================================


def get_session(config=None):
    """Get default session or create one with a given config"""
    sess = tf.get_default_session()
    if sess is None:
        sess = make_session(config=config, make_default=True)
    return sess


#TODO: This function as been changed from OpenAI baselines, I think config.gpu_options.allow_growth is deprecated
def make_session(config=None, num_cpu=None, make_default=False, graph=None):
    """Returns a session that will use <num_cpu> CPU's only"""
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    if config is None:
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu,
            gpu_options=gpu_options)
        # from OpenAI baselines
        # config.gpu_options.allow_growth = True

    if make_default:
        return tf.InteractiveSession(config=config, graph=graph)
    else:
        return tf.Session(config=config, graph=graph)


def single_threaded_session():
    """Returns a session which will only use a single CPU"""
    return make_session(num_cpu=1)


# ================================================================
# Initialize graph functions
# ================================================================

ALREADY_INITIALIZED = set()


def initialize():
    """Initialize all the uninitialized variables in the global scope."""
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)


def normc_initializer(std=1.0, axis=0):

    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)

    return _initializer


# ================================================================
# Save and load graph variables
# ================================================================


def save_variables(save_path, variables=None, sess=None):
    sess = sess or get_session()
    variables = variables or tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    ps = sess.run(variables)
    save_dict = {v.name: value for v, value in zip(variables, ps)}
    dirname = os.path.dirname(save_path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(save_dict, save_path)


def load_variables(load_path, variables=None, sess=None):
    sess = sess or get_session()
    variables = variables or tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    loaded_params = joblib.load(os.path.expanduser(load_path))
    restores = []
    if isinstance(loaded_params, list):
        assert len(loaded_params) == len(
            variables), 'number of variables loaded mismatches len(variables)'
        for d, v in zip(loaded_params, variables):
            restores.append(v.assign(d))
    else:
        for v in variables:
            restores.append(v.assign(loaded_params[v.name]))

    sess.run(restores)


# ================================================================
# Tensorboard monitoring
# ================================================================


def launch_tensorboard_in_background(log_dir):
    '''
    To log the Tensorflow graph when using rl-algs
    algorithms, you can run the following code
    in your main script:
        import threading, time
        def start_tensorboard(session):
            time.sleep(10) # Wait until graph is setup
            tb_path = osp.join(logger.get_dir(), 'tb')
            summary_writer = tf.summary.FileWriter(tb_path, graph=session.graph)
            summary_op = tf.summary.merge_all()
            launch_tensorboard_in_background(tb_path)
        session = tf.get_default_session()
        t = threading.Thread(target=start_tensorboard, args=([session]))
        t.start()
    '''
    import subprocess
    subprocess.Popen(['tensorboard', '--logdir', log_dir])


# ================================================================
# Shape adjustment for feeding into tf placeholders
# ================================================================
def adjust_shape(placeholder, data):
    '''
    adjust shape of the data to the shape of the placeholder if possible.
    If shape is incompatible, AssertionError is thrown

    Parameters:
        placeholder     tensorflow input placeholder

        data            input data to be (potentially) reshaped to be fed into placeholder

    Returns:
        reshaped data
    '''

    if not isinstance(data, np.ndarray) and not isinstance(data, list):
        return data
    if isinstance(data, list):
        data = np.array(data)

    placeholder_shape = [x or -1 for x in placeholder.shape.as_list()]


    assert _check_shape(placeholder_shape, data.shape), \
        'Shape of data {} is not compatible with shape of the placeholder {}'.format(data.shape, placeholder_shape)

    return np.reshape(data, placeholder_shape)


def _check_shape(placeholder_shape, data_shape):
    ''' check if two shapes are compatible (i.e. differ only by dimensions of size 1, or by the batch dimension)'''

    squeezed_placeholder_shape = _squeeze_shape(placeholder_shape)
    squeezed_data_shape = _squeeze_shape(data_shape)

    for i, s_data in enumerate(squeezed_data_shape):
        s_placeholder = squeezed_placeholder_shape[i]
        if s_placeholder != -1 and s_data != s_placeholder:
            return False

    return True


def _squeeze_shape(shape):
    return [x for x in shape if x != 1]


# ================================================================
# Observations from open AI
# ================================================================

# ================================================================
# My helper functions to create nodes in the tf.Graph
# ================================================================


def get_placeholder(batch_size, shape, dtype, name):
    placeholder = tf.placeholder(
        shape=(batch_size,) + shape, dtype=dtype, name=name)

    #from input.py in baselines deepq, they recast the tensor to tf.float
    # placeholder = tf.to_float(placeholder)
    return placeholder


if __name__ == '__main__':
    make_session()
    get_session()