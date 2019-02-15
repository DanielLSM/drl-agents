import os
import pickle
import zipfile
import random
import numpy as np
import tensorflow as tf

# ================================================================
# Seeding
# ================================================================


def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        tf.set_random_seed(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)


def set_seeds(seed):
    """ My simple seeds version without multiprocessing """
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ================================================================
# Pretty printings
# ================================================================


def pretty_eta(seconds_left):
    """Print the number of seconds in human readable format.

    Examples:
    2 days
    2 hours and 37 minutes
    less than a minute

    Paramters
    ---------
    seconds_left: int
        Number of seconds to be converted to the ETA
    Returns
    -------
    eta: str
        String representing the pretty ETA.
    """
    minutes_left = seconds_left // 60
    seconds_left %= 60
    hours_left = minutes_left // 60
    minutes_left %= 60
    days_left = hours_left // 24
    hours_left %= 24

    def helper(cnt, name):
        return "{} {}{}".format(str(cnt), name, ('s' if cnt > 1 else ''))

    if days_left > 0:
        msg = helper(days_left, 'day')
        if hours_left > 0:
            msg += ' and ' + helper(hours_left, 'hour')
        return msg
    if hours_left > 0:
        msg = helper(hours_left, 'hour')
        if minutes_left > 0:
            msg += ' and ' + helper(minutes_left, 'minute')
        return msg
    if minutes_left > 0:
        return helper(minutes_left, 'minute')
    return 'less than a minute'


# ================================================================
# Save and load compressed python objects
# ================================================================


def relatively_safe_pickle_dump(obj, path, compression=False):
    """This is just like regular pickle dump, except from the fact that failure cases are
    different:

        - It's never possible that we end up with a pickle in corrupted state.
        - If a there was a different file at the path, that file will remain unchanged in the
          even of failure (provided that filesystem rename is atomic).
        - it is sometimes possible that we end up with useless temp file which needs to be
          deleted manually (it will be removed automatically on the next function call)

    The indended use case is periodic checkpoints of experiment state, such that we never
    corrupt previous checkpoints if the current one fails.

    Parameters
    ----------
    obj: object
        object to pickle
    path: str
        path to the output file
    compression: bool
        if true pickle will be compressed
    """
    temp_storage = path + ".relatively_safe"
    if compression:
        # Using gzip here would be simpler, but the size is limited to 2GB
        with tempfile.NamedTemporaryFile() as uncompressed_file:
            pickle.dump(obj, uncompressed_file)
            uncompressed_file.file.flush()
            with zipfile.ZipFile(
                    temp_storage, "w",
                    compression=zipfile.ZIP_DEFLATED) as myzip:
                myzip.write(uncompressed_file.name, "data")
    else:
        with open(temp_storage, "wb") as f:
            pickle.dump(obj, f)
    os.rename(temp_storage, path)


def pickle_load(path, compression=False):
    """Unpickle a possible compressed pickle.

    Parameters
    ----------
    path: str
        path to the output file
    compression: bool
        if true assumes that pickle was compressed when created and attempts decompression.

    Returns
    -------
    obj: object
        the unpickled object
    """

    if compression:
        with zipfile.ZipFile(
                path, "r", compression=zipfile.ZIP_DEFLATED) as myzip:
            with myzip.open("data") as f:
                return pickle.load(f)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)


# ================================================================
# Exploration from OpenAI
# ================================================================


class LinearSchedule(object):

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)