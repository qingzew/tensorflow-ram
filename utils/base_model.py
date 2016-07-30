import re
import tensorflow as tf

class BaseModel(object):
    def __init__(self):
        self.need_save = None
        self.need_save_fine = None

    def _activation_summary(self, x):
        """Helper to create summaries for activations.

        Creates a summary that provides a histogram of activations.
        Creates a summary that measure the sparsity of activations.

        Args:
            x: Tensor
        Returns:
            nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
        tf.histogram_summary(tensor_name + '/activations', x)
        tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


    def _variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.

        Args:
            name: name of the variable
            shape: list of ints
            initializer: initializer for Variable

        Returns:
            Variable Tensor
        """
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer = initializer)
        return var

    def _variable_with_weight_decay(self, name, shape, method = 'normal', mean = 0, stddev = 0.01, wd = 0.0005):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.

        Args:
            name: name of the variable
            shape: list of ints
            stddev: standard deviation of a truncated Gaussian
            wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.

        Returns:
            Variable Tensor
        """
        if method == 'normal':
            var = self._variable_on_cpu(name, shape,
                               tf.random_normal_initializer(mean = mean, stddev = stddev))
        elif method == 'xavier':
            var = self._variable_on_cpu(name, shape,
                               tf.contrib.layers.xavier_initializer())

        if wd is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name = 'weight_loss')
            tf.add_to_collection('losses', weight_decay)

        return var

