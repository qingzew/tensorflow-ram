# =============================================
# using multiple GPU's with synchronous updates.
# =============================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
DECAY_STEP = 100 #  decay step
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
# LEARNING_RATE_DECAY_FACTOR = 0.1 # Learning rate decay factor.
LEARNING_RATE_DECAY_FACTOR = 0.9999  # Learning rate decay factor.

class BaseTrainer(object):
    def __init__(self,
                model,
                reader,
                train_dir = '/tmp/train',
                max_steps = 10000,
                test_steps = 100,
                gpus = [0],
                truncation = False,
                max_grad_norm = 5,
                restore_file = None,
                checkpoint_file_prefix = None,
                log_device_placement = False):

        self.__model = model
        self.__reader = reader
        self.__train_dir = train_dir
        self.__batch_size = self.reader.batch_size
        self.__max_steps = max_steps
        self.__test_steps = test_steps
        self.__gpus = gpus
        self.__truncation = truncation
        self.__max_grad_norm = max_grad_norm
        self.__restore_file = restore_file

        if checkpoint_file_prefix is None:
            self.__checkpoint_file_prefix = self.model.__class__.__name__ + '.ckpt'
        else:
            self.__checkpoint_file_prefix = checkpoint_file_prefix + '.ckpt'

        self.__log_device_placement = log_device_placement



    def average_gradients(self, grads):
        """Calculate the average gradient for each shared variable across all towers.

        Note that this function provides a synchronization point across all towers.

        Args:
            grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                if g is None:
                    continue
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(0, grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


    def tower_loss_and_gradients(self, scope):
        """Calculate the total loss and gradients on a single tower running the model.

        Args:
            scope: unique prefix string identifying the tower, e.g. 'tower_0'

        Returns:
            Tensor of shape [] containing the total loss for a batch of data
        """
        # Get images and labels
        feed_dict  = self.reader.next_train()
        images = feed_dict['images']
        labels = feed_dict['labels']

        show = labels
        # Build inference Graph.
        logits, _ = self.model.inference(images)

        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.
        total_loss = self.model.loss(logits, labels)

        # Assemble all of the losses for the current tower only.
        losses = tf.get_collection('losses', scope)

        # Calculate the total loss for the current tower.
        # total_loss = tf.add_n(losses, name = 'total_loss')

        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name = 'avg')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on tensorboard.
            loss_name = re.sub('%s_[0-9]*/' % 'tower', '', l.op.name)
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.scalar_summary(loss_name +' (raw)', l)
            tf.scalar_summary(loss_name, loss_averages.average(l))

        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)

        tvars = tf.trainable_variables()
        if self.truncation:
            cost = tf.reduce_sum(total_loss) / self.batch_size
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                                self.max_grad_norm)
        else:
            cost = tf.reduce_sum(total_loss) / self.batch_size
            grads = tf.gradients(cost, tvars)

        return total_loss, zip(grads, tvars), show


    def evaluate(self):
        feed_dict  = self.reader.next_test()
        images = feed_dict['images']
        labels = feed_dict['labels']

        # Build inference Graph.
        prob, _ = self.model.inference(images)
        correct = tf.nn.in_top_k(prob, labels, 1)

        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def precision(self, num_correct):
        precision = num_correct / (self.test_steps * self.batch_size)
        print('%s: testing output' %(datetime.now()))
        print('     Num examples: %d Num corrcet: %d Precision @ 1: %0.04f' %
                (self.test_steps * self.batch_size, num_correct, precision))

    def train(self):
        with tf.device('/cpu:0'):
            # Create a variable to count the number of train() calls. This equals the
            # number of batches processed * num_gpus.
            self.global_step = tf.get_variable('global_step', [],
                                               initializer = tf.constant_initializer(0), trainable = False)

            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                            self.global_step,
                                            DECAY_STEP,
                                            LEARNING_RATE_DECAY_FACTOR,
                                            staircase = True)

            # Create an optimizer that performs gradient descent.
            opt = tf.train.GradientDescentOptimizer(lr)
            # opt = tf.train.MomentumOptimizer(lr, 0.9)

            # Calculate the gradients for each model tower.
            tower_grads = []
            for i in self.gpus:
                with tf.device('/gpu:%d' % i), tf.name_scope('%s_%d' % ('tower', i)) as scope:
                    # Calculate the loss for one tower of the model. This function
                    # constructs the entire model but shares the variables across
                    # all towers.
                    loss, grads, debug = self.tower_loss_and_gradients(scope)
                    tower_grads.append(grads)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    #summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)


            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads = self.average_gradients(tower_grads)
            #apply_gradient_op = self.apply_gradients(grads)
            apply_gradient_op = opt.apply_gradients(grads, global_step = self.global_step)


            # Add a summary to track the learning rate.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            summaries.append(tf.scalar_summary('learning_rate', lr))

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    summaries.append(tf.histogram_summary(var.op.name + '/gradients', grad))

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                summaries.append(tf.histogram_summary(var.op.name, var))

            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, self.global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            # Build the summary operation from the last tower summaries.
            summary_op = tf.merge_summary(summaries)

            # Group all updates to into a single train op.
            train_op = tf.group(apply_gradient_op, variables_averages_op)

            # evaluate_op = self.evaluate()

            # Start running operations on the Graph. allow_soft_placement must be set to
            # True to build towers on GPU, as some of the ops do not have GPU
            # implementations.
            self.sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True,
                                                      log_device_placement = self.log_device_placement))


            # Build an initialization operation to run below.
            init = tf.initialize_all_variables()
            self.sess.run(init)

            # Create a saver.
            saver = tf.train.Saver(tf.all_variables())
            if self.restore_file is not None:
                saver.restore(self.sess, self.restore_file)

            summary_writer = tf.train.SummaryWriter(self.train_dir, self.sess.graph)

            # Start the queue runners.
            tf.train.start_queue_runners(sess = self.sess)

            for step in xrange(self.max_steps):
                start_time = time.time()
                _, loss_value, show = self.sess.run([train_op, loss, debug])
                # print(show)

                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step != 0 and step % 100 == 0:
                    num_examples_per_step = self.batch_size * len(self.gpus)
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / len(self.gpus)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ' 'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

                if step != 0 and step % 100 == 0:
                    summary_str = self.sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                if step % 1000 == 0:
                    cnt = 0
                    for i in xrange(self.test_steps):
                        pass
                        # cnt += self.sess.run(evaluate_op)

                    self.precision(cnt);

                # Save the model checkpoint periodically.
                if (step != 0 and step % 1000 == 0) or (step + 1) == self.max_steps:
                    path = os.path.join(self.train_dir, self.checkpoint_file_prefix)
                    saver.save(self.sess, path, global_step = step)


    # model
    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, value):
        self.__model = value

    @model.deleter
    def model(self):
        del self.__model

    # reader
    @property
    def reader(self):
        return self.__reader

    @reader.setter
    def reader(self, value):
        self.__reader = value
        self.__batch_size = self.reader.batch_size

    @reader.deleter
    def reader(self):
        del self.__reader

    # train_dir
    @property
    def train_dir(self):
        return self.__train_dir

    @train_dir.setter
    def train_dir(self, value):
        self.__train_dir = value

    @train_dir.deleter
    def train_dir(self):
        del self.__train_dir

    # batch_size
    @property
    def batch_size(self):
        return self.__batch_size

    # max_steps
    @property
    def max_steps(self):
        return self.__max_steps

    @max_steps.setter
    def max_steps(self, value):
        self.__max_steps = value

    @max_steps.deleter
    def max_steps(self):
        del self.__max_steps

    # test_steps
    @property
    def test_steps(self):
        return self.__test_steps

    @test_steps.setter
    def test_steps(self, value):
        self.__test_steps = value

    @test_steps.deleter
    def test_steps(self):
        del self.__test_steps

    # gpus
    @property
    def gpus(self):
        return self.__gpus

    @gpus.setter
    def gpus(self, value):
        self.__gpus = value

    @gpus.deleter
    def gpus(self):
        del self.__gpus

    # truncation
    @property
    def truncation(self):
        return self.__truncation

    @truncation.setter
    def truncation(self, value):
        self.truncation = value

    @truncation.deleter
    def truncation(self):
        del self.__truncation

    # max_grad_norm
    @property
    def max_grad_norm(self):
        return self.__max_grad_norm

    @max_grad_norm.setter
    def max_grad_norm(self, value):
        self.truncation = value

    @max_grad_norm.deleter
    def max_grad_norm(self):
        del self.__max_grad_norm

    # restore_file_path
    @property
    def restore_file(self):
        return self.__restore_file

    @restore_file.setter
    def restore_file(self, value):
        self.__restore_file = value

    @restore_file.deleter
    def restore_file(self):
        del self.__restore_file

    # checkpoint_file_prefix
    @property
    def checkpoint_file_prefix(self):
        return self.__checkpoint_file_prefix

    @checkpoint_file_prefix.setter
    def checkpoint_file_name(self, value):
        self.__checkpoint_file_prefix = value

    @checkpoint_file_prefix.deleter
    def checkpoint_file_prefix(self):
        del self.__checkpoint_file_prefix

    # log_device_placement
    @property
    def log_device_placement(self):
        return self.__log_device_placement

    @log_device_placement.setter
    def log_device_placement(self, value):
        self.__log_device_placement = value

    @log_device_placement.deleter
    def log_device_placement(self):
        del self.__log_device_placement


def main(argv=None):
    pass

if __name__ == '__main__':
    tf.app.run()
