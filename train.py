#import re
import sys
sys.path.append('..')
from utils.reader.read_img import reader
from utils import base_trainer
import ram
import tensorflow as tf

class Trainer(base_trainer.BaseTrainer):
    def __init__(self,
                 model,
                 reader,
                 train_dir = './train',
                 max_steps = 10000,
                 test_steps = 100,
                 gpus = [3],
                 truncation = False,
                 max_grad_norm = 5,
                 restore_file = None,
                 checkpoint_file_prefix = None,
                 log_device_placement = False):
        super(Trainer, self).__init__(model, reader, train_dir,
                                      max_steps, test_steps, gpus,
                                      truncation, max_grad_norm,
                                      restore_file,
                                      checkpoint_file_prefix,
                                      log_device_placement)

    def tower_loss_and_gradients(self, scope):
        # Get images and labels
        feed_dict  = self.reader.next_train()
        x = feed_dict['images']
        labels = feed_dict['labels']

        # Build inference Graph.
        loc_mean_t, loc_t, h_t, prob, pred, debug = self.model.inference(x)

        loss, grads = self.model.grad(loc_mean_t, loc_t, h_t, prob, pred, labels)


        ## Assemble all of the losses for the current tower only.
        #losses = tf.get_collection('losses', scope)
        #print(losses)

        ## Calculate the total loss for the current tower.
        #total_loss = tf.add_n(losses, name = 'total_loss')

        ## Compute the moving average of all individual losses and the total loss.
        #loss_averages = tf.train.ExponentialMovingAverage(0.9, name = 'avg')
        #loss_averages_op = loss_averages.apply(losses + [total_loss])

        ## Attach a scalar summary to all individual losses and the total loss; do the
        ## same for the averaged version of the losses.
        #for l in losses + [total_loss]:
        #    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        #    # session. This helps the clarity of presentation on tensorboard.
        #    loss_name = re.sub('%s_[0-9]*/' % 'tower', '', l.op.name)
        #    # Name each loss as '(raw)' and name the moving average version of the loss
        #    # as the original loss name.
        #    tf.scalar_summary(loss_name +' (raw)', l)
        #    tf.scalar_summary(loss_name, loss_averages.average(l))

        #with tf.control_dependencies([loss_averages_op]):
        #    total_loss = tf.identity(total_loss)

        #return total_loss, grads
        debug = labels
        return loss, grads, debug

    def evaluate(self):
        feed_dict  = self.reader.next_train()
        x = feed_dict['images']
        labels = feed_dict['labels']

        # Build inference Graph.
        loc_mean_t, loc_t, h_t, prob, pred, debug = self.model.inference(x)
        correct = tf.nn.in_top_k(prob, labels, 1)

        return tf.reduce_sum(tf.cast(correct, tf.int32))




reader = reader.Reader('./utils/reader/read_img/translated_mnist_output/', [40, 40, 3], batch_size = 16)
model = ram.Ram()
trainer = Trainer(model, reader, gpus = [0], max_steps = 500000, truncation = False)
trainer.train()
