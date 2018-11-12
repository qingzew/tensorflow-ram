import tensorflow as tf
import sys
sys.path.append('..')
from utils import base_model
import numpy as np
#import random

# loc_init = init random in [-1, 1] (location x,y are in [-1, 1] for all application)
# hid_init = init zeros

# Step1: sensor
# x_t = rho(loc_tm1, x, patch, k)
#     = x[loc_tm1.y-patch/2:loc_tm1.y+patch/2, loc_tm1.x-patch/2:loc_tm1+patch/2]
#       x[loc_tm1.y-patch/2*2:loc_tm1.y+patch/2*2, loc_tm1.x-patch/2*2:loc_tm1+patch/2*2]
#       x[loc_tm1.y-patch/2*4 ....]
#       x[loc_tm1.y-patch/2*(2^(k-1)):loc_tm1.y+patch/2*(2^(k-1)), loc_tm1.x-patch/2*(2^(k-1)):loc_tm1.x+patch/2*(2^(k-1))]

# Step2: glimps network
# g_t = f_g(x_t, loc_tm1)                          (256 units)
#     = relu(W_1 * h_g + W_2 * h_l + (b_1+b_2))
#           where h_g = relu(W_3 * x_t + b_3)    (128 units)
#                 h_l = relu(W_4 * loc_tm1 + b_4)  (128 units)

# Step3: core network
# h_t = f_h(h_tm1, g_t)                          (256 units)
#     = W_5 * h_tm1 + W_6 * g_t + (b_5 + b_6)    (for classification)

# Step4: actions
# Step4a: location network                       (2 units)
# l_t ~ P(l_t | f_l(h_t)) = N(l_t | f_l(h_t), [[sigma^2, 0], [0, sigma^2]]) (= Gaussian with fixed variance)
#     where f_l(h_t) = W_7 * h_t + b_7
# Step4b: env action network                     (10 units for MNIST)
# a_t ~ P(a_t | f_a(h_t)) = Bernoulli(f_a(h_t))
#     where f_a(h_t) = softmax(W_8 * h_t + b_8)
# => a_T ~ P(a_T | f_a(h_T)) = Bernoulli(f_a(h_T))
#     where f_a(h_T) = softmax(W_8 * h_t + b_8)

# Step5: loss and grad
# Step5a: reinforcement learning loss and its grad
# loss1 = 1 / M * sum_i_{1..M}{r_T^i}  where r_T is 1 (if correct) or 0 (if incorrect)
# grad1 = 1 / M * sum_i_{1..M}{theano.grad( sum_t_{0..T}{logN(l_t | f_l(h_t))} ) * (R^i - b) )}
#           where R^i = r_T^i = 1 (if correct) or 0 (if incorrect)
#                 b = mean(R^i)  (the value function???)
#                 b = sum_i_{1..M}{(grad( sum_t_{0..T}{logN(l_t | f_l(h_t))} ) ** 2 ) * R^i } / sum_i_{1..M}{ grad( sum_t_{0..T}{logN(l_t | f_l(h_t))} ) ** 2 }
#                     (optimal baseline)
# Step5b: supervised loss and its grad
# loss2 = 1 / M * sum_i_{1..M} cross_entroy_loss(groundtruth, a_T)
# grad2 = theano.grad(loss2)
#
# grad1 is for location network W_7 and b_7
# grad2 is for the others, W_1, b_1, ..., W_8, b_8 except W_7 and b_7
#

class Ram(base_model.BaseModel):
    def __init__(self,
                 n_batch = 16, # number of batch
                 img_shape = [40, 40, 3],
                 k = 3, # number of glimps scales
                 patch = [8, 8], # size of glimps patch
                 n_steps = 8, # number of glimps steps
                 lambda_ = 0.8, # mixing ratio between
                 n_h_g = 128, # number of hidden units in h_g (in glimps network)
                 n_h_l = 128, # number of hidden units in h_l (in glimps network)
                 n_f_g = 256, # number of hidden units in f_g (glimps network)
                 n_f_h = 256, # number of hidden units in f_h (core network)
                 n_classes = 10, # number of classes in classification problem
                 learn_init=True):

        self.n_batch = n_batch
        self.img_shape = img_shape
        self.k = k
        self.patch = patch
        self.n_steps = n_steps
        self.lambda_ = lambda_

        self.n_h_g = n_h_g
        self.n_h_l = n_h_l
        self.n_f_g = n_f_g
        self.n_f_h = n_f_h
        #self.n_f_l = 2
        self.n_classes = n_classes
        self.sigma = 0.1

        #for f_g
        self.w_h_g = []
        for i in xrange(self.k):
            self.w_h_g.append(self._variable_with_weight_decay('w_h_g_' + str(i),
            [(self.patch[0] * (2 ** i)) * self.patch[1] * (2 ** i) * self.img_shape[2], self.n_h_g]))
        self.b_h_g = self._variable_on_cpu('b_h_g', [self.n_h_g], tf.constant_initializer(0.0))

        self.w_h_l = self._variable_with_weight_decay('w_h_l', [2, self.n_h_l])
        self.b_h_l = self._variable_on_cpu('b_h_l', [self.n_h_l], tf.constant_initializer(0.0))

        self.w_f_g_1 = self._variable_with_weight_decay('w_f_g_1', [self.n_h_g, self.n_f_g])
        self.w_f_g_2 = self._variable_with_weight_decay('w_f_g_2', [self.n_h_l, self.n_f_g])
        self.b_f_g = self._variable_on_cpu('b_f_g', [self.n_f_g], tf.constant_initializer(0.0))

        #for f_h
        self.w_f_h_1 = self._variable_with_weight_decay('w_f_h_1', [self.n_f_h, self.n_f_h])
        self.w_f_h_2 = self._variable_with_weight_decay('w_f_h_2', [self.n_f_g, self.n_f_h])
        self.b_f_h = self._variable_on_cpu('b_f_h', [self.n_f_h], tf.constant_initializer(0.0))

        #for f_l
        self.w_f_l = self._variable_with_weight_decay('w_f_l', [self.n_f_h, 2])
        self.b_f_l = self._variable_on_cpu('b_f_l', [2], tf.constant_initializer(0.0))

        #for classifier
        self.w_classifier = self._variable_with_weight_decay('w_classifier', [self.n_f_h, self.n_classes])
        self.b_classifier = self._variable_on_cpu('b_classifier', [self.n_classes], tf.constant_initializer(0.0))


        self.loc_init = tf.random_uniform([self.n_batch, 2], -1, 1)
        self.h_init =  tf.random_normal(shape = [self.n_batch, self.n_f_h], mean = 0, stddev = 0.01)

    def rho(self, loc_tm1, x):
        """
        return:
            x_t = sensor output, where
            x_t[i] = (n_batch x patch*(2**i) x patch*(2**i) x channels) for i in 0, ..., k
                    [python list, consisting of tensor variables]

        inputs:
            loc_tm1 = location estimated at t - 1
                    = l(t-1) = y(t-1), x(t-1)
                    = (n_batch x 2)
                    [tensor variable] and recurrent
            x = original image
                 = (n_batch x channels x height x width)
                [tensor variable]
        """
        x_t = []
        for i in xrange(self.k):
            croped_imgs = []
            x_t_i = []
            for b in xrange(self.n_batch):
                img = tf.identity(x[b, :, :, :])
                height = img.get_shape()[0].value
                width = img.get_shape()[1].value

                img = tf.image.pad_to_bounding_box(img, self.patch[0] * (2 ** i), self.patch[1] * (2 ** i),
                                                        height + 2 * self.patch[0] * (2 ** i), width +  2 * self.patch[1] * (2 ** i))

                loc_tm1 = tf.tanh(loc_tm1)
                x_start = tf.to_int32((1. + loc_tm1[b, 0]) / 2. * height + self.patch[0] * (2 ** i) - self.patch[0] * (2 ** i) / 2)
                y_start = tf.to_int32((1. + loc_tm1[b, 1]) / 2. * width + self.patch[1] * (2 ** i) - self.patch[1] * (2 ** i) / 2)
                begin = tf.pack([x_start, y_start, 0])

                scaled_height = self.patch[0] * (2 ** i)
                scaled_width = self.patch[1] * (2 ** i)
                size = tf.constant([scaled_height, scaled_width, img.get_shape()[2].value])

                croped_img = tf.slice(img, begin, size)
                croped_imgs.append(croped_img)

                croped_img = tf.reshape(croped_img, [-1])
                x_t_i.append(croped_img)

            x_t.append(tf.pack(x_t_i))

            croped_imgs = tf.pack(croped_imgs)
            import random
            tf.image_summary(str(random.random()) + '_croped_imgs_' + str(i), croped_imgs, max_images = 3)
        return x_t

    def f_g(self, x_t, loc_tm1):
        """
        g_t = f_g(x_t, loc_tm1)
            = relu(W_1 * h_g + W_2 * h_l + (b_1 + b_2)) where
            h_g = relu(W_3 * x_t + b_3)
            h_l = relu(W_4 * loc_tm1 + b_4)

        return:
            g_t = glimps output
                = (n_batch x num hiddens of g_t)
                [tensor variable]

        inputs:
            x_t = sensor output,
                where x_t[i] = n_batch x (channels x patch*(2**i) x patch*(2**i)) for i in 0, ..., k
                [python list, consisting of tensor variables]
            loc_tm1 = location estimated at t-1
                    = l(t-1) = y(t-1), x(t-1)
                     = (n_batch x 2)
                  [tensor variable] and recurrent
        parameters:
            W_h_g = (k x num_inputs x num hiddens of h_g)
            b_h_g = (num hiddens of h_g,)

            W_h_l = (2 x num hiddens of h_l)
            b_h_l = (num_hiddens of h_l,)

            W_f_g_1 = (num hiddens of h_g x num hiddens of g_t)
            W_f_g_2 = (num hiddens of h_l x num hiddens of g_t)
            b_f_g = (num hiddens of g_t,)

        """
        h_g = tf.matmul(x_t[0], self.w_h_g[0])
        for i in xrange(1, self.k):
            h_g += tf.matmul(x_t[i], self.w_h_g[i])
        h_g = tf.nn.bias_add(h_g, self.b_h_g)
        h_g = tf.nn.relu(h_g, name = 'h_g')

        h_l = tf.matmul(loc_tm1, self.w_h_l)
        h_l = tf.nn.bias_add(h_l, self.b_h_l)
        h_l = tf.nn.relu(h_l, 'h_l')


        g_t = tf.matmul(h_g, self.w_f_g_1) + tf.matmul(h_l, self.w_f_g_2)
        g_t = tf.nn.bias_add(g_t, self.b_f_g)
        g_t = tf.nn.relu(g_t, 'g_t')

        return g_t

    # for classification f_h uses simple rectify layer
    # for dynamic environment f_h uses LSTM layer
    def f_h(self, h_tm1, g_t):
        """
        return:
            h_t = hidden states (output of core network)
                = (n_batch x num hiddens of h_t)
                  [tensor variable] and recurrent

        inputs:
            h_tm1 = hidden states estimated at t-1
                = (n_batch x num hiddens of h_t)
                [tensor variable] and recurrent
            g_t = glimps output
                = (n_batch x num hiddens of g_t)
                [tensor variable]

        parameters:
            W_f_h_1 = (num hiddens of h_t x num hiddens of h_t)
            W_f_h_2 = (num hiddens of g_t x num hiddens of h_t)
            b_f_h = (num hiddens of h_t,)
        """

        h_t = tf.matmul(h_tm1, self.w_f_h_1) + tf.matmul(g_t, self.w_f_h_2)
        h_t = tf.nn.bias_add(h_t, self.b_f_h)
        h_t = tf.nn.relu(h_t, name = 'h_t')

        return h_t

    def f_l(self, h_t):
        """
        return:
            loc_mean_t = (mean) location estimated for t
                = l(t) = y(t), x(t)
                = (n_batch x 2)
                [tensor variable] and recurrent

        inputs:
            h_t = hidden states (output of core network)
                = (n_batch x num hiddens of h_t)
                [tensor variable] and recurrent

        parameters:
            W_f_l = (num hiddens of h_t x 2)
            b_f_l = (2,)

        """

        loc_mean_t = tf.matmul(h_t, self.w_f_l)
        loc_mean_t = tf.nn.bias_add(loc_mean_t, self.b_f_l)

        return loc_mean_t


    def grad_reinforcement(self, loc_mean_t, loc_t, h_t, prob, pred, labels):
        """
        return:
            loss = 1 / M * sum_i_{1..M}{r_T^i}  where r_T is 1 (if correct) or 0 (if incorrect)
            [scalar variable]
            grads = 1 / M * sum_i_{1..M}{grad( sum_t_{0..T}{logN(l_t | f_l(h_t))} ) * (R^i - b) } where
                R^i = r_T^i = 1 (if correct) or 0 (if incorrect)
                b = mean(R^i)  (the value function)
                b = sum_i_{1..M}{(grad( sum_t_{0..T}{logN(l_t | f_l(h_t))} ) ** 2 ) * R^i } / sum_i_{1..M}{ grad( sum_t_{0..T}{logN(l_t | f_l(h_t))} ) ** 2 }
                 (optimal baseline)
            [tensor variable]

        inputs:
            loc_mean_t
            loc_t
            h_t
            prob
            pred
            labels = (n_batch,)
            [tensor variable]
        """
        # reward estimation
        reward = tf.equal(pred, labels)
        reward = tf.cast(reward, tf.float32)

        # for baseline estimation
        log_pi_t = -(loc_t - loc_mean_t) ** 2 / (2 * np.pi * self.sigma ** 2 )
        log_pi_t = tf.reduce_mean(log_pi_t, 2)

        # jcobian of log_pi_t wrt param
        tvars = tf.trainable_variables()

        jacobian = []
        for i in xrange(self.n_batch):
            for j in xrange(self.n_steps):
                jacobian.append(tf.gradients(log_pi_t[i, j], tvars))

        grads = []
        for p in xrange(len(tvars)):
            if jacobian[0][p] is None:
                grads.append(tf.zeros(shape = tvars[p].get_shape()))
                continue

            b = tf.zeros_like(jacobian[0][p])

            # n_batch * n_steps eaquals to the number fo elments in loc_pi_t
            for i in xrange(self.n_batch):
                numerator = tf.zeros_like(jacobian[0][p])
                for j in xrange(self.n_steps):
                    numerator += (1. / self.n_steps) * (jacobian[i * self.n_steps + j][p] ** 2)

                denominator = numerator + 1
                numerator = numerator * reward[i]
                b += (1. / self.n_batch) * numerator / denominator

            # estimate grad
            grad = tf.zeros_like(jacobian[0][p])
            for i in xrange(self.n_batch):
                tmp = tf.zeros_like(jacobian[0][p])
                for j in xrange(self.n_steps):
                    tmp += (1. / self.n_steps) * jacobian[i * self.n_steps + j][p]

                grad += (1. / self.n_batch) * tmp * (reward[i] - b)


            grads.append(grad)


        loss = tf.reduce_mean(reward, 0)

        return loss, grads

    def grad_supervised(self, prob, labels):
        """
        return:
            loss = 1 / M * sum_i_{1..M} cross_entroy_loss(groundtruth, a_T)
            grads = grad(loss, params)
        inputs:
            prob
            labels = (n_batch,)
            [tensor variable]
        """
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(prob, labels, name = 'cross_entropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        for i in xrange(len(grads)):
            if grads[i] == None:
                grads[i] = tf.zeros(shape = tvars[i].get_shape())

        return loss, grads

    def grad(self, loc_mean_t, loc_t, h_t, prob, pred, labels):
        loss1, grads1 = self.grad_reinforcement(loc_mean_t, loc_t, h_t, prob, pred, labels)
        loss2, grads2 = self.grad_supervised(prob, labels)

        loss = (1 - self.lambda_) * loss1 + self.lambda_ * loss2

        grads = []
        for i in xrange(len(grads1)):
            grads.append((1 - self.lambda_) * grads1[i] + self.lambda_ * grads2[i])

        tvars = tf.trainable_variables()
        grads = zip(grads, tvars)

        tf.scalar_summary('loss', loss)
        tf.scalar_summary('loss_reinforcement', loss1)
        tf.scalar_summary('loss_supervised', loss2)

        return loss, grads

    def inference(self, x):
        #loc_t ~ gaussian(loc_mean_t, [[sigma^2, 0], [0, sigma^2]]^-1)
        #loc_t = loc_mean_t + normal(loc_mean_t.shape,
        #                                     avg = 0.0,
        #                                     std = self.sigma)
        loc_t = self.loc_init
        h_t = self.h_init

        loc_mean_ts = []
        loc_ts = []
        h_ts = []

        for i in xrange(self.n_steps):
            x_t = self.rho(loc_t, x)
            g_t = self.f_g(x_t, loc_t)
            h_t = self.f_h(h_t, g_t)

            loc_mean_t = self.f_l(h_t)
            loc_t = tf.random_normal(loc_mean_t.get_shape(), mean = loc_mean_t, stddev = self.sigma)

            loc_mean_ts.append(loc_mean_t)
            loc_ts.append(loc_t)
            h_ts.append(h_t)

        prob = tf.matmul(h_t, self.w_classifier)
        prob = tf.nn.bias_add(prob, self.b_classifier)

        pred = tf.argmax(prob, 1)

        loc_mean_ts = tf.transpose(tf.pack(loc_mean_ts), perm = [1, 0, 2])
        loc_ts = tf.transpose(tf.pack(loc_ts), perm = [1, 0, 2])
        h_ts = tf.transpose(tf.pack(h_ts), perm = [1, 0, 2])

        return loc_mean_ts, loc_ts, h_ts, prob, pred, loc_t

