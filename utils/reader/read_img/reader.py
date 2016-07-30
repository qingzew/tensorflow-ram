from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf

class Reader(object):
    def __init__(self, data_dir, image_shape, batch_size = 64, num_threads = 16, num_epochs = None,
                     crop_size = None):
        self.__data_dir = data_dir
        self.__image_shape = image_shape
        self.__batch_size = batch_size
        self.__num_threads = num_threads
        self.__num_epochs = num_epochs
        self.__crop_size = crop_size


    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features = {
                'image_raw': tf.FixedLenFeature([], tf.string),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.int64),
            })

        # Convert from a scalar string tensor
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image_size = 1
        for shape in self.image_shape:
            image_size *= shape
        image.set_shape(image_size)
        image = tf.reshape(image, self.image_shape)

        # Convert from [0, 255] -> [0, 1] floats.
        image = tf.cast(image, tf.float32) * (1. / 255)

        # Convert label from a scalar uint8 tensor to an int64 scalar.
        label = tf.cast(features['label'], tf.int64)

        return image, label

    def _generate_image_and_label_batch(self, image, label, min_queue_examples,
                                        shuffle):
        """Construct a queued batch of images and labels.

        Args:
            image: 3-D Tensor of [height, width, 3] of type.float32.
            label: 1-D Tensor of type.int32
            min_queue_examples: int32, minimum number of samples to retain
                in the queue that provides of batches of examples.
            batch_size: Number of images per batch.
            shuffle: boolean indicating whether to use a shuffling queue.

        Returns:
            images: Images. 4D tensor of [batch_size, height, width, 3] size.
            labels: Labels. 1D tensor of [batch_size] size.
        """
        # Create a queue that shuffles the examples, and then
        # read 'batch_size' images + labels from the example queue.
        if shuffle:
            images, labels = tf.train.shuffle_batch(
                [image, label],
                batch_size = self.batch_size,
                num_threads = self.num_threads,
                capacity = min_queue_examples + 3 * self.batch_size,
                min_after_dequeue = min_queue_examples)
        else:
            images, labels = tf.train.batch(
                [image, label],
                batch_size = self.batch_size,
                num_threads = self.num_threads,
                capacity = min_queue_examples + 3 * self.batch_size)

        # Display the training images in the visualizer.
        tf.image_summary('images', images, max_images = 3)

#        return images, tf.reshape(label_batch, [batch_size])
        return {'images' : images, 'labels' : labels}


    def distorted_inputs(self):
        """Construct distorted input for training using the Reader ops.
        Returns:
            images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
            labels: Labels. 1D tensor of [batch_size] size.
        """
        with tf.name_scope('input'):
            filenames = []
            for root, dirs, files in os.walk(self.__data_dir):
                for f in files:
                    filenames.append(os.path.join(root, f))

            # Create a queue that produces the filenames to read.
            self.filename_queue = tf.train.string_input_producer(filenames, num_epochs = self.num_epochs,
                                                            name = 'string_input_producer')

            # Read examples from files in the filename queue.
            image, label = self.read_and_decode()

            # Image processing for training the network. Note the many random
            # distortions applied to the image.

            # Randomly crop a [height, width] section of the image.
            if self.crop_size is not None:
                distorted_image = tf.random_crop(image,
                                                 [self.crop_size, self.crop_size, 3])

            # Randomly flip the image horizontally.
            distorted_image = tf.image.random_flip_left_right(distorted_image)

            # Because these operations are not commutative, consider randomizing
            # the order their operation.
            distorted_image = tf.image.random_brightness(distorted_image, max_delta = 63)
            distorted_image = tf.image.random_contrast(distorted_image, lower = 0.2, upper = 1.8)

            # Subtract off the mean and divide by the variance of the pixels.
            float_image = tf.image.per_image_whitening(distorted_image)

            # Ensure that the random shuffling has good mixing properties.
            min_fraction_of_examples_in_queue = 0.4
            num_examples_per_epoch = 0.5 * self.batch_size
            min_queue_examples = int(num_examples_per_epoch *
                                   min_fraction_of_examples_in_queue)
            print('Filling queue with %d images before starting to train. '
                    'This will take a few minutes.' % min_queue_examples)

            # Generate a batch of images and labels by building up a queue of examples.
            return self._generate_image_and_label_batch(float_image, label, min_queue_examples,
                                                        self.batch_size, shuffle = True)

    def next_train(self):
        """
        Returns:
            images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
            labels: Labels. 1D tensor of [batch_size] size.
        """

        with tf.name_scope('train'):
            train_dir = os.path.join(self.__data_dir, 'train')
            if not os.path.exists(train_dir):
                print('no train data')
                exit(1)

            filenames =[]
            for root, dirs, files in os.walk(train_dir):
                for f in files:
                    filenames.append(os.path.join(root, f))

            # Create a queue that produces the filenames to read.
            filename_queue = tf.train.string_input_producer(filenames, num_epochs = self.num_epochs,
                                                            name = 'string_input_producer')

            # Read examples from files in the filename queue.
            image, label = self.read_and_decode(filename_queue)

            # Subtract off the mean and divide by the variance of the pixels.
#            image = tf.image.per_image_whitening(image)

            # Ensure that the random shuffling has good mixing properties.
            min_fraction_of_examples_in_queue = 0.4
            num_examples_per_epoch = 0.5 * self.batch_size
            min_queue_examples = int(num_examples_per_epoch *
                                     min_fraction_of_examples_in_queue)

            # Generate a batch of images and labels by building up a queue of examples.
            return self._generate_image_and_label_batch(image, label, min_queue_examples, shuffle = True)

    def next_test(self):
        """
        Returns:
            images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
            labels: Labels. 1D tensor of [batch_size] size.
        """

        with tf.name_scope('test'):
            test_dir = os.path.join(self.data_dir, 'test')
            if not os.path.exists(test_dir):
                print('no test data')
                exit(1)

            filenames =[]
            for root, dirs, files in os.walk(test_dir):
                for f in files:
                    filenames.append(os.path.join(root, f))

            # Create a queue that produces the filenames to read.
            filename_queue = tf.train.string_input_producer(filenames, num_epochs = self.num_epochs,
                                                            name = 'string_input_producer')

            # Read examples from files in the filename queue.
            image, label = self.read_and_decode(filename_queue)

            # Subtract off the mean and divide by the variance of the pixels.
#            image = tf.image.per_image_whitening(image)

            # Ensure that the random shuffling has good mixing properties.
            min_fraction_of_examples_in_queue = 0.4
            num_examples_per_epoch = 0.5 * self.batch_size
            min_queue_examples = int(num_examples_per_epoch *
                                     min_fraction_of_examples_in_queue)

            # Generate a batch of images and labels by building up a queue of examples.
            return self._generate_image_and_label_batch(image, label, min_queue_examples, shuffle = True)

    # data_dir
    @property
    def data_dir(self):
        return self.__data_dir

    @data_dir.setter
    def data_dir(self, value):
        self.__data_dir = value

    @data_dir.deleter
    def data_dir(self):
        del self.__data_dir

    # image_shape
    @property
    def image_shape(self):
        return self.__image_shape

    @image_shape.setter
    def image_shape(self, value):
        self.__image_shape = value

    @image_shape.deleter
    def image_shape(self):
        del self.__image_shape

    # batch_size
    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, value):
        self.__batch_size = value

    @batch_size.deleter
    def batch_size(self):
        del self.__batch_size

    # num_threads
    @property
    def num_threads(self):
        return self.__num_threads

    @num_threads.setter
    def num_threads(self, value):
        self.__num_threads = value

    @num_threads.deleter
    def num_threads(self):
        del self.__num_epochs

    # num_epochs
    @property
    def num_epochs(self):
        return self.__num_epochs

    @num_epochs.setter
    def num_epochs(self, value):
        self.__num_threads = value

    @num_epochs.deleter
    def num_epochs(self):
        del self.__num_threads

    # crop_size
    @property
    def crop_size(self):
        return self.__crop_size

    @crop_size.setter
    def crop_size(self, value):
        self.__crop_size = value

    @crop_size.deleter
    def crop_size(self):
        del self.__crop_size


def main(argv = None):
    reader = Reader('./ocr_output/', [32, 32, 3])
    print(reader.batch_size)

if __name__ == '__main__':
    tf.app.run()
