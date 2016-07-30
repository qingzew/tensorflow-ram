from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, cv2, random
import tensorflow as tf
import numpy as np

batch = 10000
image_rows = 28
image_cols = 28

tf.app.flags.DEFINE_string('src', '/tmp/data',
                           'Directory to get image files')
tf.app.flags.DEFINE_string('dst', '/tmp/data',
                           'Directory to write the converted result')
FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def convert_to(images, labels, dataset, num_examples, name):
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    dir = os.path.join(FLAGS.dst, dataset)
    if not os.path.exists(dir):
        os.makedirs(dir)

    filename = os.path.join(FLAGS.dst, dataset , name +  '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features = tf.train.Features(feature={
            # 'height': _int64_feature(rows),
            # 'width': _int64_feature(cols),
            # 'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))

        writer.write(example.SerializeToString())
    writer.close()

# def main(argv):
#    datasets = ['train', 'test']
#    for dataset in datasets:
#        file_list = []
#        dir = os.path.join(FLAGS.src, dataset)
#        for root, dirs, files in os.walk(dir):
#            for f in files:
#                path = os.path.join(root, f)
#                file_list.append(path)

#        random.shuffle(file_list)

#        cnt = 0
#        #images = np.zeros((batch, image_rows, image_cols, 3), float)
#        #labels = np.zeros((batch, 1), int)
#        images = []
#        labels = []

#        for p in file_list:
#            im = cv2.imread(p)
#            images.append(im)
#            label = p.split('/')[-2]
#            print(label)
#            labels.append(label)

#            cnt += 1
#            if cnt != 0 and cnt % batch == 0:
#                images = np.array(images)
#                labels = np.array(labels)
#                convert_to(images, labels, dataset, batch, str(cnt // batch))

#                images = []
#                labels = []


#        if cnt != 0 and cnt % batch != 0:
#            images = np.array(images)
#            labels = np.array(labels)
#            convert_to(images, labels, dataset, cnt % batch, str(cnt // batch))

#            images = []
#            labels = []

def main(argv):
    # datasets = ['train', 'test']
    datasets = ['test']
    for dataset in datasets:
        cnt = 0
        images = []
        labels = []

        dataset_path = os.path.join(FLAGS.src, dataset)

        for line in open(dataset_path):
            try:
                img_path = line.strip()
                img_path = os.path.join(FLAGS.src, img_path)
                print(img_path)
                label = line.strip().split('/')[-2]

                print(img_path)
                im = cv2.imread(img_path)
                # if im is not None and im.size == 0:
                #     print(img_path, 'can not be found')
                #     continue

                im = cv2.resize(im, (32, 32))
            except Exception as e:
                print(e)
                continue

            images.append(im)
            labels.append(label)

            cnt += 1
            if cnt != 0 and cnt % batch == 0:
                images = np.array(images)
                labels = np.array(labels)
                convert_to(images, labels, dataset, batch, str(cnt // batch))

                images = []
                labels = []


        if cnt != 0 and cnt % batch != 0:
            images = np.array(images)
            labels = np.array(labels)
            convert_to(images, labels, dataset, cnt % batch, str(cnt // batch + 1))

            images = []
            labels = []

if __name__ == '__main__':
    tf.app.run()
