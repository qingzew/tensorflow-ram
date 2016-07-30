# tensorflow-ram
this is a re-implementation of Recurrent Models of Visual Attention in tensorflow of [lim0606/lasagne-ram](https://github.com/lim0606/lasagne-ram/), which can run on multi-gpus
# usage
  first, go to the `utils/reader/read_img/` dir, and run `imgs_to_records.py --src src_dir --dst dst_dir` to make your own dataset to tfrecords
  second, in the root dir, run 'python train.py' to train on your own dataset
#reference
1. http://arxiv.org/abs/1406.6247
2. http://www.scholarpedia.org/article/Policy_gradient_methods
