#!/usr/bin/python
import os
import time
import tensorflow as tf
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
import readers
import utils
import numpy as np
FLAGS = flags.FLAGS

flags.DEFINE_string("feature_names", "rgb", "Name of the feature "
		"to use for training.")
flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")
flags.DEFINE_bool(
    "segment_labels", False,
    "If set, then --train_data_pattern must be frame-level features (but with"
    " segment_labels). Otherwise, --train_data_pattern must be aggregated "
    "video-level features. The model must also be set appropriately (i.e. to "
    "read 3D batches VS 4D batches.")


def get_reader():
  # Convert feature_names and feature_sizes to lists of values.
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)

  reader = readers.YT8MFrameFeatureReader(
      feature_names=feature_names,
      feature_sizes=feature_sizes)

  return reader


def get_input_data_tensors2(reader,
                           data_pattern,
                           batch_size=1000,
                           num_epochs=None,
                           num_readers=1):
  def parse_fn(example):
    return  reader.prepare_serialized_examples(example)
    
  with tf.name_scope("train_input"):
    #files = gfile.Glob(data_pattern)
    #if not files:
    #  raise IOError("Unable to find training files. data_pattern='" +
    #               data_pattern + "'.")
    files = tf.data.Dataset.list_files(data_pattern)
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=num_readers)
    dataset = dataset.shuffle(buffer_size=1024*batch_size)
    dataset = dataset.map(map_func=parse_fn, num_parallel_calls=num_readers)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=64*batch_size)
    iterator = dataset.make_one_shot_iterator()
    vids,x,y,z = iterator.get_next()
    xs = x.shape.as_list()
    ys = y.shape.as_list()
    zs = z.shape.as_list()
    print("tsm_shape",xs,ys,zs)
    #x = tf.reshape(x, [-1,xs[2],xs[3]])
    #y = tf.reshape(y, [-1,ys[2]])
    #z = tf.reshape(z, [-1])
    x = tf.squeeze(x, [1])
    y = tf.squeeze(y, [1])
    z = tf.squeeze(z, [1])
    vids = tf.squeeze(vids, [1])
    return x,y,z,vids

    #return  iterator.get_next()
    #filename_queue = tf.train.string_input_producer(
    #    files, num_epochs=num_epochs, shuffle=True)
    #reader = tf.TFRecordReader()
    #_, serialized_example = reader.read(filename_queue)
    #return tf.train.batch(reader.prepare_reader(filename_queue),
    #  batch_size=batch_size, capacity=batch_size*200,
    #  enqueue_many=True)

def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=1000,
                           num_epochs=None,
                           num_readers=1):
  with tf.name_scope("train_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find training files. data_pattern='" +
                   data_pattern + "'.")
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs, shuffle=True)
    training_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]
    return tf.train.shuffle_batch_join(
        training_data,
        batch_size=batch_size,
        capacity=batch_size * 5,
        min_after_dequeue=batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)

def read_and_decode(filename,batch_size,num_epochs):
  reader = tf.TFRecordReader()
  filename_queue = tf.train.string_input_producer([filename],
      shuffle=False,num_epochs=num_epochs)

  _, serialized_example = reader.read_up_to(filename_queue, batch_size)
  #my_example_features = {'sparse': tf.SparseFeature(index_key=['index'],
  #  value_key='values',
  #  dtype=tf.int64,
  #  size=[1,100])}

  tensor = tf.parse_example(
          serialized=serialized_example,
          features=g_features)
  print(tensor)
  #tensor = tf.parse_example(serialized_example,my_example_features)

  #dense_tensor = tf.sparse_tensor_to_dense(tensor['sparse'])
  #tensor = tf.reshape(tensor, tf.stack([-1,128]))
  #print tensor
  #data = tf.cast(dense_tensor, tf.int32)
  data = tf.train.batch([tensor['item_emb']],
      batch_size=batch_size, capacity=batch_size*200,
      enqueue_many=True)
  return data

if __name__ == '__main__':
  reader = get_reader()
  #data=get_input_data_tensors(reader,'/home/taisimin/frame/train*.tfrecord', 32, 10)
  #f,vid,num_frames=get_input_data_tensors2(reader,'/home/taisimin/frame/train3785.tfrecord', batch_size=1, num_epochs=1, num_readers=1)
  #f,vid,num_frames=get_input_data_tensors2(reader,'/home/taisimin/frame/train3785.tfrecord', batch_size=1, num_epochs=1, num_readers=1)
  #f,vid,num_frames=get_input_data_tensors2(reader,'/home/taisimin/frame/tmp/3/train1013.tfrecord', batch_size=1, num_epochs=1, num_readers=1)
  #f,vid,num_frames=get_input_data_tensors2(reader,'/home/taisimin/frame/validation/*/validate*', batch_size=160, num_epochs=10, num_readers=12)
  f,label,num_frames,vid=get_input_data_tensors2(reader,'/home/taisimin/frame/tmp/*/train*.tfrecord', batch_size=1, num_epochs=10, num_readers=12)
  #f,label,num_frames,vid=get_input_data_tensors2(reader,'/da3/rec/yangyang1-pd/data/tfrecord_all/*tfrecord', batch_size=1, num_epochs=10, num_readers=12)
  #data = read_and_decode("test.tfrecord",10, 1)
  config = tf.ConfigProto( allow_soft_placement=True,log_device_placement=True)
  config.gpu_options.allow_growth=True  

  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer()) 
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    num = 0
    try:
      i = 0
      start = time.time()
      print_interval=100
      stat = [[],[],[],[]]
      while True:
        d,v,n = sess.run([f,vid,num_frames])
        i += 1
        #print("d",d)
        print("vid",v)
        #print("len(d)",len(d))
        #print("len(v)",len(v))
        #print("len(n)",len(n))
        #print(len(d[0]))
        #print("max_frame={},frame={}".format(len(d[0][0]), n[0]))
        #print("framef=",len(d[0][0][0]))
        #aa = []
        #for fid,frame in enumerate(d[0][0]):
        #  aa.append('fid={},mean={},std={}'.format(fid, np.mean(frame), np.std(frame)))
        x,y,t_max,t_min = np.mean(d[0]), np.std(d[0]),np.max(d[0]),np.min(d[0])
        num += len(d)
        #print(x, y,t_max,t_min)
        stat[0].append(x)
        stat[1].append(y)
        stat[2].append(t_max)
        stat[3].append(t_min)
        #stat.append([x,y,t_max,t_min])
        #break
    except tf.errors.OutOfRangeError:
      print("OutOfRangeError",num)
      print("avg",np.mean(stat[0]), np.mean(stat[1]), np.mean(stat[2]), np.mean(stat[3]))
    finally:
      coord.request_stop()
      coord.join(threads)
