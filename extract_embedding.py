# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary for generating predictions over a set of videos."""

import os
import glob
import json
import tarfile
import time

import numpy
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

import eval_util
import losses
import readers
import utils

FLAGS = flags.FLAGS

if __name__ == '__main__':
  # Input
  flags.DEFINE_string("train_dir", "",
                      "The directory to load the model files from. We assume "
                      "that you have already run eval.py onto this, such that "
                      "inference_model.* files already exist.")
  flags.DEFINE_string("model_name", "",
                      "The directory to load the model files from. We assume "
                      "that you have already run eval.py onto this, such that "
                      "inference_model.* files already exist.")
  flags.DEFINE_string(
      "input_data_pattern", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_string("input_model_tgz", "",
                      "If given, must be path to a .tgz file that was written "
                      "by this binary using flag --output_model_tgz. In this "
                      "case, the .tgz file will be untarred to "
                      "--untar_model_dir and the model will be used for "
                      "inference.")
  flags.DEFINE_string("untar_model_dir", "/tmp/yt8m-model",
                      "If --input_model_tgz is given, then this directory will "
                      "be created and the contents of the .tgz file will be "
                      "untarred here.")

  # Output
  flags.DEFINE_string("output_file", "",
                      "The file to save the predictions to.")
  flags.DEFINE_string("output_model_tgz", "",
                      "If given, should be a filename with a .tgz extension, "
                      "the model graph and checkpoint will be bundled in this "
                      "gzip tar. This file can be uploaded to Kaggle for the "
                      "top 10 participants.")
  flags.DEFINE_integer("top_k", 20,
                       "How many predictions to output per video.")

  # Other flags.
  flags.DEFINE_integer(
      "batch_size", 8192,
      "How many examples to process per batch.")
  flags.DEFINE_integer("num_readers", 1,
                       "How many threads to use for reading input files.")

def format_lines(video_ids, predictions, top_k):
  batch_size = len(video_ids)
  for video_index in range(batch_size):
    top_indices = numpy.argpartition(predictions[video_index], -top_k)[-top_k:]
    line = [(class_index, predictions[video_index][class_index])
            for class_index in top_indices]
    line = sorted(line, key=lambda p: -p[1])
    yield video_ids[video_index].decode('utf-8') + "," + " ".join(
        "%i %g" % (label, score) for (label, score) in line) + "\n"

def get_input_data_tensors2(reader,
                           data_pattern,
                           batch_size=1000,
                           num_epochs=None,
                           num_readers=1):
  def parse_fn(example):
    return reader.prepare_serialized_examples(example)
    #return ret['video_matrix'],ret['labels'],ret['num_frames']
    
  with tf.name_scope("input"):
    files = tf.data.Dataset.list_files(data_pattern)
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=num_readers)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(map_func=parse_fn, num_parallel_calls=num_readers)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=64)
    iterator = dataset.make_one_shot_iterator()
    batch_video_ids, batch_video_matrix, batch_labels, batch_frames = iterator.get_next()
    batch_video_ids = tf.squeeze(batch_video_ids,[1])
    batch_video_matrix = tf.squeeze(batch_video_matrix,[1])
    batch_frames = tf.squeeze(batch_frames,[1])
    return batch_video_ids, batch_video_matrix, batch_frames
    #_,x,y,z = iterator.get_next()
    #print("==================",x,y,z)
    #x = tf.squeeze(x,[1])
    #y = tf.squeeze(y,[1])
    #z = tf.squeeze(z,[1])
    ##xs = x.shape.as_list()
    ##ys = y.shape.as_list()
    ##zs = z.shape.as_list()
    ##print("tsm_shape",xs,ys,zs)
    ##x = tf.reshape(x, [-1,xs[2],xs[3]])
    ##y = tf.reshape(y, [-1,ys[2]])
    ##z = tf.reshape(z, [-1])
    #return _,x,y,z


def get_input_data_tensors(reader, data_pattern, batch_size, num_readers=10):
  """Creates the section of the graph which reads the input data.

  Args:
    reader: A class which parses the input data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  with tf.name_scope("input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find input files. data_pattern='" +
                    data_pattern + "'")
    logging.info("number of input files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=1, shuffle=False)
    examples_and_labels = [reader.prepare_reader(filename_queue)
                           for _ in range(num_readers)]

    video_id_batch, video_batch, unused_labels, num_frames_batch = (
        tf.train.batch_join(examples_and_labels,
                            batch_size=batch_size,
                            allow_smaller_final_batch=True,
                            enqueue_many=True))
    return video_id_batch, video_batch, num_frames_batch

def inference(reader, train_dir, data_pattern, out_file_location, batch_size, top_k):
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess, gfile.Open(out_file_location, "w+") as out_file:
    video_id_batch, video_batch, num_frames_batch = get_input_data_tensors(reader, data_pattern, batch_size)
    print("========tsm===========",video_id_batch, video_batch, num_frames_batch)
    checkpoint_file = os.path.join(FLAGS.train_dir, "inference_model" if FLAGS.model_name == "" else FLAGS.model_name)
    if not gfile.Exists(checkpoint_file + ".meta"):
      raise IOError("Cannot find %s. Did you run eval.py?" % checkpoint_file)
    meta_graph_location = checkpoint_file + ".meta"
    logging.info("loading meta-graph: " + meta_graph_location)

    with tf.device("/gpu:0"):
      saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=False)

    logging.info("restoring variables from " + checkpoint_file)
    saver.restore(sess, checkpoint_file)
    input_tensor = tf.get_collection("input_batch_raw")[0]
    num_frames_tensor = tf.get_collection("num_frames")[0]
    predictions_tensor = tf.get_collection("predictions")[0]

    # Workaround for num_epochs issue.
    def set_up_init_ops(variables):
      init_op_list = []
      for variable in list(variables):
        if "train_input" in variable.name:
          init_op_list.append(tf.assign(variable, 1))
          variables.remove(variable)
      init_op_list.append(tf.variables_initializer(variables))
      return init_op_list

    sess.run(set_up_init_ops(tf.get_collection_ref(
        tf.GraphKeys.LOCAL_VARIABLES)))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    num_examples_processed = 0
    start_time = time.time()
    embedding_ops = tf.get_collection("embedding")[0]
    try:
      while not coord.should_stop():
          video_id_batch_val, video_batch_val,num_frames_batch_val = sess.run([video_id_batch, video_batch, num_frames_batch])
          #print(video_id_batch_val,len(num_frames_batch_val))
          embs = sess.run([embedding_ops], feed_dict={input_tensor: video_batch_val, num_frames_tensor: num_frames_batch_val})
          #print(len(embs),len(embs[0]),len(embs[0][0]), len(video_id_batch_val),  embs[0][0])
          for vid,emb in zip(video_id_batch_val, embs[0]):
            v = str(vid, encoding='utf-8').split('/')[-1].split('.')[0]
            #print(vid,len(emb), emb)
            out_file.write('{}\t{}\n'.format(v, ' '.join([str(x) for x in emb])))
          num_examples_processed += len(num_frames_batch_val)

    except tf.errors.OutOfRangeError:
        logging.info('Done with inference. The output file was written to ' + out_file_location)
    finally:
        logging.info('finally The output file was written to ' + out_file_location)
        coord.request_stop()

    out_file.close()
    coord.join(threads)
    sess.close()


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)
  if FLAGS.input_model_tgz:
    if FLAGS.train_dir:
      raise ValueError("You cannot supply --train_dir if supplying "
                       "--input_model_tgz")
    # Untar.
    if not os.path.exists(FLAGS.untar_model_dir):
      os.makedirs(FLAGS.untar_model_dir)
    tarfile.open(FLAGS.input_model_tgz).extractall(FLAGS.untar_model_dir)
    FLAGS.train_dir = FLAGS.untar_model_dir

  flags_dict_file = os.path.join(FLAGS.train_dir, "model_flags.json")
  if not os.path.exists(flags_dict_file):
    raise IOError("Cannot find %s. Did you run eval.py?" % flags_dict_file)
  flags_dict = json.loads(open(flags_dict_file).read())

  # convert feature_names and feature_sizes to lists of values
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      flags_dict["feature_names"], flags_dict["feature_sizes"])

  if flags_dict["frame_features"]:
    reader = readers.YT8MFrameFeatureReader(feature_names=feature_names,
                                            feature_sizes=feature_sizes)
  else:
    reader = readers.YT8MAggregatedFeatureReader(feature_names=feature_names,
                                                 feature_sizes=feature_sizes)

  if FLAGS.output_file is "":
    raise ValueError("'output_file' was not specified. "
      "Unable to continue with inference.")

  if FLAGS.input_data_pattern is "":
    raise ValueError("'input_data_pattern' was not specified. "
      "Unable to continue with inference.")

  inference(reader, FLAGS.train_dir, FLAGS.input_data_pattern,
    FLAGS.output_file, FLAGS.batch_size, FLAGS.top_k)


if __name__ == "__main__":
  app.run()
