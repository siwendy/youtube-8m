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
"""Produces tfrecord files similar to the YouTube-8M dataset.

It processes a CSV file containing lines like "<video_file>,<labels>", where
<video_file> must be a path of a video, and <labels> must be an integer list
joined with semi-colon ";". It processes all videos and outputs tfrecord file
onto --output_file.

It assumes that you have OpenCV installed and properly linked with ffmpeg (i.e.
function `cv2.VideoCapture().open('/path/to/some/video')` should return True).

The binary only processes the video stream (images) and not the audio stream.
"""

import csv
import os
import sys
import numpy as np
import cv2
import time
import feature_extractor
import numpy
import tensorflow as tf
from tensorflow import flags
from multiprocessing import Pool, Manager
import subprocess

FLAGS = flags.FLAGS

if __name__ == '__main__':
  # Required flags for input and output.
  flags.DEFINE_string(
      'output_file', None,
      'File containing tfrecords will be written at this path.')
  flags.DEFINE_string(
      'data_tmp', None,
      'File containing tfrecords will be written at this path.')
  flags.DEFINE_string(
      'wav_tmp', None,
      'File containing tfrecords will be written at this path.')
  flags.DEFINE_string(
      'input_videos_csv', None,
      'CSV file with lines "<video_file>,<labels>", where '
      '<video_file> must be a path of a video and <labels> '
      'must be an integer list joined with semi-colon ";"')
  # Optional flags.
  flags.DEFINE_string('model_dir', os.path.join(os.getenv('HOME'), 'yt8m'),
                      'Directory to store model files. It defaults to ~/yt8m')

  # The following flags are set to match the YouTube-8M dataset format.
  flags.DEFINE_integer('frames_per_second', 1,
                       'This many frames per second will be processed')
  flags.DEFINE_boolean(
      'skip_frame_level_features', False,
      'If set, frame-level features will not be written: only '
      'video-level features will be written with feature '
      'names mean_*')
  flags.DEFINE_string(
      'labels_feature_key', 'labels',
      'Labels will be written to context feature with this '
      'key, as int64 list feature.')
  flags.DEFINE_string(
      'image_feature_key', 'rgb',
      'Image features will be written to sequence feature with '
      'this key, as bytes list feature, with only one entry, '
      'containing quantized feature string.')
  flags.DEFINE_string(
      'video_file_feature_key', 'id',
      'Input <video_file> will be written to context feature '
      'with this key, as bytes list feature, with only one '
      'entry, containing the file path of the video. This '
      'can be used for debugging but not for training or eval.')
  flags.DEFINE_boolean(
      'insert_zero_audio_features', True,
      'If set, inserts features with name "audio" to be 128-D '
      'zero vectors. This allows you to use YouTube-8M '
      'pre-trained model.')
  flags.DEFINE_boolean(
      'pca_enable', True,
      'If set, audio feature will be pca  '
      ''
      '.')
  flags.DEFINE_string(
      'model_path',os.path.join(os.getenv('HOME'), 'audioset'),
      'If set, inserts features with name "audio" to be 128-D '
      'zero vectors. This allows you to use YouTube-8M '
      'pre-trained model.')


workders_queue = {}
# worker进程数，别设置多了，和自己的cpu核心数有关。
MAX_WORKER_NUM = 10

# 生产者与消费者模型的应用
def init_worker_queue():
  # 为每一个worker进程分配自己的队列
  for i in range(0, MAX_WORKER_NUM):
    workders_queue[i] = Manager().Queue()


def read_data():
  index = 0
  for line in open(FLAGS.input_videos_csv):
    q = workders_queue[index % MAX_WORKER_NUM]
    vid,label = line.strip().split(',')
    #print('w=',vid)
    q.put(vid)
    index += 1
  for q in workders_queue.values():
    q.put(None)


def workder_data(worker_id):
  print("worker_id:{}".format(worker_id))
  n = 0
  begin = time.time()
  fos = open('{}/stat/worker_{}'.format(FLAGS.data_tmp, worker_id), 'w')
  while True:
    q = workders_queue[worker_id]
    data = q.get()
    n += 1
    if data == None:
      print('finished',worker_id)
      break
    video_path = data
    vid = video_path.split('/')[-1].split('.')[0]
    video = '/'.join([FLAGS.data_tmp, str(worker_id), vid])
    wav_cmd   = ['ffmpeg', '-loglevel quiet', '-y', '-i {}'.format(video_path), '-f wav {}.wav'.format(video)]
    frame_cmd = ['ffmpeg', '-loglevel quiet', '-y', '-i {}'.format(video_path), '-t 300 -vframes 300 -r 1 {}_%d.jpg'.format(video)]
    res1 = subprocess.call(' '.join(wav_cmd), shell=True)
    res2 = subprocess.call(' '.join(frame_cmd), shell=True)
    fos.write('{}\t{}\t{}\n'.format(vid, res1, res2))
  print('worker:{} total_num={} qps={:.3f}'.format(worker_id, n, n/float(time.time() - begin)))
  fos.close()

def _int64_list_feature(int64_list):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=int64_list))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _make_bytes(int_array):
  if bytes == str:  # Python2
    return ''.join(map(chr, int_array))
  else:
    return bytes(int_array)


def quantize(features, min_quantized_value=-2.0, max_quantized_value=2.0):
  """Quantizes float32 `features` into string."""
  assert features.dtype == 'float32'
  assert len(features.shape) == 1  # 1-D array
  features = numpy.clip(features, min_quantized_value, max_quantized_value)
  quantize_range = max_quantized_value - min_quantized_value
  features = (features - min_quantized_value) * (255.0 / quantize_range)
  features = [int(round(f)) for f in features]

  return _make_bytes(features)

if __name__ == '__main__':
  start_time = time.time()
  init_worker_queue()
  p = Pool(MAX_WORKER_NUM)
  p.apply_async(read_data)
  for i in range(0, MAX_WORKER_NUM):
    dirs = '{}/{}'.format(FLAGS.data_tmp, i)
    if not os.path.exists(dirs):
      os.mkdir(dirs)
  dirs = '{}/stat'.format(FLAGS.data_tmp)
  if not os.path.exists(dirs):
    os.mkdir(dirs)
  for i in range(0, MAX_WORKER_NUM):
    p.apply_async(workder_data, args=(i,))
  p.close()
  p.join()
  end_time = time.time()
  print('handle time:%ss' % (end_time - start_time))
