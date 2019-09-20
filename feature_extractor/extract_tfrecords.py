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
onto --output_tfrecords_file.

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
from tensorflow import app
from tensorflow import flags
from audio_feature import AudioFeature 
from vggish import mel_features,vggish_input,vggish_params,vggish_postprocess,vggish_slim 
from multiprocessing import Pool, Manager
FLAGS = flags.FLAGS

# In OpenCV3.X, this is available as cv2.CAP_PROP_POS_MSEC
# In OpenCV2.X, this is available as cv2.cv.CV_CAP_PROP_POS_MSEC
CAP_PROP_POS_MSEC = 0

if __name__ == '__main__':
  # Required flags for input and output.
  flags.DEFINE_string(
      'output_tfrecords_file', None,
      'File containing tfrecords will be written at this path.')
  flags.DEFINE_string(
      'wav_tmp', None,
      'File containing tfrecords will be written at this path.')
  flags.DEFINE_string(
      'input_videos_file', None,
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

workers_queue = {}
# worker进程数，别设置多了，和自己的cpu核心数有关。
MAX_WORKER_NUM = 10
gpu_workers_queue = {}
gpu_dict = {}


def frame_iterator(filename, every_ms=1000, max_num_frames=300):
  """Uses OpenCV to iterate over all frames of filename at a given frequency.

  Args:
    filename: Path to video file (e.g. mp4)
    every_ms: The duration (in milliseconds) to skip between frames.
    max_num_frames: Maximum number of frames to process, taken from the
      beginning of the video.

  Yields:
    RGB frame with shape (image height, image width, channels)
  """
  video_capture = cv2.VideoCapture()
  if not video_capture.open(filename):
    print >> sys.stderr, 'Error: Cannot open video file ' + filename
    return
  last_ts = -99999  # The timestamp of last retrieved frame.
  num_retrieved = 0

  while num_retrieved < max_num_frames:
    # Skip frames
    while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
      if not video_capture.read()[0]:
        return

    last_ts = video_capture.get(CAP_PROP_POS_MSEC)
    has_frames, frame = video_capture.read()
    if not has_frames:
      break
    yield frame
    num_retrieved += 1


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

    
def read_input_fn():
  index = 0
  for line in open(FLAGS.input_videos_file):
    q = workers_queue[index % MAX_WORKER_NUM]
    vid,video_file,_,_ = line.strip().split('\t')
    #print('vid', vid, "q=", q)
    q.put([vid,video_file])
    index += 1
  for q in workers_queue.values():
    q.put(None)

def read_frame_and_audio_fn(worker_id):
  print("worker_id:{}".format(worker_id))
  n = 0
  begin = time.time()
  print_num = 10
  MAX_GPU_WORKER_NUM = len(gpu_workers_queue)
  batch_frame_sec = 0
  batch_audio_sec = 0
  while True:
    q = workers_queue[worker_id]
    data = q.get()
    if data == None:
      print('{} finished,count={}'.format(worker_id, n))
      break
    n += 1
    vid,video_file = data
    rgbs = []
    batch_begin = time.time()
    for i in range(1,301):
      img_file = '{}_{}.jpg'.format(video_file, i)
      if not os.path.exists(img_file):
        break
      rgbs.append(cv2.imread(img_file))
    batch_frame_end = time.time()
    batch_frame_sec += batch_frame_end - batch_begin

    batch_audio_begin = batch_frame_end
    audio_file = '{}.wav'.format(video_file)
    input_batch = vggish_input.wavfile_to_examples(audio_file)
    batch_audio_end = time.time()
    batch_audio_sec += batch_audio_end - batch_audio_begin

    q = gpu_workers_queue[n % MAX_GPU_WORKER_NUM]
    q.put([vid, rgbs, input_batch])
    if n % print_num == 0:
      print('read_frame_and_audio:{} total_num={} qps={:.3f} frame_sec_per_video={:.3f} audio_sec_per_video={:.3f}'.format(
        worker_id, n, print_num/float(batch_audio_sec + batch_frame_sec)
        ,batch_frame_sec/print_num,batch_audio_sec/print_num))
      batch_frame_sec = 0
      batch_audio_sec = 0
  for q in gpu_workers_queue.values():
    q.put(None)
  print('read_frame_and_audio:{} total_num={} qps={:.3f}'.format(worker_id, n, n/float(time.time() - begin)))


def generate_tfrecord(gpu):
  #set GPU id before importing tensorflow!!!!!!!!!!!!!
  os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_dict[gpu])
  import tensorflow as tf
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  extractor = feature_extractor.YouTube8MFeatureExtractor(FLAGS.model_dir)
  audio_fea = AudioFeature(FLAGS.wav_tmp,FLAGS.model_path)
  audio_fea.load_model(sess)
  writer = [tf.python_io.TFRecordWriter('{}/{}.tfrecords'.format(FLAGS.output_tfrecords_file, i)) for i in range(0,100)]
  total_written = 0
  total_error = 0
  begin = time.time()
  end_of_num = 0
  print_num = 10
  batch_sec = 0
  batch_frame_sec = 0
  batch_audio_sec = 0
  while True:
    data = gpu_workers_queue[gpu].get()
    if data is None:
      end_of_num += 1
      if end_of_num == MAX_WORKER_NUM:
        break
      continue
    batch_begin = time.time()
    video_file,rgbs,audio_input_batch = data
    labels = "0"
    sum_rgb_features = None
    rgb_features = []
    for rgb in rgbs:
      features = extractor.extract_rgb_frame_features(rgb[:, :, ::-1])
      if sum_rgb_features is None:
        sum_rgb_features = features
      else:
        sum_rgb_features += features
      rgb_features.append(_bytes_feature(quantize(features)))
    if not rgb_features:
      print('Could not get features for ' + video_file, file=sys.stderr)
      total_error += 1
      continue
    batch_end = time.time()
    batch_frame_sec += batch_end - batch_begin
    mean_rgb_features = sum_rgb_features / len(rgb_features)
    # Create SequenceExample proto and write to output.
    feature_list = {
        FLAGS.image_feature_key: tf.train.FeatureList(feature=rgb_features),
    }
    context_features = {
        FLAGS.labels_feature_key:
            _int64_list_feature(sorted(map(int, labels.split(';')))),
        FLAGS.video_file_feature_key:
            _bytes_feature(_make_bytes(map(ord, video_file))),
        'mean_' + FLAGS.image_feature_key:
            tf.train.Feature(
                float_list=tf.train.FloatList(value=mean_rgb_features)),
    }
    batch_audio_begin = time.time()
    raw_audio_features  = audio_fea.audio_emb_input_batch(sess,audio_input_batch,FLAGS.pca_enable)
    batch_end = time.time()
    batch_audio_sec += batch_end - batch_audio_begin
    if not (type(raw_audio_features) == type(None)):
      sum_audio_features = None
      audio_features = []
      #print("audio_feature",type(raw_audio_features))
      for i in range(raw_audio_features.shape[0]):
        if sum_audio_features is None:
          sum_audio_features = raw_audio_features[i]
        else:
          sum_audio_features += raw_audio_features[i]
        audio_features.append(_bytes_feature(quantize(raw_audio_features[i])))
      mean_audio_features = sum_audio_features / len(audio_features)
        
      feature_list['audio'] = tf.train.FeatureList(
          feature=audio_features)
      context_features['mean_audio'] = tf.train.Feature(
          float_list=tf.train.FloatList(value=mean_audio_features))
    else:
      zero_vec = [0] * 128
      feature_list['audio'] = tf.train.FeatureList(
          feature=[_bytes_feature(_make_bytes(zero_vec))] * len(rgb_features))
      context_features['mean_audio'] = tf.train.Feature(
          float_list=tf.train.FloatList(value=zero_vec))   

    if FLAGS.skip_frame_level_features:
      example = tf.train.SequenceExample(
          context=tf.train.Features(feature=context_features))
    else:
      example = tf.train.SequenceExample(
          context=tf.train.Features(feature=context_features),
          feature_lists=tf.train.FeatureLists(feature_list=feature_list))
    #print('=======read one features5555=====',video_file)
    writer[int(video_file[-2:])].write(example.SerializeToString())
    total_written += 1
    sec = time.time() - batch_begin
    batch_sec += sec
    if total_written % print_num  == 0:
      end = time.time()
      print('Successfully encoded %i out of %i videos, qps=%f , sec=%d frame_sec_per_video=%.3f audio_sec_per_video=%.3f' %
        (total_written, total_written + total_error, float(print_num)/batch_sec, batch_sec,
          batch_frame_sec / print_num, batch_audio_sec / print_num))
      batch_frame_sec = 0
      batch_audio_sec = 0
      batch_sec = 0
      batch_begin = end
  for w in writer:
    w.close()
  end = time.time()
  print('Successfully encoded %i out of %i videos, sec=%d' %
    (total_written, total_written + total_error, end - begin))

# 生产者与消费者模型的应用
def init_worker_queue():
  # 为每一个worker进程分配自己的队列
  #print('inti gpu workerwait1:', gpu_workers_queue)
  for i in range(0, MAX_WORKER_NUM):
    workers_queue[i] = Manager().Queue()



def main(unused_argv):
  gpus = '0,1'
  for i,gpu in enumerate(gpus.split(',')):
    gpu_workers_queue[i] = Manager().Queue()
    gpu_dict[i] = gpu
  init_worker_queue()
  MAX_GPU_WORKER_NUM = len(gpu_workers_queue)
  p = Pool(MAX_WORKER_NUM + 1 + MAX_GPU_WORKER_NUM)
  p.apply_async(read_input_fn)
  for i in range(0, MAX_WORKER_NUM):
    p.apply_async(read_frame_and_audio_fn, args=(i,))
  for i in range(0, MAX_GPU_WORKER_NUM):
    p.apply_async(generate_tfrecord,args=(i,))
  p.close()
  p.join()


if __name__ == '__main__':
  app.run(main)
