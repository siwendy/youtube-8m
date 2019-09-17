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
import time
import numpy
#import tensorflow as tf
#from tensorflow import flags
from multiprocessing import Pool, Manager
import subprocess

import argparse
import random

arg_parser = argparse.ArgumentParser(description='filter_parser')
arg_parser.add_argument('--data_tmp', type=str,
    default=False, help='mapper')
arg_parser.add_argument('--input_videos_csv', type=str,
    default=False, help='reducer')
FLAGS = arg_parser.parse_args()

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
  #fos = open('{}/stat/worker_{}'.format(FLAGS.data_tmp, worker_id), 'w',  buffering=0)
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
    #print(' '.join(wav_cmd))
    print('id={} num={} video={}'.format(worker_id, n, video))
    res1 = subprocess.call(' '.join(wav_cmd), shell=True)
    res2 = subprocess.call(' '.join(frame_cmd), shell=True)
    fos.write('{}\t{}\t{}\t{}\n'.format(vid, video, res1, res2))
  print('worker:{} total_num={} qps={:.3f}'.format(worker_id, n, n/float(time.time() - begin)))
  fos.close()


if __name__ == '__main__':
  #res1 = subprocess.call('ffmpeg -loglevel quiet -y -i /da5/taisimin/data/tuitui/DAM/src/data/40/972784854442646640.mp4 -f wav frame//9/972784854442646640.wav video=frame//9/972784854442646640', shell=True)
  #print(res1)
  #return -1
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
