#!/bib/bash/python

import sys
import json
import hashlib
import base64
import re,os
import threading
import multiprocessing


#encoding: utf-8
from multiprocessing import Pool, Manager
from time import sleep, time
import csv
from requests import get, head

# 模拟一共有1W封邮件

workders_queue = {}
# worker进程数，别设置多了，和自己的cpu核心数有关。
MAX_WORKER_NUM = 1

# 生产者与消费者模型的应用
def init_worker_queue():
  # 为每一个worker进程分配自己的队列
  for i in range(0, MAX_WORKER_NUM):
    workders_queue[i] = Manager().Queue()

# 读取email的数据，并放入每个worker对应的队列里。
def read_data():
  index = 0
  vid_done = {}
  #for line in open(sys.argv[2]):
  #  vid_done[line.strip().split('\t')[0]] = 1
  for line in open(sys.argv[1]):
    #print('input:')
    q = workders_queue[index % MAX_WORKER_NUM]
    #print(index%MAX_WORKER_NUM)
    index += 1
    encode_id,video_id = line.strip('\r\n').split('\t')
    if video_id == '':
      continue
    q.put([encode_id,video_id])
  for q in workders_queue.values():
    q.put(None)

# 处理队列里的数据，将结果录入结果队列里
def workder_data(worker_id):
  print("worker_id:{}".format(worker_id))
  n = 0
  fos = open('video_title/video_id_{}.txt'.format(worker_id), 'w')
  print_num = 500
  begin = time()
  while True:
    q = workders_queue[worker_id]
    data = q.get()
    n += 1
    if data == None:
      print('finished',worker_id)
      break
    try:
      vid,video_id = data
    except:
      #print("===",data)
      continue
    if n % print_num == 0:
      end = time()
      ws = end - begin
      begin = end
      print('qps={} sec={}'.format(print_num/float(ws), ws))
    url = 'https://noembed.com/embed?url=https://www.youtube.com/watch?v={}'.format(video_id)
    #print(url)
    try:
      r = get(url, stream=True)
    except:
      #fos.write('{}\t{}\n'.format(vid, video_id))
      print('failed get={}\turl={}'.format(vid, url))
      continue
    print(worker_id, url)
    json_r = json.loads(str(r.content,encoding='utf-8'))
    title = json_r['title'].strip() if 'title' in json_r else ''
    fos.write('{}\t{}\t{}\n'.format(vid,video_id,title))
  fos.close()
  print('worker:{} download num={} download_per_sec={:.3f}'.format(worker_id, n, 100/(end_time - start_time)))

# 从结果队列里读取数据，并写入csv文件里，方便以后分析数据。
if __name__ == '__main__':
  worker_id = sys.argv[2]
  fos = open('video_title/video_id_{}.txt'.format(worker_id), 'w', buffering=1)
  for line in open(sys.argv[1]):
    vid,video_id = line.strip('\r\n').split('\t')
    if video_id == '':
      continue
    url = 'https://noembed.com/embed?url=https://www.youtube.com/watch?v={}'.format(video_id)
    #print(url)
    try:
      r = get(url, stream=True, timeout=10)
    except:
      #fos.write('{}\t{}\n'.format(vid, video_id))
      print('failed get={}\turl={}'.format(vid, url))
      continue
    json_r = json.loads(str(r.content,encoding='utf-8'))
    title = json_r['title'].strip() if 'title' in json_r else ''
    #print(worker_id, title, url)
    fos.write('{}\t{}\t{}\n'.format(vid,video_id,title))
  fos.close()

  #start_time = time()
  #init_worker_queue()
  #p = Pool(MAX_WORKER_NUM)
  #p.apply_async(read_data)
  #for i in range(0, MAX_WORKER_NUM):
  #  p.apply_async(workder_data, args=(i,))
  #p.close()
  #p.join()
  #end_time = time()
  #print('handle time:%ss' % (end_time - start_time))