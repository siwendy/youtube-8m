#!/bin/sh

#NVIDIA_VISIBLE_DEVICES=1 python extract_request_main.py --input_videos_csv ./submit.csv --output_tfrecords_file test_record  
NVIDIA_VISIBLE_DEVICES=1 python extract_tfrecords_main.py --input_videos_csv ./test.csv --output_tfrecords_file test.tfrecord  

