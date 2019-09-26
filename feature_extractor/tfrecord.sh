#!/bin/bash
# @Author: zhuyuanyuan@360.cn
# @Create time: 2018-10-25 11:58:43
# @Last Changed: 2019-04-22 11:47:59

WORKDIR=`dirname $0`
INPUT_CSV=test.csv
OUTPUT=test.tfrecord
WAV_TMP=tmp

#安装环境
#1.pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pydub resampy pysoundfile 

mkdir  ${WAV_TMP}

export CUDA_VISIBLE_DEVICES=2;python extract_tfrecords_with_audio.py --input_videos_csv ${INPUT_CSV}  --output_tfrecords_file $OUTPUT   --wav_tmp $WAV_TMP
