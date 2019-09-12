#!/bin/bash
# @Author: zhuyuanyuan@360.cn
# @Create time: 2018-10-25 11:58:43
# @Last Changed: 2019-04-22 11:47:59

source ~/.bashrc
WORKDIR=`dirname $0`



python=/home/hdp-svideo-algo/anaconda3/envs/yangyang1-pd/bin/python
base_dir=`pwd`
i=0
video_tmp="/da1/home/yangyang1-pd/data"
wav_tmp="../../../data/wav_tmp"
submit_csv="${base_dir}/../../../data/submit_sec.csv"
#submit_csv="./submit.csv"
all_csv="${base_dir}/../../../data/all_sec.csv"
source activate yangyang1-pd
rm ${all_csv}

for ((i=51; i<60; i++))
do
    video_path=${video_tmp}/$i
    ls ${video_path} | grep "mp4" | head -10 |  awk '{print $1",1"}' | python ../../src/add_prefix.py ${video_path} > ${submit_csv}
    cat ${submit_csv} >> ${all_csv}
    #cat ${submit_csv}
    export CUDA_VISIBLE_DEVICES=3;kernprof -l  extract_tfrecords_with_audio.py --input_videos_csv ${submit_csv}  --output_tfrecords_path /da1/home/yangyang1-pd/projects --first_vid $i  
    #export CUDA_VISIBLE_DEVICES=3; ${python} -m cProfile -s cumulative  ./extract_tfrecords_with_audio.py --input_videos_csv ${submit_csv}  --output_tfrecords_path /da1/home/yangyang1-pd/projects --first_vid $i  

    #export CUDA_VISIBLE_DEVICES=3;python  ${base_dir}/../plugins/feature_extractor/extract_tfrecords_with_audio.py --input_videos_csv ${submit_csv}  --output_tfrecords_path /da1/home/yangyang1-pd/projects --first_vid $i  
    break
done  
