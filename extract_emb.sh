#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
MODEL=/home/taisimin/linrongc/nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic
INPUT_DATA=/da5/public/taisimin/tfrecord_all/*.tfrecord
OUTPUT_DATA=output

python extract_embedding.py --train_dir ${MODEL} --input_data_pattern "$INPUT_DATA" --output_file $OUTPUT_DATA --batch_size 32
