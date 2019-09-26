#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
INPUT_DATA="/da3/rec/yangyang1-pd/data/tfrecord_all/*.tfrecord"
MODEL=nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic_whiteening
LOG_FILE=log.emb
python extract_embedding.py --reverse_whiteening True --train_dir "trained_model/${MODEL}" --input_data_pattern "$INPUT_DATA" --output_file "output_emb.$MODEL" --batch_size 32 > ${LOG_FILE} 2>&1 &
