#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
MODEL=/home/taisimin/linrongc/nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic
INPUT_DATA=/da5/public/taisimin/tfrecord_all/*.tfrecord
OUTPUT_DATA=output

#python extract_embedding.py --train_dir ${MODEL} --input_data_pattern "$INPUT_DATA" --output_file $OUTPUT_DATA --batch_size 32
#python extract_embedding.py --train_dir ${MODEL} --input_data_pattern "/da5/public/taisimin/tfrecord_all/68.tfrecord" --output_file 'output.68' --batch_size 32
MODEL=/home/taisimin/linrongc/model_test
python extract_embedding.py --train_dir ${MODEL} --input_data_pattern "/da5/public/taisimin/tfrecord_all/*.tfrecord" --output_file 'output.all.b1' --batch_size 32 > log.emb 2>&1
#MODEL=/home/taisimin/linrongc/nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic
#python extract_embedding.py --train_dir ${MODEL} --input_data_pattern "/da5/public/taisimin/tfrecord_all/*.tfrecord" --output_file 'output.all.b2' --batch_size 32 >> log.emb 2>&1
#python extract_embedding.py --train_dir ${MODEL} --input_data_pattern "/da5/public/taisimin/tfrecord_all/68.tfrecord" --output_file 'output.68.b3' --batch_size 32 >> log.emb 2>&1
#python extract_embedding.py --train_dir ${MODEL} --input_data_pattern "emb.tfrecord" --output_file emb.tfrecord.output --batch_size 32
#python extract_embedding.py --train_dir ${MODEL} --input_data_pattern "test.csv.tfrecord" --output_file test.csv.tfrecord.output --batch_size 32
