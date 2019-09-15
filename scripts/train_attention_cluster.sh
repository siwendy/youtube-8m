#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
num_gpu=1

datapath=/media/linrongc/dream/data/yt8m/2/frame/train
eval_path=$HOME/datasets/competitions/yt8m/2/val
test_path=/media/linrongc/dream/data/yt8m/2/frame/test
datapath=/home/taisimin/frame/tmp/*/

model_name=AttentionClusterModel
parameters="--video_cluster_size=128 --audio_cluster_size=16 --shift_operation=True --filter_size=2 --cluster_dropout=0.7 --ff_dropout=0.8 --hidden_size=512"

train_dir=trained_model/AttentionClusterModel
result_folder=results

echo "model name: " $model_name
echo "model parameters: " $parameters

echo "training directory: " $train_dir
echo "data path: " $datapath
echo "evaluation path: " $eval_path

python train.py ${parameters} --model=${model_name}  --num_readers=8 --learning_rate_decay_examples 2000000 \
                --video_level_classifier_model=LogisticModel --label_loss=CrossEntropyLoss --start_new_model=False \
                --train_data_pattern=/home/taisimin/frame/tmp/*/train*.tfrecord --train_dir=${train_dir} --frame_features=True \
                --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=80 --base_learning_rate=0.0002 \
                --learning_rate_decay=0.8 --l2_penalty=1e-5 --max_step=700000 --num_gpu=${num_gpu} > log.${model_name} 2>&1 &

#python eval.py ${parameters} --batch_size=80 --video_level_classifier_model=LogisticModel --l2_penalty=1e-5\
#               --label_loss=CrossEntropyLoss --eval_data_pattern=${eval_path}/validate*.tfrecord --train_dir ${train_dir} \
#               --run_once=True
#
#mkdir -p $result_folder
#python inference.py --output_model_tgz ${result_folder}/${train_dir}.tgz \
#                    --output_file ${result_folder}/${train_dir}.csv \
#                    --input_data_pattern=${test_path}/test*.tfrecord --train_dir ${train_dir} \
#                    --batch_size=80 --num_readers=8