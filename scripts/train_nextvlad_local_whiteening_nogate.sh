#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
num_gpu=1

datapath=/media/linrongc/dream/data/yt8m/2/frame/train
eval_path=$HOME/datasets/competitions/yt8m/2/val
test_path=/media/linrongc/dream/data/yt8m/2/frame/test

model_name=NeXtVLADModel
parameters="--groups=8 --nextvlad_cluster_size=128 --nextvlad_hidden_size=2048 \
            --expansion=2 --gating_reduction=8 --drop_rate=0.5 --reverse_whiteening=True --enable_gate=False"

train_dir=trained_model/nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic_whiteening_no_gate
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
                --learning_rate_decay=0.8 --l2_penalty=1e-5 --max_step=700000 --num_gpu=${num_gpu} > log.${model_name}.whiteening.no_gate 2>&1 &
