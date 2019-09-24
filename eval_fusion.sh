export CUDA_VISIBLE_DEVICES=3
eval_path="/home/taisimin/frame/validation/*"
train_dir="trained_model/SelfAttentionFusionNeXtVLADModel"
parameters="--groups=8 --nextvlad_cluster_size=128 --nextvlad_hidden_size=2048 \
            --expansion=2 --gating_reduction=8 --drop_rate=0.5"

  #--eval_data_pattern="/home/taisimin/frame/validation/*/validate*.tfrecord" \
  #--eval_data_pattern="/home/taisimin/frame/validation/0/validate3843.tfrecord" \
python eval.py ${parameters} --batch_size=80 --video_level_classifier_model=LogisticModel --l2_penalty=1e-5 \
               --label_loss=CrossEntropyLoss --eval_data_pattern=${eval_path}/validate*.tfrecord --train_dir ${train_dir} \
               --run_once=True --num_readers 12 --num_epochs 1
