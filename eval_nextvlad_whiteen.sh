export CUDA_VISIBLE_DEVICES=1
eval_path="/home/taisimin/frame/validation/*"
train_dir="/da5/public/taisimin/trained_model/nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic_whiteening"
parameters="--groups=8 --nextvlad_cluster_size=128 --nextvlad_hidden_size=2048 \
            --expansion=2 --gating_reduction=8 --drop_rate=0.5 --reverse_whiteening=True"

  #--eval_data_pattern="/home/taisimin/frame/validation/*/validate*.tfrecord" \
  #--eval_data_pattern="/home/taisimin/frame/validation/0/validate3843.tfrecord" \
python eval.py ${parameters} --batch_size=80 --video_level_classifier_model=LogisticModel --l2_penalty=1e-5 \
               --label_loss=CrossEntropyLoss --eval_data_pattern=${eval_path}/validate*.tfrecord --train_dir ${train_dir} \
               --run_once=True --num_readers 12 --num_epochs 1
