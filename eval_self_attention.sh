export CUDA_VISIBLE_DEVICES=2
eval_path="/home/taisimin/frame/validation/*"
train_dir="trained_model/SelfAttentionModel"
parameters="--self_attention_n_head=8 --self_attention_n_layer=2 --self_attention_filter_size=2 --self_attention_cluster_dropout=0.5 --self_attention_ff_dropout=0.5 --self_attention_hidden_size=512"

  #--eval_data_pattern="/home/taisimin/frame/validation/*/validate*.tfrecord" \
  #--eval_data_pattern="/home/taisimin/frame/validation/0/validate3843.tfrecord" \
python eval.py ${parameters} --batch_size=80 --video_level_classifier_model=LogisticModel --l2_penalty=1e-5 \
               --label_loss=CrossEntropyLoss --eval_data_pattern=${eval_path}/validate*.tfrecord --train_dir ${train_dir} \
               --run_once=True --num_readers 12 --num_epochs 1
