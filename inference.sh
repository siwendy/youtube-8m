export CUDA_VISIBLE_DEVICES=1
eval_path="/da3/rec/yangyang1-pd/data/tfrecord_all"
train_dir="/da5/public/taisimin/trained_model/nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic_whiteening"
parameters="--groups=8 --nextvlad_cluster_size=128 --nextvlad_hidden_size=2048 \
            --expansion=2 --gating_reduction=8 --drop_rate=0.5 --reverse_whiteening=True"

python inference.py  --batch_size=80 --label_loss=CrossEntropyLoss \
 --input_data_pattern=${eval_path}/*.tfrecord --train_dir ${train_dir} \
 --output_file="output.csv"
 #--num_readers 12 --num_epochs 1
