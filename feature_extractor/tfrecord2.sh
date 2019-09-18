#export CUDA_VISIBLE_DEVICES=1;python3 extract_tfrecords_main.py --input_videos_csv  ../submit.csv  --skip_frame_level_features --output_tfrecords_file ../../../data/ugc_emb.tfrecord   
#export CUDA_VISIBLE_DEVICES=1;python3 extract_tfrecords.py --input_videos_file  worker  --output_tfrecords_file output_tf > log.tf 2>&1 &
export CUDA_VISIBLE_DEVICES=0;python3 extract_tfrecords.py --input_videos_file  a --output_tfrecords_file output_tf 
