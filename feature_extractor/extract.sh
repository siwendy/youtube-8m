export CUDA_VISIBLE_DEVICES=1;python3 extract_tfrecords_main.py --input_videos_csv  ../submit.csv  --skip_frame_level_features --output_tfrecords_file ../../../data/ugc_emb.tfrecord   
