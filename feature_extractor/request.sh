export CUDA_VISIBLE_DEVICES=1;python3 docker_request_video.py --input_videos_csv  submit1.csv  --skip_frame_level_features --output_tfrecords_file ../data/ugc_emb.tfrecord   
