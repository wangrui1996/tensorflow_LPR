python tools/create_crnn_ctc_tfrecord.py \
  --image_dir /home/rui/work/crnn_ctc_ocr.Tensorflow/data --anno_file /home/rui/work/crnn_ctc_ocr.Tensorflow/train_data/image_list.txt --data_dir ./tfrecords/ \
  --validation_split_fraction 0.1
