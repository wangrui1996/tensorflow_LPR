"""
Write text features and labels into tensorflow records
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import json

import tensorflow as tf
import cv2
import numpy as np

from tools.config import config, default

_IMAGE_HEIGHT = config.image_height
_IMAGE_WIDTH = config.image_width

_CROP_SIZE = 5

tf.app.flags.DEFINE_string(
    'image_dir', default.image_dir, 'Dataset root folder with images.')

tf.app.flags.DEFINE_string(
    'anno_file', default.anno_file, 'Path of dataset annotation file.')

tf.app.flags.DEFINE_string(
    'data_dir', default.data_dir, 'Directory where tfrecords are written to.')

tf.app.flags.DEFINE_float(
    'validation_split_fraction', default.validation_split_fraction, 'Fraction of training data to use for validation.')

tf.app.flags.DEFINE_boolean(
    'shuffle_list', default.shuffle_list, 'Whether shuffle data in annotation file list.')

tf.app.flags.DEFINE_string(
    'char_map_json_file', default.char_map_json_file, 'Path to char map json file')

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _string_to_int(label, char_map_dict=None):
    if char_map_dict is None:
        # convert string label to int list by char map
        char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    int_list = []
    for c in label:
        int_list.append(char_map_dict[c])
    return int_list

def _write_tfrecord(dataset_split, anno_lines, char_map_dict=None, training=False):
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    tfrecords_path = os.path.join(FLAGS.data_dir, dataset_split + '.tfrecord')
    with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
        for i, line in enumerate(anno_lines):
            line = line.strip()
            image_name = line.split()[0]
            image_path = os.path.join(FLAGS.image_dir, image_name)
            label = line.split()[1].lower()

            image = cv2.imread(image_path)
            if image is None:
                print("Can not read image of : '{}'".format(image_path))
                exit(0)
                continue # skip bad image.
            height = _IMAGE_HEIGHT
            width = _IMAGE_WIDTH
            if training:
                height = height + _CROP_SIZE
                width = width + _CROP_SIZE
            image = cv2.resize(image, (width, height))
            is_success, image_buffer = cv2.imencode('.jpg', image)
            if not is_success:
                continue

            # convert string object to bytes in py3
            image_name = image_name if sys.version_info[0] < 3 else image_name.encode('utf-8')
            features = tf.train.Features(feature={
               'labels': _int64_feature(_string_to_int(label, char_map_dict)),
               'images': _bytes_feature(image_buffer.tostring()),
               'imagenames': _bytes_feature(image_name)
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
            sys.stdout.write('\r>>Writing to {:s}.tfrecords {:d}/{:d}'.format(dataset_split, i + 1, len(anno_lines)))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.write('>> {:s}.tfrecords write finish.'.format(dataset_split))
        sys.stdout.flush()

def _convert_dataset():
    char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    
    with open(FLAGS.anno_file, 'r') as anno_fp:
        anno_lines = anno_fp.readlines()

    if FLAGS.shuffle_list:
        random.shuffle(anno_lines)
    
    # split data in annotation list to train and val
    split_index = int(len(anno_lines) * (1 - FLAGS.validation_split_fraction))
    train_anno_lines = anno_lines[:split_index - 1]
    validation_anno_lines = anno_lines[split_index:]

    dataset_anno_lines = {'train' : train_anno_lines, 'validation' : validation_anno_lines}
    for dataset_split in ['train', 'validation']:
        if dataset_split == "validation":
            _write_tfrecord(dataset_split, dataset_anno_lines[dataset_split], char_map_dict, False)
        else:
            _write_tfrecord(dataset_split, dataset_anno_lines[dataset_split], char_map_dict, True)

def main(unused_argv):
    _convert_dataset()

if __name__ == '__main__':
    tf.app.run()
