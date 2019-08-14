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

from tools.config import config, default, generate_config
generate_config(default.network, default.dataset)
_IMAGE_HEIGHT = config.image_height
_IMAGE_WIDTH = config.image_width

_CROP_SIZE = 5


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
        char_map_dict = json.load(open(config.char_map_json_file, 'r'))
    int_list = []
    for c in label:
        int_list.append(char_map_dict[c])
    return int_list

def write_tfrecord(dataset_split, anno_lines, char_map_dict=None, training=False):
    if not os.path.exists(config.data_store_path):
        os.makedirs(config.data_store_path)

    tfrecords_path = os.path.join(config.data_store_path, dataset_split + '.tfrecord')
    with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
        for i, line in enumerate(anno_lines):
            line = line.strip()
            image_name = line.split()[0]
            image_path = os.path.join(config.data_store_path, "images", dataset_split, image_name)
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

def convert_dataset():
    char_map_dict = json.load(open(config.plate_map_json_file, 'r'))

    trainval_targets = config.trainval_targets
    train_ratios = config.train_ratio
    train_images_lines = []
    for index in range(len(trainval_targets)):
        dataset_name = trainval_targets[index]
        train_ratio = train_ratios[index]
        image_list_path = os.path.join(config.data_store_path, "{}.txt".format(dataset_name))
        with open(image_list_path, 'r') as image_list:
            images_lines = image_list.readlines()
            if config.shuffle_list:
                random.shuffle(images_lines)
        # split data in annotation list to train and val
        split_index = int(len(images_lines) * train_ratio)
        train_images_lines = train_images_lines + images_lines[:split_index]
        validation_images_lines = images_lines[split_index:]
        if len(validation_images_lines) != 0:
            write_tfrecord(dataset_name, validation_images_lines, char_map_dict, False)
    write_tfrecord('train', train_images_lines, char_map_dict, True)


def main():
    convert_dataset()

if __name__ == '__main__':
    main()