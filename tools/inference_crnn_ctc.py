from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

import cv2
import numpy as np
from src.models import pyramidcnnmodel
from tools.config import config, generate_config, default

generate_config(default.network, default.dataset)

os.environ["CUDA_VISIBLE_DEVICES"]=""

_IMAGE_HEIGHT = 64
_IMAGE_WIDTH = 128

image_dir = './test_data/images/'

# ------------------------------------Basic prameters------------------------------------
image_list = './test_data/image_list.txt'

tf.app.flags.DEFINE_string(
    'model_dir', './model/', 'Base directory for the model.')


def _sparse_matrix_to_list(sparse_matrix, char_map_dict=None):
    indices = sparse_matrix.indices
    values = sparse_matrix.values
    dense_shape = sparse_matrix.dense_shape

    # the last index in sparse_matrix is ctc blanck note
    if char_map_dict is None:
        char_map_dict = json.load(open(config.plate_map_json_file, 'r'))
    assert(isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')    

    dense_matrix =  len(char_map_dict.keys()) * np.ones(dense_shape, dtype=np.int32)
    for i, indice in enumerate(indices):
        dense_matrix[indice[0], indice[1]] = values[i]
    string_list = []
    for row in dense_matrix:
        string = []
        for val in row:
            string.append(_int_to_string(val, char_map_dict))
        string_list.append(''.join(s for s in string if s != '*'))
    return string_list

def _int_to_string(value, char_map_dict=None):
    if char_map_dict is None:
        char_map_dict = json.load(open(config.plate_map_json_file, 'r'))
    assert(isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')
    
    for key in char_map_dict.keys():
        if char_map_dict[key] == int(value):
            return str(key)
        elif len(char_map_dict.keys()) == int(value):
            return "" 
    raise ValueError('char map dict not has {:d} value. convert index to char failed.'.format(value))

def _inference_crnn_ctc():
    input_images = tf.placeholder(dtype=tf.float32, shape=[1, _IMAGE_HEIGHT, _IMAGE_WIDTH, 3])
    char_map_dict = json.load(open(config.plate_map_json_file, 'r'))
    # initialise the net model

    training = tf.placeholder(tf.bool, name='training')
    net_out = pyramidcnnmodel.build_network(input_images, len(char_map_dict.keys()) + 1, training)

    input_sequence_length = tf.placeholder(tf.int32, shape=[1], name='input_sequence_length')

    ctc_decoded, ct_log_prob = tf.nn.ctc_beam_search_decoder(net_out, input_sequence_length, merge_repeated=False)

    init_op = tf.global_variables_initializer()
    with open(image_list, 'r') as fd:
       image_names = [line.strip() for line in fd.readlines()]
    if config.stn:
        model_save_path = default.model_stn_save_path
    else:
        model_save_path = default.model_save_path
    # set checkpoint saver
    save_path = tf.train.latest_checkpoint(model_save_path)
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['CRNN_CTC/locnet'])
    init_fn = tf.contrib.framework.assign_from_checkpoint_fn(save_path, variables_to_restore)
    with tf.Session() as sess:
        sess.run(init_op)
        # restore all variables
        #saver.restore(sess=sess, save_path=save_path)
        try:
            init_fn(sess)
        except:
            print("can not find model weight")

        for image_name in image_names:
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (_IMAGE_WIDTH, _IMAGE_HEIGHT))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)
            image = np.array(image, dtype=np.float32)
            seq_len = np.array([_IMAGE_WIDTH / 8], dtype=np.int32)

            preds = sess.run(ctc_decoded, feed_dict={input_images:image, input_sequence_length:seq_len, training:False})
 
            preds = _sparse_matrix_to_list(preds[0], char_map_dict)

            print('Predict {:s} image as: {:s}'.format(image_name, preds[0]))
        
def main(unused_argv):
    _inference_crnn_ctc()

if __name__ == '__main__':
    tf.app.run() 
