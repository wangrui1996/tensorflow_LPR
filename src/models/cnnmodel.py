from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from src.stn.transformer import spatial_transformer_network
from tools.config import config, default
_BATCH_DECAY = 0.999


def build_network(images, num_classes=default.num_classes, training=None, stn=False):
    tf.logging.info("Loading CNN Model")

    if stn:
        tf.logging.info("Start to loading stn network")
        # locnet
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=None):
            with tf.variable_scope('locnet'):
                n_fc = 6
                B, H, W, C = images.shape
                # identity transform
                initial = np.array([[1., 0, 0], [0, 1., 0]])
                initial = initial.astype('float32').flatten()
                # Output Layer Transformation
                # localization network

                # 64 x 128
                avg_net = slim.avg_pool2d(images, kernel_size=2, stride=2, scope="loc_pool1")
                # 32 x 64
                conv1_1_net = slim.conv2d(avg_net, 32, kernel_size=3, stride=4, scope='loc_conv11')
                # 8 x 16
                conv1_2_net = slim.conv2d(images, 32, kernel_size=5, stride=8, scope='loc_conv11')

                loc_concat_net = tf.concat(3, [conv1_1_net, conv1_2_net], name='loc_concat')
                #loc_net = slim.repeat(images, 2, slim.conv2d, 32, kernel_size=3, stride=1, scope='loc_conv1')
                #loc_net = slim.max_pool2d(loc_net, kernel_size=2, stride=2, scope='loc_pool1')
                # 8 x 16
                loc_net = slim.conv2d(conv1_2_net, 128, kernel_size=3, stride=1, scope='loc_conv3')
                loc_net = slim.batch_norm(loc_net, decay=_BATCH_DECAY, is_training=training, scope='loc_bn1')
                loc_net = slim.conv2d(loc_net, 32, kernel_size=3, stride=1, scope='loc_conv4')
                loc_net = slim.batch_norm(loc_net, decay=_BATCH_DECAY, is_training=training, scope='loc_bn2')
                loc_net = slim.max_pool2d(loc_net, kernel_size=5, stride=4, scope='loc_pool3')
                # 2 x 4
                loc_net = slim.conv2d(loc_net, 16, kernel_size=3, stride=1, scope='loc_conv5')
                loc_net = tf.reduce_mean(input_tensor=loc_net, axis=[1, 2], keep_dims=False, name="loc_se_pool1")
                loc_net = tf.reshape(loc_net, [loc_net.shape[0], -1])
                loc_B, loc_W = loc_net.shape
                W_fc1 = tf.Variable(tf.zeros([loc_W, n_fc]), name='W_fc1')
                b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
                loc_net = tf.matmul(loc_net, W_fc1) + b_fc1
                loc_output = spatial_transformer_network(images, loc_net)
                images = loc_output
                tf.logging.info("stn network loaded...")

        # 1 x 2

    # first apply the cnn feature extraction stage
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        biases_initializer=None):
        tf.logging.info("Start to loading cnn feature extraction network")
        net = slim.repeat(images, 2, slim.conv2d, 32, kernel_size=3, stride=1, scope='conv1')
        net = slim.max_pool2d(net, kernel_size=2, stride=2, scope='pool1')
        # 32 x 64
        net = slim.repeat(net, 2, slim.conv2d, 64, kernel_size=3, stride=1, scope='conv2')
        net = slim.max_pool2d(net, kernel_size=2, stride=2, scope='pool2')
        # 16 x 32
        net = slim.repeat(net, 2, slim.conv2d, 128, kernel_size=3, stride=1, scope='conv3')
        net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], scope='pool3')
        # 8 x 32
        net = slim.conv2d(net, 256, kernel_size=3, stride=1, scope='conv4')
        net = slim.batch_norm(net, decay=_BATCH_DECAY, is_training=training, scope='bn4')
        net = slim.conv2d(net, 256, kernel_size=3, stride=1, scope='conv5')
        net = slim.batch_norm(net, decay=_BATCH_DECAY, is_training=training, scope='bn5')
        net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], scope='pool5')
        # 4 x 32
        net = slim.conv2d(net, 256, padding="VALID", kernel_size=2, stride=2, scope='conv6')
        # 2 x 32
        cnn_out = slim.conv2d(net, 512, padding="VALID", kernel_size=[2, 1], stride=1, scope='conv7')
        # 1 x 32
        tf.logging.info("feature network loaded")
    # second apply the map to sequence stage
    shape = cnn_out.get_shape().as_list()
    assert shape[1] == 1
    sequence = tf.squeeze(cnn_out, axis=1)
    # third apply the sequence label stage
    shape = sequence.get_shape().as_list()
    B, W, C = shape
    with tf.variable_scope('Softmax_Layers'):
        # forward lstm cell
        # Doing the affine projection
        w = tf.Variable(tf.truncated_normal([C, num_classes], stddev=0.01), name="w")
        b = tf.Variable(tf.truncated_normal([num_classes], stddev=0.01), name="b")
        logits = tf.matmul(sequence, w) + b

        logits = tf.reshape(logits, [B, W, num_classes])
        # Swap batch and batch axis
        net_out = tf.transpose(logits, (1, 0, 2), name='transpose_time_major')
    return net_out
