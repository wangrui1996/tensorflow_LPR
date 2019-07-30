from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import slim
from tools.config import config, default
_BATCH_DECAY = 0.999


def build_network(images, num_classes=default.num_classes, phase="test"):
    # first apply the cnn feature extraction stage
    is_training = True if phase == 'train' else False
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        biases_initializer=None):
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
        net = slim.batch_norm(net, decay=_BATCH_DECAY, is_training=is_training, scope='bn4')
        net = slim.conv2d(net, 256, kernel_size=3, stride=1, scope='conv5')
        net = slim.batch_norm(net, decay=_BATCH_DECAY, is_training=is_training, scope='bn5')
        net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], scope='pool5')
        # 4 x 32
        net = slim.conv2d(net, 256, padding="VALID", kernel_size=2, stride=2, scope='conv6')
        # 2 x 32
        cnn_out = slim.conv2d(net, 512, padding="VALID", kernel_size=[2, 1], stride=1, scope='conv7')
        # 1 x 32
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
