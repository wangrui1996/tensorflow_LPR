from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from src.utils.stn import spatial_transformer_network
from src.utils.distribution import channel_wise_attention
from tools.config import config, default
_BATCH_DECAY = 0.999


def AtrousBlock(input_tensor, filters, rate, block_id, stride=1):
    x = slim.conv2d(input_tensor, filters, kernel_size=3, stride=stride, rate=rate, biases_initializer=None,scope=block_id + '_dilation')
    return x

def CFE(input_tensor, filters, block_id, training):
    rate = [3, 5, 7]
    cfe0 = slim.conv2d(input_tensor, filters, kernel_size=1, biases_initializer=None,
                scope=block_id + '_cf20')
    cfe1 = AtrousBlock(input_tensor, filters, rate[0], block_id + '_cfe1')
    cfe2 = AtrousBlock(input_tensor, filters, rate[1], block_id + '_cfe2')
    cfe3 = AtrousBlock(input_tensor, filters, rate[2], block_id + '_cfe3')
    cfe_concat = tf.concat([cfe0, cfe1, cfe2, cfe3], axis=-1, name=block_id + 'concatcfe')
    cfe_concat = slim.batch_norm(cfe_concat, decay=_BATCH_DECAY, is_training=training, scope='bn1')
    return cfe_concat

def BilinearUpsampling(input_tensor, upsampling=(1,1), name=None):
    _, H, W, _ = input_tensor.get_shape().as_list()
    return tf.image.resize_bilinear(input_tensor, [H * upsampling[0], W * upsampling[1]], align_corners=True, name=name)

def BilinearDownsampling(input_tensor, upsampling=(1,1), name=None):
    _, H, W, _ = input_tensor.get_shape().as_list()
    return tf.image.resize_bilinear(input_tensor, [int(H / upsampling[0]), int(W / upsampling[1])], align_corners=True, name=name)


def ChannelWiseAttention(input_tensor,name):
    H, W, C = map(int, input_tensor.get_shape()[1:])
    attention = tf.reduce_mean(input_tensor, axis=[1, 2], keep_dims=True, name=name + "_GlobalAveragePooling2D")
    #attention = GlobalAveragePooling2D(name=name+'_GlobalAveragePooling2D')(inputs)
    attention = slim.fully_connected(attention, C / 4)
    attention = slim.fully_connected(attention, C, activation_fn=tf.nn.sigmoid)
    #attention = Dense(C, activation='sigmoid',activity_regularizer=l1_reg)(attention)
    attention = tf.reshape(attention, [1, 1, C], name=name+"_repeat")
    #attention = Reshape((1, 1, C),name=name+'_reshape')(attention)
    attention = tf.tile(attention, multiples=[1, H, W, 1], name=name + "_repeat")
    #attention = Repeat(repeat_list=[1, H, W, 1],name=name+'_repeat')(attention)
    attention = tf.multiply(attention, input_tensor, name=name + "_multiply")
    #attention = Multiply(name=name + '_multiply')([attention, inputs])
    return attention


def SpatialAttention(input_tensor, training, name):
    k = 9
    H, W, C = map(int,input_tensor.get_shape()[1:])
    attention1 = slim.conv2d(input_tensor, C / 2, kernel_size=(1, k), scope=name + '_1_conv1')
    #attention1 = Conv2D(C / 2, (1, k), padding='same', name=name+'_1_conv1')(inputs)
    attention1 = slim.batch_norm(attention1, decay=_BATCH_DECAY, is_training=training, scope=name+'_1_conv1')
    #attention1 = BN(attention1,'attention1_1')
    attention1 = slim.conv2d(attention1, 1, kernel_size=(k, 1), scope=name + '_1_conv2')
    #attention1 = Conv2D(1, (k, 1), padding='same', name=name + '_1_conv2')(attention1)
    attention1 = slim.batch_norm(attention1, decay=_BATCH_DECAY, is_training=training, scope='attention1_2')
    #attention1 = BN(attention1, 'attention1_2')
    attention2 = slim.conv2d(input_tensor, C / 2, kernel_size=(k, 1), scope=name + '_2_conv1')
    #attention2 = Conv2D(C / 2, (k, 1), padding='same', name=name + '_2_conv1')(inputs)
    attention2 = slim.batch_norm(attention2, decay=_BATCH_DECAY, is_training=training, scope='attention2_1')
    #attention2 = BN(attention2, 'attention2_1')
    attention2 = slim.conv2d(attention2, 1, kernel_size=(1, k), scope=name + '_2_conv2')
    #attention2 = Conv2D(1, (1, k), padding='same', name=name + '_2_conv2')(attention2)
    attention2 = slim.batch_norm(attention2, decay=_BATCH_DECAY, is_training=training, scope='attention2_2')
    #attention2 = BN(attention2, 'attention2_2')
    attention = tf.add(attention1, attention2, name=name + '_add')
    #attention = Add(name=name+'_add')([attention1,attention2])
    attention = tf.nn.sigmoid(attention, name="sigmoid")
    #attention = Activation('sigmoid')(attention)
    attention = tf.tile(attention, multiples=[1, 1, 1, C])
    #attention = Repeat(repeat_list=[1, 1, 1, C])(attention)
    return attention

def build_network(images, num_classes=default.num_classes, training=None):
    tf.logging.info("Loading CNN Model")

    if config.stn:
        tf.logging.info("Start to loading stn network")
        # locnet
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=None):
            with tf.variable_scope('Loc_Net'):
                n_fc = 6
                #B, H, W, C = images.shape
                # identity transform
                initial = np.array([[1., 0, 0], [0, 1., 0]])
                initial = initial.astype('float32').flatten()
                # Output Layer Transformation
                # localization network

                # 64 x 128
                avg_net = slim.avg_pool2d(images, kernel_size=2, stride=2, scope="pool1")
                # 32 x 64
                conv1_1_net = slim.conv2d(avg_net, 32, kernel_size=3, stride=4, scope='conv1_1')
                # 8 x 16
                conv1_2_net = slim.conv2d(images, 32, kernel_size=5, stride=8, scope='conv1_2')

                loc_concat_net = tf.concat([conv1_1_net, conv1_2_net], 3, name='concat')
                #loc_net = slim.repeat(images, 2, slim.conv2d, 32, kernel_size=3, stride=1, scope='loc_conv1')
                #loc_net = slim.max_pool2d(loc_net, kernel_size=2, stride=2, scope='loc_pool1')
                # 8 x 16
                loc_net = slim.conv2d(conv1_2_net, 128, kernel_size=3, stride=1, scope='conv3')
                loc_net = slim.batch_norm(loc_net, decay=_BATCH_DECAY, is_training=training, scope='bn1')
                loc_net = slim.conv2d(loc_net, 32, kernel_size=3, stride=1, scope='conv4')
                loc_net = slim.batch_norm(loc_net, decay=_BATCH_DECAY, is_training=training, scope='bn2')
                loc_net = slim.max_pool2d(loc_net, kernel_size=5, stride=4, scope='pool3')
                # 2 x 4
                loc_net = slim.conv2d(loc_net, 16, kernel_size=3, stride=1, scope='conv5')
                loc_net = tf.reduce_mean(input_tensor=loc_net, axis=[1, 2], keep_dims=False, name="se_pool1")
                loc_net = tf.reshape(loc_net, [loc_net.shape[0], -1])
                loc_B, loc_W = loc_net.shape
                W_fc1 = tf.Variable(tf.zeros([loc_W, n_fc]), name='W_fc1')
                b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
                loc_net = tf.matmul(loc_net, W_fc1) + b_fc1
                loc_output = spatial_transformer_network(images, loc_net)
                images = loc_output
                tf.logging.info("stn network loaded...")

        # 1 x 2

    if config.rgb:
        tf.logging.info("Start to loading Init rgb network")
        # rgbnet
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=None):
            with tf.variable_scope('RGB_Net'):
                # identity transform
                # 64 x 128
                avg_net = slim.avg_pool2d(images, kernel_size=2, stride=2, scope="pool1")
                # 32 x 64
                conv1_1_net = slim.conv2d(avg_net, 32, kernel_size=3, stride=4, scope='conv1_1')
                # 8 x 16
                conv1_2_net = slim.conv2d(images, 32, kernel_size=5, stride=8, scope='conv1_2')

                rgb_concat_net = tf.concat([conv1_1_net, conv1_2_net], 3, name='concat')
                #loc_net = slim.repeat(images, 2, slim.conv2d, 32, kernel_size=3, stride=1, scope='loc_conv1')
                #loc_net = slim.max_pool2d(loc_net, kernel_size=2, stride=2, scope='loc_pool1')
                # 8 x 16
                rgb_output = channel_wise_attention(rgb_concat_net, images, "RGB")
                images = rgb_output
                tf.logging.info("stn network loaded...")

        # 1 x 2

    # first apply the cnn feature extraction stage
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        biases_initializer=None):
        with tf.variable_scope('FEN'):
            tf.logging.info("Start to loading cnn feature extraction network")
            net = slim.repeat(images, 2, slim.conv2d, 64, kernel_size=3, stride=1, scope='conv1')

            net = slim.max_pool2d(net, kernel_size=2, stride=2, scope='pool1')
            # 32 x 64
            net = slim.repeat(net, 2, slim.conv2d, 128, kernel_size=3, stride=1, scope='conv2')
            C1 = net

            net = slim.max_pool2d(net, kernel_size=2, stride=2, scope='pool2')
            # 16 x 32
            net = slim.repeat(net, 3, slim.conv2d, 256, kernel_size=3, stride=1, scope='conv3')
            C2 = net

            net = slim.max_pool2d(net, kernel_size=2, stride=2, scope='pool3')
            # 8 x 16
            net = slim.repeat(net, 3, slim.conv2d, 512, kernel_size=3, stride=1, scope='conv4')
            C3 = net

            net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], scope='pool4')
            # 4 x 16
            net = slim.repeat(net, 3, slim.conv2d, 512, kernel_size=3, stride=1, scope='conv5')
            C4 = net
            net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], scope='pool5')
            # 1 x 16
            C5 = net

            C1 = slim.conv2d(C1, 64, kernel_size=3, scope='C1_conv')
            C1 = slim.batch_norm(C1, decay=_BATCH_DECAY, is_training=training, scope='C1_BN')
            C2 = slim.conv2d(C2, 64, kernel_size=3, scope='C2_conv')
            C2 = slim.batch_norm(C2, decay=_BATCH_DECAY, is_training=training, scope='C2_BN')

            if config.with_CPFE:
                C1_cfe = CFE(C1, 32, 'C3_cfe', training)
                C2_cfe = CFE(C2, 32, 'C4_cfe', training)
                C3_cfe = CFE(C3, 32, 'C5_cfe', training)
                C1_cfe = BilinearDownsampling(C1_cfe, upsampling=(4, 4), name="C5_cfe_up4")
                C2_cfe = BilinearUpsampling(C2_cfe, upsampling=(2, 2), name="C4_cfe_up2")
                C123 = tf.concat([C1_cfe, C2_cfe, C3_cfe], axis=-1, name='C123_aspp_concat')

            C123 = slim.conv2d(C123, 64, kernel_size=1, scope='C123_conv')
            C123 = slim.batch_norm(C123, decay=_BATCH_DECAY, is_training=training, scope='C123_BN')
            #C123 = BilinearUpsampling(C345, upsampling=(4, 4), name="C123_up4")

            if config.with_SA:
                C5 = BilinearUpsampling(C5, upsampling=(4,1), name="C2_up2")
                C45 = tf.concat([C4, C5], axis=-1, name='C12_concat')
                #C12 = tf.con`(name='C12_concat', axis=-1)([C1, C2])
                C45 = slim.conv2d(C45, 64, kernel_size=3, scope='C12_conv')
                #C12 = Conv2D(64, (3, 3), padding='same', name='C12_conv')(C12)
                C45 = slim.batch_norm(C45, decay=_BATCH_DECAY, is_training=training, scope='C12')
                #C12 = BN(C12, 'C12')
                #C45 = tf.multiply(SA, C45, name="C12_atten_multiply")
                #C12 = Multiply(name='C12_atten_mutiply')([SA, C12])
                C45 = BilinearUpsampling(C45, upsampling=(2, 1), name="C45_up3")
                SA = SpatialAttention(C45, training, name="spatial_attention")
                if config.with_CA:
                    C45 = ChannelWiseAttention(C45, name="C345_ChannelWiseAttention_withcpfe")
            C123 = tf.multiply(SA, C123, name="C123_atten_multiply")

            net = tf.concat([C123, C45], axis=-1, name="fuse_concat")
            net = slim.conv2d(net, 256, padding="VALID", kernel_size=3, stride=[2, 1], scope='conv6')
            # 2 x 32
            cnn_out = slim.conv2d(net, 512, padding="VALID", kernel_size=[4, 1], stride=1, scope='conv7')
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
