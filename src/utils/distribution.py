import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim

def channel_wise_attention(body, input, name):
    H, W, C = map(int, input.get_shape()[1:])
    if C <= 3:
        C_N = 64
    else:
        C_N = C
    body = tf.reduce_mean(input_tensor=body, axis=[1, 2], keep_dims=True, name=name + "global_avg_pool")
    body = slim.fully_connected(body, C_N)
    body = slim.fully_connected(body, C , activation_fn = tf.nn.sigmoid)
    #attention = Dense(C, activation='sigmoid', activity_regularizer=l1_reg)(attention)
    body = tf.reshape(body, [-1, 1, 1, C], name=name+"_reshape")
    #attention = Reshape((1, 1, C), name=name + '_reshape')(attention)
    body = tf.tile(body, multiples=[1, H, W, 1], name=name + "_repeat")
    #attention = Repeat(repeat_list=[1, H, W, 1], name=name + '_repeat')(attention)
    W_fc1 = tf.Variable(tf.zeros([H, W, C]), name=name + 'W_fc1')
    b_fc1 = tf.Variable(tf.ones([H, W, C]), name=name + 'b_fc1')
    body = tf.multiply(body * W_fc1 + b_fc1, input, name=name + "_multiply")
    bias_out = tf.Variable(tf.ones([H, W, C]), name=name + 'bias_out')
    body = body + bias_out
    #attention = Multiply(name=name + '_multiply')([attention, inputs])
    return body