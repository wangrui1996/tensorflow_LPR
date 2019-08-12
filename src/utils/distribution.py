import tensorflow as tf
from tensorflow.contrib import slim

def channel_wise_attention(inputs, name):
    H, W, C = map(int, inputs.get_shape()[1:])
    body = tf.reduce_mean(input_tensor=inputs, axis=[1, 2], keep_dims=True, name=name + "global_avg_pool")
    body = slim.fully_connected(body, C / 4)
    body = slim.fully_connected(body, C , activation_fn = tf.nn.sigmoid)
    #attention = Dense(C, activation='sigmoid', activity_regularizer=l1_reg)(attention)
    body = tf.reshape(body, [1, 1, C], name=name+"_reshape")
    #attention = Reshape((1, 1, C), name=name + '_reshape')(attention)
    body = tf.tile(body, multiples=[1, H, W, 1], name=name + "_repeat")
    #attention = Repeat(repeat_list=[1, H, W, 1], name=name + '_repeat')(attention)
    body = tf.multiply(body, inputs, name=name + "_multiply")
    bias = tf.Variable(tf.truncated_normal([C], stddev=0.01), name=name + "bias")
    body = body + bias
    #attention = Multiply(name=name + '_multiply')([attention, inputs])
    return body