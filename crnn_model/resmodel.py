"""
Implement for the crnn model mentioned in "An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition"

https://arxiv.org/abs/1507.05717v1
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import slim
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
def _get_block_sizes(resnet_size):
    choices = {
        18:  [2, 2, 2, 2],
        34:  [3, 4, 6, 3],
        50:  [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }
    try:
        return choices[resnet_size]
    except KeyError:
        err = ('Could not find layers for selected Resnet size.\n'
               'Size received: {}; sizes allowed: {}.'.format(resnet_size, choices.keys()))
        raise ValueError(err)

################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.compat.v1.layers.batch_normalization(
      inputs=inputs, axis=3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)

def fixed_padding(inputs, kernel_size):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end], [0, 0]])
  return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size)

  return tf.compat.v1.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.compat.v1.variance_scaling_initializer())

def _building_block_v1(inputs, filters, training, projection_shortcut, strides):
    shortcut = inputs
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters,
                                  kernel_size=3, strides=strides)
    inputs = batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters,
                                  kernel_size=3, strides=1)
    inputs = batch_norm(inputs, training)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs


def _building_block_v2(inputs, filters, training, projection_shortcut, strides):
    shortcut = inputs
    inputs = batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters,
                                  kernel_size=3, strides=strides)

    inputs = batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters,
                                  kernel_size=3, strides=1)
    return inputs + shortcut

def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
                         strides):
    shortcut = inputs
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=1)
    inputs = batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides)
    inputs = batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=4 * filters, kernel_size=1, strides=1)
    inputs = batch_norm(inputs, training)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs

def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
                         strides):
  shortcut = inputs
  inputs = batch_norm(inputs, training)
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1)

  inputs = batch_norm(inputs, training)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides)

  inputs = batch_norm(inputs, training)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1)

  return inputs + shortcut

def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name):

    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return tf.compat.v1.layers.conv2d(
            inputs=inputs, filters=filters_out, kernel_size=strides, strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.compat.v1.variance_scaling_initializer())

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1)

    return tf.identity(inputs, name)

class CRNNCTCNetwork(object):
    def __init__(self, phase, hidden_num, layers_num, num_classes,
                 resnet_size, bottleneck, num_filters,
                 kernel_size, conv_stride, first_pool_size,
                 first_pool_stride,
                 block_sizes, block_strides,
                 resnet_version = DEFAULT_VERSION):

        self.phase_ = phase.lower()
        self.hidden_num_ = hidden_num
        self.layers_num_ = layers_num
        self.num_classes_ = num_classes

        self.resnet_size_ = resnet_size
        self.bottleneck = bottleneck
        self.num_filters_ = num_filters
        self.kernel_size_ = kernel_size
        self.conv_stride_ = conv_stride
        self.first_pool_size_ = first_pool_size
        self.first_pool_stride_ = first_pool_stride
        self.block_sizes_ = block_sizes
        self.block_strides_ = block_strides
        self.resnet_version_ = resnet_version
        self.pre_activation_ = resnet_version == 2

        # ----------------------------- resnet ---------------------
        if bottleneck:
            if self.resnet_version_ == 1:
                self.block_fn = _bottleneck_block_v1
            else:
                self.block_fn = _bottleneck_block_v2
        else:
            if self.resnet_version_ == 1:
                self.block_fn = _building_block_v1
            else:
                self.block_fn = _building_block_v2



    def _model_variable_scope(self):
        return tf.compat.v1.variable_scope('resnet_model')

    def __feature_sequence_extraction(self, inputs):
        training = True if self.phase_ == 'train' else False
        with self._model_variable_scope():
            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=self.num_filters_, kernel_size=self.kernel_size_,
                strides=self.conv_stride_)
            # / 2
            inputs = tf.identity(inputs, 'initial_conv')

            if self.resnet_version_ == 1:
                inputs = batch_norm(inputs, training)
                inputs = tf.nn.relu(inputs)

            if self.first_pool_size_:
                inputs = tf.compat.v1.layers.max_pooling2d(inputs=inputs, pool_size=self.first_pool_size_,
                                                           strides=self.first_pool_stride_, padding='SAME')
                # /2
                inputs = tf.identity(inputs, 'initial_max_pool')

            for i, num_blocks in enumerate(self.block_sizes_):
                num_filters = self.num_filters_ * (2 ** i)
                inputs = block_layer(
                    inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
                    block_fn=self.block_fn, blocks=num_blocks,
                    strides=self.block_strides_[i], training=training,
                    name='block_layer{}'.format(i + 1))

            # Only apply the BN and ReLU for model that does pre_activation in each
            # building/bottleneck block, eg resnet V2.
            if self.pre_activation_:
                inputs = batch_norm(inputs, training)
                inputs = tf.nn.relu(inputs)

            # The current top layer has shape
            # `batch_size x pool_size x pool_size x final_size`.
            # ResNet does an Average Pooling layer over pool_size,
            # but that is the same as doing a reduce_mean. We do a reduce_mean
            # here because it performs better than AveragePooling2D.
            axes = [1]
            inputs = tf.reduce_mean(input_tensor=inputs, axis=axes, keepdims=True)
            inputs = tf.identity(inputs, 'final_reduce_mean')
        return inputs
    
    def __map_to_sequence(self, input_tensor):
        shape = input_tensor.get_shape().as_list()
        assert shape[1] == 1  # H of the feature map must equal to 1
        return tf.squeeze(input_tensor, axis=1)

    def __sequence_label(self, input_tensor, input_sequence_length):
        with tf.variable_scope('LSTM_Layers'):
            # forward lstm cell
            fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.hidden_num_]*self.layers_num_]
            # Backward direction cells
            bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.hidden_num_]*self.layers_num_]
            stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                fw_cell_list, bw_cell_list, input_tensor, sequence_length=input_sequence_length, dtype=tf.float32)

            [batch_size, _, hidden_num] = input_tensor.get_shape().as_list()
            rnn_reshaped = tf.reshape(stack_lstm_layer, [-1, hidden_num])

            # Doing the affine projection
            w = tf.Variable(tf.truncated_normal([hidden_num, self.num_classes_], stddev=0.01), name="w")
            logits = tf.matmul(rnn_reshaped, w)
           
            logits = tf.reshape(logits, [batch_size, -1, self.num_classes_])
            raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')

            # Swap batch and batch axis
            rnn_out = tf.transpose(logits, (1, 0, 2), name='transpose_time_major')
        return rnn_out, raw_pred

    def build_network(self, images, sequence_length=None):
        # first apply the cnn feature extraction stage
        cnn_out = self.__feature_sequence_extraction(images)
        # second apply the map to sequence stage
        sequence = self.__map_to_sequence(input_tensor=cnn_out)
        # third apply the sequence label stage
        net_out, raw_pred = self.__sequence_label(input_tensor=sequence, input_sequence_length=sequence_length)
        return net_out
