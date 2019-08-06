from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import slim
from tools.config import config, default
_BATCH_DECAY = 0.999

def Conv_unit(**kwargs):
    #name = kwargs.get('name')
    #_weight = mx.symbol.Variable(name+'_weight')
    #_bias = mx.symbol.Variable(name+'_bias', lr_mult=2.0, wd_mult=0.0)
    #body = mx.sym.Convolution(weight = _weight, bias = _bias, **kwargs)
    body = slim.conv2d(**kwargs)
    return body


def Act(data, act_type, name):
    if act_type=='prelu':
        body = tf.nn.leaky_relu(data, alpha=0.01, name=name)
    else:
        body = tf.nn.relu(data, name=name)
    return body

def residual_unit_v1(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    act_type = kwargs.get('version_act', 'prelu')
    bn_kwargs = {
        "center":True,
        "scale":True,
        "epsilon": 2e-5,
        "renorm_decay": bn_mom
    }
    if bottle_neck:
        conv1 = Conv_unit(inputs=data, num_outputs=int(num_filter*0.25), kernel_size=(1,1), stride=stride, pad=(0,0),
                     normalizer_fn = slim.batch_norm, normalizer_params=bn_kwargs ,scope=name + '_conv1')
   #     bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
   #     act1 = Act(data=conv1, act_type=act_type, name=name + '_relu1')
        conv2 = Conv_unit(inputs=conv1, num_outputs=int(num_filter*0.25), kernel_size=(3,3), stride=(1,1), pad=(1,1),
                     normalizer_fn = slim.batch_norm, normalizer_params=bn_kwargs ,scope=name + '_conv2')
  #      bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
 #       act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
        bn3 = Conv_unit(inputs=conv2, num_outputs=num_filter, kernel_size=(1,1), stride=(1,1), pad=(0,0),
                     normalizer_fn = slim.batch_norm, normalizer_params=bn_kwargs ,scope=name + '_conv3')
  #      bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

        if use_se:
          #se begin
          body = tf.reduce_mean(input_tensor=bn3, axis=[1, 2], keep_dims=True, name=name+"_se_pool1")
          #body = mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv_unit(inputs=body, num_outputs=num_filter//16, kernel_size=(1,1), stride=(1,1),
                                    scope=name+"_se_conv1")
          #body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
          body = Conv_unit(inputs=body, num_outputs=num_filter, kernel_size=(1,1), stride=(1,1),
                      activation_fn=tf.nn.sigmoid,  scope=name+"_se_conv2")
          #body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn3 = bn3 * body
     #     bn3 = mx.symbol.broadcast_mul(bn3, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            shortcut = Conv_unit(inputs=data, num_outputs=num_filter, kernel_size=(1,1), stride=stride,
                                normalizer_fn = slim.batch_norm, normalizer_params=bn_kwargs, scope=name+'_conv1sc')
            #shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc')
     #   if memonger:
     #       shortcut._set_attr(mirror_stage='True')
        return Act(data=bn3 + shortcut, act_type=act_type, name=name + '_relu3')
    else:
        conv1 = Conv_unit(inputs=data, num_outputs=num_filter, kernel_size=(3,3), stride=stride,
                                      normalizer_fn = slim.batch_norm, normalizer_params=bn_kwargs,  scope=name + '_conv1')
        #bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        #act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
        bn2 = Conv_unit(inputs=conv1, num_outputs=num_filter, kernel_size=(3,3), stride=(1,1),
                                      normalizer_fn = slim.batch_norm, normalizer_params=bn_kwargs, scope=name + '_conv2')
        #bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        if use_se:
          #se begin
          body = tf.reduce_mean(input_tensor=bn2, axis=[1, 2], keep_dims=True, name=name + "_se_pool1")
          #body = mx.sym.Pooling(data=bn2, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv_unit(inputs=body, num_outputs=num_filter//16, kernel=(1,1), stride=(1,1),
                                    scope=name+"_se_conv1")
          #body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
          body = Conv_unit(inputs=body, num_outputs=num_filter, kernel=(1,1), stride=(1,1),
                                    scope=name+"_se_conv2")
          #body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn2 = bn2 * body
          #bn2 = mx.symbol.broadcast_mul(bn2, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            shortcut = Conv_unit(inputs=data, num_outputs=num_filter, kernel=(1,1), stride=stride,
                                             normalizer_fn = slim.batch_norm, normalizer_params=bn_kwargs, scope=name+'_conv1sc')
            #shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc')
        #if memonger:
        #    shortcut._set_attr(mirror_stage='True')
        return Act(data=bn2 + shortcut, act_type=act_type, name=name + '_relu3')



def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
  uv = kwargs.get('version_unit', 1)
  version_input = kwargs.get('version_input', 0)
  if uv==1:
    if version_input==0:
      return residual_unit_v1(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
    else:
      return 0
  elif uv==2:
    return 0
  else:
    return 0

def resnet(data, units, num_stages, filter_list, filter_kernel, filter_stride, bottle_neck, training):
    bn_mom = config.bn_mom
    kwargs = {'version_se' : config.net_se,
        'version_input': config.net_input,
        'version_output': config.net_output,
        'version_unit': config.net_unit,
        'version_act': config.net_act,
        'bn_mom': bn_mom,
        }
    bn_kwargs = {
        "center": True,
        "scale": True,
        "epsilon": 2e-5,
        "renorm_decay": bn_mom,
        "is_training": training
    }
    version_se = kwargs.get('version_se', 1)
    version_input = kwargs.get('version_input', 0)
    assert version_input>=0
    version_output = kwargs.get('version_output', 'E')
    version_unit = kwargs.get('version_unit', 3)
    act_type = kwargs.get('version_act', 'prelu')
    print("version_se: ", version_se, " version_input: ",version_input, " version_output: ",
          version_output, "version_unit: ", version_unit, " act_type: ", act_type)
    num_unit = len(units)
    assert(num_unit == num_stages)
    last_shape = data.shape
    print("Input Tensor Shape: ", data.shape)
    # 64 x 128
    if version_input==0:
      #data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
      data = data-127.5
      data = data*0.0078125
      body = Conv_unit(inputs=data, num_outputs=filter_list[0], kernel_size=(7, 7), stride=filter_stride[0],
                                normalizer_fn = slim.batch_norm, normalizer_params=bn_kwargs , scope="conv0")
      # 32 x 64
   #   body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
   #   body = Act(data=body, act_type=act_type, name='relu0')
      #body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    elif version_input==2:
      exit(0)
   #   data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
   #   body = Conv(data=data, num_filter=filter_list[0], kernel=(3,3), stride=(1,1), pad=(1,1),
   #                             no_bias=True, name="conv0", workspace=workspace)
   #   body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
   #   body = Act(data=body, act_type=act_type, name='relu0')
    else:
      #data = mx.sym.identity(data=data, name='id')
      data = data-127.5
      data = data*0.0078125
      body = Conv_unit(inputs=data, num_outputs=filter_list[0], kernel_size=(3,3), stride=(1,1),
                       normalizer_fn = slim.batch_norm, normalizer_params=bn_kwargs, scope="conv0")
      # 64 x 128
      #body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
      #body = Act(data=body, act_type=act_type, name='relu0')

    print(" stage pre: ", last_shape, " ======> ", body.shape)
    last_shape = body.shape
    for i in range(num_stages):
      #if version_input==0:
      #  body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
      #                       name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
      #else:
      #  body = residual_unit(body, filter_list[i+1], (2, 2), False,
      #    name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
      body = Conv_unit(inputs=body, num_outputs=filter_list[i+1], kernel_size=filter_kernel[i+1], stride=filter_stride[i+1],
                       normalizer_fn=slim.batch_norm, normalizer_params=bn_kwargs, scope="stage%d_unit%d" % (i + 1, 1))
      #body = residual_unit(body, filter_list[i+1], (2, 2), False,
      #  name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
      for j in range(units[i]-1):
        body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i+1, j+2),
          bottle_neck=bottle_neck, **kwargs)
      print(" stage {}: {} ======> {}".format(i, last_shape, body.shape))
      last_shape = body.shape
    if bottle_neck:
      body = Conv_unit(inputs=body, num_outputs=512, kernel_size=(1,1), stride=(1,1),
                                normalizer_fn = slim.batch_norm, normalizer_params=bn_kwargs, scope="convd")
      #body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bnd')
      #body = Act(data=body, act_type=act_type, name='relud')

    #fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)
    body_height = body.shape[1]
    if body_height != 1:
        body = Conv_unit(inputs=body, num_outputs=512, kernel_size=(body_height,1), stride=(1,1),
                                normalizer_fn = slim.batch_norm, normalizer_params=bn_kwargs, scope="convout")
    print(" stage output shape: {} ======> {}".format(last_shape, " ======> ", body.shape))
    return body

def build_network(images, num_classes=default.num_classes, training=None):
    num_layers = config.num_layers
    filter_kernel = [(2,2), (2,2), (2,1), (2,1), (2,1)]
    filter_stride = [(2,2), (2,2), (2,1), (2,1), (2,1)]
    if num_layers >= 500:
        filter_list = [64, 256, 512, 1024, 2048]
        bottle_neck = True
    else:
        filter_list = [64, 64, 128, 256, 512]
        bottle_neck = False
    num_stages = 4
    if num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 49:
        units = [3, 4, 14, 3]
    elif num_layers == 50:
        units = [3, 4, 14, 3]
    elif num_layers == 74:
        units = [3, 6, 24, 3]
    elif num_layers == 90:
        units = [3, 8, 30, 3]
    elif num_layers == 98:
        units = [3, 4, 38, 3]
    elif num_layers == 99:
        units = [3, 8, 35, 3]
    elif num_layers == 100:
        units = [3, 13, 30, 3]
    elif num_layers == 134:
        units = [3, 10, 50, 3]
    elif num_layers == 136:
        units = [3, 13, 48, 3]
    elif num_layers == 140:
        units = [3, 15, 48, 3]
    elif num_layers == 124:
        units = [3, 13, 40, 5]
    elif num_layers == 160:
        units = [3, 24, 49, 3]
    elif num_layers == 101:
        units = [3, 4, 23, 3]
    elif num_layers == 152:
        units = [3, 8, 36, 3]
    elif num_layers == 200:
        units = [3, 24, 36, 3]
    elif num_layers == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    cnn_out = resnet(
        data=images,
        units=units,
        num_stages=num_stages,
        filter_list=filter_list,
        filter_kernel=filter_kernel,
        filter_stride=filter_stride,
        bottle_neck=bottle_neck,
        training=training)
  #  with slim.arg_scope([slim.conv2d],
   #                     weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
   #                     weights_regularizer=slim.l2_regularizer(0.0005),
   #                     biases_initializer=None):
   #     cnn_out = slim.conv2d(net, 512, padding="VALID", kernel_size=[2, 1], stride=1, scope='conv7')
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
