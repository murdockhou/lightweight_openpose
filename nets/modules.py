#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: modules.py
@time: 2019/8/12 下午4:13
@desc:
'''

import tensorflow as tf
def conv(inputs, out_channels, kernel_size=3, bn=True, dilation=1, stride=1, relu=True, bias=True, training=True, name=''):

    net = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=stride, padding='same',
                                 dilation_rate=dilation, use_bias=bias, name=name+'_conv2d')(inputs)
    if bn:
        net = tf.keras.layers.BatchNormalization(name=name+'_bn')(net, training=training)
    if relu:
        net = tf.keras.layers.ReLU(name=name+'_relu')(net)

    return net

def conv_dw(inputs, out_channels, kernel_size=3, bn=True, dilation=1, stride=1, relu='relu',  training=True, name=''):
    if relu is not None and relu not in ['relu', 'elu']:
        print ('Error! Activation must be None or in [\'relu\', \'elu\'], but got {} actually.'.format(relu))


    net = tf.keras.layers.SeparableConv2D(filters=out_channels, kernel_size=kernel_size, strides=stride, padding='same',
                                          dilation_rate=dilation, use_bias=False, name=name+'_separa_conv2d')(inputs)
    if bn:
        net = tf.keras.layers.BatchNormalization(name=name+'_bn')(net, training=training)
    if relu == 'relu':
        net = tf.keras.layers.ReLU(name=name+'_relu')(net)
    elif relu == 'elu':
        net = tf.keras.layers.ELU(name=name+'_elu')(net)

    return net

def refinement_stage_block(inputs, out_channels, training=True, name=''):
    net_initial = conv(inputs, out_channels, kernel_size=1, training=training, name=name+'_initial')
    net_truck = conv(net_initial, out_channels, training=training, name=name+'_truck1')
    net_truck = conv(net_truck, out_channels, dilation=2, name=name+'_truck2')
    out = tf.keras.layers.Add(name=name+'_add')([net_initial, net_truck])
    return out