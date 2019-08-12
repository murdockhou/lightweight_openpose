#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: lightweight_openpose.py
@time: 2019/8/12 下午4:09
@desc:
'''
import tensorflow as tf
from nets.modules import conv, conv_dw, refinement_stage_block

def lightweight_openpose(inputs, num_joints, num_pafs, is_training=True):
    name = 'BackBone'
    net = conv(inputs, 32, stride=2, bias=False, training=is_training, name=name+'_conv1')
    net = conv_dw(net, 64, bn=True, relu='relu', training=is_training, name=name+'_convdw1')
    net = conv_dw(net, 128, stride=2, bn=True, relu='relu', training=is_training, name=name + '_convdw2')
    net = conv_dw(net, 128, bn=True, relu='relu', training=is_training, name=name + '_convdw3')
    net = conv_dw(net, 256, stride=2, bn=True, relu='relu', training=is_training, name=name + '_convdw4')
    net = conv_dw(net, 256, bn=True, relu='relu', training=is_training, name=name + '_convdw5')
    net = conv_dw(net, 512, bn=True, relu='relu', training=is_training, name=name + '_convdw6')
    net = conv_dw(net, 512, bn=True, relu='relu', dilation=2, training=is_training, name=name + '_convdw7')
    net = conv_dw(net, 512, bn=True, relu='relu', training=is_training, name=name + '_convdw8')
    net = conv_dw(net, 512, bn=True, relu='relu', training=is_training, name=name + '_convdw9')
    net = conv_dw(net, 512, bn=True, relu='relu', training=is_training, name=name + '_convdw10')
    net = conv_dw(net, 512, bn=True, relu='relu', training=is_training, name=name + '_convdw11')

    name = 'Cpm'
    net_align = conv(net, 128, kernel_size=1, bn=False, training=is_training, name=name + '_align')
    net_truck = conv_dw(net_align, 128, bn=False, relu='elu', training=is_training, name=name + '_truck1')
    net_truck = conv_dw(net_truck, 128, bn=False, relu='elu', training=is_training, name=name + '_truck2')
    net_truck = conv_dw(net_truck, 128, bn=False, relu='elu', training=is_training, name=name + '_truck3')
    net_add = tf.keras.layers.Add(name=name+'_add')([net_align, net_truck])
    net_cpm = conv(net_add, 128, bn=False, training=is_training, name=name+'_cpm')

    name = 'InitialStage'
    trunk_features = conv(net_cpm, 128, bn=False, training=is_training, name=name+'_conv1')
    trunk_features = conv(trunk_features, 128, bn=False, training=is_training, name=name+'_conv2')
    trunk_features = conv(trunk_features, 128, bn=False, training=is_training, name=name+'_conv3')
    heatmaps1 = conv(trunk_features, 512, kernel_size=1, bn=False, training=is_training, name=name+'_conv4')
    heatmaps1 = conv(heatmaps1, num_joints, kernel_size=1, bn=False, relu=False, training=is_training, name=name+'_heat')
    pafs1 = conv(trunk_features, 512, kernel_size=1, bn=False, training=is_training, name=name+'_conv5')
    pafs1 = conv(pafs1, num_pafs, kernel_size=1, bn=False, relu=False, training=is_training, name=name+'_paf')
    stage_out = tf.keras.layers.Concatenate(name=name+'_concat')([net_cpm, heatmaps1, pafs1])

    name = 'RefinementStage'
    net = refinement_stage_block(stage_out, 128, training=is_training, name=name+'_block1')
    net = refinement_stage_block(net, 128, training=is_training, name=name + '_block2')
    net = refinement_stage_block(net, 128, training=is_training, name=name + '_block3')
    net = refinement_stage_block(net, 128, training=is_training, name=name + '_block4')
    net = refinement_stage_block(net, 128, training=is_training, name=name + '_block5')
    heatmaps2 = conv(net, 512, kernel_size=1, bn=False, training=is_training, name=name + '_conv1')
    heatmaps2 = conv(heatmaps2, num_joints, kernel_size=1, bn=False, relu=False, training=is_training, name=name + '_heat')
    pafs2 = conv(net, 512, kernel_size=1, bn=False, training=is_training, name=name + '_conv2')
    pafs2 = conv(pafs2, num_pafs, kernel_size=1, bn=False, relu=False, training=is_training, name=name + '_paf')

    return [[heatmaps1, pafs1], [heatmaps2, pafs2]]

if __name__ == '__main__':
    inputs = tf.keras.Input(shape=(256,256,3), dtype=tf.float32)
    outputs = lightweight_openpose(inputs, 14, 26)
    model = tf.keras.Model(inputs, outputs)
    model.summary()