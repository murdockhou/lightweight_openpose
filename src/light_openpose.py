# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: light_openpose.py
@time: 18-12-4 11:34
'''
import tensorflow as tf


def conv_bn_relu(inputs, filters, kernel_size,  stride=1, dilation_rate=1, kernel_initializer=tf.glorot_normal_initializer(),
                padding='same', use_bias = True, bn=True, relu='relu', is_training=True, scope=None):

    net = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=stride, padding=padding,
                           dilation_rate=dilation_rate, use_bias=use_bias,
                           kernel_initializer=kernel_initializer, name=scope)
    if bn:
        net = tf.layers.batch_normalization(inputs=net, training=is_training)
    if relu == 'relu':
        net = tf.nn.relu(net)
    elif relu == 'elu':
        net = tf.nn.elu(net)
    else:
        net = net

    return net

def dw_bn_relu(inputs, filters, kernel_size, stride=1, dilation_rate=1, kernel_initializer=tf.glorot_normal_initializer(),
               padding='same', use_bias = True, bn=True, relu='relu', is_training=True, scope=None, format='depth_wise'):

    if format == 'separable':
        net = tf.layers.separable_conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=stride,
                                         padding=padding, dilation_rate=dilation_rate, use_bias=use_bias,
                                         depthwise_initializer=kernel_initializer, pointwise_initializer=kernel_initializer, name=scope)
        if bn:
            net = tf.layers.batch_normalization(inputs=net, training=is_training)
        if relu == 'relu':
            net = tf.nn.relu(net)
        elif relu == 'elu':
            net = tf.nn.elu(net)
        else:
            net = net
    elif format == 'depth_wise':
        # depthwise_filter = tf.Variable([kernel_size, kernel_size, inputs.get_shape()[-1], 1])
        with tf.variable_scope(scope):
            depthwise_filter = tf.get_variable(scope+'_3x3_weight', shape=[kernel_size, kernel_size, inputs.get_shape()[-1], 1],
                                               dtype=tf.float32, trainable=True, initializer=kernel_initializer)
        net = tf.nn.depthwise_conv2d(inputs,
                                     filter = depthwise_filter,
                                     strides = [1,stride,stride,1],
                                     padding='SAME',
                                     rate=[dilation_rate, dilation_rate],
                                     name=scope+'_3x3')

        if bn:
            net = tf.layers.batch_normalization(inputs=net, training=is_training)
        if relu == 'relu':
            net = tf.nn.relu(net)
        elif relu == 'elu':
            net = tf.nn.elu(net)
        else:
            net = net

        net = tf.layers.conv2d(net, filters=filters, kernel_size=1, use_bias=use_bias, kernel_initializer=kernel_initializer, name=scope+'_1x1')
        if bn:
            net = tf.layers.batch_normalization(inputs=net, training=is_training)
        if relu == 'relu':
            net = tf.nn.relu(net)
        elif relu == 'elu':
            net = tf.nn.elu(net)
        else:
            net = net
    else:
        net = inputs
        assert (format == 'separable' or format == 'depth_wise')

    return net

def create_light_weight_openpose(inputs, heatmap_channels, paf_channels, is_training=True):

    with tf.variable_scope('light_weight_open_pose', reuse=tf.AUTO_REUSE):
        conv1 = conv_bn_relu(inputs=inputs, filters=32, kernel_size=3, stride=2, use_bias=False, is_training=is_training, scope='conv1')
        conv2_1 = dw_bn_relu(inputs=conv1, filters=64, kernel_size=3, stride=1, use_bias=False, is_training=is_training, scope='conv2_1')
        conv2_2 = dw_bn_relu(inputs=conv2_1, filters=128, kernel_size=3, stride=2, use_bias=False, is_training=is_training, scope='conv2_2')
        conv3_1 = dw_bn_relu(inputs=conv2_2, filters=128, kernel_size=3, stride=1, use_bias=False, is_training=is_training, scope='conv3_1')
        conv3_2 = dw_bn_relu(inputs=conv3_1, filters=256, kernel_size=3, stride=2, use_bias=False, is_training=is_training, scope='conv3_2')
        conv4_1 = dw_bn_relu(inputs=conv3_2, filters=256, kernel_size=3, stride=1, use_bias=False, is_training=is_training, scope='conv4_1')
        conv4_2 = dw_bn_relu(inputs=conv4_1, filters=512, kernel_size=3, stride=1, use_bias=False, is_training=is_training,
                             kernel_initializer=tf.variance_scaling_initializer(scale=2.), scope='conv4_2')
        conv5_1 = dw_bn_relu(inputs=conv4_2, filters=512, kernel_size=3, stride=1, use_bias=False, dilation_rate=2, is_training=is_training,
                             kernel_initializer=tf.variance_scaling_initializer(scale=2.), scope='conv5_1')
        conv5_2 = dw_bn_relu(inputs=conv5_1, filters=512, kernel_size=3, stride=1, use_bias=False, is_training=is_training,
                             kernel_initializer=tf.variance_scaling_initializer(scale=2.), scope='conv5_2')
        conv5_3 = dw_bn_relu(inputs=conv5_2, filters=512, kernel_size=3, stride=1, use_bias=False, is_training=is_training,
                             kernel_initializer=tf.variance_scaling_initializer(scale=2.), scope='conv5_3')
        conv5_4 = dw_bn_relu(inputs=conv5_3, filters=512, kernel_size=3, stride=1, use_bias=False, is_training=is_training,
                             kernel_initializer=tf.variance_scaling_initializer(scale=2.), scope='conv5_4')
        conv5_5 = dw_bn_relu(inputs=conv5_4, filters=512, kernel_size=3, stride=1, use_bias=False, is_training=is_training,
                             kernel_initializer=tf.variance_scaling_initializer(scale=2.), scope='conv5_5')

        conv4_3_cpm = conv_bn_relu(inputs=conv5_5, filters=128, kernel_size=1, stride=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                   relu='elu', use_bias=True, bn=False, scope='conv4_3_cpm')
        conv4_3_cpm_red = conv_bn_relu(inputs=conv4_3_cpm, filters=128, kernel_size=1, stride=1, bn=False,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01), relu='elu', scope='conv4_3_cpm_red')
        conv4_3_cpm_dw1 = dw_bn_relu(inputs=conv4_3_cpm_red, filters=128, kernel_size=3, relu='elu', bn=False, scope='conv4_3_cpm_dw1')
        conv4_3_cpm_dw2 = dw_bn_relu(inputs=conv4_3_cpm_dw1, filters=128, kernel_size=3, relu='elu', bn=False, scope='conv4_3_cpm_dw2')
        conv4_3_cpm_dw3 = dw_bn_relu(inputs=conv4_3_cpm_dw2, filters=128, kernel_size=3, relu='elu', bn=False, scope='conv4_3_cpm_dw3')
        conv4_3_res = tf.add(conv4_3_cpm, conv4_3_cpm_dw3, name='conv4_3_cpm_res')

        with tf.variable_scope('initial_stage'):
            conv4_4_cpm = conv_bn_relu(inputs=conv4_3_res, filters=128, kernel_size=3,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                       bn=False, relu='relu', scope='conv4_4_cpm')
            conv5_1_cpm = conv_bn_relu(inputs=conv4_4_cpm, filters=128, kernel_size=3,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                       bn=False, relu='relu', scope='conv5_1_cpm_l1')
            conv5_2_cpm = conv_bn_relu(inputs=conv5_1_cpm, filters=128, kernel_size=3,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                       bn=False, relu='relu', scope='conv5_2_cpm_l1')
            conv5_3_cpm = conv_bn_relu(inputs=conv5_2_cpm, filters=128, kernel_size=3,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                       bn=False, relu='relu', scope='conv5_3_cpm_l1')
            conv5_4_cpm_l1 = conv_bn_relu(inputs=conv5_3_cpm, filters=512, kernel_size=1,
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                          bn=False, relu='relu', scope='conv5_4_cpm_l1')
            conv5_4_cpm_l2 = conv_bn_relu(inputs=conv5_3_cpm, filters=512, kernel_size=1,
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                          bn=False, relu='relu', scope='conv5_4_cpm_l2')
            cpm1 = tf.layers.conv2d(inputs=conv5_4_cpm_l1, filters=heatmap_channels, kernel_size=1,padding='same',
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01), activation=None, name='cpm1')
            paf1 = tf.layers.conv2d(inputs=conv5_4_cpm_l2, filters=paf_channels, kernel_size=1, padding='same',
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01), activation=None,
                                    name='paf1')

        concat_stage2 = tf.concat([conv4_4_cpm, cpm1, paf1], axis=-1)

        with tf.variable_scope('second_stage'):
            stage2_conv = concat_stage2
            for i in range(5):
                conv1x1 = conv_bn_relu(inputs=stage2_conv, filters=128, kernel_size=1, bn=False, relu='relu', scope='_' + str(i) + '_conv1x1')
                conv3x3 = conv_bn_relu(inputs=conv1x1, filters=128, kernel_size=3, bn=True, is_training=is_training, relu='relu', scope='_' + str(i) +'_conv3x3'.format(i))
                conv3x3_dila2 = conv_bn_relu(inputs=conv3x3, filters=128, kernel_size=3, dilation_rate=2, bn=True, is_training=is_training, relu='relu',
                                             scope='_' + str(i) +'_conv3x3_dilation2'.format(i))
                stage2_conv = tf.add(conv1x1, conv3x3_dila2)

            cpm2 = tf.layers.conv2d(inputs=stage2_conv, filters=128, kernel_size=1,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                    activation=tf.nn.relu)
            cpm2 = tf.layers.conv2d(inputs=cpm2, filters=heatmap_channels, kernel_size=1,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                    name='cpm2')

            paf2 = tf.layers.conv2d(inputs=stage2_conv, filters=128, kernel_size=1,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                    activation=tf.nn.relu)
            paf2 = tf.layers.conv2d(inputs=paf2, filters=paf_channels, kernel_size=1,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                    name='paf2')



    return cpm1, paf1, cpm2, paf2



#
# import numpy as np
# with tf.device('/cpu:0'):
#     a = np.random.rand(1,368,368,3)
#     a = tf.convert_to_tensor(a, dtype=tf.float32)
#     cpm1, paf1, cpm,paf = create_light_weight_openpose(a, 4, 6)
#     var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#     global_list = tf.global_variables()
#     bn_vars = [g for g in global_list if 'moving_mean' in g.name]
#     bn_vars += [g for g in global_list if 'moving_variance' in g.name]
#     for var in bn_vars:
#         print (var)
#     print (len(bn_vars))
#     print ('#########################')
#     for var in var_list:
#         print(var)
#     print (len(var_list))
    # summary_writer = tf.summary.FileWriter('light_openpose')
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     e,f, c,d = sess.run([cpm1, paf1, cpm, paf])
    #     summary_writer.add_graph(sess.graph)
    #     print (e.shape, f.shape, c.shape, d.shape)

