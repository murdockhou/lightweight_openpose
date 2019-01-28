# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: lightweight_openpose.py
@time: 19-1-21 16:59
'''
import tensorflow as tf

def conv_bn_relu(inputs, filters, kernel_size,  stride=1, dilation_rate=1, kernel_initializer=tf.glorot_normal_initializer(),
                padding='same', use_bias = True, bn=True, relu='relu', is_training=True, scope=None):

    net = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=stride, padding=padding,
                           dilation_rate=dilation_rate, use_bias=use_bias,
                           kernel_initializer=kernel_initializer, name=scope)
    if bn:
        net = tf.layers.batch_normalization(inputs=net, training=is_training, name=scope + '_bn')
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
            net = tf.layers.batch_normalization(inputs=net, training=is_training, name=scope + '_bn')
        if relu == 'relu':
            net = tf.nn.relu(net)
        elif relu == 'elu':
            net = tf.nn.elu(net)
        else:
            net = net
    elif format == 'depth_wise':
        # depthwise_filter = tf.Variable([kernel_size, kernel_size, inputs.get_shape()[-1], 1])
        with tf.variable_scope(scope):
            depthwise_filter = tf.get_variable('_3x3_weight', shape=[kernel_size, kernel_size, inputs.get_shape()[-1], 1],
                                               dtype=tf.float32, trainable=True, initializer=kernel_initializer)
        net = tf.nn.depthwise_conv2d(inputs,
                                     filter = depthwise_filter,
                                     strides = [1,stride,stride,1],
                                     padding='SAME',
                                     rate=[dilation_rate, dilation_rate],
                                     name=scope+'_3x3')

        if bn:
            net = tf.layers.batch_normalization(inputs=net, training=is_training, name=scope + '_bn_3x3')
        if relu == 'relu':
            net = tf.nn.relu(net)
        elif relu == 'elu':
            net = tf.nn.elu(net)
        else:
            net = net

        net = tf.layers.conv2d(net, filters=filters, kernel_size=1, use_bias=use_bias, kernel_initializer=kernel_initializer, name=scope+'_1x1')
        if bn:
            net = tf.layers.batch_normalization(inputs=net, training=is_training, name=scope + '_bn_1x1')
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

def light_openpose(inputs, joints = 14, paf = 26, dilation_rate = 1, is_training=True):

    with tf.variable_scope('light_openpose', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('backbone'):
            conv1   = conv_bn_relu(inputs=inputs, filters=32, kernel_size=3, stride=2, use_bias=False, is_training=is_training, scope='conv1')
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
        with tf.variable_scope('initial_stage', reuse=tf.AUTO_REUSE):
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

            # add initial output
            cpm_outputs = []
            paf_outputs = []
            conv5_4_cpm_l1_upsample = tf.layers.conv2d_transpose(inputs=conv5_4_cpm_l1, filters=128, kernel_size=3, strides=2,
                                                          padding='same',
                                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                          name='conv5_4_cpm_l1_upsample')
            conv5_4_cpm_l1_upsample_output = tf.layers.conv2d(inputs=conv5_4_cpm_l1_upsample, filters=128, kernel_size=1,
                                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                  name='conv5_4_cpm_l1_upsample_cpm1')
            conv5_4_cpm_output = tf.layers.conv2d(inputs=conv5_4_cpm_l1_upsample_output, filters=joints, kernel_size=1,
                                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                  name='conv5_4_cpm_l1_upsample_output_cpm')
            cpm_outputs.append(conv5_4_cpm_output)

            # paf
            conv5_4_cpm_l2_upsample = tf.layers.conv2d_transpose(inputs=conv5_4_cpm_l2, filters=128, kernel_size=3, strides=2,
                                                                 padding='same',
                                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                                 name='conv5_4_cpm_l2_upsample')
            conv5_4_cpm_l2_upsample_output = tf.layers.conv2d(inputs=conv5_4_cpm_l2_upsample, filters=128, kernel_size=1,
                                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                  name='conv5_4_cpm_l2_upsample_paf1')
            conv5_4_paf_output = tf.layers.conv2d(inputs=conv5_4_cpm_l2_upsample_output, filters=paf, kernel_size=1,
                                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                  name='conv5_4_cpm_l2_upsample_output_paf')
            paf_outputs.append(conv5_4_paf_output)

            # downsampling initial output
            conv5_4_cpm_output_down = tf.layers.max_pooling2d(conv5_4_cpm_output, 2, 2, 'same', name='conv5_4_cpmout_down')
            conv5_4_paf_output_down = tf.layers.max_pooling2d(conv5_4_paf_output, 2, 2, 'same', name='conv5_4_pafout_down')
            stage_inp = tf.concat([conv4_4_cpm, conv5_4_cpm_output_down, conv5_4_paf_output_down], axis=-1, name='stage_input')
            previous  = stage_inp

        with tf.variable_scope('stage_1'):
            for i in range(5):
                conv1x1 = conv_bn_relu(inputs=stage_inp, filters=128, kernel_size=1, bn=False, relu='relu',
                                       scope='_' + str(i) + '_conv1x1')
                conv3x3 = conv_bn_relu(inputs=conv1x1, filters=128, kernel_size=3, bn=True, is_training=is_training,
                                       relu='relu', scope='_' + str(i) + '_conv3x3'.format(i))
                conv3x3_dila2 = conv_bn_relu(inputs=conv3x3, filters=128, kernel_size=3, dilation_rate=2, bn=True,
                                             is_training=is_training, relu='relu',
                                             scope='_' + str(i) + '_conv3x3_dilation2'.format(i))
                stage_inp = tf.add(conv1x1, conv3x3_dila2)

            # upsampling
            transform_conv = tf.layers.conv2d_transpose(inputs=stage_inp, filters=128, kernel_size=3, strides=2, padding='same',
                                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),name='transform_conv')
            # cpm
            conv5_4_cpm_output_upchannels = tf.layers.conv2d(inputs=conv5_4_cpm_output, filters=128, kernel_size=1,
                                                             kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                             name='conv5_4_cpm_out_up_channels')
            conv5_4_add_transform = tf.add(conv5_4_cpm_output_upchannels, transform_conv, name='conv5_4_cpm_add_transform')
            stagecpm1x1_1 = tf.layers.conv2d(inputs=conv5_4_add_transform, filters=128, kernel_size=1,
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        activation=tf.nn.relu, name='cpm1x1_1')
            stagecpm1x1_2 = tf.layers.conv2d(inputs=stagecpm1x1_1, filters=128, kernel_size=1,
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        activation=tf.nn.relu, name='cpm1x1_2')
            stage1_out_cpm  = tf.layers.conv2d(inputs=stagecpm1x1_2, filters=joints, kernel_size=1,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                   name='cpm')
            cpm_outputs.append(stage1_out_cpm)
            # paf
            conv4_4_paf_output_upchannels = tf.layers.conv2d(inputs=conv5_4_paf_output, filters=128, kernel_size=1,
                                                             kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                             name='conv4_4_paf_out_up_channels')
            conv4_4_add_transform = tf.add(conv4_4_paf_output_upchannels, transform_conv, name='conv4_4_paf_add_transform')
            stagepaf1x1_1 = tf.layers.conv2d(inputs=conv4_4_add_transform, filters=128, kernel_size=1,
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                             activation=tf.nn.relu, name='paf1x1_1')
            stagepaf1x1_2 = tf.layers.conv2d(inputs=stagepaf1x1_1, filters=128, kernel_size=1,
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                             activation=tf.nn.relu, name='paf1x1_2')
            stage1_out_paf = tf.layers.conv2d(inputs=stagepaf1x1_2, filters=paf, kernel_size=1,
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                             name='paf')
            paf_outputs.append(stage1_out_paf)


            cpm_stage_out = tf.layers.max_pooling2d(inputs=stage1_out_cpm, pool_size=2, strides=2, padding='same', name='stage_out_cpm_down')
            paf_stage_out = tf.layers.max_pooling2d(inputs=stage1_out_paf, pool_size=2, strides=2, padding='same', name='stage_out_paf_down')
            stage_inp = tf.concat([cpm_stage_out, paf_stage_out, previous], axis=-1, name='stage_concate')
        return cpm_outputs, paf_outputs, stage1_out_cpm, stage1_out_paf
        # with tf.variable_scope('stage_2'):
        #     for i in range(5):
        #         conv1x1 = conv_bn_relu(inputs=stage_inp, filters=128, kernel_size=1, bn=False, relu='relu', dilation_rate=dilation_rate,
        #                                scope='_' + str(i) + '_conv1x1')
        #         conv3x3 = conv_bn_relu(inputs=conv1x1, filters=128, kernel_size=3, bn=True, is_training=is_training, dilation_rate=dilation_rate,
        #                                relu='relu', scope='_' + str(i) + '_conv3x3'.format(i))
        #         conv3x3_dila2 = conv_bn_relu(inputs=conv3x3, filters=128, kernel_size=3, dilation_rate=2*dilation_rate, bn=True,
        #                                      is_training=is_training, relu='relu',
        #                                      scope='_' + str(i) + '_conv3x3_dilation2'.format(i))
        #         stage_inp = tf.add(conv1x1, conv3x3_dila2)
        #
        #     # upsampling
        #     transform_conv = tf.layers.conv2d_transpose(inputs=stage_inp, filters=128, kernel_size=3, strides=2,
        #                                                 padding='same',
        #                                                 kernel_initializer=tf.random_normal_initializer(
        #                                                     stddev=0.01), name='transform_conv')
        #     # cpm
        #     stage1_cpm_output_upchannels = tf.layers.conv2d(inputs=stage1_out_cpm, filters=128, kernel_size=1,
        #                                                      kernel_initializer=tf.random_normal_initializer(
        #                                                          stddev=0.01),
        #                                                      name='stage1_cpm_out_up_channels')
        #     stage1_add_transform = tf.add(stage1_cpm_output_upchannels, transform_conv,
        #                                    name='stage1_cpm_add_transform')
        #     stagecpm1x1_1 = tf.layers.conv2d(inputs=stage1_add_transform, filters=128, kernel_size=1,
        #                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        #                                      activation=tf.nn.relu, name='cpm1x1_1')
        #     stagecpm1x1_2 = tf.layers.conv2d(inputs=stagecpm1x1_1, filters=128, kernel_size=1,
        #                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        #                                      activation=tf.nn.relu, name='cpm1x1_2')
        #     stage2_out_cpm = tf.layers.conv2d(inputs=stagecpm1x1_2, filters=joints, kernel_size=1,
        #                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        #                                      name='cpm')
        #     cpm_outputs.append(stage2_out_cpm)
        #     # paf
        #     stage1_paf_output_upchannels = tf.layers.conv2d(inputs=stage1_out_paf, filters=128, kernel_size=1,
        #                                                      kernel_initializer=tf.random_normal_initializer(
        #                                                          stddev=0.01),
        #                                                      name='stage1_paf_out_up_channels')
        #     stage1_add_transform = tf.add(stage1_paf_output_upchannels, transform_conv,
        #                                    name='stage1_paf_add_transform')
        #     stagepaf1x1_1 = tf.layers.conv2d(inputs=stage1_add_transform, filters=128, kernel_size=1,
        #                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        #                                      activation=tf.nn.relu, name='paf1x1_1')
        #     stagepaf1x1_2 = tf.layers.conv2d(inputs=stagepaf1x1_1, filters=128, kernel_size=1,
        #                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        #                                      activation=tf.nn.relu, name='paf1x1_2')
        #     stage2_out_paf = tf.layers.conv2d(inputs=stagepaf1x1_2, filters=paf, kernel_size=1,
        #                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        #                                      name='paf')
        #     paf_outputs.append(stage2_out_paf)

        # return cpm_outputs, paf_outputs, stage2_out_cpm, stage2_out_paf

if __name__ == '__main__':
    import numpy as np
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    a = np.random.rand(1,368,368,3)
    a = tf.convert_to_tensor(a, dtype=tf.float32)
    _1, _2, _3, _4 = light_openpose(a, is_training=True)
    summary_writer = tf.summary.FileWriter('head_neck')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cpms, pafs, cpm, paf = sess.run([_1, _2, _3, _4])
        for c in cpms:
            print ('cpm: ', c.shape)
        for p in pafs:
            print ('paf: ', p.shape)
        print (cpm.shape)
        print (paf.shape)
        summary_writer.add_graph(sess.graph)
