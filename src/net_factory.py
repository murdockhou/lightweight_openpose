
import tensorflow as tf

def conv(inputs, out_channels, kernel_size=3, bn=True, dilation=1, stride=1, relu=True, bias=True, training=True, name=''):
    net = tf.layers.conv2d(inputs, filters=out_channels, kernel_size=kernel_size,
                           strides=stride, dilation_rate=dilation, padding='same',
                           use_bias=bias, name=name)
    if bn:
        net = tf.layers.batch_normalization(net, training=training)
    if relu:
        net = tf.nn.relu(net)
    return net

def conv_dw(inputs, out_channels, kernel_size=3, stride=1, dilation=1, training=True, name=''):
    net = tf.layers.separable_conv2d(inputs, filters=out_channels, kernel_size=kernel_size,
                                     strides=stride, dilation_rate=dilation, padding='same',
                                     use_bias=False, name=name)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    return net

def conv_dw_no_bn(inputs, out_channels, kernel_size=3, stride=1, dilation=1, training=True, name=''):

    net = tf.layers.separable_conv2d(inputs, filters=out_channels, kernel_size=kernel_size,
                                     strides=stride, dilation_rate=dilation, padding='same',
                                     use_bias=False, name=name)

    net = tf.nn.elu(net)

    return net

def RefinementStageBlock(inputs, out_channels, training=True):
    net_initial = conv(inputs, out_channels, kernel_size=1, bn=False)
    net_truck   = conv(net_initial, out_channels, training=training)
    net_truck   = conv(net_truck, out_channels, dilation=2, training=training)
    return tf.add(net_initial, net_truck)