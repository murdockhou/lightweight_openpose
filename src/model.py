import sys
sys.path.append('../')

from src.net_factory import conv, conv_dw, conv_dw_no_bn, RefinementStageBlock

import tensorflow as tf

def lightweight_openpose(inputs, num_joints, num_pafs, is_training=True):
    with tf.variable_scope('BackBone', reuse=tf.AUTO_REUSE):
        net = conv(inputs, 32, stride=2, bias=False, training=is_training)
        net = conv_dw(net, 64, training=is_training)
        net = conv_dw(net, 128, stride=2, training=is_training)
        net = conv_dw(net, 128, training=is_training)
        net = conv_dw(net, 256, stride=2, training=is_training)
        net = conv_dw(net, 256, training=is_training)
        net = conv_dw(net, 512, training=is_training)
        net = conv_dw(net, 512, dilation=2, training=is_training)
        net = conv_dw(net, 512, training=is_training)
        net = conv_dw(net, 512, training=is_training)
        net = conv_dw(net, 512, training=is_training)
        net = conv_dw(net, 512, training=is_training)
    with tf.variable_scope('Cpm', reuse=tf.AUTO_REUSE):
        net_align = conv(net, 128, kernel_size=1, bn=False)
        net_truck = conv_dw_no_bn(net_align, 128)
        net_truck = conv_dw_no_bn(net_truck, 128)
        net_truck = conv_dw_no_bn(net_truck, 128)
        net_cpm   = conv(tf.add(net_align, net_truck), 128, bn=False)
    with tf.variable_scope('InitialStage', reuse=tf.AUTO_REUSE):
        trunk_features = conv(net_cpm, 128, bn=False)
        trunk_features = conv(trunk_features, 128, bn=False)
        trunk_features = conv(trunk_features, 128, bn=False)
        heatmaps1 = conv(trunk_features, 512, kernel_size=1, bn=False)
        heatmaps1 = conv(heatmaps1, num_joints, kernel_size=1, bn=False, relu=False)
        pafs1     = conv(trunk_features, 512, kernel_size=1, bn=False)
        pafs1     = conv(pafs1, num_pafs, kernel_size=1, bn=False, relu=False)
        outs1    = tf.concat([heatmaps1, pafs1], axis=-1)
    with tf.variable_scope('RefinementStage', reuse=tf.AUTO_REUSE):
        net = RefinementStageBlock(outs1, 128, training=is_training)
        net = RefinementStageBlock(net, 128, training=is_training)
        net = RefinementStageBlock(net, 128, training=is_training)
        net = RefinementStageBlock(net, 128, training=is_training)
        net = RefinementStageBlock(net, 128, training=is_training)
        heatmaps2 = conv(net, 128, kernel_size=1, bn=False)
        heatmaps2 = conv(heatmaps2, num_joints, kernel_size=1, bn=False, relu=False)
        pafs2     = conv(net, 128, kernel_size=1, bn=False)
        pafs2     = conv(pafs2, num_pafs, kernel_size=1, bn=False, bias=False)

    return heatmaps2, pafs2

if __name__ == '__main__':
    import hiddenlayer as hl
    import hiddenlayer.transforms as ht
    import os
    # Hide GPUs. Not needed for this demo.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    with tf.Session() as sess:
        with tf.Graph().as_default() as tf_graph:
            # Setup input placeholder
            inputs = tf.placeholder(tf.float32, shape=(1, 256, 256, 3))
            # Build model
            predictions, _ = lightweight_openpose(inputs, 14, 26, True)
            # Build HiddenLayer graph
            hl_graph = hl.build_graph(tf_graph)

    hl_graph.save('lightweight', format='png')

