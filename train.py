# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: train.py
@time: 18-10-31 09:57
'''

import tensorflow as tf

from src.dataset import get_dataset_pipeline
from src.parameters import params
from src.log_info import get_logger

from src.light_pose import create_light_weight_openpose
import os
from datetime import datetime
import math
import time
# from src.learning_rate import cosine, cosine_tf

def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    current_time = datetime.now().strftime('%Y%m%d-%H%M')
    if params['finetuning'] is None:
        checkpoint_dir = os.path.join(params['checkpoint_path'], current_time)
    else:
        checkpoint_dir = os.path.join(params['checkpoint_path'], params['finetuning'])

    print ('Checkpoint Dir == {}'.format(checkpoint_dir))

    graph = tf.Graph()
    train_steps_per_epoch = params['train_nums'] // params['batch_size']
    valid_steps_per_epoch = params['valid_nums'] // params['valid_batch_size']
    print ('train steps per epoch == {}, valid steps per epoch == {}'.format(train_steps_per_epoch, valid_steps_per_epoch))

    with graph.as_default():

        # train dataset
        dataset = get_dataset_pipeline(mode='train')
        # iter = dataset.make_initializable_iterator()
        iter    = dataset.make_one_shot_iterator()
        imgs, heatmaps, pafs = iter.get_next()

        #valid dataset
        valid_dataset = get_dataset_pipeline(mode='valid')
        valid_iter    = valid_dataset.make_one_shot_iterator()
        valid_imgs, valid_heatmaps, valid_pafs = valid_iter.get_next()


        imgs_placeholder     = tf.placeholder(tf.float32, shape=[None, params['height'], params['width'], 3])

        heatmaps_placeholder = tf.placeholder(tf.float32, shape=[None, params['height']//params['input_scale'],
                                                                 params['width']//params['input_scale'], params['num_keypoints']])

        pafs_placeholder     = tf.placeholder(tf.float32, shape=[None, params['height']//params['input_scale'],
                                                                 params['width']//params['input_scale'], params['paf_channels']])


        pred_cpm1, pred_paf1, pred_cpm, pred_paf = create_light_weight_openpose(inputs=imgs_placeholder,
                                                                                heatmap_channels=params['num_keypoints'],
                                                                                paf_channels=params['paf_channels'],
                                                                                is_training=True)

        valid_imgs_placeholder = tf.placeholder(tf.float32, shape=[None, params['height'], params['width'], 3])

        valid_heatmaps_placeholder = tf.placeholder(tf.float32, shape=[None, params['height'] // params['input_scale'],
                                                                 params['width'] // params['input_scale'],
                                                                 params['num_keypoints']])

        valid_pafs_placeholder = tf.placeholder(tf.float32, shape=[None, params['height'] // params['input_scale'],
                                                             params['width'] // params['input_scale'],
                                                             params['paf_channels']])

        valid_pred_cpm1, valid_pred_paf1, valid_pred_cpm, valid_pred_paf = create_light_weight_openpose(inputs=valid_imgs_placeholder,
                                                                                heatmap_channels=params['num_keypoints'],
                                                                                paf_channels=params['paf_channels'],
                                                                                is_training=False)

        cpm1_loss = tf.nn.l2_loss(pred_cpm1 - heatmaps_placeholder)
        cpm2_loss = tf.nn.l2_loss(pred_cpm - heatmaps_placeholder)
        paf1_loss = tf.nn.l2_loss(pred_paf1 - pafs_placeholder)
        paf2_loss = tf.nn.l2_loss(pred_paf - pafs_placeholder)
        loss = cpm1_loss + cpm2_loss + paf1_loss + paf2_loss

        valid_loss = tf.nn.l2_loss(valid_pred_cpm1 - valid_heatmaps_placeholder) + tf.nn.l2_loss(valid_pred_cpm - valid_heatmaps_placeholder) +\
                     tf.nn.l2_loss(valid_pred_paf1 - valid_pafs_placeholder) + tf.nn.l2_loss(valid_pred_paf - valid_pafs_placeholder)


        # step lr
        global_step  = tf.Variable(0, trainable=False)

        # decay_steps = train_steps_per_epoch
        # decay_rate  = 0.1
        # staircase   = True
        # learning_rate  = tf.train.exponential_decay(learning_rate=params['lr'], global_step=global_step,
        #                                             decay_steps=decay_steps,
        #                                             decay_rate=decay_rate,
        #                                             staircase=staircase,
        #                                             name='lr')
        learning_rate = tf.Variable(params['lr'], trainable=False, name='lr')
        optimizer    = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)
        update_ops   = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)

        # saver
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        global_list = tf.global_variables()
        bn_vars = [g for g in global_list if 'moving_mean' in g.name]
        bn_vars += [g for g in global_list if 'moving_variance' in g.name]

        if params['finetuning'] is not None:
            saver       = tf.train.Saver(var_list=var_list+bn_vars, max_to_keep=50)
            saver_alter = tf.train.Saver(var_list=var_list + bn_vars, max_to_keep=params['epoch'])
        else:
            saver       = tf.train.Saver(max_to_keep=50)
            saver_alter = tf.train.Saver(var_list=var_list + bn_vars, max_to_keep=params['epoch'])
        # logging
        logger = get_logger()
        logger.info('(height, width) : ({}, {})'.format(params['height'], params['width']))
        logger.info('learning_rate: {} '.format(params['lr']))
        logger.info('optimizer: adam')
        logger.info('lr decay way: no decay, keep constant')
        # logger.info('decay rate:   {}'.format(decay_rate))
        # logger.info('decay step:   {}'.format(train_steps_per_epoch))
        logger.info('batch size:   {}'.format(params['batch_size']))
        logger.info('checkpoint saver number: {}'.format(params['epoch']))
        logger.info('train epochs: {}'.format(params['epoch']))
        logger.info('total train nums: {}'.format(params['train_nums']))
        logger.info(' ')

        # summary
        tf.summary.scalar('total_loss', loss)
        tf.summary.scalar('cpm_loss', cpm2_loss)
        tf.summary.scalar('paf_loss', paf2_loss)
        tf.summary.scalar('lr', learning_rate)
        tf.summary.image('img', imgs_placeholder, max_outputs=3)
        tf.summary.image('label_heat', tf.reduce_sum(heatmaps_placeholder, axis=3, keepdims=True), max_outputs=3)
        tf.summary.image('right_shoulder', tf.expand_dims(heatmaps_placeholder[...,0], axis=3), max_outputs=3)
        tf.summary.image('pred_heat', tf.reduce_sum(pred_cpm, axis=3, keepdims=True), max_outputs=3)

        true_paf_0 = pafs_placeholder[...,0]
        pred_paf_0 = pred_paf[...,0]
        true_paf_0 = (true_paf_0 - tf.reduce_min(true_paf_0)) / (tf.reduce_max(true_paf_0) - tf.reduce_min(true_paf_0))
        pred_paf_0 = (pred_paf_0 - tf.reduce_min(pred_paf_0)) / (tf.reduce_max(pred_paf_0) - tf.reduce_min(pred_paf_0))
        tf.summary.image('label_paf0', tf.expand_dims(true_paf_0, axis=3), max_outputs=3)
        tf.summary.image('pred_paf0', tf.expand_dims(pred_paf_0, axis=3), max_outputs=3)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(checkpoint_dir, graph)

        # init
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        config  = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # learning rate parameter
        epoch_gap      = 0
        decay_lr       = False
        pre_valid_loss = 0

        with tf.Session(graph=graph, config=config) as sess:
            sess.run(init_op)
            sess.graph.finalize()

            # coord = tf.train.Coordinator()

            train_step = 0
            valid_step = 0
            valid_losses = 0

            s_time = time.time()
            state = 'train'

            if params['finetuning'] is not None:
                saver.restore(sess, checkpoint_dir)
                print ('#------------------Successfully restored pre_trained Model.---------------------#')
                print ('#------------------------Current lr, global_step = {}---------------------------#'.format(sess.run([learning_rate, global_step])))

            while (1):
                if state == 'train' and train_step < train_steps_per_epoch * params['epoch']:

                    train_step += 1
                    _imgs, _heatmaps, _pafs = sess.run([imgs, heatmaps, pafs])

                    predcpm1, predpaf1, predcpm2, predpaf2, \
                    total_loss, loss_cpm1, loss_paf1, loss_cpm2, loss_paf2, \
                    lr, _ = sess.run(
                        [pred_cpm1, pred_paf1, pred_cpm, pred_paf,
                         loss, cpm1_loss, paf1_loss, cpm2_loss, paf2_loss,
                         learning_rate, train_op],
                        feed_dict={imgs_placeholder: _imgs,
                                   pafs_placeholder: _pafs,
                                   heatmaps_placeholder: _heatmaps}
                    )

                    if train_step % 10 == 0:
                        merge_op = sess.run(
                            summary_op,
                            feed_dict={imgs_placeholder: _imgs,
                                       pafs_placeholder: _pafs,
                                       heatmaps_placeholder: _heatmaps}
                        )
                        summary_writer.add_summary(merge_op, train_step)
                        summary_writer.flush()

                        print('step {}: loss = {:.8f}, cpm2_loss = {:.8f}, paf2_loss = {:.8f}, lr = {:.10f}'.format
                              (train_step, total_loss / params['batch_size'], loss_cpm2 / params['batch_size'],
                               loss_paf2 / params['batch_size'], lr))

                    if train_step % train_steps_per_epoch == 0:
                        end_time = time.time()
                        print('One epoch spend time == {} hours.'.format((end_time - s_time) / 60 / 60))
                        save_alter_path = saver_alter.save(sess, checkpoint_dir + '/model_alter.ckpt', global_step=train_step)
                        save_path = saver.save(sess, checkpoint_dir + '/model.ckpt', global_step=train_step)
                        print('For one epoch, Model saved in      {}'.format(save_path))
                        print('For one epoch, Model also saved in {}'.format(save_alter_path))
                        print('Start run validation......')
                        state = 'valid'

                elif state == 'valid' and valid_step < valid_steps_per_epoch * params['epoch']:

                    _validimgs, _validheatmaps, _validpafs = sess.run([valid_imgs, valid_heatmaps, valid_pafs])
                    valid_total_loss = sess.run(valid_loss,
                                                feed_dict={valid_imgs_placeholder: _validimgs,
                                                           valid_pafs_placeholder: _validpafs,
                                                           valid_heatmaps_placeholder: _validheatmaps}
                                                )
                    valid_step += 1
                    valid_losses += valid_total_loss
                    print ('valid step ', valid_step)
                    if valid_step % 300 == 0:
                        valid_losses /= (valid_steps_per_epoch * params['valid_batch_size'])
                        print('-----------------------Valid Loss == {}'.format(valid_losses))
                        logger.info('valid loss == {}'.format(valid_losses))

                        valid_losses = 0.
                        state = 'train'
                        s_time = time.time()
                else:
                    assert (state == 'train' or state == 'valid')

train()


