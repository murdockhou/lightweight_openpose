# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: tridentnet_train.py
@time: 19-1-9 18:42
'''

import tensorflow as tf

from src.dataset import get_dataset_pipeline
from src.train_config import train_config as params

from src.lightweight_openpose import light_openpose

import os
from datetime import datetime
import logging
import time


def get_logger(log):

    if not os.path.exists(os.path.dirname(log)):
        os.mkdir(os.path.dirname(log))

    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关

    log_name = log

    # print(log_path)
    logfile   = log_name
    fh        = logging.FileHandler(logfile, mode='a')
    fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    # formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    # fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    # logger.warning('this ')
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y %m %d %H:%M', time.localtime(time.time()))
    logger.info(' ')
    logger.info('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    logger.info(rq)
    return logger

def train():
    # params['input_scale'] = 4
    train_steps_per_epoch = params['train_nums'] // params['batch_size']
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    current_time = datetime.now().strftime('%Y%m%d-%H%M')
    if params['finetuning'] is None:
        checkpoint_dir = os.path.join(params['checkpoint_path'], current_time)
    else:
        checkpoint_dir = os.path.join(params['checkpoint_path'], params['finetuning'])

    print ('Checkpoint Dir == {}'.format(checkpoint_dir))

    graph = tf.Graph()

    with graph.as_default():

        # train dataset
        dataset = get_dataset_pipeline(params, mode='train')
        iter = dataset.make_initializable_iterator()
        # iter = dataset.make_one_shot_iterator()
        imgs, heatmaps, pafs = iter.get_next()

        # valid dataset
        valid_dataset = get_dataset_pipeline(params, mode='valid')
        valid_iter = valid_dataset.make_initializable_iterator()
        # valid_iter = valid_dataset.make_one_shot_iterator()
        valid_imgs, valid_heatmaps, valid_pafs = valid_iter.get_next()

        imgs_placeholder = tf.placeholder(tf.float32, shape=[None, params['height'], params['width'], 3], name='img')

        heatmaps_placeholder = tf.placeholder(tf.float32, shape=[None, params['height'] // params['input_scale'],
                                                                 params['width'] // params['input_scale'],
                                                                 params['num_kps']], name='cpm')
        pafs_placeholder = tf.placeholder(tf.float32, shape=[None, params['height'] // params['input_scale'],
                                                             params['width'] // params['input_scale'],
                                                             params['paf']], name='paf')

        cpms1, pafs1, cpm, paf = light_openpose(
            inputs=imgs_placeholder,
            joints=params['num_kps'],
            paf=params['paf'],
            dilation_rate=1,
            is_training=True)
        # outputs_2, final_output_2 = create_head_count_model(
        #     inputs=imgs_placeholder,
        #     dilation_rate=2,
        #     is_training=True)
        # outputs_3, final_output_3 = create_head_count_model(
        #     inputs=imgs_placeholder,
        #     dilation_rate=3,
        #     is_training=True)

        valid_imgs_placeholder = tf.placeholder(tf.float32, shape=[None, params['height'], params['width'], 3], name='v_img')

        valid_heatmaps_placeholder = tf.placeholder(tf.float32, shape=[None, params['height'] // params['input_scale'],
                                                                 params['width'] // params['input_scale'],
                                                                 params['num_kps']], name='v_cpm')
        valid_pafs_placeholder = tf.placeholder(tf.float32, shape=[None, params['height'] // params['input_scale'],
                                                             params['width'] // params['input_scale'],
                                                             params['paf']], name='v_paf')

        _1, _2, valid_cpm, valid_paf = light_openpose(
            inputs=valid_imgs_placeholder,
            joints=params['num_kps'],
            paf=params['paf'],
            dilation_rate=1,
            is_training=False)

        # define train_loss
        cpm_train_loss = []
        paf_train_loss = []
        for mid_output in cpms1:
            cpm_train_loss.append(tf.nn.l2_loss(mid_output - heatmaps_placeholder))
        for mid_output in pafs1:
            paf_train_loss.append(tf.nn.l2_loss(mid_output - pafs_placeholder))
        loss = tf.reduce_sum(cpm_train_loss) + tf.reduce_sum(paf_train_loss)
        final_cpm_loss_1 = tf.nn.l2_loss(cpm - heatmaps_placeholder)
        final_paf_loss_1 = tf.nn.l2_loss(paf - pafs_placeholder)

        # define valid_loss, only contain cpm loss
        _loss = []
        for mid_output in _1:
            _loss.append(tf.nn.l2_loss(mid_output - valid_heatmaps_placeholder))
        # for mid_output in _2:
        #     _loss.append(tf.nn.l2_loss(mid_output - valid_pafs_placeholder))
        valid_loss = tf.reduce_sum(_loss)

        global_step  = tf.Variable(0, trainable=False)

        # step lr
        boundaries = [train_steps_per_epoch * 5, train_steps_per_epoch * 10]
        values = [1e-4, 1e-5, 1e-6]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=values)

        # learning_rate = tf.Variable(params['lr'], trainable=False, name='lr')
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
            saver       = tf.train.Saver(var_list=var_list+bn_vars, max_to_keep=params['save_models'])
            saver_alter = tf.train.Saver(var_list=var_list + bn_vars, max_to_keep=params['save_models'])
        else:
            saver       = tf.train.Saver(max_to_keep=params['save_models'])
            saver_alter = tf.train.Saver(var_list=var_list + bn_vars, max_to_keep=params['save_models'])

        # logging
        logger = get_logger(params['log_path'])
        logger.info('(height, width) : ({}, {})'.format(params['height'], params['width']))
        logger.info('learning_rate: {} '.format(values))
        logger.info('boundaries: {}'.format(boundaries))
        logger.info('optimizer: adam')
        logger.info('paf_width_thre: {}'.format(params['paf_width_thre']))
        logger.info('sigma: {}'.format(params['sigma']))
        logger.info('input_scale: {}'.format(params['input_scale']))
        logger.info('batch size:   {}'.format(params['batch_size']))
        logger.info('checkpoint saver number: {}'.format(params['save_models']))
        logger.info('step_per_epochs: {}'.format(train_steps_per_epoch))
        logger.info('flip, gray, rotate, scale')
        logger.info(' ')

        # summary
        tf.summary.scalar('total_loss', loss)
        tf.summary.scalar('final_output_loss_1', final_cpm_loss_1)
        tf.summary.scalar('lr', learning_rate)
        tf.summary.image('img', imgs_placeholder, max_outputs=3)
        tf.summary.image('label_heat', tf.reduce_sum(heatmaps_placeholder, axis=3, keepdims=True), max_outputs=3)
        tf.summary.image('pred_heat', tf.reduce_sum(cpm, axis=3, keepdims=True), max_outputs=3)

        true_paf_0 = pafs_placeholder[..., 0]
        pred_paf_0 = paf[..., 0]
        true_paf_0 = (true_paf_0 - tf.reduce_min(true_paf_0)) / (tf.reduce_max(true_paf_0) - tf.reduce_min(true_paf_0))
        pred_paf_0 = (pred_paf_0 - tf.reduce_min(pred_paf_0)) / (tf.reduce_max(pred_paf_0) - tf.reduce_min(pred_paf_0))
        tf.summary.image('label_paf', tf.expand_dims(true_paf_0, axis=3), max_outputs=3)
        tf.summary.image('pred_paf', tf.expand_dims(pred_paf_0, axis=3), max_outputs=3)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(checkpoint_dir, graph)

        # init
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        config  = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(graph=graph, config=config) as sess:
            sess.run(init_op)

            train_step = 0

            if params['finetuning'] is not None:
                saver.restore(sess, checkpoint_dir)
                print ('#------------------Successfully restored pre_trained Model.---------------------#')
                print ('#------------------------Current lr, global_step = {}---------------------------#'.format(sess.run([learning_rate, global_step])))
                # checkpoint_dir = '/media/ulsee/E/head_count/tridentnet_small/'
            for epoch in range(1, 101, 1):
                sess.run(iter.initializer)

                try:
                    while True:
                        train_step += 1
                        _imgs, _heatmaps, _pafs = sess.run([imgs, heatmaps, pafs])

                        total_loss, cpm_lists, paf_lists, cpm_loss_1, paf_loss_1,\
                        lr, _, merge_op = sess.run(
                            [loss, cpm_train_loss, paf_train_loss, final_cpm_loss_1, final_paf_loss_1,
                             learning_rate, train_op, summary_op],
                            feed_dict={imgs_placeholder: _imgs,
                                       heatmaps_placeholder: _heatmaps,
                                       pafs_placeholder: _pafs})

                        if train_step % 10 == 0:
                            summary_writer.add_summary(merge_op, train_step)
                            summary_writer.flush()
                            print('Epoch_{}_Step_{}/{}: lr = {:.8f} cpm_1 loss = {:.6f}, paf_1 loss = {:.6f}'.format
                                  (epoch, train_step, train_steps_per_epoch*(epoch-0), lr, cpm_loss_1, paf_loss_1))
                            print ('....cpm_list = {}, paf_list = {}.... '.format(cpm_lists, paf_lists))

                except tf.errors.OutOfRangeError:
                    saver_path       = saver.save(sess, checkpoint_dir + '/model.ckpt', global_step=train_step)
                    saver_alter_path = saver_alter.save(sess, checkpoint_dir + '/model_alter.ckpt', global_step=train_step)
                    print ('For one train epoch, saved model in {}'.format(saver_path))
                    print ('Also saved model in                 {}'.format(saver_alter_path))
                finally:
                    pass

                sess.run(valid_iter.initializer)
                print ('start run validation...')
                valid_step = 0
                valid_losses = 0
                try:
                    while True:
                        _validimgs, _validheatmaps, _validpafs = sess.run([valid_imgs, valid_heatmaps, valid_pafs])
                        valid_total_loss = sess.run(valid_loss, feed_dict={valid_imgs_placeholder: _validimgs,
                                                                           valid_heatmaps_placeholder: _validheatmaps,
                                                                           valid_pafs_placeholder: _validpafs})
                        valid_step += 1
                        valid_losses += valid_total_loss
                except tf.errors.OutOfRangeError:
                    pass
                finally:
                    pass

                valid_losses /= (valid_step * params['valid_batch_size'])
                print('Epoch {}, Valid Loss == {}'.format(epoch, valid_losses))
                logger.info('Epoch = {}, lr = {}, valid loss == {}'.format(epoch , lr, valid_losses))
                logger.info('...... model = {}'.format(saver_path))

if __name__ == '__main__':
    train()


