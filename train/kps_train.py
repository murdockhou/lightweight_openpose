#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: kps_train.py
@time: 2019/8/12 下午7:12
@desc:
'''
import tensorflow as tf

from configs.ai_config import ai_config
import os

def train(model, optimizer, dataset, epochs, cur_time='8888-88-88-88-88', max_keeps=200):
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    if ai_config['finetune'] is not None:
        manager = tf.train.CheckpointManager(ckpt, ai_config['finetune'], max_to_keep=max_keeps)
        # print (manager.checkpoints)
        # you can through manager.checkpoints print all ckpts that have saved, it returns a list
        # Then can set which 'index' ckpt you need to restore
        ckpt.restore(manager.checkpoints[-1])
    else:
        manager = tf.train.CheckpointManager(ckpt, os.path.join(ai_config['ckpt'], cur_time), max_to_keep=max_keeps)

    for epoch in range(epochs):
        for step, (img, heatmap, paf) in enumerate(dataset):
            with tf.GradientTape() as tape:
                outputs = model(img)
                heatmap_loss = 0
                paf_loss = 0
                for output in outputs:
                    heatmap_loss += tf.reduce_sum(tf.nn.l2_loss(output[0]-heatmap))
                    paf_loss += tf.reduce_sum(tf.nn.l2_loss(output[1]-paf))
                loss = heatmap_loss + paf_loss
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            ckpt.step.assign_add(1)
            tf.summary.scalar('loss', loss, step=int(ckpt.step))
            # print ('--------------------------------------------------------------------------------------')
            if int(ckpt.step) % 10 == 0:
                tf.summary.image('img', img, step=int(ckpt.step), max_outputs=3)
                tf.summary.image('gt_heatmap', tf.reduce_sum(heatmap, axis=-1, keepdims=True),
                                 step=int(ckpt.step), max_outputs=3)
                tf.summary.image('pred_heatmap', tf.reduce_sum(outputs[-1][0], axis=-1, keepdims=True),
                                 step=int(ckpt.step), max_outputs=3)
                tf.summary.image('gt_paf[1]', tf.expand_dims(
                    (paf[...,1] - tf.reduce_min(paf[...,1])) / (tf.reduce_max(paf[...,1]) - tf.reduce_min(paf[...,1])),
                    axis=-1), step=int(ckpt.step), max_outputs=3)
                tf.summary.image('pred_paf[1]', tf.expand_dims(
                    (outputs[-1][1][..., 1] - tf.reduce_min(outputs[-1][1][..., 1])) / (tf.reduce_max(outputs[-1][1][..., 1]) - tf.reduce_min(outputs[-1][1][..., 1])),
                    axis=-1), step=int(ckpt.step), max_outputs=3)

            if int(ckpt.step) % 20 == 0:
                print('for epoch {:<5d} step {:<10d}, loss = {:<10f}, heat loss = {:<10f}, paf loss = {:<10f}'.format(
                    epoch, int(ckpt.step), loss, heatmap_loss, paf_loss))
            if int(ckpt.step) % 5000 == 0:
                save_path = manager.save()
                print('Saved ckpt for step {} : {}'.format(int(ckpt.step), save_path))
