#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: dataset_test.py
@time: 2019/8/12 下午6:06
@desc:
'''

from data.dataset import get_dataset

# dataset = get_dataset()
# for epoch in range(10):
#     for step, (img, heatmap, paf) in enumerate(dataset):
#         print ('step {}/{}'.format(step, epoch))
#         print (img.shape)
#         print (heatmap.shape)
#         print (paf.shape)
#         break
#     break

# from utils.utils import *
# from utils.data_aug import data_aug
# import cv2, os
# json_file = '/media/hsw/E/datasets/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json'
# img_path = '/media/hsw/E/datasets/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902'
# img_ids, id_kps_dict = read_json(json_file)
#
# colors = [[0,0,255],[255,0,0],[0,255,0]]
# for img_id in img_ids:
#     print ('--------------------------------------------------------------')
#     kps = id_kps_dict[img_id]
#     # print('original, box:')
#
#     img = cv2.imread(os.path.join(img_path, img_id + '.jpg'))
#
#     img, kps = data_aug(img, None, kps)

# test lr schedule
import tensorflow as tf
import os

def lr_schedules(epoch, base_lr=1e-4):
    if epoch < 5:
        return base_lr
    elif epoch < 10:
        return base_lr / 10
    elif epoch < 15:
        return base_lr / 100
    elif epoch < 20:
        return base_lr / 1000
    else:
        return base_lr / 10000
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
dataset = get_dataset()
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(256,256,3)))
model.add(tf.keras.layers.MaxPool2D(padding='same'))
model.add(tf.keras.layers.MaxPool2D(padding='same'))
model.add(tf.keras.layers.MaxPool2D(padding='same'))
model.add(tf.keras.layers.Conv2D(14,3,1,'same'))
model.summary()
lr_fn = tf.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,decay_steps=100,decay_rate=0.1)
optimizer = tf.optimizers.Adam(learning_rate=1e-4)
# optimizer.lr.assign(1e-5)
for epoch in range(200):
    for step, (img, heatmap, paf) in enumerate(dataset):
        with tf.GradientTape() as tape:
            out = model(img)
            loss = tf.reduce_sum(tf.nn.l2_loss(out[0][0]-heatmap))
        grads = tape.gradient(loss, model.trainable_variables)

        optimizer.learning_rate = lr_schedules(epoch)
        print ('step / epoch: {} / {}, lr {}'.format(step, epoch, optimizer.learning_rate.numpy()))

        optimizer.apply_gradients(zip(grads, model.trainable_variables))


