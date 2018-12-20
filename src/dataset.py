# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: dataset.py
@time: 18-10-30 16:36
'''

import tensorflow as tf
import os
import json
import cv2
import numpy as np

import sys
sys.path.append('../')
from src.parameters import params
from src.get_heatmap import get_heatmap
from src.get_paf import get_paf

parameters = params
id_kps_dict = {}

def prepare(json_file):

    global id_kps_dict

    img_ids   = []

    with open(json_file) as f:
        annos = json.load(f)
        for anno in annos:
            img_ids.append(anno['image_id'])
            kps = []
            for key, val in anno['keypoint_annotations'].items():
                kps += val
            kps = np.reshape(np.asarray(kps), (-1, 14, 3))
            id_kps_dict[anno['image_id']] = kps

    return img_ids

def _img_preprocessing(img, kps):
    h, w, c = img.shape

    # run random horizontally flip with 0.5 probability
    rd_filp = np.random.randint(1, 11)
    if rd_filp > 5:
        new_single_kps = np.zeros((4, 3))
        img = cv2.flip(img, 1)
        new_kps = {}
        for key, value in kps.items():
            single_kps = value
            assert (single_kps.shape == (4, 3))
            for i in range(4):
                point = single_kps[i]
                point[1] = point[1]
                point[0] = w - point[0] - 1
                single_kps[i] = point
            # exchange left and right point
            new_single_kps[0] = single_kps[1]
            new_single_kps[1] = single_kps[0]
            new_single_kps[2] = single_kps[2]
            new_single_kps[3] = single_kps[3]
            new_kps[key] = new_single_kps
        kps = new_kps

    # run scale hue, saturation, brightness with coefficients uniformly drawn from [0.6, 1.4]
    img_zero = np.zeros([h, w, c], img.dtype)
    rd_adjust1 = np.random.uniform(0.6, 1.5)
    rd_adjust2 = np.random.uniform(0.6, 1.5)
    img = cv2.addWeighted(img, rd_adjust1, img_zero, abs(rd_adjust1-1), rd_adjust2)

    # run random rotate 90 degree or -90 degree
    rd_rotate = np.random.randint(1, 10)
    if rd_rotate > 6:
        img, kps = _rotate(img, kps, 90)
    elif rd_rotate < 4:
        img, kps = _rotate(img, kps, -90)

    return img, kps

def _rotate(img, kps, degree):
    h, w, c = img.shape
    # get the rotation matrix
    matrix = cv2.getRotationMatrix2D((w//2, h//2), degree, 1.0)

    cos = np.abs(matrix[0,0])
    sin = np.abs(matrix[0,1])

    # compute the new bounding dimensions of img
    nw = int(h*sin + w*cos)
    nh = int(h*cos + w*sin)

    # adjust rotation matrix
    matrix[0, 2] += (nw / 2) - w//2
    matrix[1, 2] += (nh / 2) - h//2

    img = cv2.warpAffine(img, matrix, (nw, nh))

    m11 = matrix[0, 0]
    m12 = matrix[0, 1]
    m13 = matrix[0, 2]
    m21 = matrix[1, 0]
    m22 = matrix[1, 1]
    m23 = matrix[1, 2]

    for key, value in kps.items():
        single_kps = value
        assert (single_kps.shape == (4, 3))
        for i in range(4):
            point = single_kps[i]
            ori_x = point[0]
            ori_y = point[1]
            dx = m11 * ori_x + m12 * ori_y + m13
            dy = m21 * ori_x + m22 * ori_y + m23
            point[0] = dx
            point[1] = dy
            single_kps[i] = point

        kps[key] = single_kps
    return img, kps


def _parse_function(img_id, mode):

    global parameters
    global id_kps_dict
    # print ('###########################################')
    if type(img_id) == type(b'123'):
        img_id = str(img_id, encoding='utf-8')
    if type(mode) == type(b'123'):
        mode   = str(mode, encoding='utf-8')

    # read img_data and convert BGR to RGB
    if mode == 'train':
        img_data = cv2.imread(os.path.join(parameters['train_data_path'], img_id + '.jpg'))
        data_aug = False
    elif mode == 'valid':
        img_data = cv2.imread(os.path.join(parameters['valid_data_path'], img_id + '.jpg'))
        data_aug = False
    else:
        img_data = None
        data_aug = None
        print('parse_function mode must be train or valid.')
        exit(-1)

    b,g,r    = cv2.split(img_data)
    img_data = cv2.merge([r,g,b])

    # get kps
    keypoints = id_kps_dict[img_id]
    kps = dict()
    for i in range(keypoints.shape[0]):
        val = []
        val.append(keypoints[i, 0, :])
        val.append(keypoints[i, 3, :])
        val.append(keypoints[i, 12, :])
        val.append(keypoints[i, 13, :])
        kps[str(i)] = np.reshape(np.asarray(val), newshape=(-1, 3))
    keypoints = kps

    if data_aug:
        img_data, keypoints = _img_preprocessing(img_data, keypoints)

    h, w, c  = img_data.shape
    img      = cv2.resize(img_data, (parameters['width'], parameters['height']))
    img      = np.asarray(img, dtype=np.float32) / 255

    heatmap  = get_heatmap(keypoints, h, w)
    paf      = get_paf(keypoints, h, w)

    return img, heatmap, paf

def get_dataset_pipeline(mode='train'):

    global parameters

    if mode == 'train':
        json_file = parameters['train_json_file']
        batch_size = parameters['batch_size']
    elif mode == 'valid':
        json_file = parameters['valid_json_file']
        batch_size = parameters['valid_batch_size']
    else:
        json_file = None
        batch_size = None
        print('Dataset mode must be train or valid.')
        exit(-1)

    img_ids = prepare(json_file)
    np.random.shuffle(img_ids)
    dataset = tf.data.Dataset.from_tensor_slices(img_ids)

    dataset = dataset.map(
        lambda  img_id: tuple(
            tf.py_func(
                func=_parse_function,
                inp = [img_id, mode],
                Tout=[tf.float32, tf.float32, tf.float32])),
                num_parallel_calls=12)

    dataset = dataset.batch(batch_size, drop_remainder=True).repeat(-1)
    dataset = dataset.prefetch(buffer_size=batch_size*50)

    return dataset

# json_file = parameters['valid_json_file']
# img_ids   = prepare(json_file)
# n = 0
# print ('--------{}---------'.format(len(img_ids)))


# test number of dataset
# import time
# tf.enable_eager_execution()
# input_pipeline = get_train_dataset_pipeline('train')
# iter = input_pipeline.make_one_shot_iterator()
# count = 0
# #
# try:
#     print ('start training dataset....')
#     while (1):
#         imgs, hp, paf = iter.get_next()
#         # print (imgs.shape, count)
#         count += 1
#
# except tf.errors.OutOfRangeError:
#     print (count)
#
# input_pipeline = get_train_dataset_pipeline('valid')
# iter = input_pipeline.make_one_shot_iterator()
# count = 0
# #
# try:
#     print ('start valid dataset....')
#     while (1):
#         imgs, hp, paf = iter.get_next()
#         # print(imgs.shape, count)
#         count += 1
#         print (count)
#
# except tf.errors.OutOfRangeError:
#     print(count)

# input_pipeline = get_dataset_pipeline(mode='valid')
# iter = input_pipeline.make_one_shot_iterator()
# imgs, heatmaps, pafs = iter.get_next()
# count = 0
# label = np.random.rand(100, 368, 368, 3)
# loss = tf.nn.l2_loss(imgs-label)
# heatmaps_placeholder = tf.placeholder(tf.float32, shape=[None, params['height']//params['input_scale'],
#                                                                  params['width']//params['input_scale'], params['num_keypoints']])
#
# pafs_placeholder     = tf.placeholder(tf.float32, shape=[None, params['height']//params['input_scale'],
#                                                                  params['width']//params['input_scale'], params['paf_channels']])
# #
# with tf.Session() as sess:
#     sess.run(loss)
#     try:
#         print ('start training dataset....')
#         while (1):
#             _img, _h, _p = sess.run([imgs, heatmaps, pafs])
#             sess.run(loss, feed_dict={heatmaps_placeholder:_h, pafs_placeholder:_p})
#             count += 1
#             print (count)
#     except tf.errors.OutOfRangeError:
#         print (count)

# img_id = b'0f128955dd4210efae7d604fe399f5f1d63fa6fb'
# json_file = parameters['train_json_file']
# img_ids   = prepare(json_file)
# img, heatmap, paf = _parse_function(img_id)
# print (heatmap.shape)
# print (paf.shape)
# # cv2.imwrite('img.jpg', img*255)
# # cv2.imwrite('heat.jpg', np.sum(heatmap, axis=-1, keepdims=True) * 255)
# pp = paf[:,:,0]
# pp = np.expand_dims(pp, axis=2)
# pp = (pp - np.min(pp))/(np.max(pp) - np.min(pp)) * 255
# paf = np.expand_dims(paf[...,0], axis=2)
# cv2.imwrite('paf.jpg', pp)

# json_file = parameters['train_json_file']
# img_ids   = prepare(json_file)
# print (img_ids[0])
# np.random.shuffle(img_ids)
# print ('$$$$')
# print (img_ids[0])