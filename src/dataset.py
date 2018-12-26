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
from src.img_preprocess import rotate, flip

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
        img, kps = flip(img, kps, 1)

    # run scale hue, saturation, brightness with coefficients uniformly drawn from [0.6, 1.4]
    img_zero = np.zeros([h, w, c], img.dtype)
    rd_adjust1 = np.random.uniform(0.6, 1.5)
    rd_adjust2 = np.random.uniform(0.6, 1.5)
    img = cv2.addWeighted(img, rd_adjust1, img_zero, abs(rd_adjust1-1), rd_adjust2)

    # run random rotate 90 degree or -90 degree
    rd_rotate = np.random.randint(1, 10)
    if rd_rotate > 6:
        img, kps = rotate(img, kps, 90)
    elif rd_rotate < 4:
        img, kps = rotate(img, kps, -90)

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

    dataset = dataset.batch(batch_size, drop_remainder=True).repeat(1)
    dataset = dataset.prefetch(buffer_size=batch_size*50)

    return dataset
