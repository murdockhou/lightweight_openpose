#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: dataset.py
@time: 2019/8/12 下午5:08
@desc:
'''
import tensorflow as tf
import os
import cv2
import numpy as np

from utils.utils import read_json, get_heatmap, get_paf
from utils.data_aug import data_aug
from configs.ai_config import ai_config as train_config

img_path = None
id_kps_dict = None
params = train_config
def get_dataset(model = 'train'):
    global id_kps_dict, params, img_path

    json_file = params['train_json_file']
    img_path = params['train_img_path']
    img_ids, id_kps_dict = read_json(json_file)

    dataset = tf.data.Dataset.from_tensor_slices(img_ids)
    dataset = dataset.shuffle(buffer_size=1000).repeat(1)
    dataset = dataset.map(tf_parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(params['batch_size'], drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def tf_parse_func(img_id):
    [img, heatmap, paf] = tf.py_function(paser_func, [img_id], [tf.float32, tf.float32, tf.float32])
    return img, heatmap, paf

def paser_func(img_id):
    global img_path, id_kps_dict, params

    if not type(img_id) == type('123'):
        img_id = img_id.numpy()
        if type(img_id) == type(b'123'):
            img_id = str(img_id, encoding='utf-8')

    kps = id_kps_dict[img_id]
    img_ori = cv2.imread(os.path.join(img_path, img_id+'.jpg'))
    # data augmentation
    img_ori, kps = data_aug(img_ori, None, kps)

    # padding img
    img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape

    # 只在最右边或者最下边填充0, 这样不会影响box或者点的坐标值, 所以无需再对box或点的坐标做改变
    if w > h:
        img = cv2.copyMakeBorder(img, 0, w - h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        img = cv2.copyMakeBorder(img, 0, 0, 0, h - w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    orih, oriw, oric = img.shape
    neth, netw = params['height'], params['width']
    outh, outw = neth // params['scale'], netw // params['scale']

    # create heatmap label
    heatmap = get_heatmap(kps, orih, oriw, outh, outw, params['num_joints'], sigma=params['sigma'])

    # create paf label
    paf = get_paf(kps, orih, oriw, outh, outw, params['pafs'], width=params['paf_width'])

    # TODO add mask

    # create image inputs
    img = cv2.resize(img, (netw, neth), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.

    return img, heatmap, paf
