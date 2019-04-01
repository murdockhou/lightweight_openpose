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

from src.get_heatmap import get_heatmap
from src.get_paf import get_paf
from src.img_aug import img_aug_fuc

id_kps_dict = {}
parameters = {}
id_body_annos = {}


def set_params(params):
    global parameters
    parameters = params


def prepare(json_file):

    global id_kps_dict
    global id_body_annos

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
            id_body_annos[anno['image_id']] = anno['human_annotations']

    return img_ids


def _parse_function(img_id, mode):

    global id_kps_dict, parameters

    if type(img_id) == type(b'123'):
        img_id = str(img_id, encoding='utf-8')
    if type(mode) == type(b'123'):
        mode   = str(mode, encoding='utf-8')

    # read img_data and convert BGR to RGB
    if mode == 'train':
        img_data = cv2.imread(os.path.join(parameters['train_data_path'], img_id + '.jpg'))
        data_aug = True
        sigma = parameters['sigma']
    elif mode == 'valid':
        img_data = cv2.imread(os.path.join(parameters['valid_data_path'], img_id + '.jpg'))
        data_aug = False
        sigma = 1.
    else:
        img_data = None
        data_aug = None
        sigma    = None
        print('parse_function mode must be train or valid.')
        exit(-1)

    h, w, _ = img_data.shape

    # get kps
    kps_channels = parameters['num_kps']
    paf_channels = parameters['paf']
    keypoints = id_kps_dict[img_id]

    keypoints = np.reshape(np.asarray(keypoints), newshape=(-1, kps_channels, 3))

    if data_aug:
        img_data, keypoints = img_aug_fuc(img_data, keypoints)

    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

    img      = cv2.resize(img_data, (parameters['width'], parameters['height']))
    img      = np.asarray(img, dtype=np.float32) / 255.
    heatmap_height = parameters['height'] // parameters['input_scale']
    heatmap_width  = parameters['width'] // parameters['input_scale']

    heatmap = get_heatmap(keypoints, h, w, heatmap_height, heatmap_width, kps_channels, sigma)
    paf     = get_paf(keypoints, h, w, heatmap_height, heatmap_width, paf_channels, parameters['paf_width_thre'])

    # add head mask info
    mask = np.zeros((heatmap_height, heatmap_width, 1), dtype=np.float32)
    for key, value in id_body_annos[img_id].items():
        body_box = value
        body_box[0] /= heatmap_width
        body_box[1] /= heatmap_height
        body_box[2] /= heatmap_width
        body_box[3] /= heatmap_height

        minx = int(max(1, body_box[0] - 5))
        miny = int(max(1, body_box[1] - 5))
        maxx = int(min(heatmap_width - 1, body_box[2] + 5))
        maxy = int(min(heatmap_height - 1, body_box[3] + 5))

        mask[miny:maxy, minx:maxx, :] = True
    
    labels = np.concatenate([heatmap, paf, mask], axis=-1)
    return img, labels

def get_dataset_pipeline(parameters, epochs=1, mode='train'):

    set_params(parameters)
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
                Tout=[tf.float32, tf.float32])),
        num_parallel_calls=12)

    dataset = dataset.batch(batch_size, drop_remainder=True).repeat(epochs)
    dataset = dataset.prefetch(buffer_size=batch_size*12*4)

    return dataset

if __name__ == '__main__':
    from src.train_config import train_config as params
    set_params(params)
    imgids = prepare(params['train_json_file'])
    for i in range(10):
        _parse_function(imgids[i], 'train')
#
# tf.enable_eager_execution()
# dd = get_dataset_pipeline(params)
# iter = dd.make_one_shot_iterator()
# count = 0
# try:
#     print ('start training dataset....')
#     while (1):
#         imgs, hp = iter.get_next()
#         print (imgs.shape, hp.shape)
#         count += 1
#         # print (count)
#         exit(0)
#
# except tf.errors.OutOfRangeError:
#     print (count)
