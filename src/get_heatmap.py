# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: get_heatmap.py
@time: 18-10-30 17:29
'''

import numpy as np
import math

import sys
sys.path.append('../')
from src.parameters import params

parameter = params

def get_heatmap(keypoints, ori_height, ori_width):

    global parameter

    heatmap_height = parameter['height'] // parameter['input_scale']
    heatmap_width  = parameter['width'] // parameter['input_scale']

    factorx = heatmap_width / ori_width
    factory = heatmap_height / ori_height

    heatmap = np.zeros((heatmap_height, heatmap_width, parameter['num_keypoints']), dtype=np.float32)
    for i in range(parameter['num_keypoints']):
        single_heatmap = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)
        for key, val in keypoints.items():
            if val[i, 0] != 0 or val[i, 1] != 0:
                # print (val[i, 0], val[i, 1])
                center_x = val[i, 0] * factorx
                center_y = val[i, 1] * factory
                # print (center_x, center_y)
                single_heatmap = gaussian(single_heatmap, center_x, center_y, sigma=parameter['sigma'])
                # center = [center_x, center_y]
                # single_heatmap = create_gauss_map(center, heatmap_width, heatmap_height, sigma=params['sigma'])
        heatmap[:, :, i] = single_heatmap

    # heatmap[:, :, -1] = np.maximum(1-np.max(heatmap[:,:,:-1], axis=2), 0.)

    return heatmap

def create_gauss_map(center, size_x, size_y, sigma):
    """
    create gaussian map
    :param center: gaussian map center, (x, y) means the coord
    :param size_x: map size x
    :param size_y: map size y
    :return: confidence map
    """
    x_range = [i for i in range(int(size_x))]
    y_range = [i for i in range(int(size_y))]
    xx, yy = np.meshgrid(x_range, y_range)
    d2 = (xx - center[0])**2 + (yy - center[1])**2
    exponent = d2 / 2.0 / sigma / sigma
    mask = exponent <= 4.6052
    confid_map = np.exp(-exponent)
    confid_map = np.multiply(mask, confid_map)
    return confid_map

def gaussian(heatmap, center_x, center_y, sigma=6.):

    th = 1.6052
    delta = math.sqrt(th * 2)

    height = heatmap.shape[0]
    width  = heatmap.shape[1]

    x0 = int(max(0, center_x - delta * sigma))
    y0 = int(max(0, center_y - delta * sigma))

    x1 = int(min(width, center_x + delta * sigma))
    y1 = int(min(height, center_y + delta * sigma))

    for y in range(y0, y1):
        for x in range(x0, x1):
            d   = (x - center_x) ** 2 + (y - center_y) ** 2
            exp = d / 2.0 / sigma / sigma

            if exp > th:
                continue
            heatmap[y][x] = max(heatmap[y][x], math.exp(-exp))
            heatmap[y][x] = min(heatmap[y][x], 1.0)

    return heatmap

