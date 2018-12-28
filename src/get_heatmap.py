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

def get_heatmap(keypoints, ori_height, ori_width, heatmap_height, heatmap_width, heatmap_channels, sigma):



    factorx = heatmap_width / ori_width
    factory = heatmap_height / ori_height

    heatmap = np.zeros((heatmap_height, heatmap_width, heatmap_channels), dtype=np.float32)
    for i in range(heatmap_channels):
        single_heatmap = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)
        for key, val in keypoints.items():
            if val[i, 0] != 0 or val[i, 1] != 0:

                center_x = val[i, 0] * factorx
                center_y = val[i, 1] * factory

                single_heatmap = gaussian(single_heatmap, center_x, center_y, sigma=sigma)

        heatmap[:, :, i] = single_heatmap

    return heatmap

def gaussian(heatmap, center_x, center_y, sigma):

    th = 4.6052
    delta = math.sqrt(th * 2)

    height = heatmap.shape[0]
    width = heatmap.shape[1]

    x0 = int(max(0, center_x - delta * sigma + 0.5))
    y0 = int(max(0, center_y - delta * sigma + 0.5))

    x1 = int(min(width - 1, center_x + delta * sigma + 0.5))
    y1 = int(min(height - 1, center_y + delta * sigma + 0.5))

    ## fast way
    arr_heat = heatmap[y0:y1 + 1, x0:x1 + 1]
    exp_factor = 1 / 2.0 / sigma / sigma
    x_vec = (np.arange(x0, x1 + 1) - center_x) ** 2
    y_vec = (np.arange(y0, y1 + 1) - center_y) ** 2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0
    heatmap[y0:y1 + 1, x0:x1 + 1] = np.maximum(arr_heat, arr_exp)

    ## slow way
    # for y in range(y0, y1):
    #     for x in range(x0, x1):
    #         d = (x - center_x) ** 2 + (y - center_y) ** 2
    #         exp = d / 2.0 / sigma / sigma
    #         if exp > th:
    #             continue
    #
    #         heatmap[y][x] = max(heatmap[y][x], math.exp(-exp))
    #         heatmap[y][x] = min(heatmap[y][x], 1.0)

    return heatmap

