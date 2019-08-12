#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: utils.py
@time: 2019/8/12 下午5:11
@desc:
'''

import json
import numpy as np
import math

def read_json(json_file):
    img_ids = []
    id_kps_dict = {}
    with open(json_file) as f:
        annos = json.load(f)
        for anno in annos:
            img_ids.append(anno['image_id'])
            kps = []
            for key, val in anno['keypoint_annotations'].items():
                kps += val
            kps = np.reshape(np.asarray(kps), (-1, 14, 3))
            id_kps_dict[anno['image_id']] = kps
            # id_body_annos[anno['image_id']] = anno['human_annotations']


    return img_ids, id_kps_dict

def get_heatmap(keypoints, ori_height, ori_width, heatmap_height, heatmap_width, heatmap_channels, sigma):
    '''
    function that create gaussian filter heatmap based keypoints.
    :param keypoints: ndarray with shape [person_num, joints_num, 3], each joint contains three attribute, [x, y, v]
    :param ori_height: ori_img height
    :param ori_width: ori_img width
    :param heatmap_height: heatmap_height
    :param heatmap_width: heatmap_width
    :param heatmap_channels: number of joints
    :param sigma: parameter about gaussian function
    :return: heatmap
            A ndarray with shape [heatmap_height, heatmap_width, heatmap_channels]
    '''
    factorx = heatmap_width / ori_width
    factory = heatmap_height / ori_height

    heatmap = np.zeros((heatmap_height, heatmap_width, heatmap_channels), dtype=np.float32)

    for i in range(heatmap_channels):

        single_heatmap = np.zeros(shape=(heatmap_height, heatmap_width), dtype=np.float32)
        for j in range(keypoints.shape[0]):
            people = keypoints[j]
            center_x = people[i][0] * factorx
            center_y = people[i][1] * factory

            if center_x >= heatmap_width or center_y >= heatmap_height:
                continue
            if center_x < 0 or center_y < 0:
                continue
            if center_x == 0 and center_y == 0:
                continue
            if people[i][2] == 3:
                continue

            single_heatmap = gaussian(single_heatmap, center_x, center_y, sigma=sigma)

        heatmap[:, :, i] = single_heatmap

    return heatmap

def gaussian(heatmap, center_x, center_y, sigma):
    # sigma = 1.0 , 半径范围为3.5个像素
    # sigma = 2.0, 半径范围为6.5个像素
    # sigma = 0.5, 半径范围为2.0个像素


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

def get_paf(keypoints, ori_height, ori_width, paf_height, paf_width, paf_channels, width):
    '''
    function that create paf based keypoints
    :param keypoints: ndarray with shape [person_num, joints_num, 3], each joint contains three attribute, [x, y, v]
    :param ori_height: ori_img height
    :param ori_width:  ori_img width
    :param paf_height: paf_height
    :param paf_width:  paf_width
    :param paf_channels: how many paf_channels will return. the number of paf_channels is 2 * connect_num, which
                         connect_num is edges num of points.
    :param width: the threshold that controls the area about paf connection.
    :return:
        A ndarray with shape [paf_height, paf_width, paf_channels].
    '''
    factorx = paf_width / ori_width
    factory = paf_height / ori_height

    # pt1 = [0,0,0,1,1,2,3]
    # pt2 = [1,2,3,6,7,4,5]
    pt1 = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 13, 13, 13]
    pt2 = [1, 2, 4, 5, 7, 8, 10, 11, 13, 0, 3, 6, 9]

    pafs = np.zeros((paf_channels, paf_height, paf_width), dtype=np.float32)
    # print ('---------------------------------------------------')
    for i in range(len(pt1)):
        count = np.zeros((paf_height, paf_width))
        for j in range(keypoints.shape[0]):
            val = keypoints[j]
            if (val[pt1[i], 0] == 0 and val[pt1[i], 1] == 0) or (val[pt2[i], 0] == 0  and val[pt2[i], 1] == 0) or \
                val[pt1[i], 2] == 3 or val[pt2[i], 2] == 3:
                continue
            center_x = val[pt1[i], 0] * factorx
            center_y = val[pt1[i], 1] * factory
            centerA  = np.asarray([center_x, center_y])

            center_x = val[pt2[i], 0] * factorx
            center_y = val[pt2[i], 1] * factory
            centerB  = np.asarray([center_x, center_y])

            paf_map, mask = create_paf_map(centerA, centerB, paf_width, paf_height, width)
            pafs[2 * i:2 * i + 2, :, :] += paf_map
            count[mask == True] += 1

        mask = count == 0
        count[mask == True] = 1
        pafs[2 * i:2 * i + 2, :, :] = np.divide(pafs[2 * i:2 * i + 2, :, :], count[np.newaxis, :, :])

    pafs = np.transpose(pafs, (1, 2, 0))
    return pafs

def create_paf_map(centerA, centerB, size_x, size_y, thresh):
    """
    creat paf vector
    :param centerA:  start point coord
    :param centerB: end point coord
    :param size_x:  map size x
    :param size_y:  map size y
    :param thresh:  width of paf vector
    :return: paf map:
             mask: mask indicate where should have data
    """
    centerA = centerA.astype(float)
    centerB = centerB.astype(float)
    paf_map = np.zeros((2, size_y, size_x))
    norm = np.linalg.norm(centerB - centerA)
    if norm == 0.0:
        return paf_map
    limb_vec_unit = (centerB - centerA) / norm

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thresh)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thresh)), size_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thresh)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thresh)), size_y)

    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)
    xx = xx.astype(np.int32)
    yy = yy.astype(np.int32)
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thresh  # mask is 2D

    paf_map[:, yy, xx] = np.repeat(mask[np.newaxis, :, :], 2, axis=0)
    paf_map[:, yy, xx] *= limb_vec_unit[:, np.newaxis, np.newaxis]
    mask = np.logical_or.reduce((np.abs(paf_map[0, :, :]) > 0, np.abs(paf_map[1, :, :]) > 0))
    return paf_map, mask