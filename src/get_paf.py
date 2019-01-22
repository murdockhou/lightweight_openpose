# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: get_paf.py
@time: 18-10-30 17:48
'''

import numpy as np

def get_paf(keypoints, ori_height, ori_width, paf_height, paf_width, paf_channels, paf_width_thre):
    '''
    function that create paf based keypoints
    :param keypoints: ndarray with shape [person_num, joints_num, 3], each joint contains three attribute, [x, y, v]
    :param ori_height: ori_img height
    :param ori_width:  ori_img width
    :param paf_height: paf_height
    :param paf_width:  paf_width
    :param paf_channels: how many paf_channels will return. the number of paf_channels is 2 * connect_num, which
                         connect_num is edges num of points.
    :param paf_width_thre: the threshold that controls the area about paf connection.
    :return:
        A ndarray with shape [paf_height, paf_width, paf_channels].
    '''
    factorx = paf_width / ori_width
    factory = paf_height / ori_height

    # pt1 = [0,0,0,1,1,2,3]
    # pt2 = [1,2,3,6,7,4,5]
    pt1 = [1]
    pt2 = [0]

    pafs = np.zeros((paf_channels, paf_height, paf_width), dtype=np.float32)
    # print ('---------------------------------------------------')
    for i in range(len(pt1)):
        count = np.zeros((paf_height, paf_width))
        for j in range(keypoints.shape[0]):
            val = keypoints[j]
            if (val[pt1[i], 0] == 0 and val[pt1[i], 1] == 0) or (val[pt2[i], 0] == 0  and val[pt2[i], 1] == 0):
                continue
            center_x = val[pt1[i], 0] * factorx
            center_y = val[pt1[i], 1] * factory
            centerA  = np.asarray([center_x, center_y])

            center_x = val[pt2[i], 0] * factorx
            center_y = val[pt2[i], 1] * factory
            centerB  = np.asarray([center_x, center_y])

            paf_map, mask = create_paf_map(centerA, centerB, paf_width, paf_height, paf_width_thre)
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
    mask = np.logical_or.reduce((np.abs(paf_map[0, :, :]) > 0, np.abs(paf_map[0, :, :]) > 0))
    return paf_map, mask