# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: img_preprocess.py
@time: 18-12-21 15:18
'''

import cv2
import numpy as np

def rotate(img, kps, degree):

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

def flip(img, kps, code=1):

    # flip image horizontially
    if code == 1:
        h, w, c = img.shape
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

    return img, kps

def aug_scale_pad(img, kps, scale):
    height, width, channel = img.shape
    size = int(max(height, width)*scale)

    # compute pad size of every side
    pad_top = int((size - height)/2)
    pad_down = size - height - pad_top
    pad_left = int((size - width)/2)
    pad_right = size - width - pad_left

    img = cv2.copyMakeBorder(img, pad_top, pad_down, pad_left, pad_right, cv2.BORDER_REPLICATE)

    keypoints_return = {}
    for key, val in kps.items():
        keypoints_return[key] = val + np.asarray([pad_left, pad_top, 0])

    return img, keypoints_return
