#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: data_aug.py
@time: 2019/8/12 下午7:48
@desc:
'''
import numpy as np
import random
import copy
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap

def data_aug(img, bboxs=None, keypoints=None):
    '''
    :param img: 需要进行数据增强的图像
    :param bboxs: list, [ [x1, y1, x2, y2], ..., [xn1, yn1, xn2, yn2] ]
    :param keypoints: 关键点, COCO format or Ai-challenger format, ndarray, shape = [N, 14, 3]
    :return:
    '''

    is_flip = [random.randint(0, 1), random.randint(0, 1)]
    seq = iaa.Sequential([
        iaa.Multiply((0.7, 1.5)),
        iaa.Grayscale(iap.Choice(a=[0, 1], p=[0.8, 0.2]), from_colorspace='BGR'),
        iaa.Fliplr(is_flip[0]),
        iaa.Flipud(is_flip[1]),
        iaa.Affine(rotate=(-15, 15), scale=(0.8, 1.2), mode='constant'),
    ])

    seq_det = seq.to_deterministic()
    bbs = None
    kps = None

    if bboxs is not None:
        assert type(bboxs) == type([])
        bbs = ia.BoundingBoxesOnImage([], shape=img.shape)
        for box in bboxs:
            bbs.bounding_boxes.append(ia.BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))

    if keypoints is not None:
        kps = ia.KeypointsOnImage([], shape=img.shape)
        for j in range(keypoints.shape[0]):
            single_person_keypoints = keypoints[j]
            for i in range(keypoints.shape[1]):
                joint = single_person_keypoints[i]
                kps.keypoints.append(ia.Keypoint(x=joint[0], y=joint[1]))

    img_aug = seq_det.augment_image(img)
    if bbs is not None:
        bbs_aug = seq_det.augment_bounding_boxes(bbs)
        bboxs = []
        for i in range(len(bbs_aug.bounding_boxes)):
            box_aug = bbs_aug.bounding_boxes[i]
            box = [box_aug.x1, box_aug.y1, box_aug.x2, box_aug.y2]
            bboxs.append(box)

    if kps is not None:
        kps_aug = seq_det.augment_keypoints(kps)
        kps_ori = copy.copy(keypoints)
        keypoints = []
        for i in range(len(kps_aug.keypoints)):
            point = kps_aug.keypoints[i]
            keypoints.append([point.x, point.y, 1])
        keypoints = np.reshape(np.asarray(keypoints), newshape=kps_ori.shape)
        # keep ori keypoint visiable attribute
        for i in range(kps_ori.shape[0]):
            for joint in range(kps_ori.shape[1]):
                keypoints[i][joint][2] = kps_ori[i][joint][2]

        # if flip, change keypoint order (left <-> right)
        # ai-format: [ 0-right_shoulder, 1-right_elbow, 2-right_wrist,
        #              3-left_shoulder, 4-left_elbow, 5-left_wrist,
        #              6-right_hip, 7-right_knee, 8-right_ankle,
        #              9-left_hip, 10-left_knee, 11-left_ankle,
        #              12-head, 13-neck ]
        # coco-format: TODO add coco-foramt change index
        change_index = [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]]
        for flip in is_flip:
            if flip:
                for i in range(kps_ori.shape[0]):
                    for index in change_index:
                        right_point = copy.copy(keypoints[i][index[0]])
                        keypoints[i][index[0]] = keypoints[i][index[1]]
                        keypoints[i][index[1]] = right_point

    # test
    # import cv2
    # if bbs is not None:
    #     img_before = bbs.draw_on_image(img, color=(0, 255, 0), thickness=2)
    #     img_after = bbs_aug.draw_on_image(img_aug, color=(0,0,255), thickness=2)
    #     cv2.imshow('box ori', img_before)
    #     cv2.imshow('box after', img_after)
    #     cv2.waitKey(0)
    # if kps is not None:
    #     img_before = kps.draw_on_image(img, color=(0, 255, 0), size=5)
    #     img_after = kps_aug.draw_on_image(img_aug, color=(0, 0, 255), size=5)
    #     for i in range(kps_ori.shape[0]):
    #         for joint in range(kps_ori.shape[1]):
    #             point = kps_ori[i][joint]
    #             cv2.putText(img_before, str(joint), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 250), 1)
    #             point = keypoints[i][joint]
    #             cv2.putText(img_after, str(joint), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 250), 1)
    #     cv2.imshow('kps ori', img_before)
    #     cv2.imshow('kps after', img_after)
    #     cv2.waitKey(0)

    if bboxs is None and keypoints is not None:
        return img_aug, keypoints
    if bboxs is not None and keypoints is None:
        return img_aug, bboxs
    return img_aug, bboxs, keypoints