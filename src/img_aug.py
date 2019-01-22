# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: img_aug.py
@time: 19-1-4 16:12
'''

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np

def img_aug_fuc(img, kps):
    '''
    function that used for data augmentation
    :param img: ori_image that was prepared to run aug, shape must be [h, w, c]
    :param kps: a ndarray that contains keypoints on this image. shape must be [person_num, joints_num, 3].
    person_num means that there contains 'person_num' person on this image
    joints_num means that we want to detect how many joints. e.g. like 1 for single head point or 14 for all body points in ai-challenger format.
    3 means [x, y, v], v is the visable attribute for one joint point.
    :return: img and kps after augmentation.
    '''

    # 定义一个变换序列, 包括
    # 1. 改变亮度
    # 2. (-90, 90)内旋转并缩放
    # 3. 随机变成灰度图
    # 4. 随机水平翻转
    # 5. 随机上下翻转
    seq = iaa.Sequential([
        iaa.Multiply((0.7, 1.5)),  # 改变亮度,不影响关键点 (后面可逐步增大至4, 5, 6)
        iaa.Affine(
            rotate=(-45, 45),
            scale=(0.7, 1.35),
            mode='constant'
        ),  # 旋转然后缩放
        iaa.Grayscale((0.0, 1.0)),  # 随机变成灰度图
        iaa.Fliplr(iap.Choice(a=[0, 1], p=[0.5, 0.5])),  # 50% 水平flip
        iaa.Flipud(iap.Choice(a=[0, 1], p=[0.8, 0.2])), # 20% vertical flip
    ])
    seq_det = seq.to_deterministic()

    # 定义keypoints
    keypoints = ia.KeypointsOnImage([], shape=img.shape)
    person_num, joints_num, _ = kps.shape
    for person in range(person_num):
        for joint in range(joints_num):
            point = kps[person][joint]
            keypoints.keypoints.append(ia.Keypoint(x=point[0], y=point[1]))

    # 执行augmentation
    img_aug = seq_det.augment_image(img)
    kps_aug = seq_det.augment_keypoints([keypoints])[0]

    # 返回augmentation之后的keypoints
    ret_kps = []
    for i in range(len(keypoints.keypoints)):
        point_after = kps_aug.keypoints[i]
        ret_kps.append([point_after.x, point_after.y, 1])
    ret_kps = np.reshape(np.asarray(ret_kps), newshape=(-1, joints_num, 3))

    if joints_num > 2:
        import copy
        # exchange left and right point
        # [ 0-right_shoulder, 1-right_elbow, 2-right_wrist, 3-left_shoulder, 4-left_elbow, 5-left_wrist,
        # 6-right_hip, 7-right_knee, 8-right_ankle, 9-left_hip, 10-left_knee, 11-left_ankle, 12-head, 13-neck ]
        change_index = [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]]
        for person in range(person_num):
            for index in change_index:
                left_point = copy.copy(ret_kps[person][index[0]])
                ret_kps[person][index[0]] = ret_kps[person][index[1]]
                ret_kps[person][index[1]] = left_point

    assert img_aug.shape == img.shape
    assert ret_kps.shape == kps.shape

    # import cv2
    # image_before = keypoints.draw_on_image(img, size=7)
    # # image_after = kps_aug.draw_on_image(img_aug, size=7)
    # cv2.imshow('before', image_before)
    # cv2.waitKey(0)
    # for person in range(person_num):
    #     for joint in range(joints_num):
    #         point = ret_kps[person][joint]
    #         cv2.circle(img_aug, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
    #         cv2.putText(img_aug, str(joint), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    # cv2.imshow('after', img_aug)
    # cv2.waitKey(0)

    return img_aug, ret_kps