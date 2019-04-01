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
import copy
import cv2

def img_aug_fuc(img, kps, bboxs = None):
    '''
    function that used for data augmentation
    :param img: ori_image that was prepared to run aug, shape must be [h, w, c]
    :param kps: a ndarray that contains keypoints on this image. shape must be [person_num, joints_num, 3].
    person_num means that there contains 'person_num' person on this image
    joints_num means that we want to detect how many joints. e.g. like 1 for single head point or 14 for all body points in ai-challenger format.
    3 means [x, y, v], v is the visable attribute for one joint point.
    :param bboxs:  a list of lists. [[xmin, ymin, xmax, ymax], ...]
    :return: img , kps, bboxs after augmentation.
    '''

    kps_ori = np.copy(kps)

    # 定义一个变换序列, 包括
    # 1. 改变亮度
    # 2. (-90, 90)内旋转并缩放
    # 3. 随机变成灰度图
    # 4. 随机水平翻转
    # 5. 随机上下翻转
    p_flip = np.random.randint(0, 2, (2))
    # p_flip = [1, 0]
    seq = iaa.Sequential([
        iaa.Fliplr(p_flip[0]),  # 50% 水平flip
        iaa.Flipud(p_flip[1]),  # 50% vertical flip
        iaa.Multiply((0.7, 1.5)),  # 改变亮度,不影响关键点 (后面可逐步增大至4, 5, 6)
        iaa.Affine(
            rotate=(-45, 45),
            scale=(0.5, 1.5),
            mode='constant'
        ),  # 旋转然后缩放
        iaa.Grayscale((0.0, 1.0)),  # 随机变成灰度图
    ])
    seq_det = seq.to_deterministic()

    # 定义keypoints
    keypoints = ia.KeypointsOnImage([], shape=img.shape)
    person_num, joints_num, _ = kps.shape
    for person in range(person_num):
        for joint in range(joints_num):
            point = kps[person][joint]
            keypoints.keypoints.append(ia.Keypoint(x=point[0], y=point[1]))

    # 定义bbox
    if bboxs:
        assert type(bboxs) == type([])
        bbs = ia.BoundingBoxesOnImage([], shape=img.shape)
        for value in bboxs:
            bbs.bounding_boxes.append(ia.BoundingBox(x1=value[0], y1=value[1], x2=value[2], y2=value[3]))
    # 执行augmentation
    img_aug = seq_det.augment_image(img).copy()
    kps_aug = seq_det.augment_keypoints([keypoints])[0]
    bboxs_ret = []
    if bboxs:
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
        for i in range(len(bbs_aug.bounding_boxes)):
            # oribox = bbs.bounding_boxes[i]
            boxaug = bbs_aug.bounding_boxes[i]
            box = [boxaug.x1, boxaug.y1, boxaug.x2, boxaug.y2]
            bboxs_ret.append(box)

            # print('{} -> {}'.format(oribox, boxaug))
        # test
        # print(bboxs_ret)
        # img_before = bbs.draw_on_image(img, thickness=2)
        # cv2.imshow('before', img_before)
        # img_after  = bbs_aug.draw_on_image(img_aug, thickness=2, color=[0,0,255])
        # cv2.imshow('after', img_after)
        # cv2.waitKey(0)

    # 返回augmentation之后的keypoints
    ret_kps = []
    for i in range(len(keypoints.keypoints)):
        point_after = kps_aug.keypoints[i]
        ret_kps.append([point_after.x, point_after.y, 1])
    ret_kps = np.reshape(np.asarray(ret_kps), newshape=(-1, joints_num, 3))

    assert img_aug.shape == img.shape
    assert ret_kps.shape == kps_ori.shape

    # keep ori keypoint visiable attribute
    for person in range(ret_kps.shape[0]):
        for joint in range(ret_kps.shape[1]):
            ret_kps[person][joint][2] = int(kps_ori[person][joint][2])
    # import cv2
    # img_test = img_aug.copy()
    # for person in range(ret_kps.shape[0]):
    #     for joint in range(ret_kps.shape[1]):
    #         point = ret_kps[person][joint]
    #         cv2.circle(img_test, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
    #         cv2.putText(img_test, str(joint), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    # cv2.imshow('after 0 ', img_test)
    # cv2.waitKey(0)

    if joints_num > 2:
        import copy
        # exchange left and right point
        # [ 0-right_shoulder, 1-right_elbow, 2-right_wrist, 3-left_shoulder, 4-left_elbow, 5-left_wrist,
        # 6-right_hip, 7-right_knee, 8-right_ankle, 9-left_hip, 10-left_knee, 11-left_ankle, 12-head, 13-neck ]
        change_index = [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]]
        for p in p_flip:
            if p == 1:
                for person in range(ret_kps.shape[0]):
                    for index in change_index:
                        left_point = copy.copy(ret_kps[person][index[0]])
                        ret_kps[person][index[0]] = ret_kps[person][index[1]]
                        ret_kps[person][index[1]] = left_point

    # image_before = keypoints.draw_on_image(img, size=7)
    # # image_after = kps_aug.draw_on_image(img_aug, size=7)
    # for person in range(ret_kps.shape[0]):
    #     for joint in range(ret_kps.shape[1]):
    #         point = kps_ori[person][joint]
    #         # print (point)
    #         cv2.circle(img, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
    #         cv2.putText(img, str(joint) + str(point[2]), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    #
    # cv2.imshow('before', img)
    # cv2.waitKey(0)
    # for person in range(ret_kps.shape[0]):
    #     for joint in range(ret_kps.shape[1]):
    #         point = ret_kps[person][joint]
    #         # print (point)
    #         cv2.circle(img_aug, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
    #         cv2.putText(img_aug, str(point[2]), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    # cv2.imshow('after', img_aug)
    # cv2.waitKey(0)

    # print (ret_kps)

    return img_aug, ret_kps, bboxs_ret
